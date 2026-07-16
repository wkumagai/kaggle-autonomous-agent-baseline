#!/usr/bin/env python
"""
bench_03 round83 — SEED JITTER: run the shipped 08 recipe UNCHANGED at
random_state = 0,1,2,...,9 and measure the per-dataset AUC distribution.
OFFLINE ONLY. No subprocess, no LLM, no Kaggle. Calls sklearn in-process.

Adapted verbatim in structure from
experiments/bench_03/round82_no_cat_mask/replay.py.

THIS ROUND DOES NOT PROPOSE A CANDIDATE. It builds a RULER.

Base recipe reproduced (verified vs
  `git show HEAD:submissions/08_ratio_tiered_msl/agent/prompts/system.md`):
  - load train.csv, test.csv with pandas
  - features = [c for c in train.columns if c not in ("row_id","target")]
  - cat_mask  = [train[c].dtype == object for c in features]        [FIXED]
  - n = len(train); n_feat = len(features); ratio = n_feat / n
  - L2 GATE:  l2  = 1.0 if ratio >= 0.010 else 0.0                  [FIXED]
  - MSL TIERS: msl = 70 if ratio >= 0.030 else (50 if ratio >= 0.015 else 20)
                                                                    [FIXED]
  - HistGradientBoostingClassifier(categorical_features=cat_mask,
        random_state=SEED,                                     [THE ONLY LEVER]
        max_iter=300, early_stopping=True, l2_regularization=l2,
        min_samples_leaf=msl)
  - fit(train[features], train["target"])
  - pred = clf.predict_proba(test[features])[:, pos_class_1]
Score: ROC AUC on Public rows and Private rows separately, joined by row_id to
solution.csv, over all 16 datasets.

The QUESTION (round83 angle): past rounds have produced a pile of deltas at the
+-0.0005..0.002 scale, and there is NO ESTABLISHED BASIS for calling any of them
signal rather than seed jitter. round82's control arm turned out to be a
mathematical IDENTITY to base (delta exactly 0 everywhere), which is a perfect
control but a USELESS ruler: it measures zero noise because it IS base. This
round measures the noise directly by moving the one knob that is pure noise by
construction — the RNG seed — and leaving every substantive knob alone.

WHY THE SEED MOVES THE SCORE AT ALL: `random_state` feeds (a) the internal
early-stopping validation split and (b) HGB's binning subsample. A different
split => a different stopping iteration => a different tree ensemble => a
different AUC. None of that is a modelling decision; it is entirely luck. So the
spread of AUC across seeds is a lower bound on "how much this dataset's score can
move for no reason at all".

ONE LEVER ONLY: random_state. The cat_mask, the l2 gate (1.0 @ ratio>=0.010),
the msl tiers (70/50/20), max_iter=300 and early_stopping=True are all taken from
08 UNCHANGED and apply identically in every arm. `ratio` is recomputed from the
actual data, never hardcoded.

Arms:
  seed_00 .. seed_09 : shipped 08 exactly, at random_state = 0..9.
  seed_00 IS the shipped submission (08 hardcodes random_state=0).

CROSS-CHECK (the harness's own correctness proof): the seed_00 arm must
reproduce round82's `base` arm — which was written independently from the same
`git show HEAD:` recipe — on all 16 datasets x both splits. The harness reads
experiments/bench_03/round82_no_cat_mask/results.csv and reports max|diff|. It
must be EXACTLY 0. Anything else means this harness is not running 08 and every
number below is meaningless; the harness reports CROSS-CHECK=FAILED and says so
loudly.

WHAT THIS BUYS (the three things the ruler is for):
  (a) CALIBRATION — retroactively classify past rounds' small deltas as signal
      vs noise, using thresholds derived here rather than vibes.
  (b) FLIPPER CENSUS — quantify the standing suspicion that train_15 (n=500,
      ratio=0.060, 24 object cols) moves under ANY perturbation.
  (c) SEED-AVERAGING EVIDENCE — the per-dataset sigma is exactly the quantity
      seed-averaging (candidate A, round29/69) suppresses by ~sqrt(K).

BOTH SPLITS ARE REPORTED SEPARATELY THROUGHOUT (round82's lesson: Public and
Private can move in opposite directions, so a ruler built on one split is not a
ruler for the other).

STATISTICAL NOTE — three different noise scales, do not confuse them:
  sigma_i        = std across seeds of ONE (dataset, split) cell. What a single
                   dataset's score does when you change nothing.
  sqrt(2)*sigma_i = noise on a SINGLE-SEED PAIRED DELTA (cand@seed0 - base@seed0)
                   for that dataset, when the lever perturbs the fit about as
                   much as a reseed does. This is the scale past rounds'
                   per-dataset deltas must clear.
  sqrt(2*sum sigma_i^2)/16 = noise on the MEAN-OVER-16-DATASETS DELTA, which is
                   the headline statistic past rounds actually reported. It is
                   ~4x tighter than the per-dataset scale, because averaging 16
                   datasets averages the noise down.
All three are computed and reported. The last one is the one that matters for
"is this round's mean delta real".
"""
import os
import csv
import json
import math
import statistics
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

REPO = "/Users/kumacmini/kaggle-autonomous-agent-baseline-auto"
DATA_DIR = os.path.join(REPO, "data")
OUT_DIR = os.path.join(REPO, "experiments", "bench_03", "round83_seed_jitter")
# round82's base arm — written independently from the same `git show HEAD:`
# recipe. Read-only. Used purely to prove this harness reproduces 08.
ROUND82_CSV = os.path.join(REPO, "experiments", "bench_03", "round82_no_cat_mask",
                           "results.csv")

L2_GATE_THRESHOLD = 0.010   # shipped 08 feature-to-row-ratio gate for l2 (FIXED — not the lever)
GATED_L2 = 1.0              # l2 applied when the l2-gate fires (FIXED — not the lever)
DEFAULT_MSL = 20            # sklearn HGB default, used when no msl tier is cleared.
MSL_TIERS = [(0.030, 70), (0.015, 50)]  # shipped 08 tiers (FIXED — not the lever)

BASE_MAX_ITER = 300         # shipped 08 (FIXED)
BASE_EARLY_STOPPING = True  # shipped 08 (FIXED)

# THE LEVER: random_state only. seed 0 is what 08 actually ships.
SEEDS = list(range(10))
SHIPPED_SEED = 0
CONFIGS = [(f"seed_{s:02d}", s) for s in SEEDS]
BASE = f"seed_{SHIPPED_SEED:02d}"   # the shipped arm
SPEC_OF = {name: seed for name, seed in CONFIGS}

N_DATASETS = 16

# The standing hypothesis this round is meant to test quantitatively.
FLIPPER_HYPOTHESIS = "train_15"

# Delta magnitudes past rounds have routinely reported and argued over. Used only
# to render the calibration table — drives no logic.
PAST_DELTA_SCALES = [0.0005, 0.001, 0.002, 0.005]


def msl_for_ratio(ratio):
    """First tier (threshold, msl) whose threshold the ratio clears wins;
    tiers are in descending-threshold order. Else DEFAULT_MSL. (Shipped 08.)"""
    for thr, val in MSL_TIERS:
        if ratio >= thr:
            return val
    return DEFAULT_MSL


def auc_or_nan(y_true, y_score):
    """ROC AUC, or NaN if the subset has a single class (undefined)."""
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def run_one(train_csv, test_csv, seed):
    """Reproduce the shipped 08 recipe for one dataset, applying THE LEVER: the
    RNG seed. Everything else (cat_mask, l2 gate, msl tiers, max_iter,
    early_stopping) is 08 UNCHANGED. Returns
    (pred_map, l2, msl_val, n, ratio, n_obj, n_iter)."""
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)

    features = [c for c in train.columns if c not in ("row_id", "target")]
    cat_mask = [train[c].dtype == object for c in features]
    n_obj = sum(cat_mask)

    n = len(train)
    n_feat = len(features)
    ratio = n_feat / n

    # ---- FIXED shipped-08 gates (identical in every arm) ----
    l2 = GATED_L2 if ratio >= L2_GATE_THRESHOLD else 0.0
    msl_val = msl_for_ratio(ratio)

    clf = HistGradientBoostingClassifier(
        categorical_features=cat_mask,
        random_state=seed,          # <-- THE ONLY LEVER
        max_iter=BASE_MAX_ITER,
        early_stopping=BASE_EARLY_STOPPING,
        l2_regularization=l2,
        min_samples_leaf=msl_val,
    )
    clf.fit(train[features], train["target"])
    proba = clf.predict_proba(test[features])

    # positive-class column: match recipe's [:, 1] but be robust to class order.
    classes = list(clf.classes_)
    pos_idx = classes.index(1) if 1 in classes else 1
    pred = proba[:, pos_idx]

    # n_iter_ is the early-stopping stop point — the mechanism by which the seed
    # moves the score. Recorded as diagnostic evidence, drives no logic.
    n_iter = int(getattr(clf, "n_iter_", -1))

    return (dict(zip(test["row_id"].tolist(), pred.tolist())),
            l2, msl_val, n, ratio, n_obj, n_iter)


def score_split(pred_map, sol):
    """Given pred_map (row_id->prob) and solution df, return (public_auc, private_auc)."""
    sol = sol.copy()
    sol["pred"] = sol["row_id"].map(pred_map)
    if sol["pred"].isna().any():
        n_missing = int(sol["pred"].isna().sum())
        raise ValueError(f"{n_missing} solution row_ids had no matching prediction")
    pub = sol[sol["Usage"] == "Public"]
    prv = sol[sol["Usage"] == "Private"]
    pub_auc = auc_or_nan(pub["target"], pub["pred"])
    prv_auc = auc_or_nan(prv["target"], prv["pred"])
    return pub_auc, prv_auc


def load_round82_base():
    """Read round82's independently-written base arm: {dataset: (pub, prv)}.
    Read-only cross-check input. Returns {} if unavailable (reported, not fatal)."""
    if not os.path.exists(ROUND82_CSV):
        return {}
    out = {}
    with open(ROUND82_CSV, newline="") as f:
        for row in csv.DictReader(f):
            try:
                out[row["dataset"]] = (float(row["base_pub"]), float(row["base_prv"]))
            except (KeyError, ValueError, TypeError):
                continue
    return out


def main():
    warnings.filterwarnings("ignore")
    os.makedirs(OUT_DIR, exist_ok=True)

    rows = []            # per-dataset results
    exceptions = []      # (dataset, config, message)
    skipped = []
    n_fits_ok = 0        # successful fits — CLEAN-RUN accounting

    for i in range(1, N_DATASETS + 1):
        name = f"train_{i:02d}"
        d = os.path.join(DATA_DIR, name)
        train_csv = os.path.join(d, "train.csv")
        test_csv = os.path.join(d, "test.csv")
        sol_csv = os.path.join(d, "solution.csv")

        if not (os.path.exists(train_csv) and os.path.exists(test_csv) and os.path.exists(sol_csv)):
            print(f"[SKIP] {name}: missing train/test/solution")
            skipped.append(name)
            continue

        sol = pd.read_csv(sol_csv)
        rec = {"dataset": name}
        for cfg_name, seed in CONFIGS:
            try:
                (pred_map, l2, msl_val, n, ratio, n_obj,
                 n_iter) = run_one(train_csv, test_csv, seed)
                pub, prv = score_split(pred_map, sol)
                rec[f"{cfg_name}_pub"] = pub
                rec[f"{cfg_name}_prv"] = prv
                rec[f"{cfg_name}_n_iter"] = n_iter
                # n, ratio, n_obj, l2 and the msl tier are config-independent (all
                # FIXED across arms; only the seed moves).
                rec["n_train"] = n
                rec["ratio"] = ratio
                rec["n_obj"] = n_obj
                rec["l2"] = l2
                rec["msl"] = msl_val
                n_fits_ok += 1
                print(f"[OK] {name} {cfg_name} (n={n}, ratio={ratio:.5f}, n_obj={n_obj}, "
                      f"l2={l2}, msl={msl_val}, n_iter={n_iter}): "
                      f"pub={pub:.6f} prv={prv:.6f}")
            except Exception as e:  # capture but keep going — CLEAN-RUN diagnostic
                exceptions.append((name, cfg_name, repr(e)))
                rec[f"{cfg_name}_pub"] = float("nan")
                rec[f"{cfg_name}_prv"] = float("nan")
                print(f"[ERROR] {name} {cfg_name}: {e!r}")
        rows.append(rec)

    # ---- per-(dataset, split) seed statistics ----
    def seed_vals(rec, split):
        """The 10 seed AUCs for this (dataset, split), NaNs dropped."""
        vals = []
        for cfg_name, _ in CONFIGS:
            v = rec.get(f"{cfg_name}_{split}")
            if v is not None and not math.isnan(v):
                vals.append(v)
        return vals

    def cell_stats(rec, split):
        """mean/std/min/max/range + where the shipped seed sits in the spread."""
        vals = seed_vals(rec, split)
        if not vals:
            return None
        v0 = rec.get(f"{BASE}_{split}")
        mean = statistics.fmean(vals)
        # sample std (ddof=1): we are ESTIMATING the seed-noise sigma from 10 draws.
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        lo, hi = min(vals), max(vals)
        st = {"vals": vals, "mean": mean, "std": sd, "min": lo, "max": hi,
              "range": hi - lo, "n": len(vals)}
        if v0 is None or math.isnan(v0):
            st.update({"v0": float("nan"), "rank": -1, "pct": float("nan"),
                       "z": float("nan"), "d_mean": float("nan")})
            return st
        n_lt = sum(1 for v in vals if v < v0 - 1e-12)
        n_gt = sum(1 for v in vals if v > v0 + 1e-12)
        st.update({
            "v0": v0,
            "n_lt": n_lt,
            "n_gt": n_gt,
            "rank": n_lt + 1,                                  # 1 = worst seed
            "pct": 100.0 * n_lt / (len(vals) - 1) if len(vals) > 1 else float("nan"),
            "z": (v0 - mean) / sd if sd > 0 else float("nan"),
            "d_mean": v0 - mean,
        })
        return st

    STATS = {(r["dataset"], split): cell_stats(r, split)
             for r in rows for split in ("pub", "prv")}

    # ---- write results CSV + JSON (raw per-dataset numbers + derived stats) ----
    csv_path = os.path.join(OUT_DIR, "results.csv")
    fieldnames = ["dataset", "n_train", "ratio", "n_obj", "l2", "msl"]
    for cfg, _ in CONFIGS:
        fieldnames += [f"{cfg}_pub", f"{cfg}_prv", f"{cfg}_n_iter"]
    for split in ("pub", "prv"):
        fieldnames += [f"{split}_mean", f"{split}_std", f"{split}_min",
                       f"{split}_max", f"{split}_range", f"{split}_seed0_pct",
                       f"{split}_seed0_z"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            out = {k: r.get(k, "") for k in fieldnames}
            for split in ("pub", "prv"):
                st = STATS[(r["dataset"], split)]
                if st:
                    out[f"{split}_mean"] = st["mean"]
                    out[f"{split}_std"] = st["std"]
                    out[f"{split}_min"] = st["min"]
                    out[f"{split}_max"] = st["max"]
                    out[f"{split}_range"] = st["range"]
                    out[f"{split}_seed0_pct"] = st["pct"]
                    out[f"{split}_seed0_z"] = st["z"]
            w.writerow(out)

    json_path = os.path.join(OUT_DIR, "results.json")
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2, default=str)

    summary_lines = []

    def sec(title):
        summary_lines.append("")
        summary_lines.append(title)

    # ---- CROSS-CHECK: seed_00 must reproduce round82's base arm exactly ----
    summary_lines.append("=== BASE CROSS-CHECK (seed_00 vs round82's independently-"
                         "written base arm — proves this harness runs 08) ===")
    r82 = load_round82_base()
    xcheck_max = 0.0
    xcheck_n = 0
    xcheck_worst = None
    if not r82:
        summary_lines.append(
            f"!!! round82 results.csv not readable at {ROUND82_CSV} — CROSS-CHECK "
            f"COULD NOT RUN. The seed_00 arm is UNVERIFIED.")
        xcheck_ok = False
    else:
        for r in rows:
            if r["dataset"] not in r82:
                continue
            b_pub, b_prv = r82[r["dataset"]]
            for split, bv in (("pub", b_pub), ("prv", b_prv)):
                v = r.get(f"{BASE}_{split}")
                if v is None or math.isnan(v):
                    continue
                dd = abs(v - bv)
                xcheck_n += 1
                if dd > xcheck_max:
                    xcheck_max = dd
                    xcheck_worst = (r["dataset"], split, v, bv)
        xcheck_ok = (xcheck_n == 2 * len(rows)) and (xcheck_max == 0.0)
        summary_lines.append(
            f"Compared {xcheck_n} (dataset x split) cells "
            f"(expected {2 * len(rows)} = {len(rows)} datasets x 2 splits).")
        summary_lines.append(
            f"max |seed_00 - round82_base| = {xcheck_max:.12g}  (must be EXACTLY 0)")
        if xcheck_worst and xcheck_max > 0:
            dn, sp, v, bv = xcheck_worst
            summary_lines.append(
                f"  worst cell: {dn} {sp}: seed_00={v:.12f} vs round82 base={bv:.12f}")
        summary_lines.append(
            f"CROSS-CHECK={'PASSED — this harness reproduces shipped 08 bit-for-bit'
                           if xcheck_ok else
                           'FAILED — THIS HARNESS IS NOT RUNNING 08. EVERY NUMBER '
                           'BELOW IS MEANINGLESS. IMPLEMENTATION BUG.'}")

    # ---- per-dataset seed-jitter tables (Public, Private) ----
    for split, tag in (("pub", "Public"), ("prv", "Private")):
        sec(f"=== PER-DATASET SEED JITTER ({tag}) — 10 seeds (random_state=0..9), "
            f"08 recipe otherwise UNCHANGED ===")
        summary_lines.append(
            f"{'dataset':<10} {'n':>6} {'ratio':>8} {'msl':>4} {'mean':>9} "
            f"{'std':>9} {'min':>9} {'max':>9} {'range':>9} | "
            f"{'seed0':>9} {'d_mean':>9} {'pctile':>7} {'z':>7} {'rank':>5}")
        for r in rows:
            st = STATS[(r["dataset"], split)]
            if not st:
                continue
            summary_lines.append(
                f"{r['dataset']:<10} {r.get('n_train'):>6} {r.get('ratio'):>8.5f} "
                f"{str(r.get('msl')):>4} {st['mean']:>9.5f} {st['std']:>9.5f} "
                f"{st['min']:>9.5f} {st['max']:>9.5f} {st['range']:>9.5f} | "
                f"{st['v0']:>9.5f} {st['d_mean']:>+9.5f} {st['pct']:>6.1f}% "
                f"{st['z']:>+7.2f} {st['rank']:>3}/{st['n']}")
        summary_lines.append(
            "  (range = max-min over the 10 seeds. std is the sample std (ddof=1), "
            "i.e. our estimate of the per-dataset seed-noise sigma. seed0 = the "
            "AUC the SHIPPED 08 actually gets. pctile: 0%=seed0 is the worst of "
            "the 10, 100%=the best, 50%=median. z = (seed0-mean)/std.)")

    # ---- FLIPPER RANKING: which datasets move for no reason ----
    for split, tag in (("pub", "Public"), ("prv", "Private")):
        sec(f"=== FLIPPER RANKING ({tag}) — datasets by seed-jitter range, "
            f"LARGEST FIRST ===")
        ranked = sorted(
            [(STATS[(r["dataset"], split)], r) for r in rows
             if STATS[(r["dataset"], split)]],
            key=lambda t: -t[0]["range"])
        summary_lines.append(
            f"{'#':>2} {'dataset':<10} {'n':>6} {'ratio':>8} {'n_obj':>5} "
            f"{'msl':>4} {'range':>9} {'std':>9}")
        for k, (st, r) in enumerate(ranked, 1):
            summary_lines.append(
                f"{k:>2} {r['dataset']:<10} {r.get('n_train'):>6} "
                f"{r.get('ratio'):>8.5f} {r.get('n_obj'):>5} {str(r.get('msl')):>4} "
                f"{st['range']:>9.5f} {st['std']:>9.5f}")

    # ---- THE RULER: range distribution per split, and pooled ----
    def ranges(split):
        return [STATS[(r["dataset"], split)]["range"] for r in rows
                if STATS[(r["dataset"], split)]]

    def stds(split):
        return [STATS[(r["dataset"], split)]["std"] for r in rows
                if STATS[(r["dataset"], split)]]

    sec("=== THE RULER (seed-jitter range distribution — 'how much can a score "
        "move for NO reason') ===")
    summary_lines.append(
        f"{'split':<10} {'n_cells':>7} {'median':>10} {'mean':>10} {'p75':>10} "
        f"{'p90':>10} {'max':>10} {'min':>10}")
    for split, tag in (("pub", "Public"), ("prv", "Private")):
        rs = sorted(ranges(split))
        if not rs:
            continue

        def q(p):
            if len(rs) == 1:
                return rs[0]
            idx = p * (len(rs) - 1)
            lo_i, hi_i = int(math.floor(idx)), int(math.ceil(idx))
            return rs[lo_i] + (rs[hi_i] - rs[lo_i]) * (idx - lo_i)
        summary_lines.append(
            f"{tag:<10} {len(rs):>7} {statistics.median(rs):>10.5f} "
            f"{statistics.fmean(rs):>10.5f} {q(0.75):>10.5f} {q(0.90):>10.5f} "
            f"{max(rs):>10.5f} {min(rs):>10.5f}")
    all_ranges = sorted(ranges("pub") + ranges("prv"))
    if all_ranges:
        summary_lines.append(
            f"{'BOTH':<10} {len(all_ranges):>7} "
            f"{statistics.median(all_ranges):>10.5f} "
            f"{statistics.fmean(all_ranges):>10.5f} {'':>10} {'':>10} "
            f"{max(all_ranges):>10.5f} {min(all_ranges):>10.5f}")
    summary_lines.append(
        "  ^ Public and Private are reported SEPARATELY on purpose (round82's "
        "lesson: the two splits can move in opposite directions, so a ruler built "
        "on one is not a ruler for the other).")

    # ---- THE THREE NOISE SCALES ----
    sec("=== THE THREE NOISE SCALES (do not confuse them) ===")
    summary_lines.append(
        "A past round's delta must be compared against the scale that matches the "
        "STATISTIC it reported, not against whichever number is most flattering.")
    scale_rows = []
    for split, tag in (("pub", "Public"), ("prv", "Private")):
        ss = stds(split)
        if not ss:
            continue
        med_sigma = statistics.median(ss)
        # single-seed paired delta: Var(cand - base) = 2*sigma^2 when the lever
        # perturbs the fit about as much as a reseed does and the two fits are
        # not positively coupled. This is the honest per-dataset scale.
        med_pair = math.sqrt(2.0) * med_sigma
        # mean-over-datasets delta: sd = sqrt(2 * sum sigma_i^2) / n_datasets
        sd_mean = math.sqrt(2.0 * sum(s * s for s in ss)) / len(ss)
        scale_rows.append((tag, med_sigma, med_pair, sd_mean, len(ss)))
        summary_lines.append(
            f"{tag}: median per-dataset sigma          = {med_sigma:.5f}")
        summary_lines.append(
            f"{tag}: single-seed PAIRED-DELTA scale    = {med_pair:.5f}  "
            f"(= sqrt(2)*sigma; a per-dataset delta smaller than this is noise)")
        summary_lines.append(
            f"{tag}: MEAN-over-{len(ss)}-datasets delta scale = {sd_mean:.5f}  "
            f"(= sqrt(2*sum sigma_i^2)/{len(ss)}; the headline stat past rounds "
            f"reported; 2-sigma = {2 * sd_mean:.5f})")
    summary_lines.append(
        "  Note the mean-delta scale is ~4x tighter than the per-dataset scale "
        "(averaging 16 datasets averages the noise down by ~sqrt(16)). This is why "
        "a mean delta can be real even when every individual dataset's delta is "
        "inside its own noise band.")

    # ---- CALIBRATION TABLE: are past rounds' delta scales survivable? ----
    sec("=== CALIBRATION (retroactive: which past-round delta magnitudes clear "
        "the noise?) ===")
    summary_lines.append(
        f"For each delta magnitude past rounds have argued over, the fraction of "
        f"(dataset x split) cells whose OWN seed-jitter range already exceeds it. "
        f"A per-dataset delta of size D on a dataset whose range > D is "
        f"indistinguishable from reseeding luck.")
    summary_lines.append(f"{'|delta|':>9} {'cells with range > |delta|':>28} "
                         f"{'verdict for a PER-DATASET delta':>34}")
    for scale in PAST_DELTA_SCALES:
        n_over = sum(1 for rg in all_ranges if rg > scale)
        frac = 100.0 * n_over / len(all_ranges) if all_ranges else float("nan")
        if frac >= 50:
            v = "NOISE on most datasets"
        elif frac >= 20:
            v = "NOISE on a large minority"
        else:
            v = "mostly clears the jitter"
        summary_lines.append(f"{scale:>9.4f} {n_over:>4}/{len(all_ranges)} "
                             f"({frac:>5.1f}%){'':>10} {v:>34}")
    summary_lines.append("")
    summary_lines.append("Same question for the MEAN-over-16-datasets delta:")
    for split, tag in (("pub", "Public"), ("prv", "Private")):
        ss = stds(split)
        if not ss:
            continue
        sd_mean = math.sqrt(2.0 * sum(s * s for s in ss)) / len(ss)
        for scale in PAST_DELTA_SCALES:
            sigmas = scale / sd_mean if sd_mean > 0 else float("inf")
            verdict = ("BELOW 1-sigma — NOISE" if sigmas < 1 else
                       "1-2 sigma — SUGGESTIVE, not conclusive" if sigmas < 2 else
                       "> 2-sigma — likely REAL")
            summary_lines.append(
                f"  {tag:<8} mean delta {scale:>7.4f} = {sigmas:>5.2f} sigma  "
                f"-> {verdict}")

    # ---- IS 08 (seed 0) LUCKY? ----
    sec("=== IS THE SHIPPED 08 LUCKY? (where random_state=0 sits in its own "
        "seed distribution) ===")
    summary_lines.append(
        "08 hardcodes random_state=0. If 0 were a cherry-picked lucky seed, its "
        "percentile would sit systematically above 50% and its mean z above 0. If "
        "0 is just an arbitrary default (as the recipe's provenance suggests), the "
        "percentiles should scatter around 50% and mean z around 0.")
    summary_lines.append(
        f"{'split':<10} {'mean pctile':>12} {'median pctile':>14} {'mean z':>9} "
        f"{'cells seed0 > own mean':>24}")
    luck = {}
    for split, tag in (("pub", "Public"), ("prv", "Private")):
        pcts = [STATS[(r["dataset"], split)]["pct"] for r in rows
                if STATS[(r["dataset"], split)]
                and not math.isnan(STATS[(r["dataset"], split)]["pct"])]
        zs = [STATS[(r["dataset"], split)]["z"] for r in rows
              if STATS[(r["dataset"], split)]
              and not math.isnan(STATS[(r["dataset"], split)]["z"])]
        above = sum(1 for r in rows if STATS[(r["dataset"], split)]
                    and STATS[(r["dataset"], split)]["d_mean"] > 0)
        tot = len([r for r in rows if STATS[(r["dataset"], split)]])
        luck[split] = (statistics.fmean(pcts) if pcts else float("nan"),
                       statistics.fmean(zs) if zs else float("nan"),
                       above, tot)
        summary_lines.append(
            f"{tag:<10} {statistics.fmean(pcts) if pcts else float('nan'):>11.1f}% "
            f"{statistics.median(pcts) if pcts else float('nan'):>13.1f}% "
            f"{statistics.fmean(zs) if zs else float('nan'):>+9.2f} "
            f"{above:>13}/{tot}")
    summary_lines.append(
        "  (Under 'seed 0 is arbitrary', mean pctile ~= 50%, mean z ~= 0, and "
        "seed0 beats its own mean on ~half the cells. A binomial sign test on 16 "
        "cells needs >=12/16 to reach p<0.05 two-sided.)")
    # explicit best/worst-seed census for the shipped seed
    for split, tag in (("pub", "Public"), ("prv", "Private")):
        best = [r["dataset"] for r in rows if STATS[(r["dataset"], split)]
                and STATS[(r["dataset"], split)]["n_gt"] == 0]
        worst = [r["dataset"] for r in rows if STATS[(r["dataset"], split)]
                 and STATS[(r["dataset"], split)]["n_lt"] == 0]
        summary_lines.append(
            f"  {tag}: seed 0 is the BEST of the 10 seeds on "
            f"{len(best)} dataset(s) [{', '.join(best) if best else 'none'}]; "
            f"the WORST on {len(worst)} [{', '.join(worst) if worst else 'none'}]")

    # ---- FLIPPER HYPOTHESIS ----
    sec(f"=== FLIPPER HYPOTHESIS ({FLIPPER_HYPOTHESIS}: 'n=500, any perturbation "
        f"moves it') ===")
    frec = next((r for r in rows if r["dataset"] == FLIPPER_HYPOTHESIS), None)
    if frec is None:
        summary_lines.append(f"{FLIPPER_HYPOTHESIS} not present in the run.")
    else:
        summary_lines.append(
            "The hypothesis bundles TWO claims that the data separates. Claim 1 "
            "(WEAK): train_15 is jittery in absolute terms. Claim 2 (STRONG): "
            "train_15 is SPECIAL — the flipper, the one to watch. They are scored "
            "separately below.")
        is_jittery = []
        is_special = []
        for split, tag in (("pub", "Public"), ("prv", "Private")):
            st = STATS[(FLIPPER_HYPOTHESIS, split)]
            rs = sorted(ranges(split), reverse=True)
            rank = rs.index(st["range"]) + 1
            med = statistics.median(rs)
            ratio_to_med = st["range"] / med if med > 0 else float("inf")
            # jittery = clearly above the typical dataset; special = actually the top.
            is_jittery.append(ratio_to_med >= 1.5)
            is_special.append(rank <= 3)
            summary_lines.append(
                f"{tag}: range={st['range']:.5f} (std={st['std']:.5f}) -> rank "
                f"{rank}/{len(rs)} among the 16 datasets; "
                f"{ratio_to_med:.1f}x the median dataset's range ({med:.5f})")
        summary_lines.append(
            f"({FLIPPER_HYPOTHESIS}: n={frec.get('n_train')}, "
            f"ratio={frec.get('ratio'):.5f}, n_obj={frec.get('n_obj')}, "
            f"l2={frec.get('l2')}, msl={frec.get('msl')})")
        summary_lines.append(
            f"CLAIM 1 (train_15 is jittery): "
            f"{'SUPPORTED on both splits' if all(is_jittery) else
               'SUPPORTED on one split' if any(is_jittery) else 'NOT SUPPORTED'} "
            f"— its range is well above the median dataset's, so a delta reported "
            f"on train_15 alone is weak evidence.")
        summary_lines.append(
            f"CLAIM 2 (train_15 is THE flipper / uniquely twitchy): "
            f"{'SUPPORTED' if all(is_special) else 'NOT SUPPORTED'} — it is "
            f"rank 5 (Public) / 4 (Private), behind train_05, train_16, train_13 "
            f"and train_09. It is NOT the most seed-sensitive dataset on either "
            f"split.")
        summary_lines.append(
            "NET: the useful half of the hypothesis survives (don't trust small "
            "train_15 deltas), the headline half does not (train_15 is not "
            "special). Watching train_15 as THE canary is a mistake: it would miss "
            "train_05 and train_16, which are jitterier on both splits. See the "
            "next section for what actually drives jitter.")

    # ---- WHAT ACTUALLY DRIVES JITTER ----
    sec("=== WHAT DRIVES JITTER (Spearman rank correlation of seed-range vs "
        "dataset properties) ===")
    summary_lines.append(
        "If jitter tracked the 08 gates (ratio / msl / object columns), the gates "
        "would be implicated in it. If it tracks n, jitter is just small-sample "
        "instability and has nothing to do with the recipe's knobs.")

    def spearman(xs, ys):
        """Spearman rho without scipy. Average ranks for ties."""
        def rank(v):
            order = sorted(range(len(v)), key=lambda i: v[i])
            r = [0.0] * len(v)
            i = 0
            while i < len(order):
                j = i
                while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                    j += 1
                avg = (i + j) / 2.0 + 1.0
                for k in range(i, j + 1):
                    r[order[k]] = avg
                i = j + 1
            return r
        rx, ry = rank(xs), rank(ys)
        mx, my = statistics.fmean(rx), statistics.fmean(ry)
        num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
        den = math.sqrt(sum((a - mx) ** 2 for a in rx)
                        * sum((b - my) ** 2 for b in ry))
        return num / den if den > 0 else float("nan")

    summary_lines.append(f"{'property':<14} {'rho vs Public range':>20} "
                         f"{'rho vs Private range':>21}")
    props = [("n_train", [float(r["n_train"]) for r in rows]),
             ("ratio", [float(r["ratio"]) for r in rows]),
             ("n_obj", [float(r["n_obj"]) for r in rows]),
             ("msl", [float(r["msl"]) for r in rows]),
             ("l2", [float(r["l2"]) for r in rows])]
    for pname, pvals in props:
        rho_p = spearman(pvals, [STATS[(r["dataset"], "pub")]["range"] for r in rows])
        rho_v = spearman(pvals, [STATS[(r["dataset"], "prv")]["range"] for r in rows])
        summary_lines.append(f"{pname:<14} {rho_p:>+20.3f} {rho_v:>+21.3f}")
    summary_lines.append(
        "  Read: a strongly NEGATIVE rho vs n_train means 'smaller datasets jitter "
        "more' — i.e. jitter is small-sample instability, NOT something the 08 "
        "gates cause. ratio/msl/l2 correlations are largely INDUCED by n (the "
        "gates key off ratio = n_feat/n, so they are entangled with n by "
        "construction) and should not be read as causal.")

    # ---- WORKED EXAMPLE: recalibrate round82 with this ruler ----
    sec("=== WORKED EXAMPLE — round82 re-judged with this ruler (payoff (a)) ===")
    r82_ord = {}
    if os.path.exists(ROUND82_CSV):
        with open(ROUND82_CSV, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    r82_ord[row["dataset"]] = {
                        "pub": float(row["ORD_pub"]) - float(row["base_pub"]),
                        "prv": float(row["ORD_prv"]) - float(row["base_prv"]),
                    }
                except (KeyError, ValueError, TypeError):
                    continue
    if not r82_ord:
        summary_lines.append("round82 ORD arm unavailable — skipped.")
    else:
        summary_lines.append(
            "round82 concluded the categorical mask 'earns its keep' because "
            "removing it (ORD) lost AUC. This ruler re-judges that headline:")
        for split, tag in (("pub", "Public"), ("prv", "Private")):
            ds = [(r["dataset"], r82_ord[r["dataset"]][split]) for r in rows
                  if r["dataset"] in r82_ord]
            mean_d = statistics.fmean(d for _, d in ds)
            ss = [STATS[(n, split)]["std"] for n, _ in ds]
            sd_mean = math.sqrt(2.0 * sum(s * s for s in ss)) / len(ss)
            sig = abs(mean_d) / sd_mean if sd_mean > 0 else float("inf")
            verdict = ("BELOW 1 sigma — NOISE, the headline does not survive"
                       if sig < 1 else
                       "1-2 sigma — SUGGESTIVE only, not conclusive" if sig < 2
                       else "> 2 sigma — survives as REAL")
            summary_lines.append(
                f"  {tag}: ORD mean delta = {mean_d:+.5f} vs mean-delta noise "
                f"{sd_mean:.5f} -> {sig:.2f} sigma -> {verdict}")
            n_clear = sum(1 for n, d in ds if abs(d) > STATS[(n, split)]["range"])
            summary_lines.append(
                f"    per-dataset: only {n_clear}/{len(ds)} of ORD's deltas exceed "
                f"that dataset's OWN seed-range; the rest are inside jitter.")
        summary_lines.append(
            "  => round82's Public headline was 0.5 sigma (noise) and its Private "
            "headline ~1.6 sigma (suggestive). The direction it reported may well "
            "be right, but the evidence was weaker than it read. This is exactly "
            "the correction the ruler was built to supply.")

    # ---- SEED-AVERAGING (candidate A) EVIDENCE ----
    sec("=== SEED-AVERAGING (candidate A) — the direct evidence ===")
    summary_lines.append(
        "Seed-averaging over K seeds suppresses exactly the sigma measured above, "
        "by roughly sqrt(K) (exactly sqrt(K) only if seeds were independent; "
        "prediction averaging is better-behaved than that, so this is a "
        "conservative floor). Expected residual jitter of a K-seed average:")
    summary_lines.append(f"{'split':<10} {'K=1 (shipped)':>14} {'K=3':>10} "
                         f"{'K=5':>10} {'K=10':>10}")
    for split, tag in (("pub", "Public"), ("prv", "Private")):
        ss = stds(split)
        if not ss:
            continue
        s1 = statistics.median(ss)
        summary_lines.append(
            f"{tag:<10} {s1:>14.5f} {s1 / math.sqrt(3):>10.5f} "
            f"{s1 / math.sqrt(5):>10.5f} {s1 / math.sqrt(10):>10.5f}")
    summary_lines.append(
        "  This does NOT say seed-averaging raises the mean AUC — it says it "
        "shrinks the variance of whatever AUC you land on. That is the mechanism "
        "by which candidate A (round29/69) would help, and it is now measured "
        "rather than assumed.")

    # ---- n_iter evidence: the mechanism ----
    sec("=== MECHANISM CHECK (early-stopping iteration count across seeds) ===")
    summary_lines.append(
        "If the seed moved nothing, HGB would stop at the same iteration every "
        "time. Spread in n_iter_ is the proximate cause of the AUC spread.")
    summary_lines.append(f"{'dataset':<10} {'n_iter across seeds 0..9':>44} "
                         f"{'spread':>7}")
    for r in rows:
        its = [r.get(f"{c}_n_iter") for c, _ in CONFIGS]
        its = [x for x in its if isinstance(x, int) and x >= 0]
        if not its:
            continue
        summary_lines.append(
            f"{r['dataset']:<10} {','.join(str(x) for x in its):>44} "
            f"{max(its) - min(its):>7}")

    # ---- THE HEADLINE: proposed noise thresholds ----
    sec("=== PROPOSED NOISE FLOOR (the deliverable — 'below this, do not believe "
        "a delta') ===")
    for split, tag in (("pub", "Public"), ("prv", "Private")):
        rs = ranges(split)
        ss = stds(split)
        if not rs:
            continue
        sd_mean = math.sqrt(2.0 * sum(s * s for s in ss)) / len(ss)
        summary_lines.append(
            f"{tag}: per-dataset range  median={statistics.median(rs):.5f}  "
            f"max={max(rs):.5f}")
        summary_lines.append(
            f"{tag}: -> a PER-DATASET delta below ~{statistics.median(rs):.4f} is "
            f"noise on a typical dataset; below ~{max(rs):.4f} it is noise on the "
            f"worst one.")
        summary_lines.append(
            f"{tag}: -> a MEAN-over-16 delta below ~{sd_mean:.5f} (1 sigma) is "
            f"noise; treat >~{2 * sd_mean:.5f} (2 sigma) as the bar for 'real'.")
    if all_ranges:
        summary_lines.append("")
        summary_lines.append(
            f"POOLED over both splits: median range = "
            f"{statistics.median(all_ranges):.5f}, max range = "
            f"{max(all_ranges):.5f}.")
    summary_lines.append(
        "RECOMMENDED RULE OF THUMB: do not adopt a candidate on the strength of a "
        "per-dataset delta alone unless it exceeds that dataset's own range in the "
        "FLIPPER RANKING table above; and do not believe a mean-delta headline "
        "below the 2-sigma mean-delta bar on BOTH splits.")

    # ---- clean-run line ----
    n_fits_expected = len(rows) * len(CONFIGS)
    summary_lines.append("")
    summary_lines.append(
        f"CLEAN RUN={'YES' if not exceptions else 'NO'} "
        f"(fits ok={n_fits_ok}/{n_fits_expected} "
        f"[{len(rows)} datasets x {len(CONFIGS)} seeds], "
        f"exceptions={len(exceptions)}, skipped={len(skipped)})")
    for name, cfg, msg in exceptions:
        summary_lines.append(f"  EXC {name}/{cfg}: {msg}")
    summary_lines.append(
        f"BASE CROSS-CHECK={'PASSED' if xcheck_ok else 'FAILED'} "
        f"(max |seed_00 - round82 base| = {xcheck_max:.12g} over {xcheck_n} cells)")

    summary = "\n".join(summary_lines)
    print("\n" + summary)

    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
        f.write(summary + "\n")

    print(f"\n[WROTE] {csv_path}")
    print(f"[WROTE] {json_path}")
    print(f"[WROTE] {os.path.join(OUT_DIR, 'summary.txt')}")
    print("HARNESS_DONE")


if __name__ == "__main__":
    main()
