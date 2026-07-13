#!/usr/bin/env python
"""
bench_03 round32 — SEED-AVERAGING FIRING-GATE change: from ratio-based
(cand_C: n_feat/n >= 0.015) to a CATEGORICAL-SMALL-n gate
(n_train < 4000 AND n_object_cols > 0). OFFLINE ONLY. No subprocess, no LLM,
no Kaggle. Calls sklearn in-process.

Adapted from experiments/bench_03/round30_seed_k_sweep/replay.py (dataset
loading, the shipped-08 base config reproduction, Public/Private AUC scoring
joined on row_id to solution.csv, K=5 seed-averaging, and the summary machinery
are all reused verbatim). ONLY the seed-averaging FIRING GATE is changed, and
K is FIXED at 5 (round30 established K=5 as the knee; no K sweep here).

Base recipe reproduced (== shipped 08), IDENTICAL to round30 base:
  - load train.csv, test.csv with pandas
  - features = [c for c in train.columns if c not in ("row_id","target")]
  - cat_mask  = [train[c].dtype == object for c in features]
  - n = len(train); n_feat = len(features); ratio = n_feat / n
  - L2 GATE:  l2  = 1.0 if ratio >= 0.010 else 0.0
  - MSL GATE (08 tiered): msl = 70 if ratio >= 0.030
                          else 50 if ratio >= 0.015 else 20
  - HistGradientBoostingClassifier(categorical_features=cat_mask, random_state=s,
        max_iter=300, early_stopping=True, l2_regularization=l2,
        min_samples_leaf=msl)
  - fit(train[features], train["target"])
  - pred = clf.predict_proba(test[features])[:, pos_class_1]
Score: ROC AUC on Public rows and Private rows separately, joined by row_id to
solution.csv, over all 16 datasets.

The base predictions (l2/msl 2-gate, early_stopping, features) are UNCHANGED
from round30 / shipped 08. base = each dataset's single seed-0 prediction.

THE ONLY CHANGE — the seed-averaging firing gate:
  - round30 cand_C gate : ratio = n_feat/n >= 0.015  -> fires {09,13,15}
  - round32 cand gate   : n_train < 4000 AND n_object_cols > 0
      Reads n_train and n_object_cols per dataset from
      experiments/bench_03/dataset_stats.csv (NOT from ratio).
      Expected to fire {03,05,09,13,15} and to LEAVE OUT train_16
      (n=1809 but n_object_cols=0 -> non-firing), which was the only regression
      source ratio-tiered rounds struggled with. The firing set is logged at run
      time to confirm.

When the gate fires  -> cand = mean predict_proba over K=5 seeds (random_state
0..4) of the identical shipped-08 config.
When it does NOT fire -> cand = single seed-0 (byte-identical to base == 08).

IMPLEMENTATION INVARIANT: on every NON-firing dataset cand reuses the EXACT
seed-0 array, so its delta MUST be exactly 0 on both splits — checked
explicitly. On firing datasets cand's average includes seed-0 so it generally
differs from base (expected).

Adoption criterion (reused from prior rounds): a candidate config is a CLEAN
IMPROVEMENT over base(08) iff its mean delta is positive on BOTH splits AND
there are ZERO regressions on BOTH splits (no dataset worse on either split).
A single regression on either split => not clean.

Reference: cand_C (K=5, ratio>=0.015 gate, {09,13,15}) was mean Public +0.00169
/ Private +0.00151 with zero regressions. round32 is compared against this.
"""
import os
import csv
import math
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

REPO = "/Users/kumacmini/kaggle-autonomous-agent-baseline-auto"
DATA_DIR = os.path.join(REPO, "data")
BENCH_DIR = os.path.join(REPO, "experiments", "bench_03")
OUT_DIR = os.path.join(BENCH_DIR, "round32_gate_categorical")
STATS_CSV = os.path.join(BENCH_DIR, "dataset_stats.csv")

L2_GATE_THRESHOLD = 0.010   # shipped 08 feature-to-row-ratio gate for l2 (FIXED)
GATED_L2 = 1.0              # l2 applied when the l2-gate fires (FIXED)
DEFAULT_MSL = 20            # sklearn HGB default, used when no msl tier is cleared.
BASE_SEED = 0              # shipped-08 fixed random_state.

# round32 seed-averaging firing gate (THE ONLY CHANGE vs round30):
#   n_train < N_TRAIN_MAX AND n_object_cols > 0.
N_TRAIN_MAX = 4000         # "small n" threshold
MIN_OBJECT_COLS = 0        # require strictly more than this many object columns

# K is FIXED at 5 (round30 established K=5 as the knee; no sweep).
K = 5
SEEDS = list(range(K))     # [0,1,2,3,4]

# 08 tiered min_samples_leaf, IDENTICAL across base and cand (descending order).
MSL_TIERS = [(0.030, 70), (0.015, 50)]

BASE = "base"
CAND = "cand"              # K=5 seed-avg under the categorical-small-n gate
CANDIDATES = [CAND]

N_DATASETS = 16


def load_stats(path=STATS_CSV):
    """Return {dataset_name -> {"n_train": int, "n_object_cols": int}} from
    dataset_stats.csv. The gate reads n_train and n_object_cols from here."""
    stats = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stats[row["name"]] = {
                "n_train": int(row["n_train"]),
                "n_object_cols": int(row["n_object_cols"]),
            }
    return stats


def gate_fires(n_train, n_object_cols):
    """round32 firing rule: small n AND at least one object (categorical) col."""
    return (n_train < N_TRAIN_MAX) and (n_object_cols > MIN_OBJECT_COLS)


def msl_for_ratio(ratio, tiers=MSL_TIERS):
    """First tier (threshold, msl) whose threshold the ratio clears wins;
    tiers must be given in descending-threshold order. Else DEFAULT_MSL."""
    for thr, val in tiers:
        if ratio >= thr:
            return val
    return DEFAULT_MSL


def auc_or_nan(y_true, y_score):
    """ROC AUC, or NaN if the subset has a single class (undefined)."""
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def fit_one_seed(train, test, features, cat_mask, l2, msl_val, seed):
    """Fit ONE shipped-08 HGB with the given random_state=seed; return the
    positive-class probability vector aligned to test row order. All other
    hyperparameters are byte-identical to shipped 08."""
    clf = HistGradientBoostingClassifier(
        categorical_features=cat_mask,
        random_state=seed,
        max_iter=300,
        early_stopping=True,
        l2_regularization=l2,
        min_samples_leaf=msl_val,
    )
    clf.fit(train[features], train["target"])
    proba = clf.predict_proba(test[features])
    classes = list(clf.classes_)
    pos_idx = classes.index(1) if 1 in classes else 1
    return proba[:, pos_idx]


def run_one(name, train_csv, test_csv, stats):
    """Reproduce shipped-08 base + derive cand under the round32 gate for one
    dataset. Returns (preds, l2, l2_fired, fired, msl_val, n_fits,
    n_train_stat, n_obj_stat) where preds maps config_name -> {row_id -> prob}.
      base = seed-0 prediction (== shipped 08).
      On a FIRING dataset (gate_fires): fit seeds 0..4 once; cand = mean
        predict_proba over the 5 seeds.
      On a NON-firing dataset: only seed-0 is fit and cand reuses that exact
        seed-0 array (byte-identical to base)."""
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)

    features = [c for c in train.columns if c not in ("row_id", "target")]
    cat_mask = [train[c].dtype == object for c in features]

    n = len(train)
    n_feat = len(features)
    ratio = n_feat / n

    # base config gates (UNCHANGED from shipped 08 / round30).
    l2_fired = ratio >= L2_GATE_THRESHOLD
    l2 = GATED_L2 if l2_fired else 0.0
    msl_val = msl_for_ratio(ratio)

    # round32 seed-averaging gate reads stats (n_train, n_object_cols).
    st = stats[name]
    n_train_stat = st["n_train"]
    n_obj_stat = st["n_object_cols"]
    fired = gate_fires(n_train_stat, n_obj_stat)

    row_ids = test["row_id"].tolist()

    if fired:
        seed_preds = [
            fit_one_seed(train, test, features, cat_mask, l2, msl_val, s)
            for s in SEEDS
        ]
        n_fits = len(SEEDS)
        base_vec = seed_preds[BASE_SEED]  # seed-0 == base
        avg_vec = np.mean(np.vstack(seed_preds), axis=0)
        preds = {
            BASE: dict(zip(row_ids, base_vec.tolist())),
            CAND: dict(zip(row_ids, avg_vec.tolist())),
        }
    else:
        base_vec = fit_one_seed(train, test, features, cat_mask, l2, msl_val,
                                BASE_SEED)
        n_fits = 1
        base_map = dict(zip(row_ids, base_vec.tolist()))
        preds = {BASE: base_map, CAND: base_map}  # byte-identical to base

    return (preds, l2, l2_fired, fired, msl_val, n_fits, n_train_stat,
            n_obj_stat)


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


def main():
    warnings.filterwarnings("ignore")
    os.makedirs(OUT_DIR, exist_ok=True)

    stats = load_stats()

    rows = []            # per-dataset results
    exceptions = []      # (dataset, message)
    skipped = []
    total_fits = 0

    ALL_CONFIGS = [BASE] + CANDIDATES

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
        try:
            (preds, l2, l2_fired, fired, msl_val, n_fits, n_train_stat,
             n_obj_stat) = run_one(name, train_csv, test_csv, stats)
            total_fits += n_fits
            rec["l2_fired"] = l2_fired
            rec["fires"] = bool(fired)
            rec["n_train"] = n_train_stat
            rec["n_object_cols"] = n_obj_stat
            for cfg in ALL_CONFIGS:
                pub, prv = score_split(preds[cfg], sol)
                rec[f"{cfg}_pub"] = pub
                rec[f"{cfg}_prv"] = prv
            rec["msl"] = msl_val
            print(f"[OK] {name} (n_train={n_train_stat}, n_obj={n_obj_stat}, "
                  f"fires={fired}, l2={l2}, msl={msl_val}, fits={n_fits}): "
                  f"base pub={rec['base_pub']:.6f} prv={rec['base_prv']:.6f} | "
                  f"cand pub={rec['cand_pub']:.6f} prv={rec['cand_prv']:.6f}")
        except Exception as e:  # capture but keep going — CLEAN-RUN diagnostic
            exceptions.append((name, repr(e)))
            rec["l2_fired"] = False
            rec["fires"] = False
            rec["n_train"] = stats.get(name, {}).get("n_train", "")
            rec["n_object_cols"] = stats.get(name, {}).get("n_object_cols", "")
            for cfg in ALL_CONFIGS:
                rec[f"{cfg}_pub"] = float("nan")
                rec[f"{cfg}_prv"] = float("nan")
            rec["msl"] = float("nan")
            print(f"[ERROR] {name}: {e!r}")
        rows.append(rec)

    # ---- deltas & helpers (all vs base == shipped 08) ----
    def delta(rec, cfg, split):
        b = rec.get(f"{BASE}_{split}")
        c = rec.get(f"{cfg}_{split}")
        if b is None or c is None or math.isnan(b) or math.isnan(c):
            return float("nan")
        return c - b

    def mean_delta(cfg, split):
        vals = [delta(r, cfg, split) for r in rows]
        vals = [v for v in vals if not math.isnan(v)]
        return sum(vals) / len(vals) if vals else float("nan")

    def wlt(cfg, split, eps=1e-6):
        w = l = t = 0
        for r in rows:
            dd = delta(r, cfg, split)
            if math.isnan(dd):
                continue
            if dd > eps:
                w += 1
            elif dd < -eps:
                l += 1
            else:
                t += 1
        return w, l, t

    def regressions(cfg, split, eps=1e-6):
        """Datasets where cfg is strictly worse than base(08) on this split."""
        out = []
        for r in rows:
            dd = delta(r, cfg, split)
            if not math.isnan(dd) and dd < -eps:
                out.append((r["dataset"], dd))
        return out

    def differing_datasets(cfg, eps=1e-9):
        """Datasets where cfg differs from base on EITHER split."""
        out = []
        for r in rows:
            dp = delta(r, cfg, "pub")
            dv = delta(r, cfg, "prv")
            dp = 0.0 if math.isnan(dp) else dp
            dv = 0.0 if math.isnan(dv) else dv
            if abs(dp) > eps or abs(dv) > eps:
                out.append(r["dataset"])
        return out

    firing_list = [r["dataset"] for r in rows if r.get("fires")]

    # ---- write results CSV ----
    csv_path = os.path.join(OUT_DIR, "results.csv")
    fieldnames = ["dataset", "fires", "n_train", "n_object_cols", "l2_fired",
                  "msl", "base_pub", "base_prv"]
    for cfg in CANDIDATES:
        fieldnames += [f"{cfg}_pub", f"{cfg}_d_pub", f"{cfg}_prv", f"{cfg}_d_prv"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            out = {
                "dataset": r["dataset"],
                "fires": r.get("fires", ""),
                "n_train": r.get("n_train", ""),
                "n_object_cols": r.get("n_object_cols", ""),
                "l2_fired": r.get("l2_fired", ""),
                "msl": r.get("msl", ""),
                "base_pub": r.get("base_pub", ""),
                "base_prv": r.get("base_prv", ""),
            }
            for cfg in CANDIDATES:
                out[f"{cfg}_pub"] = r.get(f"{cfg}_pub", "")
                out[f"{cfg}_prv"] = r.get(f"{cfg}_prv", "")
                out[f"{cfg}_d_pub"] = delta(r, cfg, "pub")
                out[f"{cfg}_d_prv"] = delta(r, cfg, "prv")
            w.writerow(out)

    # ---- INVARIANT check: cand on any NON-firing dataset must be
    #      byte-identical to base (delta exactly 0). ----
    invariant_violations = []
    for cfg in CANDIDATES:
        for r in rows:
            if not r.get("fires"):
                dp = delta(r, cfg, "pub")
                dv = delta(r, cfg, "prv")
                dp = 0.0 if math.isnan(dp) else dp
                dv = 0.0 if math.isnan(dv) else dv
                if dp != 0.0 or dv != 0.0:
                    invariant_violations.append((cfg, r["dataset"], dp, dv))

    summary_lines = []

    # ---- per-dataset tables (Public, Private) ----
    for split, tag in (("pub", "Public"), ("prv", "Private")):
        if split == "prv":
            summary_lines.append("")
        summary_lines.append(f"=== PER-DATASET ({tag}) ===")
        header = (f"{'dataset':<10} {'fires':>5} {'nTr':>6} {'obj':>4} "
                  f"{'msl':>4} {BASE:>9}")
        for cfg in CANDIDATES:
            header += f" {cfg:>9} {'d'+cfg:>11}"
        summary_lines.append(header)
        for r in rows:
            line = (f"{r['dataset']:<10} {str(bool(r.get('fires'))):>5} "
                    f"{str(r.get('n_train')):>6} {str(r.get('n_object_cols')):>4} "
                    f"{str(r.get('msl')):>4} "
                    f"{r[f'{BASE}_{split}']:>9.4f}")
            for cfg in CANDIDATES:
                line += (f" {r[f'{cfg}_{split}']:>9.4f} "
                         f"{delta(r, cfg, split):>+11.5f}")
            summary_lines.append(line)

    # ---- gate firings / seed-averaging scope ----
    summary_lines.append("")
    summary_lines.append("=== SEED-AVERAGING SCOPE "
                         "(round32 gate: n_train<4000 AND n_object_cols>0, K=5) "
                         "===")
    summary_lines.append(
        f"seed-averaging fires on ({len(firing_list)}): "
        f"{', '.join(firing_list) if firing_list else '(none)'}. "
        f"Non-firing datasets = single seed-0 (byte-identical to base).")
    expected_fire = {"train_03", "train_05", "train_09", "train_13", "train_15"}
    fire_set = set(firing_list)
    fire_ok = (fire_set == expected_fire)
    summary_lines.append(
        f"Expected firing set {{03,05,09,13,15}} matched: "
        f"{'YES' if fire_ok else 'NO'}"
        + ("" if fire_ok else
           f" (got {sorted(fire_set)}, expected {sorted(expected_fire)})"))
    t16_nonfire = "train_16" not in fire_set
    summary_lines.append(
        f"train_16 (n=1809, n_object_cols=0) NON-firing (obj=0): "
        f"{'YES' if t16_nonfire else 'NO'}")

    # ---- INVARIANT report ----
    summary_lines.append("")
    summary_lines.append(
        "=== INVARIANT (cand on non-firing datasets identical to base, "
        "delta 0) ===")
    if invariant_violations:
        summary_lines.append("VIOLATED! cand differs from base on a non-firing "
                             "dataset:")
        for cfg, ds, dp, dv in invariant_violations:
            summary_lines.append(
                f"  {cfg}/{ds}: pub d={dp:+.6g} prv d={dv:+.6g}")
    else:
        n_nonfire = len([r for r in rows if not r.get("fires")])
        summary_lines.append(
            f"OK: each of the {n_nonfire} non-firing datasets is byte-identical "
            f"to base (delta exactly 0). Required base-reproduction check. "
            f"PASS.")

    # ---- which datasets actually differed ----
    summary_lines.append("")
    summary_lines.append("=== DATASETS THAT ACTUALLY DIFFER (candidate vs base) ===")
    for cfg in CANDIDATES:
        diff = differing_datasets(cfg)
        summary_lines.append(
            f"{cfg}: ({len(diff)}) {', '.join(diff) if diff else '(none)'}")

    # ---- per-candidate summary vs base(08) ----
    summary_lines.append("")
    summary_lines.append("=== SUMMARY (candidate vs base == shipped 08) ===")
    for cfg in CANDIDATES:
        mp = mean_delta(cfg, "pub")
        mv = mean_delta(cfg, "prv")
        wp, lp, tp = wlt(cfg, "pub")
        wv, lv, tv = wlt(cfg, "prv")
        summary_lines.append(
            f"{cfg}: mean Public d={mp:+.5f}  mean Private d={mv:+.5f}  "
            f"Public W/L/T={wp}/{lp}/{tp}  Private W/L/T={wv}/{lv}/{tv}")

    # ---- newly-included datasets spotlight (03, 05) ----
    summary_lines.append("")
    summary_lines.append("=== NEWLY-INCLUDED DATASETS (03, 05 — non-firing under "
                         "cand_C ratio>=0.015; now firing) ===")
    for name in ("train_03", "train_05"):
        r = next((x for x in rows if x["dataset"] == name), None)
        if r is None:
            summary_lines.append(f"  {name}: (missing)")
            continue
        dp = delta(r, CAND, "pub")
        dv = delta(r, CAND, "prv")
        summary_lines.append(
            f"  {name:<10} fires={bool(r.get('fires'))}  "
            f"pub {r[f'{BASE}_pub']:.4f}->{r[f'{CAND}_pub']:.4f} ({dp:+.5f})   "
            f"prv {r[f'{BASE}_prv']:.4f}->{r[f'{CAND}_prv']:.4f} ({dv:+.5f})")

    # ---- existing firing datasets (09,13,15) — cand_C K=5 reproduction ----
    summary_lines.append("")
    summary_lines.append("=== EXISTING FIRING DATASETS (09, 13, 15 — should "
                         "reproduce round30 K=5) ===")
    ref = {
        "train_09": (0.00985, 0.00634),
        "train_13": (0.00696, 0.00510),
        "train_15": (0.01016, 0.01268),
    }
    for name in ("train_09", "train_13", "train_15"):
        r = next((x for x in rows if x["dataset"] == name), None)
        if r is None:
            summary_lines.append(f"  {name}: (missing)")
            continue
        dp = delta(r, CAND, "pub")
        dv = delta(r, CAND, "prv")
        rp, rv = ref[name]
        summary_lines.append(
            f"  {name:<10} pub {dp:+.5f} (round30 K5 {rp:+.5f})   "
            f"prv {dv:+.5f} (round30 K5 {rv:+.5f})")

    # ---- per-candidate differing-dataset deltas + regressions ----
    summary_lines.append("")
    summary_lines.append("=== PER-CANDIDATE DETAIL (datasets differing from base; "
                         "all other deltas are exactly 0) ===")
    for cfg in CANDIDATES:
        summary_lines.append(f"--- {cfg} vs {BASE} ---")
        diff = set(differing_datasets(cfg))
        for r in rows:
            if r["dataset"] not in diff:
                continue
            dp = delta(r, cfg, "pub")
            dv = delta(r, cfg, "prv")
            summary_lines.append(
                f"  {r['dataset']:<10} "
                f"pub {r[f'{BASE}_pub']:.4f}->{r[f'{cfg}_pub']:.4f} ({dp:+.5f})   "
                f"prv {r[f'{BASE}_prv']:.4f}->{r[f'{cfg}_prv']:.4f} ({dv:+.5f})")
        if not diff:
            summary_lines.append("  (no datasets differ)")
        rp = regressions(cfg, "pub")
        rv = regressions(cfg, "prv")
        pub_str = ", ".join(f"{n}({d:+.5f})" for n, d in rp) if rp else "(none)"
        prv_str = ", ".join(f"{n}({d:+.5f})" for n, d in rv) if rv else "(none)"
        summary_lines.append(f"  Public regressions:  {pub_str}")
        summary_lines.append(f"  Private regressions: {prv_str}")

    # ---- verdict ----
    summary_lines.append("")
    summary_lines.append("=== VERDICT (adoption vs base == shipped 08) ===")
    summary_lines.append(
        "Criterion: CLEAN IMPROVEMENT over 08 iff mean delta positive AND "
        "zero regressions on BOTH splits.")
    any_clean = False
    clean_names = []
    for cfg in CANDIDATES:
        mp = mean_delta(cfg, "pub")
        mv = mean_delta(cfg, "prv")
        rp = regressions(cfg, "pub")
        rv = regressions(cfg, "prv")
        mean_pos = (mp > 1e-9) and (mv > 1e-9)
        zero_regs = (not rp) and (not rv)
        clean = mean_pos and zero_regs
        if clean:
            any_clean = True
            clean_names.append(cfg)
            verdict = (f"CLEAN-IMPROVEMENT "
                       f"(mean pub={mp:+.5f}, prv={mv:+.5f}; no regressions)")
        else:
            reasons = []
            if not (mp > 1e-9):
                reasons.append(f"non-positive mean Public ({mp:+.5f})")
            if not (mv > 1e-9):
                reasons.append(f"non-positive mean Private ({mv:+.5f})")
            if rp:
                reasons.append("Public regs [" +
                               ", ".join(f"{n}({d:+.5f})" for n, d in rp) + "]")
            if rv:
                reasons.append("Private regs [" +
                               ", ".join(f"{n}({d:+.5f})" for n, d in rv) + "]")
            verdict = "NOT-CLEAN (" + "; ".join(reasons) + ")"
        summary_lines.append(f"{cfg}: {verdict}")

    # ---- comparison vs cand_C (round30 K=5, ratio>=0.015) ----
    CANDC_PUB, CANDC_PRV = 0.00169, 0.00151
    summary_lines.append("")
    summary_lines.append("=== COMPARISON vs cand_C (round30 K=5, ratio>=0.015 "
                         "gate, {09,13,15}) ===")
    summary_lines.append(
        f"cand_C reference: mean Public +{CANDC_PUB:.5f} / Private +{CANDC_PRV:.5f}, "
        f"zero regressions.")
    mp = mean_delta(CAND, "pub")
    mv = mean_delta(CAND, "prv")
    rp = regressions(CAND, "pub")
    rv = regressions(CAND, "prv")
    zero_regs = (not rp) and (not rv)
    beats_pub = mp >= CANDC_PUB - 1e-9
    beats_prv = mv >= CANDC_PRV - 1e-9
    if beats_pub and beats_prv and zero_regs:
        concl = (f"round32 BEATS cand_C on both splits AND keeps zero "
                 f"regressions -> STRONGER adoption candidate.")
    elif zero_regs and (mp > 1e-9) and (mv > 1e-9):
        concl = (f"round32 is a clean improvement but does NOT beat cand_C on "
                 f"both splits (Public {mp-CANDC_PUB:+.5f}, "
                 f"Private {mv-CANDC_PRV:+.5f} vs cand_C).")
    else:
        concl = (f"round32 does NOT dominate cand_C "
                 f"(zero_regressions={'YES' if zero_regs else 'NO'}; "
                 f"Public {mp-CANDC_PUB:+.5f}, Private {mv-CANDC_PRV:+.5f} "
                 f"vs cand_C).")
    summary_lines.append(
        f"round32 cand: mean Public {mp:+.5f} ({mp-CANDC_PUB:+.5f} vs cand_C), "
        f"mean Private {mv:+.5f} ({mv-CANDC_PRV:+.5f} vs cand_C), "
        f"zero-regression={'YES' if zero_regs else 'NO'}.")
    summary_lines.append("CONCLUSION: " + concl)

    # ---- clean-run line ----
    summary_lines.append("")
    clean_run = (not exceptions) and (not invariant_violations)
    summary_lines.append(
        f"CLEAN RUN={'YES' if clean_run else 'NO'} "
        f"(total_fits={total_fits}, exceptions={len(exceptions)}, "
        f"skipped={len(skipped)}, invariant_violations={len(invariant_violations)})")
    for name, msg in exceptions:
        summary_lines.append(f"  EXC {name}: {msg}")

    summary = "\n".join(summary_lines)
    print("\n" + summary)

    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
        f.write(summary + "\n")

    print(f"\n[WROTE] {csv_path}")
    print(f"[WROTE] {os.path.join(OUT_DIR, 'summary.txt')}")
    print("HARNESS_DONE")


if __name__ == "__main__":
    main()
