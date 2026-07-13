#!/usr/bin/env python
"""
bench_03 round33 — SEED-AVERAGING FIRING-GATE n_train THRESHOLD SWEEP.
OFFLINE ONLY. No subprocess, no LLM, no Kaggle, no network. Calls sklearn
in-process.

Adapted from experiments/bench_03/round32_gate_categorical/replay.py (dataset
loading, the shipped-08 base config reproduction, Public/Private AUC scoring
joined on row_id to solution.csv, K=5 seed-averaging, the byte-identical-on-
non-firing INVARIANT check, and the summary machinery are all reused). The
ONLY change vs round32 is that the seed-averaging firing gate's n_train
threshold is SWEPT rather than fixed.

Base recipe reproduced (== shipped 08), IDENTICAL to round32:
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
solution.csv, over all 16 datasets. base = each dataset's single seed-0
prediction (byte-identical to shipped 08). The base recipe is UNCHANGED.

THE ONLY CHANGE — the seed-averaging firing gate n_train threshold is SWEPT:
  gate(N): n_train < N  AND  n_object_cols > 0        (K FIXED at 5)
  N sweep : {2000, 4000, 8500, 15000}
  n_train and n_object_cols are read per dataset from
  experiments/bench_03/dataset_stats.csv (NOT from ratio).

Each threshold N is a SEPARATE candidate config (cand_2000, cand_4000,
cand_8500, cand_15000), all compared against the SAME base == shipped-08.
The `n_object_cols > 0` requirement stays fixed, so train_16 (obj=0) is always
excluded regardless of N.

Expected firing sets (verified at run time and logged per threshold):
  N=2000  -> {train_05, train_09, train_13, train_15}     (train_03 n=3501 out)
  N=4000  -> {train_03, train_05, train_09, train_13, train_15}   (== round32)
  N=8500  -> adds train_08 (n=8173) -> {03,05,08,09,13,15}
  N=15000 -> adds {01,02,06,07,14} -> {01,02,03,05,06,07,08,09,13,14,15}

EFFICIENCY: many datasets are shared across the four thresholds. Each dataset
that fires under the LARGEST threshold (N=15000, i.e. any categorical dataset
with n_train < 15000) is fit K=5 times ONCE and its seed predictions cached.
For every threshold N, a dataset uses the cached K=5 average if it fires under
N, else the seed-0 base (byte-identical). Datasets that never fire under any
threshold are fit seed-0 only. This keeps total fits minimal (60: 11 datasets
x 5 seeds + 5 datasets x 1 seed) with NO re-fitting across thresholds.

IMPLEMENTATION INVARIANT: for a given threshold N, any NON-firing dataset's
candidate reuses the EXACT seed-0 array, so its delta MUST be exactly 0 on both
splits — checked explicitly per threshold. On firing datasets the K=5 average
includes seed-0 so it generally differs from base (expected).

Adoption criterion (reused from round32): a threshold is a CLEAN IMPROVEMENT
over base(08) iff its mean delta is positive on BOTH splits AND there are ZERO
regressions on BOTH splits (no dataset worse on either split).

round32's N=4000 result was mean Public +0.00229 / Private +0.00215 with zero
regressions; N=4000 here must reproduce it exactly. The final ranking states
which (if any) threshold beats that on BOTH splits with zero regressions.
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
OUT_DIR = os.path.join(BENCH_DIR, "round33_gate_n_sweep")
STATS_CSV = os.path.join(BENCH_DIR, "dataset_stats.csv")

L2_GATE_THRESHOLD = 0.010   # shipped 08 feature-to-row-ratio gate for l2 (FIXED)
GATED_L2 = 1.0              # l2 applied when the l2-gate fires (FIXED)
DEFAULT_MSL = 20            # sklearn HGB default, used when no msl tier is cleared.
BASE_SEED = 0              # shipped-08 fixed random_state.

# round33 seed-averaging firing gate: n_train < N AND n_object_cols > 0.
# N is SWEPT (THE ONLY CHANGE vs round32, which fixed N=4000).
N_TRAIN_THRESHOLDS = [2000, 4000, 8500, 15000]
MAX_THRESHOLD = max(N_TRAIN_THRESHOLDS)   # 15000 — decides which datasets get K=5.
MIN_OBJECT_COLS = 0        # require strictly more than this many object columns

# K is FIXED at 5 (round30 established K=5 as the knee; no K sweep here).
K = 5
SEEDS = list(range(K))     # [0,1,2,3,4]

# 08 tiered min_samples_leaf, IDENTICAL across base and cand (descending order).
MSL_TIERS = [(0.030, 70), (0.015, 50)]

BASE = "base"


def cand_name(n_thr):
    """Candidate column name for a given n_train threshold."""
    return f"cand_{n_thr}"


CANDIDATES = [cand_name(n) for n in N_TRAIN_THRESHOLDS]

N_DATASETS = 16

# round32 N=4000 reference (this sweep must reproduce it at N=4000).
ROUND32_PUB, ROUND32_PRV = 0.00229, 0.00215

# expected firing sets per threshold (verified at run time).
EXPECTED_FIRE = {
    2000: {"train_05", "train_09", "train_13", "train_15"},
    4000: {"train_03", "train_05", "train_09", "train_13", "train_15"},
    8500: {"train_03", "train_05", "train_08", "train_09", "train_13",
           "train_15"},
    15000: {"train_01", "train_02", "train_03", "train_05", "train_06",
            "train_07", "train_08", "train_09", "train_13", "train_14",
            "train_15"},
}


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


def gate_fires(n_train, n_object_cols, n_thr):
    """round33 firing rule for threshold n_thr: small n AND >=1 object col."""
    return (n_train < n_thr) and (n_object_cols > MIN_OBJECT_COLS)


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
    """Reproduce shipped-08 base + derive all threshold candidates for one
    dataset. Returns (preds, l2, l2_fired, fires_by_thr, msl_val, n_fits,
    n_train_stat, n_obj_stat, ever_fires) where:
      preds maps config_name -> {row_id -> prob} for BASE and every cand_N.
      base = seed-0 prediction (== shipped 08).
      fires_by_thr maps N -> bool (does the gate fire for this dataset at N).
      ever_fires = fires under the LARGEST threshold (decides K=5 vs seed-0).

    A dataset that ever fires is fit seeds 0..4 ONCE and cached; each cand_N
    uses the K=5 mean if it fires under N, else the exact seed-0 array. A
    dataset that never fires is fit seed-0 only and every cand_N reuses that
    exact seed-0 array (byte-identical to base)."""
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)

    features = [c for c in train.columns if c not in ("row_id", "target")]
    cat_mask = [train[c].dtype == object for c in features]

    n = len(train)
    n_feat = len(features)
    ratio = n_feat / n

    # base config gates (UNCHANGED from shipped 08 / round32).
    l2_fired = ratio >= L2_GATE_THRESHOLD
    l2 = GATED_L2 if l2_fired else 0.0
    msl_val = msl_for_ratio(ratio)

    # round33 seed-averaging gate reads stats (n_train, n_object_cols).
    st = stats[name]
    n_train_stat = st["n_train"]
    n_obj_stat = st["n_object_cols"]
    fires_by_thr = {
        n_thr: gate_fires(n_train_stat, n_obj_stat, n_thr)
        for n_thr in N_TRAIN_THRESHOLDS
    }
    ever_fires = gate_fires(n_train_stat, n_obj_stat, MAX_THRESHOLD)

    row_ids = test["row_id"].tolist()

    if ever_fires:
        # Fit K=5 ONCE and cache; reused for every threshold that fires.
        seed_preds = [
            fit_one_seed(train, test, features, cat_mask, l2, msl_val, s)
            for s in SEEDS
        ]
        n_fits = len(SEEDS)
        base_vec = seed_preds[BASE_SEED]  # seed-0 == base
        avg_vec = np.mean(np.vstack(seed_preds), axis=0)
        base_map = dict(zip(row_ids, base_vec.tolist()))
        avg_map = dict(zip(row_ids, avg_vec.tolist()))
        preds = {BASE: base_map}
        for n_thr in N_TRAIN_THRESHOLDS:
            # fires under N -> cached K=5 average; else -> exact seed-0 base.
            preds[cand_name(n_thr)] = avg_map if fires_by_thr[n_thr] else base_map
    else:
        # Never fires under any threshold -> seed-0 only; every cand == base.
        base_vec = fit_one_seed(train, test, features, cat_mask, l2, msl_val,
                                BASE_SEED)
        n_fits = 1
        base_map = dict(zip(row_ids, base_vec.tolist()))
        preds = {BASE: base_map}
        for n_thr in N_TRAIN_THRESHOLDS:
            preds[cand_name(n_thr)] = base_map  # byte-identical to base

    return (preds, l2, l2_fired, fires_by_thr, msl_val, n_fits, n_train_stat,
            n_obj_stat, ever_fires)


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
            (preds, l2, l2_fired, fires_by_thr, msl_val, n_fits, n_train_stat,
             n_obj_stat, ever_fires) = run_one(name, train_csv, test_csv, stats)
            total_fits += n_fits
            rec["l2_fired"] = l2_fired
            rec["ever_fires"] = bool(ever_fires)
            rec["n_train"] = n_train_stat
            rec["n_object_cols"] = n_obj_stat
            for n_thr in N_TRAIN_THRESHOLDS:
                rec[f"fires_{n_thr}"] = bool(fires_by_thr[n_thr])
            for cfg in ALL_CONFIGS:
                pub, prv = score_split(preds[cfg], sol)
                rec[f"{cfg}_pub"] = pub
                rec[f"{cfg}_prv"] = prv
            rec["msl"] = msl_val
            fire_flags = "".join(
                "1" if fires_by_thr[n] else "0" for n in N_TRAIN_THRESHOLDS)
            print(f"[OK] {name} (n_train={n_train_stat}, n_obj={n_obj_stat}, "
                  f"ever_fires={ever_fires}, fires[2k/4k/8.5k/15k]={fire_flags}, "
                  f"l2={l2}, msl={msl_val}, fits={n_fits}): "
                  f"base pub={rec['base_pub']:.6f} prv={rec['base_prv']:.6f}")
        except Exception as e:  # capture but keep going — CLEAN-RUN diagnostic
            exceptions.append((name, repr(e)))
            rec["l2_fired"] = False
            rec["ever_fires"] = False
            rec["n_train"] = stats.get(name, {}).get("n_train", "")
            rec["n_object_cols"] = stats.get(name, {}).get("n_object_cols", "")
            for n_thr in N_TRAIN_THRESHOLDS:
                rec[f"fires_{n_thr}"] = False
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

    def firing_list_for(n_thr):
        return [r["dataset"] for r in rows if r.get(f"fires_{n_thr}")]

    # ---- write results CSV ----
    csv_path = os.path.join(OUT_DIR, "results.csv")
    fieldnames = ["dataset", "n_train", "n_object_cols", "l2_fired", "msl",
                  "ever_fires"]
    for n_thr in N_TRAIN_THRESHOLDS:
        fieldnames.append(f"fires_{n_thr}")
    fieldnames += ["base_pub", "base_prv"]
    for cfg in CANDIDATES:
        fieldnames += [f"{cfg}_pub", f"{cfg}_d_pub", f"{cfg}_prv", f"{cfg}_d_prv"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            out = {
                "dataset": r["dataset"],
                "n_train": r.get("n_train", ""),
                "n_object_cols": r.get("n_object_cols", ""),
                "l2_fired": r.get("l2_fired", ""),
                "msl": r.get("msl", ""),
                "ever_fires": r.get("ever_fires", ""),
                "base_pub": r.get("base_pub", ""),
                "base_prv": r.get("base_prv", ""),
            }
            for n_thr in N_TRAIN_THRESHOLDS:
                out[f"fires_{n_thr}"] = r.get(f"fires_{n_thr}", "")
            for cfg in CANDIDATES:
                out[f"{cfg}_pub"] = r.get(f"{cfg}_pub", "")
                out[f"{cfg}_prv"] = r.get(f"{cfg}_prv", "")
                out[f"{cfg}_d_pub"] = delta(r, cfg, "pub")
                out[f"{cfg}_d_prv"] = delta(r, cfg, "prv")
            w.writerow(out)

    # ---- INVARIANT check: for each threshold N, cand_N on any dataset that
    #      does NOT fire under N must be byte-identical to base (delta 0). ----
    invariant_violations = []
    for n_thr in N_TRAIN_THRESHOLDS:
        cfg = cand_name(n_thr)
        for r in rows:
            if not r.get(f"fires_{n_thr}"):
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
        summary_lines.append(f"=== PER-DATASET ({tag}) — base + 4 thresholds ===")
        header = (f"{'dataset':<10} {'nTr':>6} {'obj':>4} {'msl':>4} "
                  f"{BASE:>8}")
        for cfg in CANDIDATES:
            header += f" {cfg:>10} {'d_'+cfg.split('_')[1]:>10}"
        summary_lines.append(header)
        for r in rows:
            line = (f"{r['dataset']:<10} {str(r.get('n_train')):>6} "
                    f"{str(r.get('n_object_cols')):>4} {str(r.get('msl')):>4} "
                    f"{r[f'{BASE}_{split}']:>8.4f}")
            for cfg in CANDIDATES:
                line += (f" {r[f'{cfg}_{split}']:>10.4f} "
                         f"{delta(r, cfg, split):>+10.5f}")
            summary_lines.append(line)

    # ---- gate firings per threshold ----
    summary_lines.append("")
    summary_lines.append("=== SEED-AVERAGING FIRING SETS PER THRESHOLD "
                         "(gate: n_train<N AND n_object_cols>0, K=5) ===")
    fire_match_all = True
    for n_thr in N_TRAIN_THRESHOLDS:
        fl = firing_list_for(n_thr)
        fset = set(fl)
        expected = EXPECTED_FIRE[n_thr]
        ok = (fset == expected)
        fire_match_all = fire_match_all and ok
        summary_lines.append(
            f"N={n_thr:<6} fires on ({len(fl)}): "
            f"{', '.join(fl) if fl else '(none)'}")
        summary_lines.append(
            f"          expected {sorted(expected)} matched: "
            f"{'YES' if ok else 'NO'}"
            + ("" if ok else f" (got {sorted(fset)})"))
    # train_16 must always be excluded (obj=0).
    t16_excluded = all("train_16" not in firing_list_for(n)
                       for n in N_TRAIN_THRESHOLDS)
    summary_lines.append(
        f"train_16 (n=1809, n_object_cols=0) excluded from ALL thresholds "
        f"(obj=0): {'YES' if t16_excluded else 'NO'}")

    # ---- INVARIANT report ----
    summary_lines.append("")
    summary_lines.append(
        "=== INVARIANT (per threshold: cand_N on datasets not firing under N "
        "is identical to base, delta 0) ===")
    if invariant_violations:
        summary_lines.append("VIOLATED! cand differs from base on a non-firing "
                             "dataset:")
        for cfg, ds, dp, dv in invariant_violations:
            summary_lines.append(
                f"  {cfg}/{ds}: pub d={dp:+.6g} prv d={dv:+.6g}")
    else:
        for n_thr in N_TRAIN_THRESHOLDS:
            n_nonfire = len([r for r in rows if not r.get(f"fires_{n_thr}")])
            summary_lines.append(
                f"OK N={n_thr}: each of the {n_nonfire} datasets not firing "
                f"under N is byte-identical to base (delta exactly 0).")
        summary_lines.append("Required base-reproduction check. PASS.")

    # ---- which datasets actually differed, per threshold ----
    summary_lines.append("")
    summary_lines.append("=== DATASETS THAT ACTUALLY DIFFER (candidate vs base) ===")
    for cfg in CANDIDATES:
        diff = differing_datasets(cfg)
        summary_lines.append(
            f"{cfg}: ({len(diff)}) {', '.join(diff) if diff else '(none)'}")

    # ---- per-candidate summary vs base(08) ----
    summary_lines.append("")
    summary_lines.append("=== SUMMARY (each threshold vs base == shipped 08) ===")
    for n_thr, cfg in zip(N_TRAIN_THRESHOLDS, CANDIDATES):
        mp = mean_delta(cfg, "pub")
        mv = mean_delta(cfg, "prv")
        wp, lp, tp = wlt(cfg, "pub")
        wv, lv, tv = wlt(cfg, "prv")
        summary_lines.append(
            f"N={n_thr:<6} {cfg}: mean Public d={mp:+.5f}  mean Private d={mv:+.5f}"
            f"  Public W/L/T={wp}/{lp}/{tp}  Private W/L/T={wv}/{lv}/{tv}")

    # ---- N=4000 reproduces round32 ----
    summary_lines.append("")
    summary_lines.append("=== N=4000 REPRODUCES round32 (must match "
                         f"+{ROUND32_PUB:.5f} Pub / +{ROUND32_PRV:.5f} Prv) ===")
    cfg4000 = cand_name(4000)
    mp4 = mean_delta(cfg4000, "pub")
    mv4 = mean_delta(cfg4000, "prv")
    repro_pub = abs(mp4 - ROUND32_PUB) < 5e-6
    repro_prv = abs(mv4 - ROUND32_PRV) < 5e-6
    summary_lines.append(
        f"N=4000 mean Public {mp4:+.5f} (round32 +{ROUND32_PUB:.5f}, "
        f"match={'YES' if repro_pub else 'NO'}); mean Private {mv4:+.5f} "
        f"(round32 +{ROUND32_PRV:.5f}, match={'YES' if repro_prv else 'NO'})")

    # ---- per-candidate differing-dataset deltas + regressions ----
    summary_lines.append("")
    summary_lines.append("=== PER-THRESHOLD DETAIL (datasets differing from base; "
                         "all other deltas are exactly 0) ===")
    for n_thr, cfg in zip(N_TRAIN_THRESHOLDS, CANDIDATES):
        summary_lines.append(f"--- N={n_thr} ({cfg}) vs {BASE} ---")
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
    clean_names = []
    for n_thr, cfg in zip(N_TRAIN_THRESHOLDS, CANDIDATES):
        mp = mean_delta(cfg, "pub")
        mv = mean_delta(cfg, "prv")
        rp = regressions(cfg, "pub")
        rv = regressions(cfg, "prv")
        mean_pos = (mp > 1e-9) and (mv > 1e-9)
        zero_regs = (not rp) and (not rv)
        clean = mean_pos and zero_regs
        if clean:
            clean_names.append((n_thr, cfg, mp, mv))
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
        summary_lines.append(f"N={n_thr:<6} {cfg}: {verdict}")

    # ---- final ranking / comparison vs round32 N=4000 ----
    summary_lines.append("")
    summary_lines.append("=== FINAL RANKING (thresholds vs round32 N=4000 = "
                         f"+{ROUND32_PUB:.5f} Pub / +{ROUND32_PRV:.5f} Prv) ===")
    # rank by (mean_pub + mean_prv) among clean-improvement thresholds, then all.
    ranking = []
    for n_thr, cfg in zip(N_TRAIN_THRESHOLDS, CANDIDATES):
        mp = mean_delta(cfg, "pub")
        mv = mean_delta(cfg, "prv")
        rp = regressions(cfg, "pub")
        rv = regressions(cfg, "prv")
        zero_regs = (not rp) and (not rv)
        ranking.append((n_thr, cfg, mp, mv, zero_regs, len(rp) + len(rv)))
    ranking_sorted = sorted(ranking, key=lambda x: (x[2] + x[3]), reverse=True)
    summary_lines.append("Ranked by (mean Public + mean Private) delta:")
    for rank, (n_thr, cfg, mp, mv, zero_regs, n_regs) in enumerate(
            ranking_sorted, 1):
        summary_lines.append(
            f"  #{rank} N={n_thr:<6} Public {mp:+.5f}  Private {mv:+.5f}  "
            f"zero_regressions={'YES' if zero_regs else 'NO'}"
            + ("" if zero_regs else f" ({n_regs} regs)"))

    # which thresholds beat round32 N=4000 on BOTH splits with zero regressions.
    beats = []
    for n_thr, cfg, mp, mv, zero_regs, n_regs in ranking:
        if n_thr == 4000:
            continue
        if (mp > ROUND32_PUB + 1e-9) and (mv > ROUND32_PRV + 1e-9) and zero_regs:
            beats.append((n_thr, mp, mv))
    summary_lines.append("")
    if beats:
        parts = ", ".join(
            f"N={n}(Pub {mp:+.5f}={mp-ROUND32_PUB:+.5f}, "
            f"Prv {mv:+.5f}={mv-ROUND32_PRV:+.5f} vs N=4000)"
            for n, mp, mv in beats)
        summary_lines.append(
            f"THRESHOLD(S) BEATING round32 N=4000 on BOTH splits with zero "
            f"regressions: {parts}")
    else:
        summary_lines.append(
            "No threshold beats round32's N=4000 on BOTH splits with zero "
            "regressions. N=4000 remains best.")

    # ---- clean-run line ----
    summary_lines.append("")
    fire_ok = fire_match_all and t16_excluded
    repro_ok = repro_pub and repro_prv
    clean_run = ((not exceptions) and (not invariant_violations)
                 and fire_ok and repro_ok)
    summary_lines.append(
        f"CLEAN RUN={'YES' if clean_run else 'NO'} "
        f"(total_fits={total_fits}, exceptions={len(exceptions)}, "
        f"skipped={len(skipped)}, invariant_violations={len(invariant_violations)}, "
        f"firing_sets_match={'YES' if fire_ok else 'NO'}, "
        f"N4000_reproduces_round32={'YES' if repro_ok else 'NO'})")
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
