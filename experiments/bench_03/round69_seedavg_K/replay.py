#!/usr/bin/env python
"""
bench_03 round69 — CANDIDATE-A (SEED-AVERAGING) K-SENSITIVITY / GAIN STABILITY.
ALL 16.
OFFLINE ONLY. No subprocess, no LLM, no Kaggle, no network. Calls sklearn
in-process only. Writes ONLY under this round69 directory; NEVER touches
submissions/.

GOAL (improvement-log angle "seedavg_K")
----------------------------------------
Candidate A = seed-average the base-08 HGB: fit the base HGB K times with
random_state 0..K-1, average predict_proba (arithmetic prob-mean), gated on
gate-C = (n_object_cols > 0). The shipped-pending A uses K=10.

Open question: is the seed-avg GAIN (delta vs the shipped seed-0 base) STABLE as
K varies, and how much variance does a GIVEN K carry depending on WHICH seeds
land in the window?

Design
------
For each of the 16 datasets, fit the base-08 HGB ONCE PER SEED for seeds
0..K_max-1 (K_max=20) on FULL train and cache each seed's predict_proba over the
whole test frame. Public / Private eval splits come from solution.csv's Usage
column (identical scoring to round68/round61). base = seed-0 vector.

Arms:
  base  = seed-0 only (MUST reproduce round68/round61 base_public/base_private
          with max|dev| = 0 — asserted).
  A_K   = prob-mean over seeds 0..K-1, for K in {5, 10, 20} (arithmetic mean of
          the K cached proba vectors, == round68 Candidate-A construction).

Gain-variance (how much a given K wobbles with seed choice):
  K=5  : slide over disjoint seed-windows [0..4],[5..9],[10..14],[15..19].
  K=10 : slide over disjoint seed-windows [0..9],[10..19].
  For each window: mean(window proba vectors) -> score -> delta vs base per
  dataset -> fired-subset (gate-C) mean delta. Report the spread (min / max /
  std / mean) of that fired-subset mean delta ACROSS windows, per split. A small
  spread => the K's gain is robust to seed choice; a large spread => the gain a
  given K delivers is a lottery on which seeds you happened to draw.
  (Note A_K5 == window [0..4] and A_K10 == window [0..9] by construction.)

BASE recipe (== shipped 08, identical to round68's base arm):
  features = [c for c in train.columns if c not in ("row_id","target")]
  cat_mask = [train[c].dtype == object for c in features]
  n=len(train); ratio = len(features)/n
  l2  = 1.0 if ratio >= 0.010 else 0.0
  msl = 70 if ratio>=0.030 else 50 if ratio>=0.015 else 20
  HistGradientBoostingClassifier(categorical_features=cat_mask, random_state=k,
      max_iter=300, early_stopping=True, l2_regularization=l2,
      min_samples_leaf=msl)

REPRODUCTION (MANDATORY — proves the harness is faithful, BIT-IDENTICAL):
  base on ALL 16 must match round61 base_public/base_private, max|dev| = 0.
  If the check fails, CLEAN RUN = NO.
"""
import os

# keep the run polite / modest on CPU; the estimator is deterministic w.r.t.
# random_state regardless of thread count, so this does not affect reproduction.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

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
OUT_DIR = os.path.join(BENCH_DIR, "round69_seedavg_K")
STATS_CSV = os.path.join(BENCH_DIR, "dataset_stats.csv")
# Anchor against round61 (base column == base-08 seed-0). Reproduced bit-identically.
ROUND61_RESULTS = os.path.join(BENCH_DIR, "round61_rf_blend", "results.csv")

L2_GATE_THRESHOLD = 0.010   # shipped-08 ratio gate for l2 (FIXED)
GATED_L2 = 1.0
DEFAULT_MSL = 20
MSL_TIERS = [(0.030, 70), (0.015, 50)]
BASE_SEED = 0
K_MAX = 20                   # seed cache size (seeds 0..19)
N_DATASETS = 16
REPRO_TOL = 0.0              # BIT-IDENTICAL: base must match round61 exactly

# K-sweep points for the primary Candidate-A curve.
K_SWEEP = [5, 10, 20]

# Disjoint seed-windows for the per-K gain-variance study. Each window is a
# (start, stop) half-open slice of the cached seeds; all windows for a given K
# have exactly K seeds. A_K{K} == the FIRST window of that K by construction.
WINDOWS = {
    5:  [(0, 5), (5, 10), (10, 15), (15, 20)],
    10: [(0, 10), (10, 20)],
}


def load_stats(path=STATS_CSV):
    stats = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            stats[row["name"]] = {
                "n_train": int(row["n_train"]),
                "n_object_cols": int(row["n_object_cols"]),
            }
    return stats


def round61_base_anchors(path=ROUND61_RESULTS):
    """Read round61's base_public/base_private for ALL 16 datasets to anchor the
    base-arm reproduction at full precision. Returns name -> (pub, prv) or None."""
    if not os.path.exists(path):
        return None
    anchors = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            name = row.get("dataset")
            try:
                anchors[name] = (float(row["base_public"]),
                                 float(row["base_private"]))
            except (KeyError, ValueError):
                anchors[name] = None
    return anchors


def msl_for_ratio(ratio, tiers=MSL_TIERS):
    for thr, val in tiers:
        if ratio >= thr:
            return val
    return DEFAULT_MSL


def auc_or_nan(y_true, y_score):
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def fit_hgb(train_frame, test, features, cat_mask, l2, msl_val, seed):
    """Fit ONE shipped-08 HGB on `train_frame` and return P(class==1) on test.
    validation_fraction / max_depth / interaction_cst / tol / monotonic_cst
    left UNSET (sklearn defaults, byte-identical to shipped 08). The ONLY thing
    that varies across the seed ensemble is `random_state`."""
    clf = HistGradientBoostingClassifier(
        categorical_features=cat_mask,
        random_state=seed,
        max_iter=300,
        early_stopping=True,
        l2_regularization=l2,
        min_samples_leaf=msl_val,
    )
    clf.fit(train_frame[features], train_frame["target"])
    proba = clf.predict_proba(test[features])
    classes = list(clf.classes_)
    pos_idx = classes.index(1) if 1 in classes else 1
    return proba[:, pos_idx]


def run_one(name, train_csv, test_csv, stats):
    """Fit the base-08 HGB once per seed 0..K_MAX-1 on FULL train, caching each
    seed's P(class==1) over the whole test frame. Returns (row_ids, seed_vecs,
    meta) where seed_vecs is (K_MAX, n_test). base = seed_vecs[0]; A_K and every
    window mean are derived downstream from these SAME cached vectors."""
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)

    features = [c for c in train.columns if c not in ("row_id", "target")]
    cat_mask = [train[c].dtype == object for c in features]

    n = len(train)
    ratio = len(features) / n
    l2 = GATED_L2 if ratio >= L2_GATE_THRESHOLD else 0.0
    msl_val = msl_for_ratio(ratio)

    row_ids = test["row_id"].tolist()
    n_test = len(test)

    seed_vecs = np.zeros((K_MAX, n_test), dtype=np.float64)
    n_fits = 0
    for k in range(K_MAX):
        seed_vecs[k] = fit_hgb(train, test, features, cat_mask, l2, msl_val, k)
        n_fits += 1

    st = stats[name]
    gate_c = st["n_object_cols"] > 0
    meta = {
        "n_train": st["n_train"],
        "n_object_cols": st["n_object_cols"],
        "l2": l2,
        "msl": msl_val,
        "n_features": len(features),
        "n_cat": sum(cat_mask),
        "gate_c": gate_c,
        "n_fits": n_fits,
    }
    return row_ids, seed_vecs, meta


def score_vec(row_ids, vec, sol):
    """Map a proba vector (aligned to row_ids) onto solution rows, split by the
    Usage column, and return (public_auc, private_auc)."""
    pred_map = dict(zip(row_ids, np.asarray(vec).tolist()))
    sol = sol.copy()
    sol["pred"] = sol["row_id"].map(pred_map)
    if sol["pred"].isna().any():
        raise ValueError(f"{int(sol['pred'].isna().sum())} row_ids unmatched")
    pub = sol[sol["Usage"] == "Public"]
    prv = sol[sol["Usage"] == "Private"]
    return (auc_or_nan(pub["target"], pub["pred"]),
            auc_or_nan(prv["target"], prv["pred"]))


def main():
    warnings.filterwarnings("ignore")
    os.makedirs(OUT_DIR, exist_ok=True)

    stats = load_stats()
    anchors61 = round61_base_anchors()
    repro_available = anchors61 is not None

    # arm keys we score per dataset. window arms named W{K}_{start}.
    ksweep_arms = [f"A_K{k}" for k in K_SWEEP]
    window_arms = []
    window_arm_meta = {}   # arm_key -> (K, start, stop)
    for k, wins in WINDOWS.items():
        for (a, b) in wins:
            key = f"W{k}_{a}"
            window_arms.append(key)
            window_arm_meta[key] = (k, a, b)
    # de-dup while preserving order (A_K arms and first-window arms coincide in
    # VALUE but we keep both name spaces distinct for clarity).
    all_arms = ksweep_arms + window_arms

    rows = []
    exceptions = []
    skipped = []
    single_class_skips = []
    total_fits = 0

    for i in range(1, N_DATASETS + 1):
        name = f"train_{i:02d}"
        d = os.path.join(DATA_DIR, name)
        train_csv = os.path.join(d, "train.csv")
        test_csv = os.path.join(d, "test.csv")
        sol_csv = os.path.join(d, "solution.csv")
        if not (os.path.exists(train_csv) and os.path.exists(test_csv)
                and os.path.exists(sol_csv)):
            print(f"[SKIP] {name}: missing files")
            skipped.append(name)
            continue

        sol = pd.read_csv(sol_csv)
        rec = {"dataset": name}
        try:
            row_ids, seed_vecs, meta = run_one(name, train_csv, test_csv, stats)
            total_fits += meta["n_fits"]
            rec.update({
                "n_train": meta["n_train"],
                "n_object_cols": meta["n_object_cols"],
                "gate_c": meta["gate_c"],
                "l2": meta["l2"],
                "msl": meta["msl"],
            })

            # base = seed 0
            bpub, bprv = score_vec(row_ids, seed_vecs[BASE_SEED], sol)
            rec["base_pub"], rec["base_prv"] = bpub, bprv
            if math.isnan(bpub) or math.isnan(bprv):
                single_class_skips.append((name, "base"))

            # K-sweep arms
            for k in K_SWEEP:
                vec = seed_vecs[0:k].mean(axis=0)
                pub, prv = score_vec(row_ids, vec, sol)
                rec[f"A_K{k}_pub"], rec[f"A_K{k}_prv"] = pub, prv
                if math.isnan(pub) or math.isnan(prv):
                    single_class_skips.append((name, f"A_K{k}"))

            # window arms
            for key, (k, a, b) in window_arm_meta.items():
                vec = seed_vecs[a:b].mean(axis=0)
                pub, prv = score_vec(row_ids, vec, sol)
                rec[f"{key}_pub"], rec[f"{key}_prv"] = pub, prv
                if math.isnan(pub) or math.isnan(prv):
                    single_class_skips.append((name, key))

            print(f"[OK] {name} n_tr={meta['n_train']} obj={meta['n_object_cols']} "
                  f"gateC={meta['gate_c']} feats={meta['n_features']} "
                  f"cat={meta['n_cat']} l2={meta['l2']} msl={meta['msl']} "
                  f"fits={meta['n_fits']} | base pub={bpub:.6f} prv={bprv:.6f}  "
                  f"A_K5 pub={rec['A_K5_pub']:.6f}  "
                  f"A_K10 pub={rec['A_K10_pub']:.6f}  "
                  f"A_K20 pub={rec['A_K20_pub']:.6f}")
        except Exception as e:
            exceptions.append((name, repr(e)))
            rec.update({"n_train": stats.get(name, {}).get("n_train", ""),
                        "n_object_cols": stats.get(name, {}).get("n_object_cols", ""),
                        "gate_c": stats.get(name, {}).get("n_object_cols", 0) > 0,
                        "l2": float("nan"), "msl": float("nan")})
            rec["base_pub"] = rec["base_prv"] = float("nan")
            for arm in all_arms:
                rec[f"{arm}_pub"] = rec[f"{arm}_prv"] = float("nan")
            print(f"[ERROR] {name}: {e!r}")
        rows.append(rec)

    # ---- delta / aggregation helpers ----
    def delta(rec, arm, split):
        b = rec.get(f"base_{split}")
        c = rec.get(f"{arm}_{split}")
        if b is None or c is None or (isinstance(b, float) and math.isnan(b)) \
                or (isinstance(c, float) and math.isnan(c)):
            return float("nan")
        return c - b

    def mean_over(vals):
        vals = [v for v in vals if not (isinstance(v, float) and math.isnan(v))]
        return sum(vals) / len(vals) if vals else float("nan")

    def std_over(vals):
        vals = [v for v in vals if not (isinstance(v, float) and math.isnan(v))]
        if len(vals) < 1:
            return float("nan")
        m = sum(vals) / len(vals)
        return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))  # population std

    def mean_delta(arm, split, subset=None):
        src = rows if subset is None else [r for r in rows if subset(r)]
        return mean_over([delta(r, arm, split) for r in src])

    def wlt(arm, split, subset=None, eps=1e-6):
        src = rows if subset is None else [r for r in rows if subset(r)]
        w = l = t = 0
        for r in src:
            dd = delta(r, arm, split)
            if math.isnan(dd):
                continue
            if dd > eps:
                w += 1
            elif dd < -eps:
                l += 1
            else:
                t += 1
        return w, l, t

    is_gatec = lambda r: bool(r.get("gate_c"))

    # ---- results.csv (per-dataset per-arm) ----
    csv_path = os.path.join(OUT_DIR, "results.csv")
    fieldnames = ["dataset", "n_train", "n_object_cols", "gate_c",
                  "base_public", "base_private"]
    for k in K_SWEEP:
        fieldnames += [f"A_K{k}_public", f"A_K{k}_private",
                       f"dA_K{k}_public", f"dA_K{k}_private"]
    for key in window_arms:
        fieldnames += [f"{key}_public", f"{key}_private",
                       f"d{key}_public", f"d{key}_private"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            out = {
                "dataset": r.get("dataset", ""),
                "n_train": r.get("n_train", ""),
                "n_object_cols": r.get("n_object_cols", ""),
                "gate_c": r.get("gate_c", ""),
                "base_public": r.get("base_pub", ""),
                "base_private": r.get("base_prv", ""),
            }
            for k in K_SWEEP:
                out[f"A_K{k}_public"] = r.get(f"A_K{k}_pub", "")
                out[f"A_K{k}_private"] = r.get(f"A_K{k}_prv", "")
                out[f"dA_K{k}_public"] = delta(r, f"A_K{k}", "pub")
                out[f"dA_K{k}_private"] = delta(r, f"A_K{k}", "prv")
            for key in window_arms:
                out[f"{key}_public"] = r.get(f"{key}_pub", "")
                out[f"{key}_private"] = r.get(f"{key}_prv", "")
                out[f"d{key}_public"] = delta(r, key, "pub")
                out[f"d{key}_private"] = delta(r, key, "prv")
            w.writerow(out)

    # ---- REPRODUCTION: base vs round61 base (BIT-IDENTICAL) ----
    by_name = {r["dataset"]: r for r in rows}
    repro = {}
    repro_ok = True
    max_dev_base = 0.0
    for i in range(1, N_DATASETS + 1):
        nm = f"train_{i:02d}"
        r = by_name.get(nm)
        mine = (r.get("base_pub"), r.get("base_prv")) if r else (None, None)
        ref = anchors61.get(nm) if anchors61 else None
        if ref is None or mine[0] is None or mine[1] is None \
                or (isinstance(mine[0], float) and math.isnan(mine[0])):
            okp = okv = False
            devp = devv = float("nan")
        else:
            devp = abs(mine[0] - ref[0])
            devv = abs(mine[1] - ref[1])
            okp = devp <= REPRO_TOL
            okv = devv <= REPRO_TOL
            max_dev_base = max(max_dev_base, devp, devv)
        repro[nm] = {"mine": mine, "ref": ref, "okp": okp, "okv": okv,
                     "devp": devp, "devv": devv}
        if not (okp and okv):
            repro_ok = False

    # ---- partition sanity ----
    present = {r["dataset"] for r in rows if not (
        isinstance(r.get("base_pub"), float) and math.isnan(r.get("base_pub")))}
    all16_ok = (len(present) == N_DATASETS and not skipped)
    n_gate_c = sum(1 for r in rows if r.get("gate_c"))
    fired_names = sorted(r["dataset"] for r in rows if r.get("gate_c"))

    # ================= SUMMARY =================
    L = []
    L.append("=" * 78)
    L.append("bench_03 round69 — CANDIDATE-A SEED-AVG K-SENSITIVITY / GAIN "
             "STABILITY (ALL 16)  [OFFLINE]")
    L.append("=" * 78)
    L.append("")
    L.append("SETUP:")
    L.append("  base = single HGB seed-0 (== shipped 08 / round61 base).")
    L.append("  A_K  = prob-mean of base-08 HGB over seeds 0..K-1, K in {5,10,20}.")
    L.append("  gate-C = (n_object_cols > 0); Candidate A fires there.")
    L.append(f"  seed cache K_MAX={K_MAX} (seeds 0..{K_MAX-1}); {K_MAX} HGB fits/dataset.")
    L.append(f"  gate-C fired on {len(fired_names)} datasets: {fired_names}")
    L.append("")

    # ---- PRIMARY K-SWEEP TABLE ----
    L.append("=" * 78)
    L.append("K-SWEEP: mean Δ vs base (A_K − base), over-16 and gate-C fired-subset")
    L.append("=" * 78)
    L.append(f"{'K':>4} {'scope':<12} {'mean dPublic':>14} {'W/L/T pub':>12} "
             f"{'mean dPrivate':>14} {'W/L/T prv':>12}")
    ksweep_summary = {}
    for k in K_SWEEP:
        arm = f"A_K{k}"
        # over-16
        mp16 = mean_delta(arm, "pub")
        mv16 = mean_delta(arm, "prv")
        wp16 = wlt(arm, "pub"); wv16 = wlt(arm, "prv")
        # fired subset
        mpf = mean_delta(arm, "pub", is_gatec)
        mvf = mean_delta(arm, "prv", is_gatec)
        wpf = wlt(arm, "pub", is_gatec); wvf = wlt(arm, "prv", is_gatec)
        ksweep_summary[k] = {"over16": (mp16, mv16), "fired": (mpf, mvf)}
        L.append(f"{k:>4} {'over-16':<12} {mp16:>+14.6f} "
                 f"{f'{wp16[0]}/{wp16[1]}/{wp16[2]}':>12} {mv16:>+14.6f} "
                 f"{f'{wv16[0]}/{wv16[1]}/{wv16[2]}':>12}")
        L.append(f"{'':>4} {'fired(gateC)':<12} {mpf:>+14.6f} "
                 f"{f'{wpf[0]}/{wpf[1]}/{wpf[2]}':>12} {mvf:>+14.6f} "
                 f"{f'{wvf[0]}/{wvf[1]}/{wvf[2]}':>12}")

    # ---- WINDOW-SPREAD (gain-variance for a given K) ----
    L.append("")
    L.append("=" * 78)
    L.append("WINDOW-SPREAD: how much a GIVEN K's fired-subset mean Δ wobbles with")
    L.append("seed choice (disjoint seed-windows, each with exactly K seeds).")
    L.append("=" * 78)
    L.append("  Each row = one disjoint seed-window's gate-C fired-subset mean Δ.")
    L.append("  A_K{K} equals the FIRST window [0..K-1] by construction.")
    window_spread = {}
    for k in sorted(WINDOWS):
        wins = WINDOWS[k]
        L.append("")
        L.append(f"--- K={k}  ({len(wins)} disjoint windows) ---")
        L.append(f"{'window':>12} {'fired dPublic':>16} {'fired dPrivate':>16}")
        pub_vals = []
        prv_vals = []
        for (a, b) in wins:
            key = f"W{k}_{a}"
            mpf = mean_delta(key, "pub", is_gatec)
            mvf = mean_delta(key, "prv", is_gatec)
            pub_vals.append(mpf)
            prv_vals.append(mvf)
            L.append(f"{f'[{a}..{b-1}]':>12} {mpf:>+16.6f} {mvf:>+16.6f}")
        pub_stats = (mean_over(pub_vals), min(pub_vals), max(pub_vals),
                     max(pub_vals) - min(pub_vals), std_over(pub_vals))
        prv_stats = (mean_over(prv_vals), min(prv_vals), max(prv_vals),
                     max(prv_vals) - min(prv_vals), std_over(prv_vals))
        window_spread[k] = {"public": pub_stats, "private": prv_stats}
        L.append(f"  Public  : mean={pub_stats[0]:+.6f} min={pub_stats[1]:+.6f} "
                 f"max={pub_stats[2]:+.6f} range={pub_stats[3]:.6f} "
                 f"std={pub_stats[4]:.6f}")
        L.append(f"  Private : mean={prv_stats[0]:+.6f} min={prv_stats[1]:+.6f} "
                 f"max={prv_stats[2]:+.6f} range={prv_stats[3]:.6f} "
                 f"std={prv_stats[4]:.6f}")

    # ---- interpretation notes ----
    L.append("")
    L.append("=== READING THE SPREAD ===")
    L.append("  Compare each K's window std against that K's mean Δ. If std is a")
    L.append("  large fraction of |mean|, the gain a single K delivers is largely")
    L.append("  a lottery on which seeds were drawn; a small std => the gain is")
    L.append("  robust. K=10 should show a smaller range than K=5 if averaging")
    L.append("  more seeds stabilises the gain.")

    # ---- FULL PER-DATASET TABLE (all 16) ----
    for split, tag in (("pub", "Public"), ("prv", "Private")):
        L.append("")
        L.append(f"=== PER-DATASET dAUC ({tag}) — all 16 (A_K vs base) ===")
        L.append(f"{'dataset':<10} {'gateC':>6} {'base':>10} "
                 f"{'dA_K5':>10} {'dA_K10':>10} {'dA_K20':>10}")
        for r in rows:
            b = r.get(f"base_{split}")
            bstr = f"{b:>10.6f}" if isinstance(b, float) and not math.isnan(b) \
                else f"{'nan':>10}"

            def fmt(x):
                return f"{x:>+10.6f}" if not math.isnan(x) else f"{'nan':>10}"
            L.append(f"{r['dataset']:<10} {str(bool(r.get('gate_c'))):>6} {bstr} "
                     f"{fmt(delta(r, 'A_K5', split))} "
                     f"{fmt(delta(r, 'A_K10', split))} "
                     f"{fmt(delta(r, 'A_K20', split))}")

    # ---- REPRODUCTION CHECK: base vs round61 base ----
    L.append("")
    L.append("=== REPRODUCTION CHECK (base on ALL 16 vs round61 base, tol=0) ===")
    if not repro_available:
        L.append("  round61 results.csv NOT found -> reproduction NOT anchored (FAIL).")
    else:
        for i in range(1, N_DATASETS + 1):
            nm = f"train_{i:02d}"
            rr = repro[nm]
            mp_, mv_ = rr["mine"]
            rp_, rv_ = rr["ref"] if rr["ref"] else (float("nan"), float("nan"))
            mp_s = f"{mp_:.6f}" if isinstance(mp_, float) and not math.isnan(mp_) else "nan"
            mv_s = f"{mv_:.6f}" if isinstance(mv_, float) and not math.isnan(mv_) else "nan"
            L.append(
                f"  {nm}: Public {mp_s} vs r61 {rp_:.6f} "
                f"(|d|={rr['devp']:.2e}, {'YES' if rr['okp'] else 'NO'}); "
                f"Private {mv_s} vs r61 {rv_:.6f} "
                f"(|d|={rr['devv']:.2e}, {'YES' if rr['okv'] else 'NO'})")
        L.append(f"  max |dev| over all 16x2 = {max_dev_base:.2e}")
        L.append(f"  REPRODUCTION (base==round61 base): "
                 f"{'PASS' if repro_ok else 'FAIL'}")

    # ---- CLEAN RUN marker ----
    clean_run = ((not exceptions) and all16_ok and repro_ok and repro_available
                 and (not skipped) and (not single_class_skips))
    L.append("")
    L.append(f"CLEAN RUN: {'YES' if clean_run else 'NO'} "
             f"({len(exceptions)} exceptions)  "
             f"[total_fits={total_fits}, datasets_scored={len(present)}/16, "
             f"skipped={len(skipped)}, single_class_skips={len(single_class_skips)}, "
             f"gate_c_datasets={n_gate_c}, "
             f"reproduction={'YES' if repro_ok else 'NO'} "
             f"(base_maxdev={max_dev_base:.2e})]")
    for name, msg in exceptions:
        L.append(f"  EXC {name}: {msg}")
    for name, cfg in single_class_skips:
        L.append(f"  SINGLE-CLASS {name}/{cfg}")

    summary = "\n".join(L)
    print("\n" + summary)
    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
        f.write(summary + "\n")
    print(f"\n[WROTE] {csv_path}")
    print(f"[WROTE] {os.path.join(OUT_DIR, 'summary.txt')}")

    # compact final marker for log-tail verification
    def g(k, scope, idx):
        return ksweep_summary[k][scope][idx]
    print(f"FINAL_MARKER CLEAN_RUN={'YES' if clean_run else 'NO'} "
          f"SCORED={len(present)}/16 EXC={len(exceptions)} "
          f"TOTAL_FITS={total_fits} BASE_MAXDEV={max_dev_base:.2e} "
          f"K5_fired_pub={g(5,'fired',0):+.6f} K5_fired_prv={g(5,'fired',1):+.6f} "
          f"K10_fired_pub={g(10,'fired',0):+.6f} K10_fired_prv={g(10,'fired',1):+.6f} "
          f"K20_fired_pub={g(20,'fired',0):+.6f} K20_fired_prv={g(20,'fired',1):+.6f} "
          f"K5_pub_range={window_spread[5]['public'][3]:.6f} "
          f"K10_pub_range={window_spread[10]['public'][3]:.6f}")
    print("HARNESS_DONE")


if __name__ == "__main__":
    main()
