#!/usr/bin/env python
"""
bench_03 round54 — ROW-BOOTSTRAP BAGGING (K=10) vs SEED-AVG (K=10) vs base-08
(ALL 16).
OFFLINE ONLY. No subprocess, no LLM, no Kaggle, no network. Calls sklearn
in-process only. Writes ONLY under this round54 directory; NEVER touches
submissions/.

GOAL (improvement-log angle "bootstrap_bag")
--------------------------------------------
Prior rounds (round29+) established a seed-averaging signal: fit K base-08 HGBs
on the FULL training data varying ONLY `random_state`, then average the test
probabilities. That injects ONLY the model-internal stochasticity (subsampling
inside HGB / early-stopping validation split) as diversity. This round isolates
a DIFFERENT source of ensemble diversity: ROW-LEVEL bootstrap resampling.

Design (single new mechanism = row bootstrap):
  BASE arm     = base-08 HGB exactly (reference column), seed-0, all 16 datasets.
                 monotonic_cst / max_depth / interaction_cst / tol UNSET;
                 byte-identical to shipped 08 -> reproduces round53 base column.
  CAND arm     = "bag": row-bootstrap bagging, K=10. For k in 0..9:
                   - idx = np.random.RandomState(k).randint(0, n, size=n)
                     (sample n TRAIN row-indices WITH replacement).
                   - train_k = train.iloc[idx]  (bootstrap resample of ROWS).
                   - fit an IDENTICAL base-08 HGB with random_state=k on train_k.
                     The l2/msl gates are computed from the ORIGINAL full train
                     (NOT recomputed on the resample) so this is a clean
                     single-mechanism change vs base.
                   - predict_proba on the SAME test set.
                 Aggregate = arithmetic MEAN of the K test-proba vectors.
                 GUARD: if a bootstrap resample contains only ONE class in
                 `target`, that k is SKIPPED and the mean is taken over the
                 remaining valid k (logged per dataset).
  CONTEXT arm  = "seedavg": plain seed-avg K=10. Fit K=10 base-08 HGB on the
                 FULL train, random_state=0..9, mean proba. UN-GATED across all
                 16 for a direct head-to-head vs bag and base. (This should be
                 consistent with prior rounds' ~+0.004 gate-C seed-avg signal,
                 but here it is computed on ALL 16, not just the gate subset.)

BASE recipe reproduced (== shipped 08), identical to round53:
  features = [c for c in train.columns if c not in ("row_id","target")]
  cat_mask = [train[c].dtype == object for c in features]
  n=len(train); ratio = len(features)/n
  l2  = 1.0 if ratio >= 0.010 else 0.0
  msl = 70 if ratio>=0.030 else 50 if ratio>=0.015 else 20
  HistGradientBoostingClassifier(categorical_features=cat_mask, random_state=0,
      max_iter=300, early_stopping=True, l2_regularization=l2,
      min_samples_leaf=msl)
  pred = predict_proba(test)[:, class==1]

REPRODUCTION: the BASE column on ALL 16 must match round53's base column
(read from round53 results.csv, columns base_pub/base_prv) to < 5e-6.

ADOPTION: per candidate arm, ADOPT iff it cleanly improves — mean ΔPublic > 0
AND mean ΔPrivate > 0 with ZERO regression on EITHER split. Any negative ΔAUC
on any dataset/split, or a net-negative/negligible mean on either split =>
REJECT.
"""
import os

# keep the run polite / modest on CPU; HGB is deterministic w.r.t. random_state
# regardless of thread count, so this does not affect reproduction.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

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
OUT_DIR = os.path.join(BENCH_DIR, "round54_bootstrap_bag")
STATS_CSV = os.path.join(BENCH_DIR, "dataset_stats.csv")
ROUND53_RESULTS = os.path.join(BENCH_DIR, "round53_monotonic_cst", "results.csv")

L2_GATE_THRESHOLD = 0.010   # shipped-08 ratio gate for l2 (FIXED)
GATED_L2 = 1.0
DEFAULT_MSL = 20
MSL_TIERS = [(0.030, 70), (0.015, 50)]
BASE_SEED = 0
N_DATASETS = 16
REPRO_TOL = 5e-6
K = 10                       # ensemble size for both bag and seedavg

BASE = "base"
CAND = "bag"                 # row-bootstrap bagging K=10
CONTEXT = "seedavg"          # plain seed-avg K=10 (full data), context arm
ENSEMBLES = [CAND, CONTEXT]
ALL_CONFIGS = [BASE, CAND, CONTEXT]


def load_stats(path=STATS_CSV):
    stats = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            stats[row["name"]] = {
                "n_train": int(row["n_train"]),
                "n_object_cols": int(row["n_object_cols"]),
            }
    return stats


def round53_base_anchors(path=ROUND53_RESULTS):
    """Read round53's base_pub/base_prv for ALL 16 datasets to anchor
    reproduction at full precision. Returns dict name -> (pub, prv) or None."""
    if not os.path.exists(path):
        return None
    anchors = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            name = row.get("dataset")
            try:
                anchors[name] = (float(row["base_pub"]), float(row["base_prv"]))
            except (KeyError, ValueError):
                pass
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


def fit_hgb_proba(train_frame, test, features, cat_mask, l2, msl_val, seed):
    """Fit ONE shipped-08 HGB on `train_frame` and return P(class==1) on test.
    validation_fraction / max_depth / interaction_cst / tol / monotonic_cst
    left UNSET (sklearn defaults, byte-identical to shipped 08). The ONLY thing
    that varies across the ensemble is (a) `random_state` and (b) whether the
    training frame is a bootstrap resample (bag) or the full train (seedavg /
    base). l2/msl are always the base-08 gate values computed from the ORIGINAL
    full train."""
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
    """Returns (preds, meta). preds maps config_name -> {row_id -> prob}.

    base     = seed-0, base-08 (byte-identical to shipped 08).
    bag      = row-bootstrap bagging K=10 (mean of valid k proba vectors).
    seedavg  = plain seed-avg K=10 on FULL train (mean of proba, random_state 0..9).
    """
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

    preds = {}
    n_fits = 0

    # ---- BASE: seed-0, full train, base-08 (byte-identical to shipped 08). ----
    base_vec = fit_hgb_proba(train, test, features, cat_mask, l2, msl_val,
                             BASE_SEED)
    preds[BASE] = dict(zip(row_ids, base_vec.tolist()))
    n_fits += 1

    # ---- CAND "bag": row-bootstrap bagging K=10. ----
    # single new mechanism = bootstrap resample of TRAINING ROWS with a per-k RNG.
    bag_sum = np.zeros(n_test, dtype=np.float64)
    bag_valid = 0
    bag_skipped_k = []
    for k in range(K):
        rng = np.random.RandomState(k)
        idx = rng.randint(0, n, size=n)        # n indices, WITH replacement
        train_k = train.iloc[idx]
        # GUARD: a bootstrap resample with a single target class cannot train a
        # binary classifier meaningfully -> skip this k, average over the rest.
        if train_k["target"].nunique() < 2:
            bag_skipped_k.append(k)
            continue
        vec_k = fit_hgb_proba(train_k, test, features, cat_mask, l2, msl_val, k)
        bag_sum += vec_k
        bag_valid += 1
        n_fits += 1
    if bag_valid == 0:
        # degenerate: no valid bootstrap fit -> fall back to base (should not
        # happen for these datasets; logged via meta bag_valid==0).
        bag_vec = base_vec.astype(np.float64)
    else:
        bag_vec = bag_sum / bag_valid
    preds[CAND] = dict(zip(row_ids, bag_vec.tolist()))

    # ---- CONTEXT "seedavg": plain seed-avg K=10 on FULL train, seeds 0..9. ----
    sa_sum = np.zeros(n_test, dtype=np.float64)
    for k in range(K):
        vec_k = fit_hgb_proba(train, test, features, cat_mask, l2, msl_val, k)
        sa_sum += vec_k
        n_fits += 1
    seedavg_vec = sa_sum / K
    preds[CONTEXT] = dict(zip(row_ids, seedavg_vec.tolist()))

    st = stats[name]
    meta = {
        "n_train": st["n_train"],
        "n_object_cols": st["n_object_cols"],
        "l2": l2,
        "msl": msl_val,
        "n_features": len(features),
        "n_cat": sum(cat_mask),
        "bag_valid_k": bag_valid,
        "bag_skipped_k": bag_skipped_k,
        "n_fits": n_fits,
    }
    return preds, meta


def score_split(pred_map, sol):
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
    anchors53 = round53_base_anchors()
    rows = []
    exceptions = []
    skipped = []
    total_fits = 0
    total_bag_skips = 0
    datasets_with_bag_skips = 0

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
            preds, meta = run_one(name, train_csv, test_csv, stats)
            total_fits += meta["n_fits"]
            nskip = len(meta["bag_skipped_k"])
            total_bag_skips += nskip
            if nskip:
                datasets_with_bag_skips += 1
            rec.update({
                "n_train": meta["n_train"],
                "n_object_cols": meta["n_object_cols"],
                "l2": meta["l2"],
                "msl": meta["msl"],
                "bag_valid_k": meta["bag_valid_k"],
                "bag_skipped_k": ";".join(str(x) for x in meta["bag_skipped_k"]),
            })
            for cfg in ALL_CONFIGS:
                pub, prv = score_split(preds[cfg], sol)
                rec[f"{cfg}_pub"] = pub
                rec[f"{cfg}_prv"] = prv
            print(f"[OK] {name} n_tr={meta['n_train']} obj={meta['n_object_cols']} "
                  f"feats={meta['n_features']} cat={meta['n_cat']} "
                  f"l2={meta['l2']} msl={meta['msl']} "
                  f"bag_validk={meta['bag_valid_k']} skips={nskip} "
                  f"fits={meta['n_fits']} | "
                  f"base pub={rec['base_pub']:.6f} prv={rec['base_prv']:.6f}  "
                  f"bag pub={rec['bag_pub']:.6f} prv={rec['bag_prv']:.6f}  "
                  f"seedavg pub={rec['seedavg_pub']:.6f} prv={rec['seedavg_prv']:.6f}")
        except Exception as e:
            exceptions.append((name, repr(e)))
            rec.update({"n_train": stats.get(name, {}).get("n_train", ""),
                        "n_object_cols": stats.get(name, {}).get("n_object_cols", ""),
                        "l2": float("nan"), "msl": float("nan"),
                        "bag_valid_k": "", "bag_skipped_k": ""})
            for cfg in ALL_CONFIGS:
                rec[f"{cfg}_pub"] = float("nan")
                rec[f"{cfg}_prv"] = float("nan")
            print(f"[ERROR] {name}: {e!r}")
        rows.append(rec)

    # ---- delta helpers (arm vs base == shipped 08) ----
    def delta(rec, arm, split):
        b = rec.get(f"{BASE}_{split}")
        c = rec.get(f"{arm}_{split}")
        if b is None or c is None or math.isnan(b) or math.isnan(c):
            return float("nan")
        return c - b

    def mean_delta(arm, split):
        vals = [delta(r, arm, split) for r in rows]
        vals = [v for v in vals if not math.isnan(v)]
        return sum(vals) / len(vals) if vals else float("nan")

    def wlt(arm, split, eps=1e-9):
        w = l = t = 0
        for r in rows:
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

    def regressions(arm, split, eps=1e-6):
        return [(r["dataset"], delta(r, arm, split)) for r in rows
                if not math.isnan(delta(r, arm, split))
                and delta(r, arm, split) < -eps]

    def improvements(arm, split, eps=1e-6):
        return [(r["dataset"], delta(r, arm, split)) for r in rows
                if not math.isnan(delta(r, arm, split))
                and delta(r, arm, split) > eps]

    # ---- results.csv ----
    csv_path = os.path.join(OUT_DIR, "results.csv")
    fieldnames = ["dataset", "n_train", "n_object_cols", "l2", "msl",
                  "bag_valid_k", "bag_skipped_k",
                  "base_pub", "base_prv",
                  "bag_pub", "bag_prv", "bag_d_pub", "bag_d_prv",
                  "seedavg_pub", "seedavg_prv", "seedavg_d_pub", "seedavg_d_prv"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            out = {k: r.get(k, "") for k in
                   ["dataset", "n_train", "n_object_cols", "l2", "msl",
                    "bag_valid_k", "bag_skipped_k",
                    "base_pub", "base_prv",
                    "bag_pub", "bag_prv",
                    "seedavg_pub", "seedavg_prv"]}
            out["bag_d_pub"] = delta(r, CAND, "pub")
            out["bag_d_prv"] = delta(r, CAND, "prv")
            out["seedavg_d_pub"] = delta(r, CONTEXT, "pub")
            out["seedavg_d_prv"] = delta(r, CONTEXT, "prv")
            w.writerow(out)

    # ---- REPRODUCTION: base on ALL 16 matches round53 (tol<5e-6) ----
    repro = {}
    repro_ok = True
    repro_available = anchors53 is not None
    by_name = {r["dataset"]: r for r in rows}
    max_abs_dev = 0.0
    for i in range(1, N_DATASETS + 1):
        nm = f"train_{i:02d}"
        r = by_name.get(nm)
        mine = (r.get("base_pub"), r.get("base_prv")) if r else (None, None)
        ref = anchors53.get(nm) if anchors53 else None
        if ref is None or mine[0] is None or mine[1] is None \
                or (isinstance(mine[0], float) and math.isnan(mine[0])):
            okp = okv = False
            devp = devv = float("nan")
        else:
            devp = abs(mine[0] - ref[0])
            devv = abs(mine[1] - ref[1])
            okp = devp < REPRO_TOL
            okv = devv < REPRO_TOL
            max_abs_dev = max(max_abs_dev, devp, devv)
        repro[nm] = {"mine": mine, "ref": ref, "okp": okp, "okv": okv,
                     "devp": devp, "devv": devv}
        if not (okp and okv):
            repro_ok = False

    # ---- partition sanity (all 16 present) ----
    present = {r["dataset"] for r in rows if not (
        isinstance(r.get("base_pub"), float) and math.isnan(r.get("base_pub")))}
    all16_ok = (len(present) == N_DATASETS and not skipped)

    # ================= SUMMARY =================
    L = []
    L.append("=" * 78)
    L.append("bench_03 round54 — ROW-BOOTSTRAP BAGGING (K=10) vs SEED-AVG (K=10) "
             "vs base-08 (ALL 16)  [OFFLINE]")
    L.append("=" * 78)
    L.append("")
    L.append("SETUP:")
    L.append("  base    == shipped 08: HGB, early_stopping, cat_mask=object cols,")
    L.append("             seed-0, full train.")
    L.append("  bag     == row-bootstrap bagging K=10. Per k in 0..9: resample n")
    L.append("             TRAIN ROWS WITH replacement via RandomState(k), fit an")
    L.append("             identical base-08 HGB (random_state=k) on the resample,")
    L.append("             predict_proba on the SAME test; aggregate = MEAN of the")
    L.append("             valid-k proba vectors. l2/msl are the base-08 gate values")
    L.append("             from the ORIGINAL full train (NOT recomputed on resample).")
    L.append("             GUARD: single-target-class resample -> that k skipped.")
    L.append("  seedavg == plain seed-avg K=10: fit K base-08 HGB on the FULL train,")
    L.append("             random_state 0..9, MEAN of proba. UN-GATED across all 16.")
    L.append("  Single new mechanism vs seedavg = ROW-LEVEL bootstrap diversity.")

    # ---- BAG GUARD firing ----
    L.append("")
    L.append("=== BOOTSTRAP GUARD (single-class resamples skipped) ===")
    L.append(f"{'dataset':<10} {'valid_k':>8} {'skipped_k':>12}")
    for r in rows:
        vk = r.get("bag_valid_k", "")
        sk = r.get("bag_skipped_k", "")
        L.append(f"{r['dataset']:<10} {str(vk):>8} {str(sk) if sk else '-':>12}")
    L.append(f"  datasets with any skip: {datasets_with_bag_skips}, "
             f"total k skipped: {total_bag_skips}")

    # ---- SWEEP HEADLINE ----
    L.append("")
    L.append("=== HEADLINE (arm vs base == shipped 08, all 16) ===")
    head = {}
    for arm in ENSEMBLES:
        mp = mean_delta(arm, "pub")
        mv = mean_delta(arm, "prv")
        wp, lp, tp = wlt(arm, "pub")
        wv, lv, tv = wlt(arm, "prv")
        head[arm] = (mp, mv, (wp, lp, tp), (wv, lv, tv))
        L.append(f"  {arm:<8}: mean ΔPublic={mp:+.6f} (W/L/T {wp}/{lp}/{tp})  "
                 f"mean ΔPrivate={mv:+.6f} (W/L/T {wv}/{lv}/{tv})")

    # ---- bag vs seedavg head-to-head ----
    L.append("")
    L.append("=== HEAD-TO-HEAD: bag vs seedavg (per-dataset, both arms vs base) ===")
    L.append(f"{'dataset':<10} {'bag_dpub':>11} {'sa_dpub':>11} "
             f"{'bag_dprv':>11} {'sa_dprv':>11}")
    bag_beats_sa_pub = sa_beats_bag_pub = 0
    bag_beats_sa_prv = sa_beats_bag_prv = 0
    for r in rows:
        bdp = delta(r, CAND, "pub")
        sdp = delta(r, CONTEXT, "pub")
        bdv = delta(r, CAND, "prv")
        sdv = delta(r, CONTEXT, "prv")
        if not math.isnan(bdp) and not math.isnan(sdp):
            if bdp > sdp + 1e-9:
                bag_beats_sa_pub += 1
            elif sdp > bdp + 1e-9:
                sa_beats_bag_pub += 1
        if not math.isnan(bdv) and not math.isnan(sdv):
            if bdv > sdv + 1e-9:
                bag_beats_sa_prv += 1
            elif sdv > bdv + 1e-9:
                sa_beats_bag_prv += 1
        def fmt(x):
            return f"{x:>+11.6f}" if not math.isnan(x) else f"{'nan':>11}"
        L.append(f"{r['dataset']:<10} {fmt(bdp)} {fmt(sdp)} {fmt(bdv)} {fmt(sdv)}")
    L.append(f"  Public : bag>seedavg on {bag_beats_sa_pub}, "
             f"seedavg>bag on {sa_beats_bag_pub}")
    L.append(f"  Private: bag>seedavg on {bag_beats_sa_prv}, "
             f"seedavg>bag on {sa_beats_bag_prv}")

    # ---- NOISY SMALL-N focus ----
    L.append("")
    L.append("=== NOISY SMALL-N FOCUS (train_05, train_09, train_13) ===")
    focus = {"train_05", "train_09", "train_13"}
    for r in rows:
        if r["dataset"] in focus:
            L.append(f"  {r['dataset']} (n={r.get('n_train','')}): "
                     f"bag ΔPub={delta(r, CAND, 'pub'):+.6f} "
                     f"ΔPrv={delta(r, CAND, 'prv'):+.6f} | "
                     f"seedavg ΔPub={delta(r, CONTEXT, 'pub'):+.6f} "
                     f"ΔPrv={delta(r, CONTEXT, 'prv'):+.6f}")

    # ---- REPRODUCTION ----
    L.append("")
    L.append("=== REPRODUCTION CHECK (base on ALL 16 vs round53, tol<5e-6) ===")
    if not repro_available:
        L.append("  round53 results.csv NOT found -> reproduction NOT anchored (FAIL).")
    else:
        for i in range(1, N_DATASETS + 1):
            nm = f"train_{i:02d}"
            rr = repro[nm]
            mp_, mv_ = rr["mine"]
            rp_, rv_ = rr["ref"] if rr["ref"] else (float("nan"), float("nan"))
            L.append(
                f"  {nm}: Public {mp_:.6f} vs r53 {rp_:.6f} "
                f"(|d|={rr['devp']:.2e}, {'YES' if rr['okp'] else 'NO'}); "
                f"Private {mv_:.6f} vs r53 {rv_:.6f} "
                f"(|d|={rr['devv']:.2e}, {'YES' if rr['okv'] else 'NO'})")
        L.append(f"  max |dev| over all 16x2 = {max_abs_dev:.2e}")
        L.append(f"  REPRODUCTION: {'PASS' if repro_ok else 'FAIL'}")

    # ---- PER-DATASET DELTAS ----
    for arm in ENSEMBLES:
        for split, tag in (("pub", "Public"), ("prv", "Private")):
            L.append("")
            L.append(f"=== PER-DATASET ΔAUC ({tag}) — base vs {arm} ===")
            L.append(f"{'dataset':<10} {'base':>10} {'arm':>10} {'delta':>11}")
            for r in rows:
                b = r.get(f"{BASE}_{split}")
                c = r.get(f"{arm}_{split}")
                dd = delta(r, arm, split)
                bstr = f"{b:>10.6f}" if isinstance(b, float) and not math.isnan(b) else f"{'nan':>10}"
                cstr = f"{c:>10.6f}" if isinstance(c, float) and not math.isnan(c) else f"{'nan':>10}"
                dstr = f"{dd:>+11.6f}" if not math.isnan(dd) else f"{'nan':>11}"
                L.append(f"{r['dataset']:<10} {bstr} {cstr} {dstr}")

    # ---- REGRESSIONS / IMPROVEMENTS ----
    L.append("")
    L.append("=== REGRESSIONS (ΔAUC < -1e-6) ===")
    for arm in ENSEMBLES:
        rp = regressions(arm, "pub")
        rv = regressions(arm, "prv")
        if not rp and not rv:
            L.append(f"  {arm}: NONE on either split.")
        else:
            for n_, d_ in rp:
                L.append(f"  {arm} Public  {n_}: {d_:+.6f}")
            for n_, d_ in rv:
                L.append(f"  {arm} Private {n_}: {d_:+.6f}")

    L.append("")
    L.append("=== IMPROVEMENTS (ΔAUC > +1e-6) ===")
    for arm in ENSEMBLES:
        ip = improvements(arm, "pub")
        iv = improvements(arm, "prv")
        if not ip and not iv:
            L.append(f"  {arm}: NONE on either split.")
        else:
            for n_, d_ in ip:
                L.append(f"  {arm} Public  {n_}: {d_:+.6f}")
            for n_, d_ in iv:
                L.append(f"  {arm} Private {n_}: {d_:+.6f}")

    # ---- largest single moves ----
    def extreme(arm, split, sign):
        vals = [(r["dataset"], delta(r, arm, split)) for r in rows
                if not math.isnan(delta(r, arm, split))]
        if not vals:
            return None
        return (min(vals, key=lambda x: x[1]) if sign < 0
                else max(vals, key=lambda x: x[1]))

    L.append("")
    L.append("=== LARGEST SINGLE MOVES ===")
    for arm in ENSEMBLES:
        L.append(f"  -- {arm} --")
        for split, tag in (("pub", "Public"), ("prv", "Private")):
            worst = extreme(arm, split, -1)
            best = extreme(arm, split, +1)
            if worst:
                L.append(f"    {tag} max regression : {worst[0]} {worst[1]:+.6f}")
            if best:
                L.append(f"    {tag} max improvement: {best[0]} {best[1]:+.6f}")

    # ---- ADOPTION / VERDICT ----
    L.append("")
    L.append("=== ADOPTION ANALYSIS ===")
    L.append("Per arm: ADOPT iff mean ΔPublic > 0 AND mean ΔPrivate > 0 with ZERO")
    L.append("  regression on EITHER split. Any regression, or net-negative/")
    L.append("  negligible mean on either split => REJECT.")
    ADOPT_EPS = 1e-5   # negligible-mean guard
    adopt_flags = {}
    L.append("")
    for arm in ENSEMBLES:
        regs_pub = regressions(arm, "pub")
        regs_prv = regressions(arm, "prv")
        mp, mv = head[arm][0], head[arm][1]
        zero_regs = (not regs_pub) and (not regs_prv)
        clean_gain = (mp > ADOPT_EPS and mv > ADOPT_EPS)
        is_adopt = zero_regs and clean_gain
        adopt_flags[arm] = is_adopt
        L.append(f"  [{arm}] zero_regressions={'YES' if zero_regs else 'NO'} "
                 f"(Public regs={len(regs_pub)}, Private regs={len(regs_prv)})")
        L.append(f"  [{arm}] mean ΔPublic  = {mp:+.6f}  (clean gain: "
                 f"{'YES' if mp > ADOPT_EPS else 'NO'})")
        L.append(f"  [{arm}] mean ΔPrivate = {mv:+.6f}  (clean gain: "
                 f"{'YES' if mv > ADOPT_EPS else 'NO'})")
        L.append(f"  [{arm}] -> {'ADOPT' if is_adopt else 'REJECT'}")
        L.append("")

    bag_adopt = adopt_flags.get(CAND, False)
    is_adopt_any = any(adopt_flags.values())
    adopted = [a for a in ENSEMBLES if adopt_flags[a]]

    # bag vs seedavg verdict (does row bootstrap beat plain seed-avg on mean Δ?)
    bag_mp, bag_mv = head[CAND][0], head[CAND][1]
    sa_mp, sa_mv = head[CONTEXT][0], head[CONTEXT][1]
    bag_beats_sa_mean = (bag_mp > sa_mp) and (bag_mv > sa_mv)

    L.append("=== VERDICT ===")
    if bag_beats_sa_mean:
        L.append(f"  Row-bootstrap bagging BEATS plain seed-avg on BOTH mean splits "
                 f"(bag ΔPub {bag_mp:+.6f} > seedavg {sa_mp:+.6f}; "
                 f"bag ΔPrv {bag_mv:+.6f} > seedavg {sa_mv:+.6f}).")
    else:
        L.append(f"  Row-bootstrap bagging does NOT beat plain seed-avg on both mean "
                 f"splits (bag ΔPub {bag_mp:+.6f} vs seedavg {sa_mp:+.6f}; "
                 f"bag ΔPrv {bag_mv:+.6f} vs seedavg {sa_mv:+.6f}).")
    if is_adopt_any:
        L.append(f"  ADOPT: {', '.join(adopted)} cleanly improve BOTH splits with "
                 f"zero regression.")
    else:
        L.append("  REJECT (clean-win test): no arm improves BOTH splits with zero "
                 "regressions across all 16. Row-bootstrap injects row-level "
                 "diversity but discards ~37% of the training rows per member, "
                 "which hurts the well-fit large-n datasets even as it may help "
                 "noisy small-n ones -> no universal clean win.")

    ship = "ADOPT" if is_adopt_any else "REJECT"
    L.append("")
    L.append(f"SHIP VERDICT: {ship}")

    # ---- CLEAN RUN marker ----
    clean_run = ((not exceptions) and all16_ok and repro_ok and repro_available
                 and (not skipped))
    L.append("")
    L.append(f"CLEAN RUN: {'YES' if clean_run else 'NO'} "
             f"({len(exceptions)} exceptions)  "
             f"[total_fits={total_fits}, datasets_scored={len(present)}/16, "
             f"skipped={len(skipped)}, reproduction={'YES' if repro_ok else 'NO'}, "
             f"base_repro_maxdev={max_abs_dev:.2e}, "
             f"bag_k_skipped_total={total_bag_skips}]")
    for name, msg in exceptions:
        L.append(f"  EXC {name}: {msg}")

    summary = "\n".join(L)
    print("\n" + summary)
    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
        f.write(summary + "\n")
    print(f"\n[WROTE] {csv_path}")
    print(f"[WROTE] {os.path.join(OUT_DIR, 'summary.txt')}")
    print(f"FINAL_MARKER CLEAN_RUN={'YES' if clean_run else 'NO'} "
          f"SCORED={len(present)}/16 EXC={len(exceptions)} "
          f"REPRO_MAXDEV={max_abs_dev:.2e}")
    print("HARNESS_DONE")


if __name__ == "__main__":
    main()
