#!/usr/bin/env python
"""
bench_03 round56 — COLUMN-SUBSPACE BAGGING (K=10) vs SEED-AVG (K=10) vs base-08
(ALL 16).
OFFLINE ONLY. No subprocess, no LLM, no Kaggle, no network. Calls sklearn
in-process only. Writes ONLY under this round56 directory; NEVER touches
submissions/.

GOAL (improvement-log angle "colbag")
-------------------------------------
Prior rounds established a seed-averaging signal (round29+): fit K base-08 HGBs
on the FULL training data varying ONLY `random_state`, then average the test
probabilities — injecting ONLY the model-internal stochasticity as diversity.
round54 isolated ROW-level bootstrap resampling as a different diversity source.
This round isolates yet ANOTHER source: COLUMN (feature) subspace bagging, i.e.
random-subspace ensembling. Each member sees ALL training rows but only a random
80% subset of the feature columns.

Design (single new mechanism vs seedavg = column subspace):
  BASE arm     = base-08 HGB exactly (reference column), seed-0, FULL columns,
                 full train. monotonic_cst / max_depth / interaction_cst / tol
                 UNSET; byte-identical to shipped 08 -> reproduces round54's base
                 column EXACTLY.
  CAND arm     = "colbag": column-subspace bagging, K=10. For k in 0..9:
                   - choose ceil(0.8 * n_feat) feature columns WITHOUT
                     replacement via np.random.RandomState(k).
                   - fit an IDENTICAL base-08 HGB with random_state=k on ALL
                     training rows but only the chosen columns; the categorical
                     mask is REBUILT (object-dtype over the chosen columns).
                     l2/msl are the base-08 gate values from the FULL train (NOT
                     recomputed on the subset) so this is a clean single-mechanism
                     change vs base.
                   - predict_proba(X_test_subset)[:, class==1] per member.
                 Aggregate = arithmetic MEAN of the K test-proba vectors.
                 EDGE CASE: if ceil(0.8*n_feat) >= n_feat (tiny n_feat), the
                 member uses ALL columns (fine).
                 GUARD: if a member yields a single-class fit, that k is SKIPPED
                 and the mean is taken over the rest (expected skips = 0, since
                 every member trains on the full target).
  CONTEXT arm  = "seedavg": plain seed-avg K=10. Fit K=10 base-08 HGB on the
                 FULL train, FULL columns, random_state=0..9, mean proba. This
                 reproduces round54's seedavg column EXACTLY.

BASE recipe reproduced (== shipped 08), identical to round54:
  features = [c for c in train.columns if c not in ("row_id","target")]
  cat_mask = [train[c].dtype == object for c in features]
  n=len(train); ratio = len(features)/n
  l2  = 1.0 if ratio >= 0.010 else 0.0
  msl = 70 if ratio>=0.030 else 50 if ratio>=0.015 else 20
  HistGradientBoostingClassifier(categorical_features=cat_mask, random_state=0,
      max_iter=300, early_stopping=True, l2_regularization=l2,
      min_samples_leaf=msl)
  pred = predict_proba(test)[:, class==1]

REPRODUCTION ANCHORS: the BASE column on ALL 16 must byte-reproduce round54's
base column, AND the SEEDAVG column on ALL 16 must byte-reproduce round54's
seedavg column (read from round54 results.csv). Both max|dev| must be 0.00e+00.

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
OUT_DIR = os.path.join(BENCH_DIR, "round56_colbag")
STATS_CSV = os.path.join(BENCH_DIR, "dataset_stats.csv")
ROUND54_RESULTS = os.path.join(BENCH_DIR, "round54_bootstrap_bag", "results.csv")

L2_GATE_THRESHOLD = 0.010   # shipped-08 ratio gate for l2 (FIXED)
GATED_L2 = 1.0
DEFAULT_MSL = 20
MSL_TIERS = [(0.030, 70), (0.015, 50)]
BASE_SEED = 0
N_DATASETS = 16
REPRO_TOL = 0.0             # byte reproduction: max|dev| must be exactly 0.0
COL_FRAC = 0.8             # fraction of feature columns each colbag member sees
K = 10                       # ensemble size for both colbag and seedavg

BASE = "base"
CAND = "colbag"              # column-subspace bagging K=10
CONTEXT = "seedavg"          # plain seed-avg K=10 (full data, full columns)
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


def round54_anchors(path=ROUND54_RESULTS):
    """Read round54's base_pub/base_prv AND seedavg_pub/seedavg_prv for ALL 16
    datasets to anchor reproduction at full precision. Returns
    dict name -> {"base": (pub, prv), "seedavg": (pub, prv)} or None if the
    file is absent."""
    if not os.path.exists(path):
        return None
    anchors = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            name = row.get("dataset")
            entry = {}
            try:
                entry["base"] = (float(row["base_pub"]), float(row["base_prv"]))
            except (KeyError, ValueError):
                entry["base"] = None
            try:
                entry["seedavg"] = (float(row["seedavg_pub"]),
                                    float(row["seedavg_prv"]))
            except (KeyError, ValueError):
                entry["seedavg"] = None
            anchors[name] = entry
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
    """Fit ONE shipped-08 HGB on `train_frame[features]` and return P(class==1)
    on test. validation_fraction / max_depth / interaction_cst / tol /
    monotonic_cst left UNSET (sklearn defaults, byte-identical to shipped 08).
    The ONLY things that vary across the ensemble are (a) `random_state` and
    (b) WHICH feature COLUMNS the member is trained/scored on (colbag) vs the
    full column set (base / seedavg). l2/msl are always the base-08 gate values
    computed from the ORIGINAL full train."""
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


def fit_hgb_proba_guarded(train_frame, test, features, cat_mask, l2, msl_val,
                          seed):
    """Like fit_hgb_proba but returns None if the fit is single-class (guard for
    colbag members). With full training rows the target always has both classes,
    so this is expected never to fire — but we handle it gracefully anyway."""
    clf = HistGradientBoostingClassifier(
        categorical_features=cat_mask,
        random_state=seed,
        max_iter=300,
        early_stopping=True,
        l2_regularization=l2,
        min_samples_leaf=msl_val,
    )
    clf.fit(train_frame[features], train_frame["target"])
    classes = list(clf.classes_)
    if len(classes) < 2:
        return None
    proba = clf.predict_proba(test[features])
    pos_idx = classes.index(1) if 1 in classes else 1
    return proba[:, pos_idx]


def run_one(name, train_csv, test_csv, stats):
    """Returns (preds, meta). preds maps config_name -> {row_id -> prob}.

    base     = seed-0, full columns, base-08 (byte-identical to shipped 08).
    colbag   = column-subspace bagging K=10 (mean of valid k proba vectors).
    seedavg  = plain seed-avg K=10 on FULL train, FULL columns (mean of proba,
               random_state 0..9).
    """
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)

    features = [c for c in train.columns if c not in ("row_id", "target")]
    cat_mask = [train[c].dtype == object for c in features]

    n = len(train)
    n_feat = len(features)
    ratio = n_feat / n
    l2 = GATED_L2 if ratio >= L2_GATE_THRESHOLD else 0.0
    msl_val = msl_for_ratio(ratio)

    row_ids = test["row_id"].tolist()
    n_test = len(test)

    preds = {}
    n_fits = 0

    # ---- BASE: seed-0, FULL columns, base-08 (byte-identical to shipped 08). --
    base_vec = fit_hgb_proba(train, test, features, cat_mask, l2, msl_val,
                             BASE_SEED)
    preds[BASE] = dict(zip(row_ids, base_vec.tolist()))
    n_fits += 1

    # ---- CAND "colbag": column-subspace bagging K=10. ----
    # single new mechanism = each member sees ALL rows but a random 80% column
    # subset chosen (without replacement) by a per-k RNG. cat_mask rebuilt for
    # the chosen columns; l2/msl fixed at the FULL-train base-08 gate values.
    k_cols = math.ceil(COL_FRAC * n_feat)
    use_all_cols = k_cols >= n_feat            # tiny n_feat edge case
    colbag_sum = np.zeros(n_test, dtype=np.float64)
    colbag_valid = 0
    colbag_skipped_k = []
    for k in range(K):
        if use_all_cols:
            chosen_idx = list(range(n_feat))
        else:
            rng = np.random.RandomState(k)
            chosen_idx = sorted(
                int(i) for i in rng.choice(n_feat, size=k_cols, replace=False))
        sub_features = [features[i] for i in chosen_idx]
        sub_mask = [train[c].dtype == object for c in sub_features]
        vec_k = fit_hgb_proba_guarded(train, test, sub_features, sub_mask,
                                      l2, msl_val, k)
        if vec_k is None:
            # GUARD: single-class member -> skip, average over the rest.
            colbag_skipped_k.append(k)
            continue
        colbag_sum += vec_k
        colbag_valid += 1
        n_fits += 1
    if colbag_valid == 0:
        # degenerate: no valid member -> fall back to base (should not happen).
        colbag_vec = base_vec.astype(np.float64)
    else:
        colbag_vec = colbag_sum / colbag_valid
    preds[CAND] = dict(zip(row_ids, colbag_vec.tolist()))

    # ---- CONTEXT "seedavg": plain seed-avg K=10 on FULL train/columns, 0..9. --
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
        "n_features": n_feat,
        "n_cat": sum(cat_mask),
        "k_cols": k_cols,
        "use_all_cols": use_all_cols,
        "colbag_valid_k": colbag_valid,
        "colbag_skipped_k": colbag_skipped_k,
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
    anchors54 = round54_anchors()
    rows = []
    exceptions = []
    skipped = []
    total_fits = 0
    total_colbag_skips = 0
    datasets_with_colbag_skips = 0

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
            nskip = len(meta["colbag_skipped_k"])
            total_colbag_skips += nskip
            if nskip:
                datasets_with_colbag_skips += 1
            rec.update({
                "n_train": meta["n_train"],
                "n_object_cols": meta["n_object_cols"],
                "l2": meta["l2"],
                "msl": meta["msl"],
                "n_features": meta["n_features"],
                "k_cols": meta["k_cols"],
                "colbag_valid_k": meta["colbag_valid_k"],
                "colbag_skipped_k": ";".join(
                    str(x) for x in meta["colbag_skipped_k"]),
            })
            for cfg in ALL_CONFIGS:
                pub, prv = score_split(preds[cfg], sol)
                rec[f"{cfg}_pub"] = pub
                rec[f"{cfg}_prv"] = prv
            print(f"[OK] {name} n_tr={meta['n_train']} obj={meta['n_object_cols']} "
                  f"feats={meta['n_features']} cat={meta['n_cat']} "
                  f"kcols={meta['k_cols']}{'(all)' if meta['use_all_cols'] else ''} "
                  f"l2={meta['l2']} msl={meta['msl']} "
                  f"colbag_validk={meta['colbag_valid_k']} skips={nskip} "
                  f"fits={meta['n_fits']} | "
                  f"base pub={rec['base_pub']:.6f} prv={rec['base_prv']:.6f}  "
                  f"colbag pub={rec['colbag_pub']:.6f} prv={rec['colbag_prv']:.6f}  "
                  f"seedavg pub={rec['seedavg_pub']:.6f} prv={rec['seedavg_prv']:.6f}")
        except Exception as e:
            exceptions.append((name, repr(e)))
            rec.update({"n_train": stats.get(name, {}).get("n_train", ""),
                        "n_object_cols": stats.get(name, {}).get("n_object_cols", ""),
                        "l2": float("nan"), "msl": float("nan"),
                        "n_features": "", "k_cols": "",
                        "colbag_valid_k": "", "colbag_skipped_k": ""})
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
                  "n_features", "k_cols", "colbag_valid_k", "colbag_skipped_k",
                  "base_pub", "base_prv",
                  "colbag_pub", "colbag_prv", "colbag_d_pub", "colbag_d_prv",
                  "seedavg_pub", "seedavg_prv", "seedavg_d_pub", "seedavg_d_prv"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            out = {k: r.get(k, "") for k in
                   ["dataset", "n_train", "n_object_cols", "l2", "msl",
                    "n_features", "k_cols", "colbag_valid_k", "colbag_skipped_k",
                    "base_pub", "base_prv",
                    "colbag_pub", "colbag_prv",
                    "seedavg_pub", "seedavg_prv"]}
            out["colbag_d_pub"] = delta(r, CAND, "pub")
            out["colbag_d_prv"] = delta(r, CAND, "prv")
            out["seedavg_d_pub"] = delta(r, CONTEXT, "pub")
            out["seedavg_d_prv"] = delta(r, CONTEXT, "prv")
            w.writerow(out)

    # ---- REPRODUCTION: base AND seedavg on ALL 16 byte-match round54 ----
    # Two independent anchors: base column and seedavg column. Both must be
    # exactly reproduced (max|dev| == 0.00e+00).
    repro_available = anchors54 is not None
    by_name = {r["dataset"]: r for r in rows}
    repro = {BASE: {}, CONTEXT: {}}
    repro_ok = {BASE: True, CONTEXT: True}
    max_abs_dev = {BASE: 0.0, CONTEXT: 0.0}
    for arm in (BASE, CONTEXT):
        for i in range(1, N_DATASETS + 1):
            nm = f"train_{i:02d}"
            r = by_name.get(nm)
            mine = ((r.get(f"{arm}_pub"), r.get(f"{arm}_prv"))
                    if r else (None, None))
            ref = anchors54.get(nm, {}).get(arm) if anchors54 else None
            if ref is None or mine[0] is None or mine[1] is None \
                    or (isinstance(mine[0], float) and math.isnan(mine[0])):
                okp = okv = False
                devp = devv = float("nan")
            else:
                devp = abs(mine[0] - ref[0])
                devv = abs(mine[1] - ref[1])
                okp = devp <= REPRO_TOL
                okv = devv <= REPRO_TOL
                max_abs_dev[arm] = max(max_abs_dev[arm], devp, devv)
            repro[arm][nm] = {"mine": mine, "ref": ref, "okp": okp, "okv": okv,
                              "devp": devp, "devv": devv}
            if not (okp and okv):
                repro_ok[arm] = False
    both_repro_ok = repro_available and repro_ok[BASE] and repro_ok[CONTEXT]

    # ---- partition sanity (all 16 present) ----
    present = {r["dataset"] for r in rows if not (
        isinstance(r.get("base_pub"), float) and math.isnan(r.get("base_pub")))}
    all16_ok = (len(present) == N_DATASETS and not skipped)

    # ================= SUMMARY =================
    L = []
    L.append("=" * 78)
    L.append("bench_03 round56 — COLUMN-SUBSPACE BAGGING (K=10) vs SEED-AVG "
             "(K=10) vs base-08 (ALL 16)  [OFFLINE]")
    L.append("=" * 78)
    L.append("")
    L.append("SETUP:")
    L.append("  base    == shipped 08: HGB, early_stopping, cat_mask=object cols,")
    L.append("             seed-0, full train, FULL columns.")
    L.append("  colbag  == column-subspace bagging K=10. Per k in 0..9: choose")
    L.append("             ceil(0.8*n_feat) feature columns WITHOUT replacement via")
    L.append("             RandomState(k), fit an identical base-08 HGB")
    L.append("             (random_state=k) on ALL rows but only those columns")
    L.append("             (cat_mask rebuilt for the subset), predict_proba on the")
    L.append("             SAME test columns; aggregate = MEAN of the valid-k proba")
    L.append("             vectors. l2/msl are the base-08 gate values from the FULL")
    L.append("             train (NOT recomputed on the subset). tiny n_feat")
    L.append("             (ceil>=n_feat) -> member uses ALL columns.")
    L.append("             GUARD: single-class member -> that k skipped.")
    L.append("  seedavg == plain seed-avg K=10: fit K base-08 HGB on the FULL train,")
    L.append("             FULL columns, random_state 0..9, MEAN of proba.")
    L.append("  Single new mechanism vs seedavg = COLUMN-subspace diversity.")

    # ---- COLBAG column-subspace detail + guard firing ----
    L.append("")
    L.append("=== COLUMN-SUBSPACE DETAIL (K=10, frac=0.8) & GUARD ===")
    L.append(f"{'dataset':<10} {'n_feat':>7} {'k_cols':>7} {'valid_k':>8} "
             f"{'skipped_k':>12}")
    for r in rows:
        nf = r.get("n_features", "")
        kc = r.get("k_cols", "")
        vk = r.get("colbag_valid_k", "")
        sk = r.get("colbag_skipped_k", "")
        L.append(f"{r['dataset']:<10} {str(nf):>7} {str(kc):>7} {str(vk):>8} "
                 f"{str(sk) if sk else '-':>12}")
    L.append(f"  datasets with any skip: {datasets_with_colbag_skips}, "
             f"total k skipped: {total_colbag_skips}")

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

    # ---- colbag vs seedavg head-to-head ----
    L.append("")
    L.append("=== HEAD-TO-HEAD: colbag vs seedavg (per-dataset, both vs base) ===")
    L.append(f"{'dataset':<10} {'cb_dpub':>11} {'sa_dpub':>11} "
             f"{'cb_dprv':>11} {'sa_dprv':>11}")
    cb_beats_sa_pub = sa_beats_cb_pub = 0
    cb_beats_sa_prv = sa_beats_cb_prv = 0
    for r in rows:
        bdp = delta(r, CAND, "pub")
        sdp = delta(r, CONTEXT, "pub")
        bdv = delta(r, CAND, "prv")
        sdv = delta(r, CONTEXT, "prv")
        if not math.isnan(bdp) and not math.isnan(sdp):
            if bdp > sdp + 1e-9:
                cb_beats_sa_pub += 1
            elif sdp > bdp + 1e-9:
                sa_beats_cb_pub += 1
        if not math.isnan(bdv) and not math.isnan(sdv):
            if bdv > sdv + 1e-9:
                cb_beats_sa_prv += 1
            elif sdv > bdv + 1e-9:
                sa_beats_cb_prv += 1

        def fmt(x):
            return f"{x:>+11.6f}" if not math.isnan(x) else f"{'nan':>11}"
        L.append(f"{r['dataset']:<10} {fmt(bdp)} {fmt(sdp)} {fmt(bdv)} {fmt(sdv)}")
    L.append(f"  Public : colbag>seedavg on {cb_beats_sa_pub}, "
             f"seedavg>colbag on {sa_beats_cb_pub}")
    L.append(f"  Private: colbag>seedavg on {cb_beats_sa_prv}, "
             f"seedavg>colbag on {sa_beats_cb_prv}")

    # ---- NOISY SMALL-N focus ----
    L.append("")
    L.append("=== NOISY SMALL-N FOCUS (train_05, train_09, train_13) ===")
    focus = {"train_05", "train_09", "train_13"}
    for r in rows:
        if r["dataset"] in focus:
            L.append(f"  {r['dataset']} (n={r.get('n_train','')}): "
                     f"colbag ΔPub={delta(r, CAND, 'pub'):+.6f} "
                     f"ΔPrv={delta(r, CAND, 'prv'):+.6f} | "
                     f"seedavg ΔPub={delta(r, CONTEXT, 'pub'):+.6f} "
                     f"ΔPrv={delta(r, CONTEXT, 'prv'):+.6f}")

    # ---- REPRODUCTION ----
    L.append("")
    L.append("=== REPRODUCTION CHECK (base & seedavg on ALL 16 vs round54, "
             "byte-exact) ===")
    if not repro_available:
        L.append("  round54 results.csv NOT found -> reproduction NOT anchored "
                 "(FAIL).")
    else:
        for arm in (BASE, CONTEXT):
            L.append(f"  -- {arm} anchor --")
            for i in range(1, N_DATASETS + 1):
                nm = f"train_{i:02d}"
                rr = repro[arm][nm]
                mp_, mv_ = rr["mine"]
                rp_, rv_ = rr["ref"] if rr["ref"] else (float("nan"),
                                                        float("nan"))
                L.append(
                    f"    {nm}: Public {mp_:.6f} vs r54 {rp_:.6f} "
                    f"(|d|={rr['devp']:.2e}, {'YES' if rr['okp'] else 'NO'}); "
                    f"Private {mv_:.6f} vs r54 {rv_:.6f} "
                    f"(|d|={rr['devv']:.2e}, {'YES' if rr['okv'] else 'NO'})")
            L.append(f"    max |dev| ({arm}) over 16x2 = {max_abs_dev[arm]:.2e}  "
                     f"-> {'PASS' if repro_ok[arm] else 'FAIL'}")
        L.append(f"  REPRODUCTION (both anchors): "
                 f"{'PASS' if both_repro_ok else 'FAIL'}")

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

    is_adopt_any = any(adopt_flags.values())
    adopted = [a for a in ENSEMBLES if adopt_flags[a]]

    # colbag vs seedavg verdict (does column subspace beat plain seed-avg?)
    cb_mp, cb_mv = head[CAND][0], head[CAND][1]
    sa_mp, sa_mv = head[CONTEXT][0], head[CONTEXT][1]
    cb_beats_sa_mean = (cb_mp > sa_mp) and (cb_mv > sa_mv)
    cb_beats_base_mean = (cb_mp > 0) and (cb_mv > 0)

    L.append("=== VERDICT ===")
    if cb_beats_sa_mean:
        L.append(f"  Column-subspace bagging BEATS plain seed-avg on BOTH mean "
                 f"splits (colbag ΔPub {cb_mp:+.6f} > seedavg {sa_mp:+.6f}; "
                 f"colbag ΔPrv {cb_mv:+.6f} > seedavg {sa_mv:+.6f}).")
    else:
        L.append(f"  Column-subspace bagging does NOT beat plain seed-avg on both "
                 f"mean splits (colbag ΔPub {cb_mp:+.6f} vs seedavg {sa_mp:+.6f}; "
                 f"colbag ΔPrv {cb_mv:+.6f} vs seedavg {sa_mv:+.6f}).")
    if cb_beats_base_mean:
        L.append(f"  Column-subspace bagging beats base on both mean splits "
                 f"(ΔPub {cb_mp:+.6f}, ΔPrv {cb_mv:+.6f}).")
    else:
        L.append(f"  Column-subspace bagging does NOT beat base on both mean "
                 f"splits (ΔPub {cb_mp:+.6f}, ΔPrv {cb_mv:+.6f}).")
    if is_adopt_any:
        L.append(f"  ADOPT: {', '.join(adopted)} cleanly improve BOTH splits with "
                 f"zero regression.")
    else:
        L.append("  REJECT (clean-win test): no arm improves BOTH splits with zero "
                 "regressions across all 16.")

    ship = "ADOPT" if is_adopt_any else "REJECT"
    L.append("")
    L.append(f"SHIP VERDICT: {ship}")

    # ---- CLEAN RUN marker ----
    clean_run = ((not exceptions) and all16_ok and both_repro_ok
                 and (not skipped) and total_colbag_skips == 0)
    L.append("")
    L.append(f"CLEAN RUN: {'YES' if clean_run else 'NO'} "
             f"({len(exceptions)} exceptions)  "
             f"[total_fits={total_fits}, datasets_scored={len(present)}/16, "
             f"skipped={len(skipped)}, "
             f"reproduction={'YES' if both_repro_ok else 'NO'}, "
             f"base_repro_maxdev={max_abs_dev[BASE]:.2e}, "
             f"seedavg_repro_maxdev={max_abs_dev[CONTEXT]:.2e}, "
             f"colbag_single_class_skips={total_colbag_skips}]")
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
          f"BASE_REPRO_MAXDEV={max_abs_dev[BASE]:.2e} "
          f"SEEDAVG_REPRO_MAXDEV={max_abs_dev[CONTEXT]:.2e} "
          f"COLBAG_SKIPS={total_colbag_skips}")
    print("HARNESS_DONE")


if __name__ == "__main__":
    main()
