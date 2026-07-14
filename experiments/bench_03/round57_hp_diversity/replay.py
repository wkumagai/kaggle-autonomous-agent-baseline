#!/usr/bin/env python
"""
bench_03 round57 — HYPERPARAMETER-DIVERSITY ensembling (K=10, learning_rate grid)
vs SEED-AVG (K=10) vs base-08 (ALL 16).
OFFLINE ONLY. No subprocess, no LLM, no Kaggle, no network. Calls sklearn
in-process only. Writes ONLY under this round57 directory; NEVER touches
submissions/.

GOAL (improvement-log angle "hpdiv")
------------------------------------
Prior rounds closed THREE ensemble-diversity sources vs base-08:
  * round54  ROW-level bootstrap resampling (each member sees a resampled row set)
  * round56  COLUMN-subspace bagging (each member sees a random 80% column subset)
  * seed-avg (round29+): fit K base-08 HGBs on the FULL data varying ONLY the
             `random_state`, mean the test probabilities (model-internal noise).
This round isolates a FOURTH source: HYPERPARAMETER diversity. Fit K=10 base-08
HGBs on the FULL training data with a SINGLE fixed random_state=0 for EVERY
member, varying ONLY the `learning_rate` across a small log-spaced grid that is
log-symmetric around the base learning_rate. Then average the test
probabilities. This asks: does learning_rate diversity beat base-08, and does it
beat the seed-avg control (which injects only random_state noise)?

Design (single new mechanism vs seedavg = learning_rate variation):
  BASE arm     = base-08 HGB exactly (reference column), seed-0, FULL columns,
                 full train, DEFAULT learning_rate (0.1, UNSET). max_depth /
                 interaction_cst / tol / monotonic_cst / learning_rate left
                 UNSET; byte-identical to shipped 08 -> reproduces round56's base
                 column EXACTLY.
  CAND arm     = "hpdiv": learning_rate-diversity ensemble, K=10. For k in 0..9:
                   - random_state = 0 for EVERY member (NO seed diversity).
                   - learning_rate = BASE_LR * factors[k], where
                       factors = np.exp(np.linspace(np.log(0.5), np.log(2.0), 10))
                     (log-symmetric around the base learning_rate, i.e. member
                     rates span 0.5x .. 2.0x of base).
                   - ALL other params = base-08 (full train, FULL columns,
                     cat_mask, l2/msl gate values from the FULL train).
                   - predict_proba(X_test)[:, class==1] per member.
                 Aggregate = arithmetic MEAN of the K test-proba vectors.
                 GUARD: if a member yields a single-class fit, that k is SKIPPED
                 and the mean is taken over the rest (expected skips = 0, since
                 every member trains on the full target).
  CONTEXT arm  = "seedavg": plain seed-avg K=10. Fit K=10 base-08 HGB on the
                 FULL train, FULL columns, DEFAULT learning_rate, random_state
                 0..9, mean proba. This reproduces round56's seedavg column
                 EXACTLY.

BASE recipe reproduced (== shipped 08), identical to round56:
  features = [c for c in train.columns if c not in ("row_id","target")]
  cat_mask = [train[c].dtype == object for c in features]
  n=len(train); ratio = len(features)/n
  l2  = 1.0 if ratio >= 0.010 else 0.0
  msl = 70 if ratio>=0.030 else 50 if ratio>=0.015 else 20
  HistGradientBoostingClassifier(categorical_features=cat_mask, random_state=0,
      max_iter=300, early_stopping=True, l2_regularization=l2,
      min_samples_leaf=msl)   # learning_rate left UNSET (default 0.1)
  pred = predict_proba(test)[:, class==1]

REPRODUCTION ANCHORS: the BASE column on ALL 16 must byte-reproduce round56's
base column, AND the SEEDAVG column on ALL 16 must byte-reproduce round56's
seedavg column (read from round56 results.csv). Both max|dev| must be 0.00e+00.

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
OUT_DIR = os.path.join(BENCH_DIR, "round57_hp_diversity")
STATS_CSV = os.path.join(BENCH_DIR, "dataset_stats.csv")
ROUND56_RESULTS = os.path.join(BENCH_DIR, "round56_colbag", "results.csv")

L2_GATE_THRESHOLD = 0.010   # shipped-08 ratio gate for l2 (FIXED)
GATED_L2 = 1.0
DEFAULT_MSL = 20
MSL_TIERS = [(0.030, 70), (0.015, 50)]
BASE_SEED = 0
N_DATASETS = 16
REPRO_TOL = 0.0             # byte reproduction: max|dev| must be exactly 0.0
BASE_LR = 0.1               # base-08 HGB learning_rate (sklearn default, UNSET)
K = 10                      # ensemble size for both hpdiv and seedavg
# log-symmetric learning_rate multipliers around the base rate (0.5x .. 2.0x).
LR_FACTORS = np.exp(np.linspace(np.log(0.5), np.log(2.0), K))

BASE = "base"
CAND = "hpdiv"              # learning_rate-diversity ensemble K=10 (seed fixed 0)
CONTEXT = "seedavg"         # plain seed-avg K=10 (full data, full columns)
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


def round56_anchors(path=ROUND56_RESULTS):
    """Read round56's base_pub/base_prv AND seedavg_pub/seedavg_prv for ALL 16
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


def _make_hgb(cat_mask, l2, msl_val, seed, learning_rate=None):
    """Build a base-08 HGB. When learning_rate is None it is left UNSET (sklearn
    default 0.1) so base/seedavg stay byte-identical to shipped 08. hpdiv passes
    an explicit learning_rate — the ONLY parameter that varies across its
    members (random_state is fixed at 0 for all of them)."""
    kwargs = dict(
        categorical_features=cat_mask,
        random_state=seed,
        max_iter=300,
        early_stopping=True,
        l2_regularization=l2,
        min_samples_leaf=msl_val,
    )
    if learning_rate is not None:
        kwargs["learning_rate"] = learning_rate
    return HistGradientBoostingClassifier(**kwargs)


def fit_hgb_proba(train_frame, test, features, cat_mask, l2, msl_val, seed,
                  learning_rate=None):
    """Fit ONE shipped-08 HGB on `train_frame[features]` and return P(class==1)
    on test. validation_fraction / max_depth / interaction_cst / tol /
    monotonic_cst left UNSET (sklearn defaults, byte-identical to shipped 08).
    learning_rate is UNSET for base/seedavg (default 0.1) and varied for hpdiv.
    l2/msl are always the base-08 gate values computed from the full train."""
    clf = _make_hgb(cat_mask, l2, msl_val, seed, learning_rate)
    clf.fit(train_frame[features], train_frame["target"])
    proba = clf.predict_proba(test[features])
    classes = list(clf.classes_)
    pos_idx = classes.index(1) if 1 in classes else 1
    return proba[:, pos_idx]


def fit_hgb_proba_guarded(train_frame, test, features, cat_mask, l2, msl_val,
                          seed, learning_rate=None):
    """Like fit_hgb_proba but returns None if the fit is single-class (guard for
    hpdiv members). With full training rows the target always has both classes,
    so this is expected never to fire — but we handle it gracefully anyway."""
    clf = _make_hgb(cat_mask, l2, msl_val, seed, learning_rate)
    clf.fit(train_frame[features], train_frame["target"])
    classes = list(clf.classes_)
    if len(classes) < 2:
        return None
    proba = clf.predict_proba(test[features])
    pos_idx = classes.index(1) if 1 in classes else 1
    return proba[:, pos_idx]


def run_one(name, train_csv, test_csv, stats):
    """Returns (preds, meta). preds maps config_name -> {row_id -> prob}.

    base     = seed-0, full columns, default lr, base-08 (byte-identical to 08).
    hpdiv    = learning_rate-diversity K=10 (seed fixed 0, lr = base*factors[k]);
               mean of valid-k proba vectors.
    seedavg  = plain seed-avg K=10 on FULL train, FULL columns, default lr
               (mean of proba, random_state 0..9).
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

    # ---- BASE: seed-0, FULL columns, default lr, base-08 (byte-identical). ----
    base_vec = fit_hgb_proba(train, test, features, cat_mask, l2, msl_val,
                             BASE_SEED)
    preds[BASE] = dict(zip(row_ids, base_vec.tolist()))
    n_fits += 1

    # ---- CAND "hpdiv": learning_rate-diversity ensemble K=10. ----
    # single new mechanism = each member trains on ALL rows / FULL columns with
    # random_state FIXED at 0, varying ONLY learning_rate = BASE_LR*factors[k]
    # (log-symmetric grid around the base rate). l2/msl fixed at base-08 gate.
    hp_sum = np.zeros(n_test, dtype=np.float64)
    hp_valid = 0
    hp_skipped_k = []
    hp_lrs = []
    for k in range(K):
        lr_k = BASE_LR * float(LR_FACTORS[k])
        hp_lrs.append(lr_k)
        vec_k = fit_hgb_proba_guarded(train, test, features, cat_mask, l2,
                                      msl_val, BASE_SEED, learning_rate=lr_k)
        if vec_k is None:
            # GUARD: single-class member -> skip, average over the rest.
            hp_skipped_k.append(k)
            continue
        hp_sum += vec_k
        hp_valid += 1
        n_fits += 1
    if hp_valid == 0:
        # degenerate: no valid member -> fall back to base (should not happen).
        hpdiv_vec = base_vec.astype(np.float64)
    else:
        hpdiv_vec = hp_sum / hp_valid
    preds[CAND] = dict(zip(row_ids, hpdiv_vec.tolist()))

    # ---- CONTEXT "seedavg": plain seed-avg K=10, FULL train/columns, 0..9. ----
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
        "hpdiv_valid_k": hp_valid,
        "hpdiv_skipped_k": hp_skipped_k,
        "hp_lrs": hp_lrs,
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
    anchors56 = round56_anchors()
    rows = []
    exceptions = []
    skipped = []
    total_fits = 0
    total_hpdiv_skips = 0
    datasets_with_hpdiv_skips = 0

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
            nskip = len(meta["hpdiv_skipped_k"])
            total_hpdiv_skips += nskip
            if nskip:
                datasets_with_hpdiv_skips += 1
            rec.update({
                "n_train": meta["n_train"],
                "n_object_cols": meta["n_object_cols"],
                "l2": meta["l2"],
                "msl": meta["msl"],
                "n_features": meta["n_features"],
                "hpdiv_valid_k": meta["hpdiv_valid_k"],
                "hpdiv_skipped_k": ";".join(
                    str(x) for x in meta["hpdiv_skipped_k"]),
            })
            for cfg in ALL_CONFIGS:
                pub, prv = score_split(preds[cfg], sol)
                rec[f"{cfg}_pub"] = pub
                rec[f"{cfg}_prv"] = prv
            print(f"[OK] {name} n_tr={meta['n_train']} obj={meta['n_object_cols']} "
                  f"feats={meta['n_features']} cat={meta['n_cat']} "
                  f"l2={meta['l2']} msl={meta['msl']} "
                  f"hpdiv_validk={meta['hpdiv_valid_k']} skips={nskip} "
                  f"fits={meta['n_fits']} | "
                  f"base pub={rec['base_pub']:.6f} prv={rec['base_prv']:.6f}  "
                  f"hpdiv pub={rec['hpdiv_pub']:.6f} prv={rec['hpdiv_prv']:.6f}  "
                  f"seedavg pub={rec['seedavg_pub']:.6f} prv={rec['seedavg_prv']:.6f}")
        except Exception as e:
            exceptions.append((name, repr(e)))
            rec.update({"n_train": stats.get(name, {}).get("n_train", ""),
                        "n_object_cols": stats.get(name, {}).get("n_object_cols", ""),
                        "l2": float("nan"), "msl": float("nan"),
                        "n_features": "",
                        "hpdiv_valid_k": "", "hpdiv_skipped_k": ""})
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

    # ---- gate-C view: n_object_cols > 0 subset (secondary; full-16 primary) ----
    def is_gate_c(r):
        v = r.get("n_object_cols")
        try:
            return int(v) > 0
        except (TypeError, ValueError):
            return False

    def mean_delta_gatec(arm, split):
        vals = [delta(r, arm, split) for r in rows if is_gate_c(r)]
        vals = [v for v in vals if not math.isnan(v)]
        return sum(vals) / len(vals) if vals else float("nan")

    def wlt_gatec(arm, split, eps=1e-9):
        w = l = t = 0
        for r in rows:
            if not is_gate_c(r):
                continue
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

    # ---- results.csv ----
    csv_path = os.path.join(OUT_DIR, "results.csv")
    fieldnames = ["dataset", "n_train", "n_object_cols", "l2", "msl",
                  "n_features", "hpdiv_valid_k", "hpdiv_skipped_k",
                  "base_pub", "base_prv",
                  "hpdiv_pub", "hpdiv_prv", "hpdiv_d_pub", "hpdiv_d_prv",
                  "seedavg_pub", "seedavg_prv", "seedavg_d_pub", "seedavg_d_prv"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            out = {k: r.get(k, "") for k in
                   ["dataset", "n_train", "n_object_cols", "l2", "msl",
                    "n_features", "hpdiv_valid_k", "hpdiv_skipped_k",
                    "base_pub", "base_prv",
                    "hpdiv_pub", "hpdiv_prv",
                    "seedavg_pub", "seedavg_prv"]}
            out["hpdiv_d_pub"] = delta(r, CAND, "pub")
            out["hpdiv_d_prv"] = delta(r, CAND, "prv")
            out["seedavg_d_pub"] = delta(r, CONTEXT, "pub")
            out["seedavg_d_prv"] = delta(r, CONTEXT, "prv")
            w.writerow(out)

    # ---- REPRODUCTION: base AND seedavg on ALL 16 byte-match round56 ----
    # Two independent anchors: base column and seedavg column. Both must be
    # exactly reproduced (max|dev| == 0.00e+00).
    repro_available = anchors56 is not None
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
            ref = anchors56.get(nm, {}).get(arm) if anchors56 else None
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
    L.append("bench_03 round57 — HYPERPARAMETER-DIVERSITY (learning_rate grid, "
             "K=10) vs SEED-AVG (K=10) vs base-08 (ALL 16)  [OFFLINE]")
    L.append("=" * 78)
    L.append("")
    L.append("SETUP:")
    L.append("  base    == shipped 08: HGB, early_stopping, cat_mask=object cols,")
    L.append("             seed-0, full train, FULL columns, default learning_rate.")
    L.append("  hpdiv   == learning_rate-diversity K=10. Per k in 0..9: random_state")
    L.append("             FIXED at 0, learning_rate = BASE_LR*factors[k] with")
    L.append("             factors = exp(linspace(log(0.5), log(2.0), 10)) (log-")
    L.append("             symmetric around the base rate, 0.5x..2.0x). ALL other")
    L.append("             params = base-08 (full train, FULL columns, l2/msl gate).")
    L.append("             Aggregate = MEAN of the valid-k proba vectors.")
    L.append("             GUARD: single-class member -> that k skipped.")
    L.append("  seedavg == plain seed-avg K=10: fit K base-08 HGB on the FULL train,")
    L.append("             FULL columns, default lr, random_state 0..9, MEAN of proba.")
    L.append("  Single new mechanism vs seedavg = LEARNING_RATE diversity (seed fixed).")
    L.append("")
    L.append("  learning_rate grid (BASE_LR=%.3f):" % BASE_LR)
    L.append("    factors = [" + ", ".join(f"{x:.4f}" for x in LR_FACTORS) + "]")
    L.append("    rates   = [" + ", ".join(f"{BASE_LR * x:.4f}"
                                            for x in LR_FACTORS) + "]")

    # ---- hpdiv guard firing detail ----
    L.append("")
    L.append("=== HPDIV MEMBER DETAIL (K=10) & GUARD ===")
    L.append(f"{'dataset':<10} {'n_feat':>7} {'valid_k':>8} {'skipped_k':>12}")
    for r in rows:
        nf = r.get("n_features", "")
        vk = r.get("hpdiv_valid_k", "")
        sk = r.get("hpdiv_skipped_k", "")
        L.append(f"{r['dataset']:<10} {str(nf):>7} {str(vk):>8} "
                 f"{str(sk) if sk else '-':>12}")
    L.append(f"  datasets with any skip: {datasets_with_hpdiv_skips}, "
             f"total k skipped: {total_hpdiv_skips}")

    # ---- SWEEP HEADLINE (full 16) ----
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

    # ---- GATE-C view (n_object_cols>0), secondary ----
    L.append("")
    L.append("=== GATE-C VIEW (n_object_cols>0 subset, SECONDARY) ===")
    n_gatec = sum(1 for r in rows if is_gate_c(r))
    L.append(f"  gate-C datasets: {n_gatec}")
    for arm in ENSEMBLES:
        mp = mean_delta_gatec(arm, "pub")
        mv = mean_delta_gatec(arm, "prv")
        wp, lp, tp = wlt_gatec(arm, "pub")
        wv, lv, tv = wlt_gatec(arm, "prv")
        L.append(f"  {arm:<8}: mean ΔPublic={mp:+.6f} (W/L/T {wp}/{lp}/{tp})  "
                 f"mean ΔPrivate={mv:+.6f} (W/L/T {wv}/{lv}/{tv})")

    # ---- hpdiv vs seedavg head-to-head ----
    L.append("")
    L.append("=== HEAD-TO-HEAD: hpdiv vs seedavg (per-dataset, both vs base) ===")
    L.append(f"{'dataset':<10} {'hp_dpub':>11} {'sa_dpub':>11} "
             f"{'hp_dprv':>11} {'sa_dprv':>11}")
    hp_beats_sa_pub = sa_beats_hp_pub = 0
    hp_beats_sa_prv = sa_beats_hp_prv = 0
    for r in rows:
        bdp = delta(r, CAND, "pub")
        sdp = delta(r, CONTEXT, "pub")
        bdv = delta(r, CAND, "prv")
        sdv = delta(r, CONTEXT, "prv")
        if not math.isnan(bdp) and not math.isnan(sdp):
            if bdp > sdp + 1e-9:
                hp_beats_sa_pub += 1
            elif sdp > bdp + 1e-9:
                sa_beats_hp_pub += 1
        if not math.isnan(bdv) and not math.isnan(sdv):
            if bdv > sdv + 1e-9:
                hp_beats_sa_prv += 1
            elif sdv > bdv + 1e-9:
                sa_beats_hp_prv += 1

        def fmt(x):
            return f"{x:>+11.6f}" if not math.isnan(x) else f"{'nan':>11}"
        L.append(f"{r['dataset']:<10} {fmt(bdp)} {fmt(sdp)} {fmt(bdv)} {fmt(sdv)}")
    L.append(f"  Public : hpdiv>seedavg on {hp_beats_sa_pub}, "
             f"seedavg>hpdiv on {sa_beats_hp_pub}")
    L.append(f"  Private: hpdiv>seedavg on {hp_beats_sa_prv}, "
             f"seedavg>hpdiv on {sa_beats_hp_prv}")

    # ---- NOISY SMALL-N focus ----
    L.append("")
    L.append("=== NOISY SMALL-N FOCUS (train_05, train_09, train_13) ===")
    focus = {"train_05", "train_09", "train_13"}
    for r in rows:
        if r["dataset"] in focus:
            L.append(f"  {r['dataset']} (n={r.get('n_train','')}): "
                     f"hpdiv ΔPub={delta(r, CAND, 'pub'):+.6f} "
                     f"ΔPrv={delta(r, CAND, 'prv'):+.6f} | "
                     f"seedavg ΔPub={delta(r, CONTEXT, 'pub'):+.6f} "
                     f"ΔPrv={delta(r, CONTEXT, 'prv'):+.6f}")

    # ---- REPRODUCTION ----
    L.append("")
    L.append("=== REPRODUCTION CHECK (base & seedavg on ALL 16 vs round56, "
             "byte-exact) ===")
    if not repro_available:
        L.append("  round56 results.csv NOT found -> reproduction NOT anchored "
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
                    f"    {nm}: Public {mp_:.6f} vs r56 {rp_:.6f} "
                    f"(|d|={rr['devp']:.2e}, {'YES' if rr['okp'] else 'NO'}); "
                    f"Private {mv_:.6f} vs r56 {rv_:.6f} "
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

    # hpdiv vs seedavg verdict (does learning_rate diversity beat plain seed-avg?)
    hp_mp, hp_mv = head[CAND][0], head[CAND][1]
    sa_mp, sa_mv = head[CONTEXT][0], head[CONTEXT][1]
    hp_beats_sa_mean = (hp_mp > sa_mp) and (hp_mv > sa_mv)
    hp_beats_base_mean = (hp_mp > 0) and (hp_mv > 0)
    hp_zero_regs = (not regressions(CAND, "pub")) and (not regressions(CAND, "prv"))

    L.append("=== VERDICT ===")
    if hp_beats_sa_mean:
        L.append(f"  Learning-rate diversity BEATS plain seed-avg on BOTH mean "
                 f"splits (hpdiv ΔPub {hp_mp:+.6f} > seedavg {sa_mp:+.6f}; "
                 f"hpdiv ΔPrv {hp_mv:+.6f} > seedavg {sa_mv:+.6f}).")
    else:
        L.append(f"  Learning-rate diversity does NOT beat plain seed-avg on both "
                 f"mean splits (hpdiv ΔPub {hp_mp:+.6f} vs seedavg {sa_mp:+.6f}; "
                 f"hpdiv ΔPrv {hp_mv:+.6f} vs seedavg {sa_mv:+.6f}).")
    if hp_beats_base_mean and hp_zero_regs:
        L.append(f"  Learning-rate diversity beats base on both mean splits with "
                 f"ZERO regressions (ΔPub {hp_mp:+.6f}, ΔPrv {hp_mv:+.6f}).")
    elif hp_beats_base_mean:
        L.append(f"  Learning-rate diversity beats base on both mean splits but "
                 f"HAS regressions (ΔPub {hp_mp:+.6f}, ΔPrv {hp_mv:+.6f}).")
    else:
        L.append(f"  Learning-rate diversity does NOT beat base on both mean "
                 f"splits (ΔPub {hp_mp:+.6f}, ΔPrv {hp_mv:+.6f}).")
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
                 and (not skipped) and total_hpdiv_skips == 0)
    L.append("")
    L.append(f"CLEAN RUN: {'YES' if clean_run else 'NO'} "
             f"({len(exceptions)} exceptions)  "
             f"[total_fits={total_fits}, datasets_scored={len(present)}/16, "
             f"skipped={len(skipped)}, "
             f"reproduction={'YES' if both_repro_ok else 'NO'}, "
             f"base_repro_maxdev={max_abs_dev[BASE]:.2e}, "
             f"seedavg_repro_maxdev={max_abs_dev[CONTEXT]:.2e}, "
             f"hpdiv_single_class_skips={total_hpdiv_skips}]")
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
          f"HPDIV_SKIPS={total_hpdiv_skips}")
    print("HARNESS_DONE")


if __name__ == "__main__":
    main()
