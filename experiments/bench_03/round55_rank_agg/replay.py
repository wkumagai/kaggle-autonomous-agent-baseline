#!/usr/bin/env python
"""
bench_03 round55 — RANK-SPACE avg (K=10) vs PROB-SPACE avg (K=10) vs base-08
(ALL 16).
OFFLINE ONLY. No subprocess, no LLM, no Kaggle, no network. Calls sklearn
in-process only. Writes ONLY under this round55 directory; NEVER touches
submissions/.

GOAL (improvement-log angle "rank_agg")
---------------------------------------
Prior rounds (round45/46) only compared PROBABILITY-space aggregations of the
K seed-averaged members (mean / median / trim10). The competition metric is
ROC AUC, which is purely RANK-based, so averaging the per-seed RANKS instead of
the per-seed PROBABILITIES is a distinct, untried aggregation. This round tests
that orthogonal angle.

Hypothesis: rank-mean might beat prob-mean when the K seeds disagree on score
SCALE; likely near-equal since it is the same model varying only random_state,
but it is untried and cheap.

Design (single mechanism = aggregation function over K=10 seed-avg members):
  BASE arm     = base-08 HGB exactly (reference column), seed-0, full train.
                 monotonic_cst / max_depth / interaction_cst / tol /
                 validation_fraction UNSET; byte-identical to shipped 08 ->
                 reproduces round54 base column.
  Cached seeds = fit K=10 base-08 HGB on the FULL train, random_state=0..9,
                 collect the K proba vectors P(class==1) on the test set. This
                 is EXACTLY round54's `seedavg` computation. The K fits are done
                 ONCE and cached; BOTH aggregation arms below are computed from
                 the SAME cached vectors (NO extra fits).
  PROBMEAN arm = arithmetic MEAN of the K proba vectors (the standard shipped
                 seed-avg). Must reproduce round54's `seedavg` column exactly.
  RANKMEAN arm = for each of the K proba vectors, rank-transform it over the
                 test rows via scipy.stats.rankdata(vec, method="average")
                 (average ties), then take the arithmetic MEAN of the K rank
                 vectors elementwise -> use that mean-rank vector as the score.
                 (AUC is invariant to monotone transforms of a single vector, so
                 ranks need not be normalized; we divide by n_test purely for
                 readability -- it does not change AUC.)

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

GATING for ship interpretation:
  Results are reported BOTH un-gated (all 16, direct head-to-head) AND gated on
  gate C = (n_object_cols > 0), the established ship gate. On non-gate-C
  datasets (n_object_cols==0) the ship recommendation falls back to base-08
  (single seed-0 model), so both arms revert to base (delta 0) there. The gated
  view therefore reports the mean delta / W/L/T computed ONLY over the
  gate-C-firing datasets, which is what would actually ship.

REPRODUCTION (MANDATORY — proves the harness is faithful):
  1. BASE column on ALL 16 must match round54's base column (round54
     results.csv, base_pub/base_prv) to < 5e-6.
  2. PROBMEAN column on ALL 16 must match round54's seedavg column
     (seedavg_pub/seedavg_prv) to < 5e-6 (same seeds 0..9, same fits, same mean
     -> should be ~0 dev). If either check fails, CLEAN RUN = NO.

ADOPTION (same spirit as round54, evaluated on the gate-C-only view since that
  is what ships): an aggregation arm is ADOPT iff it cleanly improves over base
  — mean ΔPublic > 0 AND mean ΔPrivate > 0 with ZERO regression on EITHER split.
  Additionally a VERDICT reports whether RANKMEAN beats PROBMEAN on BOTH mean
  splits (gate-C view).
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
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

REPO = "/Users/kumacmini/kaggle-autonomous-agent-baseline-auto"
DATA_DIR = os.path.join(REPO, "data")
BENCH_DIR = os.path.join(REPO, "experiments", "bench_03")
OUT_DIR = os.path.join(BENCH_DIR, "round55_rank_agg")
STATS_CSV = os.path.join(BENCH_DIR, "dataset_stats.csv")
ROUND54_RESULTS = os.path.join(BENCH_DIR, "round54_bootstrap_bag", "results.csv")

L2_GATE_THRESHOLD = 0.010   # shipped-08 ratio gate for l2 (FIXED)
GATED_L2 = 1.0
DEFAULT_MSL = 20
MSL_TIERS = [(0.030, 70), (0.015, 50)]
BASE_SEED = 0
N_DATASETS = 16
REPRO_TOL = 5e-6
K = 10                       # ensemble size (seed-avg members)

BASE = "base"
PROBMEAN = "probmean"        # arithmetic mean of the K proba vectors (== seedavg)
RANKMEAN = "rankmean"        # mean of the K rank-transformed vectors
ENSEMBLES = [PROBMEAN, RANKMEAN]
ALL_CONFIGS = [BASE, PROBMEAN, RANKMEAN]


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
    datasets to anchor reproduction at full precision. Returns dict
    name -> {"base": (pub, prv), "seedavg": (pub, prv)} or None if unavailable."""
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
    """Fit ONE shipped-08 HGB on `train_frame` and return P(class==1) on test.
    validation_fraction / max_depth / interaction_cst / tol / monotonic_cst
    left UNSET (sklearn defaults, byte-identical to shipped 08). The ONLY thing
    that varies across the ensemble is `random_state`. l2/msl are always the
    base-08 gate values computed from the full train."""
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
    """Returns (preds, meta). preds maps config_name -> {row_id -> score}.

    base     = seed-0, base-08 (byte-identical to shipped 08).
    probmean = MEAN of the K cached proba vectors (== round54 seedavg).
    rankmean = MEAN of the K rank-transformed cached vectors (rankdata avg-ties).

    The K seed fits (random_state 0..9 on the FULL train) are done ONCE and
    cached; BOTH aggregation arms are computed from the SAME cached vectors.
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

    # ---- CACHE the K seed-avg member proba vectors ONCE (seeds 0..9). ----
    # This is EXACTLY round54's seedavg computation. Both aggregation arms are
    # derived from these SAME cached vectors below (NO extra fits).
    seed_vecs = np.zeros((K, n_test), dtype=np.float64)
    for k in range(K):
        seed_vecs[k] = fit_hgb_proba(train, test, features, cat_mask, l2,
                                     msl_val, k)
        n_fits += 1

    # ---- PROBMEAN: arithmetic mean of the K proba vectors (== seedavg). ----
    probmean_vec = seed_vecs.mean(axis=0)
    preds[PROBMEAN] = dict(zip(row_ids, probmean_vec.tolist()))

    # ---- RANKMEAN: mean of the K rank-transformed vectors (avg ties). ----
    # rank each proba vector over the n_test rows, then average elementwise.
    # divide by n_test only for readability; AUC is invariant to this scaling.
    rank_sum = np.zeros(n_test, dtype=np.float64)
    for k in range(K):
        rank_sum += rankdata(seed_vecs[k], method="average")
    rankmean_vec = (rank_sum / K) / n_test
    preds[RANKMEAN] = dict(zip(row_ids, rankmean_vec.tolist()))

    st = stats[name]
    meta = {
        "n_train": st["n_train"],
        "n_object_cols": st["n_object_cols"],
        "l2": l2,
        "msl": msl_val,
        "n_features": len(features),
        "n_cat": sum(cat_mask),
        "gate_c": st["n_object_cols"] > 0,
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
            rec.update({
                "n_train": meta["n_train"],
                "n_object_cols": meta["n_object_cols"],
                "gate_c": meta["gate_c"],
                "l2": meta["l2"],
                "msl": meta["msl"],
            })
            for cfg in ALL_CONFIGS:
                pub, prv = score_split(preds[cfg], sol)
                rec[f"{cfg}_pub"] = pub
                rec[f"{cfg}_prv"] = prv
            print(f"[OK] {name} n_tr={meta['n_train']} obj={meta['n_object_cols']} "
                  f"gateC={meta['gate_c']} feats={meta['n_features']} "
                  f"cat={meta['n_cat']} l2={meta['l2']} msl={meta['msl']} "
                  f"fits={meta['n_fits']} | "
                  f"base pub={rec['base_pub']:.6f} prv={rec['base_prv']:.6f}  "
                  f"probmean pub={rec['probmean_pub']:.6f} prv={rec['probmean_prv']:.6f}  "
                  f"rankmean pub={rec['rankmean_pub']:.6f} prv={rec['rankmean_prv']:.6f}")
        except Exception as e:
            exceptions.append((name, repr(e)))
            rec.update({"n_train": stats.get(name, {}).get("n_train", ""),
                        "n_object_cols": stats.get(name, {}).get("n_object_cols", ""),
                        "gate_c": stats.get(name, {}).get("n_object_cols", 0) > 0,
                        "l2": float("nan"), "msl": float("nan")})
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

    def _row_subset(gate_c_only):
        if not gate_c_only:
            return rows
        return [r for r in rows if r.get("gate_c")]

    def mean_delta(arm, split, gate_c_only=False):
        vals = [delta(r, arm, split) for r in _row_subset(gate_c_only)]
        vals = [v for v in vals if not math.isnan(v)]
        return sum(vals) / len(vals) if vals else float("nan")

    def wlt(arm, split, gate_c_only=False, eps=1e-9):
        w = l = t = 0
        for r in _row_subset(gate_c_only):
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

    def regressions(arm, split, gate_c_only=False, eps=1e-6):
        return [(r["dataset"], delta(r, arm, split))
                for r in _row_subset(gate_c_only)
                if not math.isnan(delta(r, arm, split))
                and delta(r, arm, split) < -eps]

    def improvements(arm, split, gate_c_only=False, eps=1e-6):
        return [(r["dataset"], delta(r, arm, split))
                for r in _row_subset(gate_c_only)
                if not math.isnan(delta(r, arm, split))
                and delta(r, arm, split) > eps]

    # ---- results.csv ----
    csv_path = os.path.join(OUT_DIR, "results.csv")
    fieldnames = ["dataset", "n_train", "n_object_cols", "gate_c", "l2", "msl",
                  "base_pub", "base_prv",
                  "probmean_pub", "probmean_prv", "probmean_d_pub", "probmean_d_prv",
                  "rankmean_pub", "rankmean_prv", "rankmean_d_pub", "rankmean_d_prv",
                  "rank_minus_prob_pub", "rank_minus_prob_prv"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            out = {k: r.get(k, "") for k in
                   ["dataset", "n_train", "n_object_cols", "gate_c", "l2", "msl",
                    "base_pub", "base_prv",
                    "probmean_pub", "probmean_prv",
                    "rankmean_pub", "rankmean_prv"]}
            out["probmean_d_pub"] = delta(r, PROBMEAN, "pub")
            out["probmean_d_prv"] = delta(r, PROBMEAN, "prv")
            out["rankmean_d_pub"] = delta(r, RANKMEAN, "pub")
            out["rankmean_d_prv"] = delta(r, RANKMEAN, "prv")

            def _rmp(split):
                rk = r.get(f"{RANKMEAN}_{split}")
                pm = r.get(f"{PROBMEAN}_{split}")
                if rk is None or pm is None or math.isnan(rk) or math.isnan(pm):
                    return float("nan")
                return rk - pm
            out["rank_minus_prob_pub"] = _rmp("pub")
            out["rank_minus_prob_prv"] = _rmp("prv")
            w.writerow(out)

    # ---- REPRODUCTION 1: base on ALL 16 matches round54 base (tol<5e-6) ----
    repro_available = anchors54 is not None
    by_name = {r["dataset"]: r for r in rows}

    def build_repro(arm_key, anchor_key):
        """Compare arm column (arm_key in rec) to round54 anchor_key column.
        Returns (repro_dict, repro_ok, max_abs_dev)."""
        repro = {}
        ok_all = True
        max_dev = 0.0
        for i in range(1, N_DATASETS + 1):
            nm = f"train_{i:02d}"
            r = by_name.get(nm)
            mine = (r.get(f"{arm_key}_pub"), r.get(f"{arm_key}_prv")) if r \
                else (None, None)
            ref = anchors54.get(nm, {}).get(anchor_key) if anchors54 else None
            if ref is None or mine[0] is None or mine[1] is None \
                    or (isinstance(mine[0], float) and math.isnan(mine[0])):
                okp = okv = False
                devp = devv = float("nan")
            else:
                devp = abs(mine[0] - ref[0])
                devv = abs(mine[1] - ref[1])
                okp = devp < REPRO_TOL
                okv = devv < REPRO_TOL
                max_dev = max(max_dev, devp, devv)
            repro[nm] = {"mine": mine, "ref": ref, "okp": okp, "okv": okv,
                         "devp": devp, "devv": devv}
            if not (okp and okv):
                ok_all = False
        return repro, ok_all, max_dev

    repro_base, repro_base_ok, max_dev_base = build_repro(BASE, "base")
    repro_pm, repro_pm_ok, max_dev_pm = build_repro(PROBMEAN, "seedavg")
    repro_ok = repro_base_ok and repro_pm_ok
    max_abs_dev = max(max_dev_base, max_dev_pm)

    # ---- partition sanity (all 16 present) ----
    present = {r["dataset"] for r in rows if not (
        isinstance(r.get("base_pub"), float) and math.isnan(r.get("base_pub")))}
    all16_ok = (len(present) == N_DATASETS and not skipped)
    n_gate_c = sum(1 for r in rows if r.get("gate_c"))

    # ================= SUMMARY =================
    L = []
    L.append("=" * 78)
    L.append("bench_03 round55 — RANK-SPACE avg (K=10) vs PROB-SPACE avg (K=10) "
             "vs base-08 (ALL 16)  [OFFLINE]")
    L.append("=" * 78)
    L.append("")
    L.append("SETUP:")
    L.append("  base     == shipped 08: HGB, early_stopping, cat_mask=object cols,")
    L.append("              seed-0, full train.")
    L.append("  seed cache == fit K=10 base-08 HGB on FULL train, random_state 0..9,")
    L.append("              collect the K proba vectors P(class==1) ONCE. Both arms")
    L.append("              below are derived from the SAME cached vectors (no re-fit).")
    L.append("  probmean == arithmetic MEAN of the K proba vectors (== round54 seedavg,")
    L.append("              the standard shipped seed-avg).")
    L.append("  rankmean == MEAN of the K rank-transformed vectors: each proba vector")
    L.append("              rank-transformed over test rows via rankdata(method='average'),")
    L.append("              then averaged elementwise (÷n_test for readability; AUC-invariant).")
    L.append("  Single new mechanism vs probmean = RANK-space vs PROB-space aggregation.")
    L.append("")
    L.append(f"  gate C = (n_object_cols > 0): fires on {n_gate_c}/{len(rows)} datasets. "
             "On non-gate-C")
    L.append("  datasets the ship falls back to base-08, so both arms revert to base there.")

    # ---- HEADLINE (un-gated, all 16) ----
    L.append("")
    L.append("=== HEADLINE (a) UN-GATED (arm vs base == shipped 08, all 16) ===")
    head = {}
    for arm in ENSEMBLES:
        mp = mean_delta(arm, "pub")
        mv = mean_delta(arm, "prv")
        wp, lp, tp = wlt(arm, "pub")
        wv, lv, tv = wlt(arm, "prv")
        head[arm] = (mp, mv, (wp, lp, tp), (wv, lv, tv))
        L.append(f"  {arm:<9}: mean ΔPublic={mp:+.6f} (W/L/T {wp}/{lp}/{tp})  "
                 f"mean ΔPrivate={mv:+.6f} (W/L/T {wv}/{lv}/{tv})")

    # ---- HEADLINE (gate-C only) ----
    L.append("")
    L.append(f"=== HEADLINE (b) GATE-C ONLY (n_object_cols>0, {n_gate_c} datasets) ===")
    head_g = {}
    for arm in ENSEMBLES:
        mp = mean_delta(arm, "pub", gate_c_only=True)
        mv = mean_delta(arm, "prv", gate_c_only=True)
        wp, lp, tp = wlt(arm, "pub", gate_c_only=True)
        wv, lv, tv = wlt(arm, "prv", gate_c_only=True)
        head_g[arm] = (mp, mv, (wp, lp, tp), (wv, lv, tv))
        L.append(f"  {arm:<9}: mean ΔPublic={mp:+.6f} (W/L/T {wp}/{lp}/{tp})  "
                 f"mean ΔPrivate={mv:+.6f} (W/L/T {wv}/{lv}/{tv})")

    # ---- HEAD-TO-HEAD (c): rankmean vs probmean per-dataset ----
    L.append("")
    L.append("=== HEADLINE (c) HEAD-TO-HEAD: rankmean vs probmean (per-dataset) ===")
    L.append(f"{'dataset':<10} {'gateC':>6} {'rk_dpub':>11} {'pm_dpub':>11} "
             f"{'rk-pm_pub':>11} {'rk_dprv':>11} {'pm_dprv':>11} {'rk-pm_prv':>11}")
    rk_beats_pm_pub = pm_beats_rk_pub = 0
    rk_beats_pm_prv = pm_beats_rk_prv = 0
    differ = []   # datasets where |rank - prob| > 1e-6 on any split
    for r in rows:
        rdp = delta(r, RANKMEAN, "pub")
        pdp = delta(r, PROBMEAN, "pub")
        rdv = delta(r, RANKMEAN, "prv")
        pdv = delta(r, PROBMEAN, "prv")

        def _d(a, b):
            return (a - b) if (not math.isnan(a) and not math.isnan(b)) \
                else float("nan")
        dpub = _d(r.get("rankmean_pub", float("nan")),
                  r.get("probmean_pub", float("nan")))
        dprv = _d(r.get("rankmean_prv", float("nan")),
                  r.get("probmean_prv", float("nan")))
        if not math.isnan(rdp) and not math.isnan(pdp):
            if rdp > pdp + 1e-9:
                rk_beats_pm_pub += 1
            elif pdp > rdp + 1e-9:
                pm_beats_rk_pub += 1
        if not math.isnan(rdv) and not math.isnan(pdv):
            if rdv > pdv + 1e-9:
                rk_beats_pm_prv += 1
            elif pdv > rdv + 1e-9:
                pm_beats_rk_prv += 1
        if (not math.isnan(dpub) and abs(dpub) > 1e-6) or \
                (not math.isnan(dprv) and abs(dprv) > 1e-6):
            differ.append((r["dataset"], dpub, dprv))

        def fmt(x):
            return f"{x:>+11.6f}" if not math.isnan(x) else f"{'nan':>11}"
        L.append(f"{r['dataset']:<10} {str(bool(r.get('gate_c'))):>6} "
                 f"{fmt(rdp)} {fmt(pdp)} {fmt(dpub)} "
                 f"{fmt(rdv)} {fmt(pdv)} {fmt(dprv)}")
    L.append(f"  Public : rankmean>probmean on {rk_beats_pm_pub}, "
             f"probmean>rankmean on {pm_beats_rk_pub}")
    L.append(f"  Private: rankmean>probmean on {rk_beats_pm_prv}, "
             f"probmean>rankmean on {pm_beats_rk_prv}")
    if differ:
        L.append("  datasets where rankmean and probmean differ by >1e-6:")
        for nm, dpub, dprv in differ:
            dps = f"{dpub:+.6f}" if not math.isnan(dpub) else "nan"
            dvs = f"{dprv:+.6f}" if not math.isnan(dprv) else "nan"
            L.append(f"    {nm}: rank-prob Public={dps}, Private={dvs}")
    else:
        L.append("  rankmean and probmean are IDENTICAL (|d|<=1e-6) on both "
                 "splits for every dataset.")

    # ---- REPRODUCTION 1: base vs round54 base ----
    L.append("")
    L.append("=== REPRODUCTION CHECK 1 (base on ALL 16 vs round54 base, tol<5e-6) ===")
    if not repro_available:
        L.append("  round54 results.csv NOT found -> reproduction NOT anchored (FAIL).")
    else:
        for i in range(1, N_DATASETS + 1):
            nm = f"train_{i:02d}"
            rr = repro_base[nm]
            mp_, mv_ = rr["mine"]
            rp_, rv_ = rr["ref"] if rr["ref"] else (float("nan"), float("nan"))
            L.append(
                f"  {nm}: Public {mp_:.6f} vs r54 {rp_:.6f} "
                f"(|d|={rr['devp']:.2e}, {'YES' if rr['okp'] else 'NO'}); "
                f"Private {mv_:.6f} vs r54 {rv_:.6f} "
                f"(|d|={rr['devv']:.2e}, {'YES' if rr['okv'] else 'NO'})")
        L.append(f"  max |dev| over all 16x2 = {max_dev_base:.2e}")
        L.append(f"  REPRODUCTION 1 (base): {'PASS' if repro_base_ok else 'FAIL'}")

    # ---- REPRODUCTION 2: probmean vs round54 seedavg ----
    L.append("")
    L.append("=== REPRODUCTION CHECK 2 (probmean on ALL 16 vs round54 seedavg, "
             "tol<5e-6) ===")
    if not repro_available:
        L.append("  round54 results.csv NOT found -> reproduction NOT anchored (FAIL).")
    else:
        for i in range(1, N_DATASETS + 1):
            nm = f"train_{i:02d}"
            rr = repro_pm[nm]
            mp_, mv_ = rr["mine"]
            rp_, rv_ = rr["ref"] if rr["ref"] else (float("nan"), float("nan"))
            L.append(
                f"  {nm}: Public {mp_:.6f} vs r54sa {rp_:.6f} "
                f"(|d|={rr['devp']:.2e}, {'YES' if rr['okp'] else 'NO'}); "
                f"Private {mv_:.6f} vs r54sa {rv_:.6f} "
                f"(|d|={rr['devv']:.2e}, {'YES' if rr['okv'] else 'NO'})")
        L.append(f"  max |dev| over all 16x2 = {max_dev_pm:.2e}")
        L.append(f"  REPRODUCTION 2 (probmean==seedavg): "
                 f"{'PASS' if repro_pm_ok else 'FAIL'}")

    # ---- PER-DATASET DELTAS ----
    for arm in ENSEMBLES:
        for split, tag in (("pub", "Public"), ("prv", "Private")):
            L.append("")
            L.append(f"=== PER-DATASET ΔAUC ({tag}) — base vs {arm} ===")
            L.append(f"{'dataset':<10} {'gateC':>6} {'base':>10} {'arm':>10} "
                     f"{'delta':>11}")
            for r in rows:
                b = r.get(f"{BASE}_{split}")
                c = r.get(f"{arm}_{split}")
                dd = delta(r, arm, split)
                bstr = f"{b:>10.6f}" if isinstance(b, float) and not math.isnan(b) else f"{'nan':>10}"
                cstr = f"{c:>10.6f}" if isinstance(c, float) and not math.isnan(c) else f"{'nan':>10}"
                dstr = f"{dd:>+11.6f}" if not math.isnan(dd) else f"{'nan':>11}"
                L.append(f"{r['dataset']:<10} {str(bool(r.get('gate_c'))):>6} "
                         f"{bstr} {cstr} {dstr}")

    # ---- REGRESSIONS / IMPROVEMENTS (gate-C view = ship view) ----
    L.append("")
    L.append("=== REGRESSIONS (ΔAUC < -1e-6, GATE-C view = ship view) ===")
    for arm in ENSEMBLES:
        rp = regressions(arm, "pub", gate_c_only=True)
        rv = regressions(arm, "prv", gate_c_only=True)
        if not rp and not rv:
            L.append(f"  {arm}: NONE on either split.")
        else:
            for n_, d_ in rp:
                L.append(f"  {arm} Public  {n_}: {d_:+.6f}")
            for n_, d_ in rv:
                L.append(f"  {arm} Private {n_}: {d_:+.6f}")

    L.append("")
    L.append("=== IMPROVEMENTS (ΔAUC > +1e-6, GATE-C view = ship view) ===")
    for arm in ENSEMBLES:
        ip = improvements(arm, "pub", gate_c_only=True)
        iv = improvements(arm, "prv", gate_c_only=True)
        if not ip and not iv:
            L.append(f"  {arm}: NONE on either split.")
        else:
            for n_, d_ in ip:
                L.append(f"  {arm} Public  {n_}: {d_:+.6f}")
            for n_, d_ in iv:
                L.append(f"  {arm} Private {n_}: {d_:+.6f}")

    # ---- ADOPTION / VERDICT (gate-C view) ----
    L.append("")
    L.append("=== ADOPTION ANALYSIS (GATE-C view = ship view) ===")
    L.append("Per arm: ADOPT iff mean ΔPublic > 0 AND mean ΔPrivate > 0 with ZERO")
    L.append("  regression on EITHER split, over the gate-C-firing datasets. Any")
    L.append("  regression, or net-negative/negligible mean on either split => REJECT.")
    ADOPT_EPS = 1e-5   # negligible-mean guard
    adopt_flags = {}
    L.append("")
    for arm in ENSEMBLES:
        regs_pub = regressions(arm, "pub", gate_c_only=True)
        regs_prv = regressions(arm, "prv", gate_c_only=True)
        mp, mv = head_g[arm][0], head_g[arm][1]
        zero_regs = (not regs_pub) and (not regs_prv)
        clean_gain = (mp > ADOPT_EPS and mv > ADOPT_EPS)
        is_adopt = zero_regs and clean_gain
        adopt_flags[arm] = is_adopt
        L.append(f"  [{arm}] zero_regressions={'YES' if zero_regs else 'NO'} "
                 f"(Public regs={len(regs_pub)}, Private regs={len(regs_prv)})")
        L.append(f"  [{arm}] gate-C mean ΔPublic  = {mp:+.6f}  (clean gain: "
                 f"{'YES' if mp > ADOPT_EPS else 'NO'})")
        L.append(f"  [{arm}] gate-C mean ΔPrivate = {mv:+.6f}  (clean gain: "
                 f"{'YES' if mv > ADOPT_EPS else 'NO'})")
        L.append(f"  [{arm}] -> {'ADOPT' if is_adopt else 'REJECT'}")
        L.append("")

    is_adopt_any = any(adopt_flags.values())
    adopted = [a for a in ENSEMBLES if adopt_flags[a]]

    # rankmean vs probmean verdict (gate-C mean splits)
    rk_mp, rk_mv = head_g[RANKMEAN][0], head_g[RANKMEAN][1]
    pm_mp, pm_mv = head_g[PROBMEAN][0], head_g[PROBMEAN][1]
    rk_beats_pm_mean = (rk_mp > pm_mp) and (rk_mv > pm_mv)

    L.append("=== VERDICT ===")
    if rk_beats_pm_mean:
        L.append(f"  RANKMEAN BEATS PROBMEAN on BOTH gate-C mean splits "
                 f"(rank ΔPub {rk_mp:+.6f} > prob {pm_mp:+.6f}; "
                 f"rank ΔPrv {rk_mv:+.6f} > prob {pm_mv:+.6f}).")
    else:
        L.append(f"  RANKMEAN does NOT beat PROBMEAN on both gate-C mean splits "
                 f"(rank ΔPub {rk_mp:+.6f} vs prob {pm_mp:+.6f}; "
                 f"rank ΔPrv {rk_mv:+.6f} vs prob {pm_mv:+.6f}).")
    if is_adopt_any:
        L.append(f"  ADOPT: {', '.join(adopted)} cleanly improve BOTH gate-C mean "
                 f"splits with zero regression.")
    else:
        L.append("  REJECT (clean-win test): no arm improves BOTH gate-C mean splits "
                 "with zero regressions. Rank-space and prob-space averaging of the "
                 "same K seed members (varying only random_state) are near-identical "
                 "because the members agree on rank order -> no clean win over base.")

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
             f"skipped={len(skipped)}, gate_c_datasets={n_gate_c}, "
             f"reproduction={'YES' if repro_ok else 'NO'} "
             f"(base_maxdev={max_dev_base:.2e}, probmean_maxdev={max_dev_pm:.2e})]")
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
