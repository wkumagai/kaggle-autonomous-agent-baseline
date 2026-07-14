#!/usr/bin/env python
"""
bench_03 round58 — MARGIN-SPACE avg (K=10) vs PROB-SPACE avg (K=10) vs base-08
(ALL 16).
OFFLINE ONLY. No subprocess, no LLM, no Kaggle, no network. Calls sklearn
in-process only. Writes ONLY under this round58 directory; NEVER touches
submissions/.

GOAL (improvement-log angle "margin_agg")
-----------------------------------------
Prior rounds compared aggregations of the K seed-avg members in PROBABILITY
space (round45/46: mean / median / trim10) and in RANK space (round55:
rank-mean). The raw decision function of the model — the pre-sigmoid MARGIN /
LOGIT space — has NEVER been used as the aggregation space
(grep -rl decision_function experiments/bench_03/ returns nothing). This round
tests that orthogonal, untried angle.

The competition metric is ROC AUC, which is purely RANK-based, so averaging the
per-seed raw MARGINS (decision_function) and scoring the averaged margin
directly is a valid, distinct aggregation: no sigmoid is needed because AUC is
invariant to any monotone (per-vector) transform of the final score. Averaging
in margin space weights the K members differently from averaging in probability
space, because the sigmoid is nonlinear (compresses extreme probabilities), so
margin-mean and prob-mean are NOT algebraically identical.

Hypothesis: margin-mean might differ from prob-mean when members disagree in the
saturated tails (probabilities near 0/1 map to large margins); likely near-equal
since it is the same model varying only random_state, but it is untried and
cheap.

Design (single mechanism = aggregation SPACE over K=10 seed-avg members):
  BASE arm     = base-08 HGB exactly (reference column), seed-0, full train.
                 monotonic_cst / max_depth / interaction_cst / tol /
                 validation_fraction UNSET; byte-identical to shipped 08 ->
                 reproduces round55 base column.
  Cached seeds = fit K=10 base-08 HGB on the FULL train, random_state=0..9,
                 collect BOTH the K proba vectors P(class==1) AND the K raw
                 margin vectors decision_function(test) on the test set. Each of
                 the K members is fit ONCE and BOTH outputs are cached, so the
                 two aggregation arms below come from the SAME fits (NO re-fit).
                 This guarantees PROBMEAN bit-reproduces round55 and isolates the
                 aggregation-space change as the ONLY variable.
  PROBMEAN arm = arithmetic MEAN of the K proba vectors (the standard shipped
                 seed-avg). Must reproduce round55's probmean column exactly
                 (max|dev| = 0).
  MARGINMEAN arm = arithmetic MEAN of the K raw margin vectors
                 (decision_function), scored directly by AUC (no sigmoid needed
                 — AUC is rank-based).

BASE recipe reproduced (== shipped 08), identical to round55:
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

REPRODUCTION (MANDATORY — proves the harness is faithful, must be BIT-IDENTICAL):
  1. BASE column on ALL 16 must match round55's base column (round55
     results.csv, base_pub/base_prv) with max|dev| = 0.
  2. PROBMEAN column on ALL 16 must match round55's probmean column
     (probmean_pub/probmean_prv) with max|dev| = 0 (same seeds 0..9, same fits,
     same mean -> byte-identical). If either dev is non-zero, CLEAN RUN = NO.

ADOPTION (same spirit as round55, evaluated on the gate-C-only view since that
  is what ships): an aggregation arm is ADOPT iff it cleanly improves over base
  — mean ΔPublic > 0 AND mean ΔPrivate > 0 with ZERO regression on EITHER split.
  Additionally a VERDICT reports whether MARGINMEAN beats PROBMEAN on BOTH mean
  splits (gate-C view) = a clean win of margin-space over prob-space aggregation.
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
OUT_DIR = os.path.join(BENCH_DIR, "round58_margin_agg")
STATS_CSV = os.path.join(BENCH_DIR, "dataset_stats.csv")
# Anchor against round55 (the immediately-preceding rank-agg harness), which
# encodes the exact base-08 config and the standard prob-space seed-avg column.
ROUND55_RESULTS = os.path.join(BENCH_DIR, "round55_rank_agg", "results.csv")

L2_GATE_THRESHOLD = 0.010   # shipped-08 ratio gate for l2 (FIXED)
GATED_L2 = 1.0
DEFAULT_MSL = 20
MSL_TIERS = [(0.030, 70), (0.015, 50)]
BASE_SEED = 0
N_DATASETS = 16
REPRO_TOL = 0.0             # BIT-IDENTICAL: control arms must match round55 exactly
K = 10                       # ensemble size (seed-avg members)

BASE = "base"
PROBMEAN = "probmean"        # arithmetic mean of the K proba vectors (control)
MARGINMEAN = "marginmean"    # arithmetic mean of the K decision_function vectors
ENSEMBLES = [PROBMEAN, MARGINMEAN]
ALL_CONFIGS = [BASE, PROBMEAN, MARGINMEAN]


def load_stats(path=STATS_CSV):
    stats = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            stats[row["name"]] = {
                "n_train": int(row["n_train"]),
                "n_object_cols": int(row["n_object_cols"]),
            }
    return stats


def round55_anchors(path=ROUND55_RESULTS):
    """Read round55's base_pub/base_prv AND probmean_pub/probmean_prv for ALL 16
    datasets to anchor reproduction at full precision. Returns dict
    name -> {"base": (pub, prv), "probmean": (pub, prv)} or None if unavailable."""
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
                entry["probmean"] = (float(row["probmean_pub"]),
                                     float(row["probmean_prv"]))
            except (KeyError, ValueError):
                entry["probmean"] = None
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


def fit_hgb_both(train_frame, test, features, cat_mask, l2, msl_val, seed):
    """Fit ONE shipped-08 HGB on `train_frame` and return BOTH
    (P(class==1), raw_margin) on test from the SAME fit.
    validation_fraction / max_depth / interaction_cst / tol / monotonic_cst
    left UNSET (sklearn defaults, byte-identical to shipped 08). The ONLY thing
    that varies across the ensemble is `random_state`. l2/msl are always the
    base-08 gate values computed from the full train.

    proba  = predict_proba(test)[:, class==1]  (drives PROBMEAN — control).
    margin = decision_function(test)           (drives MARGINMEAN — new).
    For binary HGB, decision_function returns the raw pre-sigmoid margin of the
    positive class as a 1-D vector of shape (n_test,)."""
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
    proba_pos = proba[:, pos_idx]
    margin = np.asarray(clf.decision_function(test[features])).ravel()
    return proba_pos, margin


def run_one(name, train_csv, test_csv, stats):
    """Returns (preds, meta). preds maps config_name -> {row_id -> score}.

    base       = seed-0, base-08 (byte-identical to shipped 08).
    probmean   = MEAN of the K cached proba vectors (== round55 probmean, control).
    marginmean = MEAN of the K cached decision_function (margin) vectors (new).

    The K seed fits (random_state 0..9 on the FULL train) are done ONCE and BOTH
    proba and margin are cached per member; BOTH aggregation arms are computed
    from the SAME cached vectors (NO extra fits). This guarantees the control
    (probmean) bit-reproduces round55 and isolates aggregation-space as the only
    variable.
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
    # Reuse the seed-0 member's proba below rather than fitting twice would be
    # tempting, but we keep BASE as an explicit independent fit to mirror round55
    # exactly (round55 fits base separately from the K=10 cache; matching its fit
    # count of 1 + K is required for the reproduction anchor to be meaningful).
    base_proba, _ = fit_hgb_both(train, test, features, cat_mask, l2, msl_val,
                                 BASE_SEED)
    preds[BASE] = dict(zip(row_ids, base_proba.tolist()))
    n_fits += 1

    # ---- CACHE the K seed-avg members ONCE (seeds 0..9), BOTH proba+margin. ----
    # The proba cache is EXACTLY round55's seedavg computation. Both aggregation
    # arms are derived from these SAME cached fits below (NO extra fits).
    seed_proba = np.zeros((K, n_test), dtype=np.float64)
    seed_margin = np.zeros((K, n_test), dtype=np.float64)
    for k in range(K):
        p_vec, m_vec = fit_hgb_both(train, test, features, cat_mask, l2,
                                    msl_val, k)
        seed_proba[k] = p_vec
        seed_margin[k] = m_vec
        n_fits += 1

    # ---- PROBMEAN (control): arithmetic mean of K proba vectors (==round55). ----
    probmean_vec = seed_proba.mean(axis=0)
    preds[PROBMEAN] = dict(zip(row_ids, probmean_vec.tolist()))

    # ---- MARGINMEAN (new): arithmetic mean of K raw margin vectors. ----
    # decision_function is the pre-sigmoid margin; averaging here weights members
    # differently than prob-space because the sigmoid is nonlinear. AUC is scored
    # directly on the averaged margin (no sigmoid needed — AUC is rank-based).
    marginmean_vec = seed_margin.mean(axis=0)
    preds[MARGINMEAN] = dict(zip(row_ids, marginmean_vec.tolist()))

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
    anchors55 = round55_anchors()
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
                  f"marginmean pub={rec['marginmean_pub']:.6f} prv={rec['marginmean_prv']:.6f}")
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
                  "marginmean_pub", "marginmean_prv",
                  "marginmean_d_pub", "marginmean_d_prv",
                  "margin_minus_prob_pub", "margin_minus_prob_prv"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            out = {k: r.get(k, "") for k in
                   ["dataset", "n_train", "n_object_cols", "gate_c", "l2", "msl",
                    "base_pub", "base_prv",
                    "probmean_pub", "probmean_prv",
                    "marginmean_pub", "marginmean_prv"]}
            out["probmean_d_pub"] = delta(r, PROBMEAN, "pub")
            out["probmean_d_prv"] = delta(r, PROBMEAN, "prv")
            out["marginmean_d_pub"] = delta(r, MARGINMEAN, "pub")
            out["marginmean_d_prv"] = delta(r, MARGINMEAN, "prv")

            def _mmp(split):
                mm = r.get(f"{MARGINMEAN}_{split}")
                pm = r.get(f"{PROBMEAN}_{split}")
                if mm is None or pm is None or math.isnan(mm) or math.isnan(pm):
                    return float("nan")
                return mm - pm
            out["margin_minus_prob_pub"] = _mmp("pub")
            out["margin_minus_prob_prv"] = _mmp("prv")
            w.writerow(out)

    # ---- REPRODUCTION: base + probmean on ALL 16 must be BIT-IDENTICAL to r55 --
    repro_available = anchors55 is not None
    by_name = {r["dataset"]: r for r in rows}

    def build_repro(arm_key, anchor_key):
        """Compare arm column (arm_key in rec) to round55 anchor_key column.
        Returns (repro_dict, repro_ok, max_abs_dev). ok requires |dev| <= REPRO_TOL
        (==0.0 -> bit-identical)."""
        repro = {}
        ok_all = True
        max_dev = 0.0
        for i in range(1, N_DATASETS + 1):
            nm = f"train_{i:02d}"
            r = by_name.get(nm)
            mine = (r.get(f"{arm_key}_pub"), r.get(f"{arm_key}_prv")) if r \
                else (None, None)
            ref = anchors55.get(nm, {}).get(anchor_key) if anchors55 else None
            if ref is None or mine[0] is None or mine[1] is None \
                    or (isinstance(mine[0], float) and math.isnan(mine[0])):
                okp = okv = False
                devp = devv = float("nan")
            else:
                devp = abs(mine[0] - ref[0])
                devv = abs(mine[1] - ref[1])
                okp = devp <= REPRO_TOL
                okv = devv <= REPRO_TOL
                max_dev = max(max_dev, devp, devv)
            repro[nm] = {"mine": mine, "ref": ref, "okp": okp, "okv": okv,
                         "devp": devp, "devv": devv}
            if not (okp and okv):
                ok_all = False
        return repro, ok_all, max_dev

    repro_base, repro_base_ok, max_dev_base = build_repro(BASE, "base")
    repro_pm, repro_pm_ok, max_dev_pm = build_repro(PROBMEAN, "probmean")
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
    L.append("bench_03 round58 — MARGIN-SPACE avg (K=10) vs PROB-SPACE avg (K=10) "
             "vs base-08 (ALL 16)  [OFFLINE]")
    L.append("=" * 78)
    L.append("")
    L.append("SETUP:")
    L.append("  base       == shipped 08: HGB, early_stopping, cat_mask=object cols,")
    L.append("                seed-0, full train.")
    L.append("  seed cache == fit K=10 base-08 HGB on FULL train, random_state 0..9,")
    L.append("                cache BOTH proba P(class==1) AND raw margin")
    L.append("                decision_function(test) per member ONCE. Both arms below")
    L.append("                are derived from the SAME K fits (no re-fit).")
    L.append("  probmean   == arithmetic MEAN of the K proba vectors (control; ==")
    L.append("                round55 probmean, the standard shipped seed-avg).")
    L.append("  marginmean == arithmetic MEAN of the K raw margin (decision_function)")
    L.append("                vectors, scored directly by AUC (no sigmoid — rank-based).")
    L.append("  Single new mechanism vs probmean = MARGIN-space vs PROB-space aggregation.")
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
        L.append(f"  {arm:<10}: mean ΔPublic={mp:+.6f} (W/L/T {wp}/{lp}/{tp})  "
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
        L.append(f"  {arm:<10}: mean ΔPublic={mp:+.6f} (W/L/T {wp}/{lp}/{tp})  "
                 f"mean ΔPrivate={mv:+.6f} (W/L/T {wv}/{lv}/{tv})")

    # ---- HEAD-TO-HEAD (c): marginmean vs probmean per-dataset ----
    L.append("")
    L.append("=== HEADLINE (c) HEAD-TO-HEAD: marginmean vs probmean (per-dataset) ===")
    L.append(f"{'dataset':<10} {'gateC':>6} {'mm_dpub':>11} {'pm_dpub':>11} "
             f"{'mm-pm_pub':>11} {'mm_dprv':>11} {'pm_dprv':>11} {'mm-pm_prv':>11}")
    mm_beats_pm_pub = pm_beats_mm_pub = 0
    mm_beats_pm_prv = pm_beats_mm_prv = 0
    differ = []   # datasets where |margin - prob| > 1e-6 on any split
    for r in rows:
        mdp = delta(r, MARGINMEAN, "pub")
        pdp = delta(r, PROBMEAN, "pub")
        mdv = delta(r, MARGINMEAN, "prv")
        pdv = delta(r, PROBMEAN, "prv")

        def _d(a, b):
            return (a - b) if (not math.isnan(a) and not math.isnan(b)) \
                else float("nan")
        dpub = _d(r.get("marginmean_pub", float("nan")),
                  r.get("probmean_pub", float("nan")))
        dprv = _d(r.get("marginmean_prv", float("nan")),
                  r.get("probmean_prv", float("nan")))
        if not math.isnan(mdp) and not math.isnan(pdp):
            if mdp > pdp + 1e-9:
                mm_beats_pm_pub += 1
            elif pdp > mdp + 1e-9:
                pm_beats_mm_pub += 1
        if not math.isnan(mdv) and not math.isnan(pdv):
            if mdv > pdv + 1e-9:
                mm_beats_pm_prv += 1
            elif pdv > mdv + 1e-9:
                pm_beats_mm_prv += 1
        if (not math.isnan(dpub) and abs(dpub) > 1e-6) or \
                (not math.isnan(dprv) and abs(dprv) > 1e-6):
            differ.append((r["dataset"], dpub, dprv))

        def fmt(x):
            return f"{x:>+11.6f}" if not math.isnan(x) else f"{'nan':>11}"
        L.append(f"{r['dataset']:<10} {str(bool(r.get('gate_c'))):>6} "
                 f"{fmt(mdp)} {fmt(pdp)} {fmt(dpub)} "
                 f"{fmt(mdv)} {fmt(pdv)} {fmt(dprv)}")
    L.append(f"  Public : marginmean>probmean on {mm_beats_pm_pub}, "
             f"probmean>marginmean on {pm_beats_mm_pub}")
    L.append(f"  Private: marginmean>probmean on {mm_beats_pm_prv}, "
             f"probmean>marginmean on {pm_beats_mm_prv}")
    if differ:
        L.append("  datasets where marginmean and probmean differ by >1e-6:")
        for nm, dpub, dprv in differ:
            dps = f"{dpub:+.6f}" if not math.isnan(dpub) else "nan"
            dvs = f"{dprv:+.6f}" if not math.isnan(dprv) else "nan"
            L.append(f"    {nm}: margin-prob Public={dps}, Private={dvs}")
    else:
        L.append("  marginmean and probmean are IDENTICAL (|d|<=1e-6) on both "
                 "splits for every dataset.")

    # ---- REPRODUCTION 1: base vs round55 base (BIT-IDENTICAL) ----
    L.append("")
    L.append("=== REPRODUCTION CHECK 1 (base on ALL 16 vs round55 base, tol=0) ===")
    if not repro_available:
        L.append("  round55 results.csv NOT found -> reproduction NOT anchored (FAIL).")
    else:
        for i in range(1, N_DATASETS + 1):
            nm = f"train_{i:02d}"
            rr = repro_base[nm]
            mp_, mv_ = rr["mine"]
            rp_, rv_ = rr["ref"] if rr["ref"] else (float("nan"), float("nan"))
            L.append(
                f"  {nm}: Public {mp_:.6f} vs r55 {rp_:.6f} "
                f"(|d|={rr['devp']:.2e}, {'YES' if rr['okp'] else 'NO'}); "
                f"Private {mv_:.6f} vs r55 {rv_:.6f} "
                f"(|d|={rr['devv']:.2e}, {'YES' if rr['okv'] else 'NO'})")
        L.append(f"  max |dev| over all 16x2 = {max_dev_base:.2e}")
        L.append(f"  REPRODUCTION 1 (base): {'PASS' if repro_base_ok else 'FAIL'}")

    # ---- REPRODUCTION 2: probmean vs round55 probmean (BIT-IDENTICAL) ----
    L.append("")
    L.append("=== REPRODUCTION CHECK 2 (probmean on ALL 16 vs round55 probmean, "
             "tol=0) ===")
    if not repro_available:
        L.append("  round55 results.csv NOT found -> reproduction NOT anchored (FAIL).")
    else:
        for i in range(1, N_DATASETS + 1):
            nm = f"train_{i:02d}"
            rr = repro_pm[nm]
            mp_, mv_ = rr["mine"]
            rp_, rv_ = rr["ref"] if rr["ref"] else (float("nan"), float("nan"))
            L.append(
                f"  {nm}: Public {mp_:.6f} vs r55pm {rp_:.6f} "
                f"(|d|={rr['devp']:.2e}, {'YES' if rr['okp'] else 'NO'}); "
                f"Private {mv_:.6f} vs r55pm {rv_:.6f} "
                f"(|d|={rr['devv']:.2e}, {'YES' if rr['okv'] else 'NO'})")
        L.append(f"  max |dev| over all 16x2 = {max_dev_pm:.2e}")
        L.append(f"  REPRODUCTION 2 (probmean==round55 probmean): "
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

    # marginmean vs probmean verdict (gate-C mean splits)
    mm_mp, mm_mv = head_g[MARGINMEAN][0], head_g[MARGINMEAN][1]
    pm_mp, pm_mv = head_g[PROBMEAN][0], head_g[PROBMEAN][1]
    mm_beats_pm_mean = (mm_mp > pm_mp) and (mm_mv > pm_mv)

    L.append("=== VERDICT ===")
    if mm_beats_pm_mean:
        L.append(f"  MARGINMEAN BEATS PROBMEAN on BOTH gate-C mean splits "
                 f"(margin ΔPub {mm_mp:+.6f} > prob {pm_mp:+.6f}; "
                 f"margin ΔPrv {mm_mv:+.6f} > prob {pm_mv:+.6f}).")
    else:
        L.append(f"  MARGINMEAN does NOT beat PROBMEAN on both gate-C mean splits "
                 f"(margin ΔPub {mm_mp:+.6f} vs prob {pm_mp:+.6f}; "
                 f"margin ΔPrv {mm_mv:+.6f} vs prob {pm_mv:+.6f}).")
    if is_adopt_any:
        L.append(f"  ADOPT: {', '.join(adopted)} cleanly improve BOTH gate-C mean "
                 f"splits with zero regression.")
    else:
        L.append("  REJECT (clean-win test): no arm improves BOTH gate-C mean splits "
                 "with zero regressions. Margin-space and prob-space averaging of the "
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
