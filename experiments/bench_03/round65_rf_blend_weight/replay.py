#!/usr/bin/env python
"""
bench_03 round65 — RF-BLEND WEIGHT SWEEP for the gate-D' RF-blend clean-win.
OFFLINE ONLY. No subprocess, no LLM, no Kaggle, no network. Calls sklearn
in-process only. Writes ONLY under this round65 directory; NEVER touches
submissions/.

GOAL (robustness of the round61 "RF blend gate-D'" lead to the BLEND WEIGHT)
---------------------------------------------------------------------------
Round61 rank-averaged the shipped-08 HGB with a RandomForestClassifier
(n_estimators=300, max_features='sqrt') using an EQUAL 0.5:0.5 rank-average.
Un-gated it was a wash, but GATED on gate-D' = (n_train<5000 AND
n_object_cols>0) it was a CLEAN WIN: gate-D' fires on train_03/05/09/13/15, all
five positive on BOTH the public and private eval splits, zero regressions.
round64 already showed the clean-win survives the RF HYPERPARAMETERS
(n_estimators x max_features). The open question THIS round answers: is that
clean-win robust to the BLEND WEIGHT, or is it specific to the equal 0.5:0.5
rank-average?

This round holds EVERYTHING in round61 identical (base-08 HGB recipe, RF feature
prep pipeline, RF at the round61 anchor, per-dataset AUC on both eval splits,
gate-D' definition, degenerate single-class handling) and sweeps ONLY the BLEND
WEIGHT:
    blend = weighted_rank_average(hgb, rf, w)
          = w * rankdata(hgb) + (1 - w) * rankdata(rf)
    w (weight on the BASE HGB) in {0.3, 0.4, 0.5, 0.6, 0.7}
AUC is rank-invariant, so the overall scale factor is irrelevant; only the
induced ordering matters. w=1.0 would be pure base (zero delta) and w=0.0 pure
RF; the 0.3-0.7 grid brackets the equal 0.5:0.5 blend on both sides.

The RF is FIXED at the round61 anchor for every weight:
    RandomForestClassifier(n_estimators=300, max_features='sqrt',
                           random_state=0, n_jobs=4)
The w=0.5 weight reproduces round61's equal rank-average exactly and is the
ANCHOR: its gate-D' per-dataset deltas must match round61's results.csv within a
small tolerance (round64 saw max|dev| ~8e-8 from RF/threading nondeterminism, so
tol=1e-6 here). The actual max|dev| is reported.

Because w only changes the cheap blend step, base-HGB and RF are each fit ONCE
per dataset and their P(class==1) vectors are REUSED across all 5 weights (no
refit per weight).

For EACH weight we report:
  * gate-D' fired datasets (expected train_03,05,09,13,15)
  * over-16 mean dPublic / dPrivate (blend - base)  [non-fired contribute 0]
  * fired-subset mean dPublic / dPrivate
  * regressions among fired datasets (dAUC < -1e-6) on each split
  * clean-win verdict: GATE-D' CLEAN-WIN = zero regressions on BOTH splits AND
    net positive fired-subset mean on BOTH splits.
Plus a final compact comparison table (one row per weight) and a robustness
verdict stating in how many of the 5 weights the clean-win holds and which w
maximizes the fired-subset mean gain.

BASE recipe (== shipped 08, weight-independent — fit ONCE per dataset):
  features = [c for c in train.columns if c not in ("row_id","target")]
  cat_mask = [train[c].dtype == object for c in features]
  n=len(train); ratio = len(feats)/n
  l2  = 1.0 if ratio >= 0.010 else 0.0
  msl = 70 if ratio>=0.030 else 50 if ratio>=0.015 else 20
  HistGradientBoostingClassifier(categorical_features=cat_mask, random_state=0,
      max_iter=300, early_stopping=True, l2_regularization=l2,
      min_samples_leaf=msl)

RF view (identical to round61, FIXED for all weights):
  numeric cols (dtype != object): SimpleImputer(median)  [no scaling]
  object  cols (dtype == object): SimpleImputer(constant '__missing__')
                                  -> OneHotEncoder(handle_unknown='ignore')
  Pipeline -> RandomForestClassifier(n_estimators=300, max_features='sqrt',
                                     random_state=0, n_jobs=4). Single fit, seed-0.

Blend = weighted_rank_average(hgb_proba, rf_proba, w). Degenerate single-class
eval split -> auc_or_nan returns nan (single-class skip).
"""
import os

# keep the run polite / modest on CPU; both estimators are deterministic w.r.t.
# random_state regardless of thread count, so this does not affect reproduction.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import csv
import math
import warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

REPO = "/Users/kumacmini/kaggle-autonomous-agent-baseline-auto"
DATA_DIR = os.path.join(REPO, "data")
BENCH_DIR = os.path.join(REPO, "experiments", "bench_03")
OUT_DIR = os.path.join(BENCH_DIR, "round65_rf_blend_weight")
STATS_CSV = os.path.join(BENCH_DIR, "dataset_stats.csv")
# Anchor the w=0.5 weight against round61's per-dataset deltas.
ROUND61_RESULTS = os.path.join(BENCH_DIR, "round61_rf_blend", "results.csv")

L2_GATE_THRESHOLD = 0.010   # shipped-08 ratio gate for l2 (FIXED)
GATED_L2 = 1.0
DEFAULT_MSL = 20
MSL_TIERS = [(0.030, 70), (0.015, 50)]
BASE_SEED = 0
RF_N_JOBS = 4
N_DATASETS = 16
WLT_EPS = 1e-6             # win/lose/tie / regression threshold (per task spec)
ANCHOR_TOL = 1e-6         # w=0.5 must match round61 within this (RF/thread noise)

# ---- RF FIXED at the round61 anchor for every weight. ----
RF_N_ESTIMATORS = 300
RF_MAX_FEATURES = "sqrt"

# ---- blend-WEIGHT sweep: w = weight on the BASE HGB. 5 weights. ----
# w=0.5 reproduces round61's equal 0.5:0.5 rank-average (the ANCHOR).
WEIGHT_GRID = [0.3, 0.4, 0.5, 0.6, 0.7]
ANCHOR_WEIGHT = 0.5


def w_key(w):
    return f"w{w:g}"


def load_stats(path=STATS_CSV):
    stats = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            stats[row["name"]] = {
                "n_train": int(row["n_train"]),
                "n_object_cols": int(row["n_object_cols"]),
            }
    return stats


def round61_delta_anchors(path=ROUND61_RESULTS):
    """Read round61's per-dataset delta_public/delta_private for ALL 16 datasets
    to anchor the w=0.5 weight at full precision. Returns dict
    name -> (dpub, dprv) or None."""
    if not os.path.exists(path):
        return None
    anchors = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            name = row.get("dataset")
            try:
                anchors[name] = (float(row["delta_public"]),
                                 float(row["delta_private"]))
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
    left UNSET (sklearn defaults, byte-identical to shipped 08)."""
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


def fit_rf(train_frame, test, features, seed, n_estimators, max_features):
    """Fit a RandomForestClassifier on a robust numeric+categorical view and
    return P(class==1) on test. Numeric: median-impute (NO scaling — trees are
    scale-invariant). Object: constant-impute + one-hot(handle_unknown='ignore').
    Wrapped in a Pipeline. FIXED at the round61 anchor (n_estimators=300,
    max_features='sqrt', random_state=0, n_jobs=4) — identical to round61."""
    num_cols = [c for c in features if train_frame[c].dtype != object]
    cat_cols = [c for c in features if train_frame[c].dtype == object]

    transformers = []
    if num_cols:
        transformers.append((
            "num",
            SimpleImputer(strategy="median"),
            num_cols,
        ))
    if cat_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("impute", SimpleImputer(strategy="constant",
                                         fill_value="__missing__")),
                ("ohe", OneHotEncoder(handle_unknown="ignore")),
            ]),
            cat_cols,
        ))

    pipe = Pipeline([
        ("prep", ColumnTransformer(transformers, remainder="drop")),
        ("clf", RandomForestClassifier(n_estimators=n_estimators,
                                       max_features=max_features,
                                       random_state=seed, n_jobs=RF_N_JOBS)),
    ])
    pipe.fit(train_frame[features], train_frame["target"])
    proba = pipe.predict_proba(test[features])
    classes = list(pipe.named_steps["clf"].classes_)
    pos_idx = classes.index(1) if 1 in classes else 1
    return proba[:, pos_idx]


def weighted_rank_average(hgb, rf, w):
    """Weighted mean of the ranks of the two score vectors, with weight `w` on
    the BASE HGB and (1-w) on the RF: w*rankdata(hgb) + (1-w)*rankdata(rf).
    AUC is rank-invariant, so the blended value is used directly as the
    'probability'. At w=0.5 this equals (rankdata(hgb)+rankdata(rf))/2, the
    equal rank-average used in round61/round64."""
    return w * rankdata(hgb) + (1.0 - w) * rankdata(rf)


def run_one(name, train_csv, test_csv, stats):
    """Returns (base_scores, blend_by_w, meta).

    base_scores = {row_id -> hgb_proba}  (weight-independent, fit once).
    blend_by_w  = {w_key -> {row_id -> blend_score}} for each weight.

    base-HGB and RF are each fit ONCE per dataset; the blend for every weight
    reuses the SAME hgb_proba / rf_proba vectors (no refit per weight)."""
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)

    features = [c for c in train.columns if c not in ("row_id", "target")]

    # ---- gates computed from the FULL train (== base-08). ----
    n = len(train)
    ratio = len(features) / n
    l2 = GATED_L2 if ratio >= L2_GATE_THRESHOLD else 0.0
    msl_val = msl_for_ratio(ratio)

    row_ids = test["row_id"].tolist()
    n_fits = 0

    # ---- BASE: seed-0, FULL train, base-08 (byte-identical to shipped 08). ----
    cat_mask = [train[c].dtype == object for c in features]
    hgb_proba = fit_hgb(train, test, features, cat_mask, l2, msl_val, BASE_SEED)
    base_scores = dict(zip(row_ids, hgb_proba.tolist()))
    n_fits += 1

    # ---- RF: FIXED at the round61 anchor, fit ONCE. ----
    rf_proba = fit_rf(train, test, features, BASE_SEED,
                      RF_N_ESTIMATORS, RF_MAX_FEATURES)
    n_fits += 1

    # ---- BLEND per weight: reuse the SAME hgb_proba / rf_proba (cheap step). --
    blend_by_w = {}
    for w in WEIGHT_GRID:
        blend_score = weighted_rank_average(hgb_proba, rf_proba, w)
        blend_by_w[w_key(w)] = dict(zip(row_ids, blend_score.tolist()))

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
    return base_scores, blend_by_w, meta


def score_split(pred_map, sol):
    sol = sol.copy()
    sol["pred"] = sol["row_id"].map(pred_map)
    if sol["pred"].isna().any():
        raise ValueError(f"{int(sol['pred'].isna().sum())} row_ids unmatched")
    pub = sol[sol["Usage"] == "Public"]
    prv = sol[sol["Usage"] == "Private"]
    return (auc_or_nan(pub["target"], pub["pred"]),
            auc_or_nan(prv["target"], prv["pred"]))


def delta(rec, wk, split):
    b = rec.get(f"base_{split}")
    c = rec.get(f"{wk}_{split}")
    if b is None or c is None or math.isnan(b) or math.isnan(c):
        return float("nan")
    return c - b


def main():
    warnings.filterwarnings("ignore")
    os.makedirs(OUT_DIR, exist_ok=True)

    stats = load_stats()
    anchors61 = round61_delta_anchors()

    rows = []           # one dict per dataset, holds base + every weight's scores
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
            base_scores, blend_by_w, meta = run_one(name, train_csv, test_csv, stats)
            total_fits += meta["n_fits"]
            rec.update({
                "n_train": meta["n_train"],
                "n_object_cols": meta["n_object_cols"],
                "gate_c": meta["gate_c"],
                "l2": meta["l2"],
                "msl": meta["msl"],
            })
            bpub, bprv = score_split(base_scores, sol)
            rec["base_pub"] = bpub
            rec["base_prv"] = bprv
            if math.isnan(bpub) or math.isnan(bprv):
                single_class_skips.append((name, "base"))
            for w in WEIGHT_GRID:
                wk = w_key(w)
                cpub, cprv = score_split(blend_by_w[wk], sol)
                rec[f"{wk}_pub"] = cpub
                rec[f"{wk}_prv"] = cprv
                if math.isnan(cpub) or math.isnan(cprv):
                    single_class_skips.append((name, wk))
            anchor_k = w_key(ANCHOR_WEIGHT)
            print(f"[OK] {name} n_tr={meta['n_train']} obj={meta['n_object_cols']} "
                  f"gateC={meta['gate_c']} feats={meta['n_features']} "
                  f"cat={meta['n_cat']} l2={meta['l2']} msl={meta['msl']} "
                  f"fits={meta['n_fits']} | base pub={bpub:.6f} prv={bprv:.6f}  "
                  f"anchor(w=0.5) pub={rec[f'{anchor_k}_pub']:.6f} "
                  f"prv={rec[f'{anchor_k}_prv']:.6f}")
        except Exception as e:
            exceptions.append((name, repr(e)))
            rec.update({"n_train": stats.get(name, {}).get("n_train", ""),
                        "n_object_cols": stats.get(name, {}).get("n_object_cols", ""),
                        "gate_c": stats.get(name, {}).get("n_object_cols", 0) > 0,
                        "l2": float("nan"), "msl": float("nan")})
            rec["base_pub"] = float("nan")
            rec["base_prv"] = float("nan")
            for w in WEIGHT_GRID:
                wk = w_key(w)
                rec[f"{wk}_pub"] = float("nan")
                rec[f"{wk}_prv"] = float("nan")
            print(f"[ERROR] {name}: {e!r}")
        rows.append(rec)

    # ---- gate-D' predicate (n_train<5000 AND n_object_cols>0). ----
    is_gateD = lambda r: (isinstance(r.get("n_train"), int)
                          and r["n_train"] < 5000
                          and isinstance(r.get("n_object_cols"), int)
                          and r["n_object_cols"] > 0)
    n_all = len(rows)
    n_gate_c = sum(1 for r in rows if r.get("gate_c"))

    # ---- per-weight aggregate helpers ----
    def w_sum_delta(wk, split, subset=None):
        src = rows if subset is None else [r for r in rows if subset(r)]
        vals = [delta(r, wk, split) for r in src]
        vals = [v for v in vals if not math.isnan(v)]
        return sum(vals) if vals else 0.0

    def w_regressions(wk, split, subset=None, eps=WLT_EPS):
        src = rows if subset is None else [r for r in rows if subset(r)]
        return [(r["dataset"], delta(r, wk, split)) for r in src
                if not math.isnan(delta(r, wk, split))
                and delta(r, wk, split) < -eps]

    fired_rows = [r for r in rows if is_gateD(r)]
    fired_names = [r["dataset"] for r in fired_rows]

    # ---- gate-D' fire-set assertion (expected train_03,05,09,13,15). ----
    EXPECTED_FIRED = ["train_03", "train_05", "train_09", "train_13", "train_15"]
    fired_matches_expected = (fired_names == EXPECTED_FIRED)

    # ---- ANCHOR CHECK: w=0.5 per-dataset deltas vs round61 results.csv --------
    anchor_k = w_key(ANCHOR_WEIGHT)
    anchor_available = anchors61 is not None
    anchor_ok = True
    anchor_maxdev = 0.0
    anchor_detail = {}
    for r in rows:
        nm = r["dataset"]
        mine = (delta(r, anchor_k, "pub"), delta(r, anchor_k, "prv"))
        ref = anchors61.get(nm) if anchors61 else None
        if ref is None or math.isnan(mine[0]) or math.isnan(mine[1]):
            okp = okv = False
            devp = devv = float("nan")
        else:
            devp = abs(mine[0] - ref[0])
            devv = abs(mine[1] - ref[1])
            okp = devp <= ANCHOR_TOL
            okv = devv <= ANCHOR_TOL
            anchor_maxdev = max(anchor_maxdev, devp, devv)
        anchor_detail[nm] = {"mine": mine, "ref": ref, "okp": okp, "okv": okv,
                             "devp": devp, "devv": devv}
        if not (okp and okv):
            anchor_ok = False

    # ---- results.csv: per-weight x per-dataset rows ----
    csv_path = os.path.join(OUT_DIR, "results.csv")
    fieldnames = ["blend_weight", "dataset",
                  "n_train", "n_object_cols", "gate_c", "gate_d", "l2", "msl",
                  "base_public", "base_private", "blend_public", "blend_private",
                  "delta_public", "delta_private"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for weight in WEIGHT_GRID:
            wk = w_key(weight)
            for r in rows:
                w.writerow({
                    "blend_weight": weight,
                    "dataset": r.get("dataset", ""),
                    "n_train": r.get("n_train", ""),
                    "n_object_cols": r.get("n_object_cols", ""),
                    "gate_c": r.get("gate_c", ""),
                    "gate_d": bool(is_gateD(r)),
                    "l2": r.get("l2", ""),
                    "msl": r.get("msl", ""),
                    "base_public": r.get("base_pub", ""),
                    "base_private": r.get("base_prv", ""),
                    "blend_public": r.get(f"{wk}_pub", ""),
                    "blend_private": r.get(f"{wk}_prv", ""),
                    "delta_public": delta(r, wk, "pub"),
                    "delta_private": delta(r, wk, "prv"),
                })

    # ---- partition sanity (all 16 present, base non-nan) ----
    present = {r["dataset"] for r in rows if not (
        isinstance(r.get("base_pub"), float) and math.isnan(r.get("base_pub")))}
    all16_ok = (len(present) == N_DATASETS and not skipped)

    # ================= SUMMARY =================
    L = []
    L.append("=" * 78)
    L.append("bench_03 round65 — RF-BLEND WEIGHT SWEEP for gate-D' RF-blend "
             "clean-win (ALL 16)  [OFFLINE]")
    L.append("=" * 78)
    L.append("")
    L.append("SETUP:")
    L.append("  base  == shipped 08: HGB, early_stopping, cat_mask=object cols,")
    L.append("           seed-0, single fit on the FULL train frame "
             "(weight-independent).")
    L.append("  RF    == FIXED round61 anchor: RandomForestClassifier(")
    L.append("           n_estimators=300, max_features='sqrt', random_state=0,")
    L.append("           n_jobs=4). RF view = numeric[median-impute, no scaling] +")
    L.append("           object[constant-impute + one-hot(ignore-unknown)] in a")
    L.append("           Pipeline. base-HGB and RF each fit ONCE per dataset.")
    L.append("  blend == weighted_rank_average(hgb, rf, w) = "
             "w*rankdata(hgb) + (1-w)*rankdata(rf)")
    L.append("           w = weight on the BASE HGB. AUC is rank-invariant, so")
    L.append("           the scale factor is irrelevant; ordering is what matters.")
    L.append("  SWEEP  = w in {0.3, 0.4, 0.5, 0.6, 0.7} (reusing the same "
             "hgb/rf proba;")
    L.append("           NO refit per weight). w=0.5 == round61 equal blend "
             "(the ANCHOR).")
    L.append(f"  gate-D' = (n_train<5000 AND n_object_cols>0). "
             f"fired={len(fired_rows)} [{','.join(fired_names)}]")
    L.append(f"  gate-D' fire-set == expected [train_03,05,09,13,15]: "
             f"{'YES' if fired_matches_expected else 'NO'}")
    L.append("")

    # ---- ANCHOR CHECK ----
    L.append("=== ANCHOR CHECK (w=0.5 per-dataset deltas vs round61 "
             f"results.csv, tol={ANCHOR_TOL:.0e}) ===")
    if not anchor_available:
        L.append("  round61 results.csv NOT found -> anchor NOT verified (FAIL).")
    else:
        for r in rows:
            nm = r["dataset"]
            ad = anchor_detail[nm]
            mp_, mv_ = ad["mine"]
            if ad["ref"] is None:
                L.append(f"  {nm}: round61 ref MISSING (NO)")
                continue
            rp_, rv_ = ad["ref"]
            gd = "*" if is_gateD(r) else " "
            L.append(
                f" {gd}{nm}: dPub {mp_:+.6f} vs r61 {rp_:+.6f} "
                f"(|d|={ad['devp']:.2e},{'Y' if ad['okp'] else 'N'}); "
                f"dPrv {mv_:+.6f} vs r61 {rv_:+.6f} "
                f"(|d|={ad['devv']:.2e},{'Y' if ad['okv'] else 'N'})")
        L.append(f"  max |dev| over all 16x2 = {anchor_maxdev:.2e}")
        L.append(f"  ANCHOR REPRODUCES ROUND61: {'PASS' if anchor_ok else 'FAIL'} "
                 f"(gate-D' subset included above, marked '*')")
    L.append("")

    # ---- PER-WEIGHT DETAIL ----
    weight_summary = []   # collected rows for the final comparison table
    for weight in WEIGHT_GRID:
        wk = w_key(weight)
        is_anchor = (weight == ANCHOR_WEIGHT)

        over16_pub = w_sum_delta(wk, "pub") / n_all if n_all else float("nan")
        over16_prv = w_sum_delta(wk, "prv") / n_all if n_all else float("nan")
        fired_sum_pub = w_sum_delta(wk, "pub", is_gateD)
        fired_sum_prv = w_sum_delta(wk, "prv", is_gateD)
        n_fired = len(fired_rows)
        fired_mean_pub = fired_sum_pub / n_fired if n_fired else float("nan")
        fired_mean_prv = fired_sum_prv / n_fired if n_fired else float("nan")
        gd_reg_pub = w_regressions(wk, "pub", is_gateD)
        gd_reg_prv = w_regressions(wk, "prv", is_gateD)
        clean_win = (not gd_reg_pub and not gd_reg_prv
                     and fired_sum_pub > 0 and fired_sum_prv > 0)
        n_fired_reg = len(gd_reg_pub) + len(gd_reg_prv)

        L.append("-" * 78)
        L.append(f"WEIGHT {wk}  (w={weight:g} on base HGB, {1.0 - weight:g} on RF)"
                 + ("   [ANCHOR == round61]" if is_anchor else ""))
        L.append("-" * 78)
        L.append(f"  gate-D' fired = {n_fired} [{','.join(fired_names)}]")
        L.append(f"  over-16   mean dPublic ={over16_pub:+.6f}   "
                 f"mean dPrivate={over16_prv:+.6f}")
        L.append(f"  fired-sub mean dPublic ={fired_mean_pub:+.6f}   "
                 f"mean dPrivate={fired_mean_prv:+.6f}")
        # per-fired-dataset deltas
        L.append(f"  {'dataset':<10} {'n_train':>8} {'n_obj':>6} "
                 f"{'dPublic':>11} {'dPrivate':>11}")
        for r in fired_rows:
            dp = delta(r, wk, "pub")
            dv = delta(r, wk, "prv")
            dps = f"{dp:>+11.6f}" if not math.isnan(dp) else f"{'nan':>11}"
            dvs = f"{dv:>+11.6f}" if not math.isnan(dv) else f"{'nan':>11}"
            L.append(f"  {r['dataset']:<10} {str(r.get('n_train','')):>8} "
                     f"{str(r.get('n_object_cols','')):>6} {dps} {dvs}")
        L.append("  regressions among fired (dAUC < -1e-6):")
        L.append("    Public : " + (", ".join(f"{n_}({d_:+.6f})"
                                              for n_, d_ in gd_reg_pub)
                                     if gd_reg_pub else "NONE"))
        L.append("    Private: " + (", ".join(f"{n_}({d_:+.6f})"
                                              for n_, d_ in gd_reg_prv)
                                     if gd_reg_prv else "NONE"))
        L.append(f"  GATE-D' CLEAN-WIN (zero reg BOTH splits, net+ BOTH): "
                 f"{'YES' if clean_win else 'NO'}")
        L.append("")

        weight_summary.append({
            "wk": wk, "weight": weight,
            "over16_pub": over16_pub, "over16_prv": over16_prv,
            "fired_pub": fired_mean_pub, "fired_prv": fired_mean_prv,
            "fired_avg": (fired_mean_pub + fired_mean_prv) / 2.0,
            "n_fired_reg": n_fired_reg, "clean_win": clean_win,
            "is_anchor": is_anchor,
        })

    # ---- FINAL COMPARISON TABLE ----
    L.append("=" * 78)
    L.append("=== FINAL COMPARISON TABLE (one row per blend weight) ===")
    L.append("=" * 78)
    L.append(f"{'w_base':>7} {'over16_dPub':>12} {'over16_dPrv':>12} "
             f"{'fired_dPub':>12} {'fired_dPrv':>12} {'firedReg':>9} "
             f"{'cleanWin':>9}")
    for c in weight_summary:
        tag = " *" if c["is_anchor"] else ""
        L.append(f"{c['weight']:>7g} "
                 f"{c['over16_pub']:>+12.6f} {c['over16_prv']:>+12.6f} "
                 f"{c['fired_pub']:>+12.6f} {c['fired_prv']:>+12.6f} "
                 f"{c['n_fired_reg']:>9} "
                 f"{('Y' if c['clean_win'] else 'N'):>9}{tag}")
    L.append("  (w_base = weight on base HGB; * = w=0.5 anchor == round61;")
    L.append("   firedReg = # regressions among fired datasets across BOTH splits)")
    L.append("")

    # ---- ROBUSTNESS VERDICT ----
    n_clean = sum(1 for c in weight_summary if c["clean_win"])
    n_w = len(weight_summary)
    # best w = the clean-win weight with the largest fired-subset mean gain
    # (avg of the two splits' fired-subset means). Falls back to all weights if
    # none clean-win, so we always report a maximizer.
    clean_weights = [c for c in weight_summary if c["clean_win"]]
    rank_pool = clean_weights if clean_weights else weight_summary
    best = max(rank_pool, key=lambda c: c["fired_avg"])
    L.append("=== ROBUSTNESS VERDICT ===")
    L.append(f"  gate-D' CLEAN-WIN holds in {n_clean}/{n_w} blend weights "
             f"(w in {{0.3,0.4,0.5,0.6,0.7}}).")
    if n_clean == n_w:
        L.append("  ROBUST: the gate-D' clean-win survives ALL swept blend "
                 "weights — it is NOT specific to the equal 0.5:0.5 blend.")
    elif n_clean == 0:
        L.append("  FRAGILE: the gate-D' clean-win holds at NO swept weight "
                 "(does not even reproduce at the w=0.5 anchor -> investigate).")
    else:
        broke = [f"w={c['weight']:g}" for c in weight_summary
                 if not c["clean_win"]]
        L.append("  PARTIAL: the gate-D' clean-win BREAKS at some weights: "
                 + ", ".join(broke) + ".")
    best_pool_note = ("among clean-win weights" if clean_weights
                      else "among ALL weights (no clean-win)")
    L.append(f"  best w (max fired-subset mean gain, {best_pool_note}): "
             f"w={best['weight']:g} "
             f"(fired dPub={best['fired_pub']:+.6f}, "
             f"dPrv={best['fired_prv']:+.6f}, "
             f"avg={best['fired_avg']:+.6f}).")

    # ---- CLEAN RUN marker ----
    clean_run = ((not exceptions) and all16_ok and anchor_ok and anchor_available
                 and fired_matches_expected
                 and (not skipped) and (not single_class_skips))
    L.append("")
    L.append(f"CLEAN RUN: {'YES' if clean_run else 'NO'} "
             f"({len(exceptions)} exceptions)  "
             f"[total_fits={total_fits}, datasets_scored={len(present)}/16, "
             f"skipped={len(skipped)}, single_class_skips={len(single_class_skips)}, "
             f"gate_c_datasets={n_gate_c}, gate_d_datasets={len(fired_rows)}, "
             f"fired_expected={'YES' if fired_matches_expected else 'NO'}, "
             f"anchor={'YES' if anchor_ok else 'NO'} "
             f"(anchor_maxdev={anchor_maxdev:.2e})]")
    for name, msg in exceptions:
        L.append(f"  EXC {name}: {msg}")
    for name, wk in single_class_skips:
        L.append(f"  SINGLE-CLASS {name}/{wk}")

    summary = "\n".join(L)
    print("\n" + summary)
    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
        f.write(summary + "\n")
    print(f"\n[WROTE] {csv_path}")
    print(f"[WROTE] {os.path.join(OUT_DIR, 'summary.txt')}")
    print(f"FINAL_MARKER CLEAN_RUN={'YES' if clean_run else 'NO'} "
          f"SCORED={len(present)}/16 EXC={len(exceptions)} "
          f"ANCHOR_MAXDEV={anchor_maxdev:.2e} CLEAN_WIN_WEIGHTS={n_clean}/{n_w} "
          f"BEST_W={best['weight']:g}")
    print("=== ROUND65 COMPLETE ===")


if __name__ == "__main__":
    main()
