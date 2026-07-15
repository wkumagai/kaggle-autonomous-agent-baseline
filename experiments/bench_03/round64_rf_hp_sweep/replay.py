#!/usr/bin/env python
"""
bench_03 round64 — RF HYPERPARAMETER SWEEP for the gate-D' RF-blend clean-win.
OFFLINE ONLY. No subprocess, no LLM, no Kaggle, no network. Calls sklearn
in-process only. Writes ONLY under this round64 directory; NEVER touches
submissions/.

GOAL (robustness of the round61 "RF blend gate-D'" lead)
--------------------------------------------------------
Round61 rank-averaged the shipped-08 HGB with a RandomForestClassifier
(n_estimators=300, max_features default='sqrt'). Un-gated it was a wash, but
GATED on gate-D' = (n_train<5000 AND n_object_cols>0) it was a CLEAN WIN:
gate-D' fires on train_03/05/09/13/15, all five positive on BOTH the public and
private eval splits, zero regressions. The open question: is that gate-D' clean
win an intrinsic property of the bagging x boosting blend, or is it fragile /
lucky at exactly n_estimators=300, max_features='sqrt'?

This round holds EVERYTHING in round61 identical (base-08 HGB recipe, RF feature
prep pipeline, rank-average blend, per-dataset AUC on both eval splits, gate-D'
definition, degenerate single-class handling) and sweeps ONLY the RandomForest
over:
    n_estimators  in {100, 300, 500}
    max_features  in {'sqrt', 0.5}
= 6 RF configs. The (300, 'sqrt') config reproduces round61 exactly and is the
ANCHOR: its gate-D' per-dataset deltas must match round61's results.csv within
~1e-9. All other RF params identical to round61 (random_state=0, n_jobs=4, same
median-impute + one-hot Pipeline).

For EACH config we report:
  * gate-D' fired datasets (expected train_03,05,09,13,15)
  * over-16 mean dPublic / dPrivate (blend - base)  [non-fired contribute 0]
  * fired-subset mean dPublic / dPrivate
  * regressions among fired datasets (dAUC < -1e-6) on each split
  * clean-win verdict: GATE-D' CLEAN-WIN = zero regressions on BOTH splits AND
    net positive fired-subset mean on BOTH splits.
Plus a final compact comparison table: one row per config.

BASE recipe (== shipped 08, config-independent — fit ONCE per dataset):
  features = [c for c in train.columns if c not in ("row_id","target")]
  cat_mask = [train[c].dtype == object for c in features]
  n=len(train); ratio = len(feats)/n
  l2  = 1.0 if ratio >= 0.010 else 0.0
  msl = 70 if ratio>=0.030 else 50 if ratio>=0.015 else 20
  HistGradientBoostingClassifier(categorical_features=cat_mask, random_state=0,
      max_iter=300, early_stopping=True, l2_regularization=l2,
      min_samples_leaf=msl)

RF view (identical to round61, only n_estimators/max_features swept):
  numeric cols (dtype != object): SimpleImputer(median)  [no scaling]
  object  cols (dtype == object): SimpleImputer(constant '__missing__')
                                  -> OneHotEncoder(handle_unknown='ignore')
  Pipeline -> RandomForestClassifier(n_estimators=<swept>, max_features=<swept>,
                                     random_state=0, n_jobs=4). Single fit, seed-0.

Blend = rank-average(hgb_proba, rf_proba) = elementwise mean of rankdata of each.
Degenerate single-class eval split -> auc_or_nan returns nan (single-class skip).
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
OUT_DIR = os.path.join(BENCH_DIR, "round64_rf_hp_sweep")
STATS_CSV = os.path.join(BENCH_DIR, "dataset_stats.csv")
# Anchor the (300,'sqrt') config against round61's per-dataset deltas.
ROUND61_RESULTS = os.path.join(BENCH_DIR, "round61_rf_blend", "results.csv")

L2_GATE_THRESHOLD = 0.010   # shipped-08 ratio gate for l2 (FIXED)
GATED_L2 = 1.0
DEFAULT_MSL = 20
MSL_TIERS = [(0.030, 70), (0.015, 50)]
BASE_SEED = 0
RF_N_JOBS = 4
N_DATASETS = 16
WLT_EPS = 1e-6             # win/lose/tie / regression threshold (per task spec)
ANCHOR_TOL = 1e-9         # (300,'sqrt') must match round61 within this

# ---- RF hyperparameter sweep: 6 configs. (300,'sqrt') == round61 anchor. ----
RF_N_ESTIMATORS_GRID = [100, 300, 500]
RF_MAX_FEATURES_GRID = ["sqrt", 0.5]
# Cartesian product, ordered so 'sqrt' block comes first (300,'sqrt' is anchor).
CONFIGS = [(ne, mf) for mf in RF_MAX_FEATURES_GRID for ne in RF_N_ESTIMATORS_GRID]
ANCHOR_CONFIG = (300, "sqrt")


def cfg_key(ne, mf):
    mf_s = mf if isinstance(mf, str) else f"{mf:g}"
    return f"ne{ne}_mf{mf_s}"


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
    to anchor the (300,'sqrt') config at full precision. Returns dict
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
    Wrapped in a Pipeline. Only n_estimators / max_features vary vs round61; all
    other RF params (random_state, n_jobs) and the prep pipeline are identical."""
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


def rank_average(a, b):
    """Elementwise mean of the ranks of two score vectors. AUC is rank-invariant
    so the averaged rank is used directly as the blended 'probability'."""
    return (rankdata(a) + rankdata(b)) / 2.0


def run_one(name, train_csv, test_csv, stats):
    """Returns (base_scores, rf_scores_by_cfg, meta).

    base_scores  = {row_id -> hgb_proba}  (config-independent, fit once).
    rf_scores_by_cfg = {cfg_key -> {row_id -> blend_score}} for each RF config.
    """
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

    # ---- BLEND per RF config: rank-average(same HGB proba, RF proba). ----
    blend_by_cfg = {}
    for ne, mf in CONFIGS:
        rf_proba = fit_rf(train, test, features, BASE_SEED, ne, mf)
        blend_score = rank_average(hgb_proba, rf_proba)
        blend_by_cfg[cfg_key(ne, mf)] = dict(zip(row_ids, blend_score.tolist()))
        n_fits += 1

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
    return base_scores, blend_by_cfg, meta


def score_split(pred_map, sol):
    sol = sol.copy()
    sol["pred"] = sol["row_id"].map(pred_map)
    if sol["pred"].isna().any():
        raise ValueError(f"{int(sol['pred'].isna().sum())} row_ids unmatched")
    pub = sol[sol["Usage"] == "Public"]
    prv = sol[sol["Usage"] == "Private"]
    return (auc_or_nan(pub["target"], pub["pred"]),
            auc_or_nan(prv["target"], prv["pred"]))


def delta(rec, cfgk, split):
    b = rec.get(f"base_{split}")
    c = rec.get(f"{cfgk}_{split}")
    if b is None or c is None or math.isnan(b) or math.isnan(c):
        return float("nan")
    return c - b


def main():
    warnings.filterwarnings("ignore")
    os.makedirs(OUT_DIR, exist_ok=True)

    stats = load_stats()
    anchors61 = round61_delta_anchors()

    rows = []           # one dict per dataset, holds base + every config's scores
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
            base_scores, blend_by_cfg, meta = run_one(name, train_csv, test_csv, stats)
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
            for ne, mf in CONFIGS:
                ck = cfg_key(ne, mf)
                cpub, cprv = score_split(blend_by_cfg[ck], sol)
                rec[f"{ck}_pub"] = cpub
                rec[f"{ck}_prv"] = cprv
                if math.isnan(cpub) or math.isnan(cprv):
                    single_class_skips.append((name, ck))
            anchor_k = cfg_key(*ANCHOR_CONFIG)
            print(f"[OK] {name} n_tr={meta['n_train']} obj={meta['n_object_cols']} "
                  f"gateC={meta['gate_c']} feats={meta['n_features']} "
                  f"cat={meta['n_cat']} l2={meta['l2']} msl={meta['msl']} "
                  f"fits={meta['n_fits']} | base pub={bpub:.6f} prv={bprv:.6f}  "
                  f"anchor(300,sqrt) pub={rec[f'{anchor_k}_pub']:.6f} "
                  f"prv={rec[f'{anchor_k}_prv']:.6f}")
        except Exception as e:
            exceptions.append((name, repr(e)))
            rec.update({"n_train": stats.get(name, {}).get("n_train", ""),
                        "n_object_cols": stats.get(name, {}).get("n_object_cols", ""),
                        "gate_c": stats.get(name, {}).get("n_object_cols", 0) > 0,
                        "l2": float("nan"), "msl": float("nan")})
            rec["base_pub"] = float("nan")
            rec["base_prv"] = float("nan")
            for ne, mf in CONFIGS:
                ck = cfg_key(ne, mf)
                rec[f"{ck}_pub"] = float("nan")
                rec[f"{ck}_prv"] = float("nan")
            print(f"[ERROR] {name}: {e!r}")
        rows.append(rec)

    # ---- gate-D' predicate (n_train<5000 AND n_object_cols>0). ----
    is_gateD = lambda r: (isinstance(r.get("n_train"), int)
                          and r["n_train"] < 5000
                          and isinstance(r.get("n_object_cols"), int)
                          and r["n_object_cols"] > 0)
    is_gatec = lambda r: bool(r.get("gate_c"))
    n_all = len(rows)
    n_gate_c = sum(1 for r in rows if r.get("gate_c"))

    # ---- per-config aggregate helpers ----
    def cfg_mean_delta(cfgk, split, subset=None):
        src = rows if subset is None else [r for r in rows if subset(r)]
        vals = [delta(r, cfgk, split) for r in src]
        vals = [v for v in vals if not math.isnan(v)]
        return sum(vals) / len(vals) if vals else float("nan")

    def cfg_sum_delta(cfgk, split, subset=None):
        src = rows if subset is None else [r for r in rows if subset(r)]
        vals = [delta(r, cfgk, split) for r in src]
        vals = [v for v in vals if not math.isnan(v)]
        return sum(vals) if vals else 0.0

    def cfg_wlt(cfgk, split, subset=None, eps=WLT_EPS):
        src = rows if subset is None else [r for r in rows if subset(r)]
        w = l = t = 0
        for r in src:
            dd = delta(r, cfgk, split)
            if math.isnan(dd):
                continue
            if dd > eps:
                w += 1
            elif dd < -eps:
                l += 1
            else:
                t += 1
        return w, l, t

    def cfg_regressions(cfgk, split, subset=None, eps=WLT_EPS):
        src = rows if subset is None else [r for r in rows if subset(r)]
        return [(r["dataset"], delta(r, cfgk, split)) for r in src
                if not math.isnan(delta(r, cfgk, split))
                and delta(r, cfgk, split) < -eps]

    fired_rows = [r for r in rows if is_gateD(r)]
    fired_names = [r["dataset"] for r in fired_rows]

    # ---- ANCHOR CHECK: (300,'sqrt') per-dataset deltas vs round61 results.csv --
    anchor_k = cfg_key(*ANCHOR_CONFIG)
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

    # ---- results.csv: per-config x per-dataset rows ----
    csv_path = os.path.join(OUT_DIR, "results.csv")
    fieldnames = ["config", "n_estimators", "max_features", "dataset",
                  "n_train", "n_object_cols", "gate_c", "gate_d", "l2", "msl",
                  "base_public", "base_private", "blend_public", "blend_private",
                  "delta_public", "delta_private"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for ne, mf in CONFIGS:
            ck = cfg_key(ne, mf)
            for r in rows:
                w.writerow({
                    "config": ck,
                    "n_estimators": ne,
                    "max_features": mf,
                    "dataset": r.get("dataset", ""),
                    "n_train": r.get("n_train", ""),
                    "n_object_cols": r.get("n_object_cols", ""),
                    "gate_c": r.get("gate_c", ""),
                    "gate_d": bool(is_gateD(r)),
                    "l2": r.get("l2", ""),
                    "msl": r.get("msl", ""),
                    "base_public": r.get("base_pub", ""),
                    "base_private": r.get("base_prv", ""),
                    "blend_public": r.get(f"{ck}_pub", ""),
                    "blend_private": r.get(f"{ck}_prv", ""),
                    "delta_public": delta(r, ck, "pub"),
                    "delta_private": delta(r, ck, "prv"),
                })

    # ---- partition sanity (all 16 present, base non-nan) ----
    present = {r["dataset"] for r in rows if not (
        isinstance(r.get("base_pub"), float) and math.isnan(r.get("base_pub")))}
    all16_ok = (len(present) == N_DATASETS and not skipped)

    # ================= SUMMARY =================
    L = []
    L.append("=" * 78)
    L.append("bench_03 round64 — RF HYPERPARAMETER SWEEP for gate-D' RF-blend "
             "clean-win (ALL 16)  [OFFLINE]")
    L.append("=" * 78)
    L.append("")
    L.append("SETUP:")
    L.append("  base  == shipped 08: HGB, early_stopping, cat_mask=object cols,")
    L.append("           seed-0, single fit on the FULL train frame "
             "(config-independent).")
    L.append("  blend == rank-average(base HGB proba, RandomForest proba), seed-0.")
    L.append("           RF view = numeric[median-impute, no scaling] +")
    L.append("           object[constant-impute + one-hot(ignore-unknown)] in a")
    L.append("           Pipeline -> RandomForestClassifier(random_state=0, n_jobs=4).")
    L.append("  SWEEP  = n_estimators in {100,300,500} x max_features in "
             "{'sqrt',0.5} = 6 configs.")
    L.append("           (300,'sqrt') reproduces round61 (the ANCHOR).")
    L.append(f"  gate-D' = (n_train<5000 AND n_object_cols>0). "
             f"fired={len(fired_rows)} [{','.join(fired_names)}]")
    L.append("")

    # ---- ANCHOR CHECK ----
    L.append("=== ANCHOR CHECK ((300,'sqrt') per-dataset deltas vs round61 "
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

    # ---- PER-CONFIG DETAIL ----
    config_summary = []   # collected rows for the final comparison table
    for ne, mf in CONFIGS:
        ck = cfg_key(ne, mf)
        mf_s = mf if isinstance(mf, str) else f"{mf:g}"
        is_anchor = (ne, mf) == ANCHOR_CONFIG

        over16_pub = cfg_sum_delta(ck, "pub") / n_all if n_all else float("nan")
        over16_prv = cfg_sum_delta(ck, "prv") / n_all if n_all else float("nan")
        fired_sum_pub = cfg_sum_delta(ck, "pub", is_gateD)
        fired_sum_prv = cfg_sum_delta(ck, "prv", is_gateD)
        n_fired = len(fired_rows)
        fired_mean_pub = fired_sum_pub / n_fired if n_fired else float("nan")
        fired_mean_prv = fired_sum_prv / n_fired if n_fired else float("nan")
        gd_reg_pub = cfg_regressions(ck, "pub", is_gateD)
        gd_reg_prv = cfg_regressions(ck, "prv", is_gateD)
        clean_win = (not gd_reg_pub and not gd_reg_prv
                     and fired_sum_pub > 0 and fired_sum_prv > 0)
        n_fired_reg = len(gd_reg_pub) + len(gd_reg_prv)

        L.append("-" * 78)
        L.append(f"CONFIG {ck}  (n_estimators={ne}, max_features={mf_s})"
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
            dp = delta(r, ck, "pub")
            dv = delta(r, ck, "prv")
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

        config_summary.append({
            "cfg": ck, "n_est": ne, "max_feat": mf_s,
            "over16_pub": over16_pub, "over16_prv": over16_prv,
            "fired_pub": fired_mean_pub, "fired_prv": fired_mean_prv,
            "n_fired_reg": n_fired_reg, "clean_win": clean_win,
            "is_anchor": is_anchor,
        })

    # ---- FINAL COMPARISON TABLE ----
    L.append("=" * 78)
    L.append("=== FINAL COMPARISON TABLE (one row per RF config) ===")
    L.append("=" * 78)
    L.append(f"{'n_est':>6} {'max_feat':>9} {'over16_dPub':>12} "
             f"{'over16_dPrv':>12} {'firedReg':>9} {'cleanWin':>9}")
    for c in config_summary:
        tag = " *" if c["is_anchor"] else ""
        L.append(f"{c['n_est']:>6} {c['max_feat']:>9} "
                 f"{c['over16_pub']:>+12.6f} {c['over16_prv']:>+12.6f} "
                 f"{c['n_fired_reg']:>9} "
                 f"{('Y' if c['clean_win'] else 'N'):>9}{tag}")
    L.append("  (* = (300,'sqrt') anchor == round61; firedReg = # regressions "
             "among fired datasets across BOTH splits)")
    L.append("")
    # secondary table with fired-subset means for completeness
    L.append("=== FIRED-SUBSET MEANS (gate-D' datasets only) ===")
    L.append(f"{'n_est':>6} {'max_feat':>9} {'fired_dPub':>12} "
             f"{'fired_dPrv':>12} {'cleanWin':>9}")
    for c in config_summary:
        L.append(f"{c['n_est']:>6} {c['max_feat']:>9} "
                 f"{c['fired_pub']:>+12.6f} {c['fired_prv']:>+12.6f} "
                 f"{('Y' if c['clean_win'] else 'N'):>9}")
    L.append("")

    # ---- ROBUSTNESS VERDICT ----
    n_clean = sum(1 for c in config_summary if c["clean_win"])
    n_cfg = len(config_summary)
    L.append("=== ROBUSTNESS VERDICT ===")
    L.append(f"  gate-D' CLEAN-WIN holds in {n_clean}/{n_cfg} RF configs.")
    if n_clean == n_cfg:
        L.append("  ROBUST: the gate-D' clean-win survives ALL swept RF "
                 "hyperparameters (n_estimators in {100,300,500} x "
                 "max_features in {'sqrt',0.5}).")
    elif n_clean == 0:
        L.append("  FRAGILE: the gate-D' clean-win holds in NO swept config "
                 "(does not even reproduce at the anchor -> investigate).")
    else:
        broke = [f"({c['n_est']},{c['max_feat']})" for c in config_summary
                 if not c["clean_win"]]
        L.append("  PARTIAL: the gate-D' clean-win BREAKS for some configs: "
                 + ", ".join(broke) + ".")

    # ---- CLEAN RUN marker ----
    clean_run = ((not exceptions) and all16_ok and anchor_ok and anchor_available
                 and (not skipped) and (not single_class_skips))
    L.append("")
    L.append(f"CLEAN RUN: {'YES' if clean_run else 'NO'} "
             f"({len(exceptions)} exceptions)  "
             f"[total_fits={total_fits}, datasets_scored={len(present)}/16, "
             f"skipped={len(skipped)}, single_class_skips={len(single_class_skips)}, "
             f"gate_c_datasets={n_gate_c}, gate_d_datasets={len(fired_rows)}, "
             f"anchor={'YES' if anchor_ok else 'NO'} "
             f"(anchor_maxdev={anchor_maxdev:.2e})]")
    for name, msg in exceptions:
        L.append(f"  EXC {name}: {msg}")
    for name, cfgk in single_class_skips:
        L.append(f"  SINGLE-CLASS {name}/{cfgk}")

    summary = "\n".join(L)
    print("\n" + summary)
    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
        f.write(summary + "\n")
    print(f"\n[WROTE] {csv_path}")
    print(f"[WROTE] {os.path.join(OUT_DIR, 'summary.txt')}")
    print(f"FINAL_MARKER CLEAN_RUN={'YES' if clean_run else 'NO'} "
          f"SCORED={len(present)}/16 EXC={len(exceptions)} "
          f"ANCHOR_MAXDEV={anchor_maxdev:.2e} CLEAN_WIN_CONFIGS={n_clean}/{n_cfg}")
    print("=== ROUND64 COMPLETE ===")


if __name__ == "__main__":
    main()
