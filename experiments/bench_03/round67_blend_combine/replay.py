#!/usr/bin/env python
"""
bench_03 round67 — RF x HGB BLEND *COMBINE-SPACE* ROBUSTNESS — ALL 16.
OFFLINE ONLY. No subprocess, no LLM, no Kaggle, no network. Calls sklearn
in-process only. Writes ONLY under this round67 directory; NEVER touches
submissions/.

GOAL (improvement-log angle "blend_combine")
--------------------------------------------
Candidate B is the "gate-D' RF blend": for datasets where
`n_train < 5000 AND n_object_cols > 0` (the gate-D' firing set), blend the
shipped base-08 HGB probabilities with a RandomForest by *rank-average*; all
other datasets keep base HGB unchanged (delta 0). Prior rounds de-risked B
along three axes:
  - round64: RF hyperparameters
  - round65: blend weight
  - round66: gate threshold
The untested 4th dimension is the **combine function** — the way the two
probability vectors are *fused*. round61 fused them by rank-average. Is
rank-average the right fusion, or do prob-average / logit-average change the
gate-mean delta materially?  (Direct analogy: round58 asked the same question
for seed-averaging and found the aggregation space — prob / rank / margin — was
immaterial. This round tests that hypothesis for the RF x HGB blend.)

Design (single mechanism swept = the COMBINE FUNCTION):
  For each of the 16 datasets we fit EXACTLY round61's two estimators and take
  the two class==1 probability vectors on the test frame:
      hgb_proba = base-08 HGB proba   (seed 0, full train — byte-identical to 08)
      rf_proba  = RandomForest proba  (fit_rf, n_estimators=300, seed 0)
  and then compute THREE blended score vectors from those SAME two vectors:
      rank-avg  = (rankdata(hgb) + rankdata(rf)) / 2      [== round61, the anchor]
      prob-avg  = (hgb + rf) / 2
      logit-avg = (logit(hgb) + logit(rf)) / 2,  logit(p)=log(p/(1-p)),
                  p clipped to [1e-6, 1-1e-6].  AUC is rank-invariant so the
                  blended logit vector is used directly as the score.
  base = hgb_proba alone (reference column).

Nothing about the fits changes across the three combine spaces — only the fusion
of the two fixed proba vectors. rank-avg therefore reproduces round61 exactly.

Gate-D' application (== candidate B ships):
  gate-D' fires iff  n_train < 5000 AND n_object_cols > 0.  On NON-firing
  datasets the shipped score is base HGB, so the gated delta is 0 there. On
  firing datasets the gated delta is (blend - base) for the chosen combine
  space. The reported GATE-D' MEAN DELTA is the gated delta averaged over ALL
  16 datasets (non-firing contribute 0) — i.e. sum_over_firing(blend-base)/16.
  This is exactly candidate B's effect on the full 16-dataset benchmark.

REPRODUCTION ANCHOR (MANDATORY — proves the harness is faithful):
  The rank-avg arm must reproduce round61. round62's desk-eval reported the
  gate-D' mean delta for the rank-avg blend as dPublic +0.002230 /
  dPrivate +0.002976 over the 5-dataset firing set (averaged over all 16). We
  ALSO reconstruct that anchor directly from round61's results.csv per-dataset
  deltas restricted to the firing set, and print max|dev| between THIS run's
  rank-avg gate-D' mean deltas and both. dev ~ 0 => rank-avg reproduces round61.

Degenerate cases handled exactly as round61: single-class eval split ->
auc_or_nan returns nan, counted as a single-class skip (not scored).
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
OUT_DIR = os.path.join(BENCH_DIR, "round67_blend_combine")
STATS_CSV = os.path.join(BENCH_DIR, "dataset_stats.csv")
# Reproduction anchor: round61's rank-avg blend fit on the FULL train frame.
ROUND61_RESULTS = os.path.join(BENCH_DIR, "round61_rf_blend", "results.csv")

L2_GATE_THRESHOLD = 0.010   # shipped-08 ratio gate for l2 (FIXED)
GATED_L2 = 1.0
DEFAULT_MSL = 20
MSL_TIERS = [(0.030, 70), (0.015, 50)]
BASE_SEED = 0
RF_N_ESTIMATORS = 300
RF_N_JOBS = 4
N_DATASETS = 16
LOGIT_CLIP = 1e-6           # clip p to [LOGIT_CLIP, 1-LOGIT_CLIP] before logit
WLT_EPS = 1e-6              # win/lose/tie threshold (per task spec)
IMMATERIAL_EPS = 5e-4       # |delta_of_deltas| <= this => combine space immaterial

# Gate-D' firing predicate.
GATE_D_MAX_N_TRAIN = 5000
GATE_D_MIN_OBJ = 1

# round62 desk-eval reported gate-D' mean delta for the rank-avg blend.
ROUND61_ANCHOR_PUB = 0.002230
ROUND61_ANCHOR_PRV = 0.002976

BASE = "base"
RANK = "rank"
PROB = "prob"
LOGIT = "logit"
BLENDS = [RANK, PROB, LOGIT]
ALL_CONFIGS = [BASE] + BLENDS


def load_stats(path=STATS_CSV):
    stats = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            stats[row["name"]] = {
                "n_train": int(row["n_train"]),
                "n_object_cols": int(row["n_object_cols"]),
            }
    return stats


def gate_d_fires(n_train, n_object_cols):
    return (n_train < GATE_D_MAX_N_TRAIN) and (n_object_cols >= GATE_D_MIN_OBJ)


def round61_firing_anchor(path=ROUND61_RESULTS):
    """Reconstruct the gate-D' mean delta anchor directly from round61's
    results.csv: gated delta = round61 blend-vs-base delta on firing datasets,
    0 elsewhere, averaged over ALL 16. round61's `blend` column IS the rank-avg
    blend, so this equals THIS run's rank-avg gate-D' mean delta if faithful.
    Returns (anchor_pub, anchor_prv, n_firing) or None."""
    if not os.path.exists(path):
        return None
    sp = sv = 0.0
    n_all = 0
    n_fire = 0
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            n_all += 1
            try:
                nt = int(row["n_train"])
                noc = int(row["n_object_cols"])
            except (KeyError, ValueError):
                continue
            if gate_d_fires(nt, noc):
                n_fire += 1
                sp += float(row["delta_public"])
                sv += float(row["delta_private"])
    if n_all == 0:
        return None
    return (sp / n_all, sv / n_all, n_fire)


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


def fit_rf(train_frame, test, features, seed):
    """Fit a RandomForestClassifier on a robust numeric+categorical view and
    return P(class==1) on test. Numeric: median-impute (NO scaling — trees are
    scale-invariant). Object: constant-impute + one-hot(handle_unknown='ignore').
    Wrapped in a Pipeline. Byte-identical to round61's fit_rf."""
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
        ("clf", RandomForestClassifier(n_estimators=RF_N_ESTIMATORS,
                                       random_state=seed, n_jobs=RF_N_JOBS)),
    ])
    pipe.fit(train_frame[features], train_frame["target"])
    proba = pipe.predict_proba(test[features])
    classes = list(pipe.named_steps["clf"].classes_)
    pos_idx = classes.index(1) if 1 in classes else 1
    return proba[:, pos_idx]


def combine_rank(hgb, rf):
    """Elementwise mean of the ranks of the two proba vectors (== round61)."""
    return (rankdata(hgb) + rankdata(rf)) / 2.0


def combine_prob(hgb, rf):
    """Elementwise mean of the two probability vectors."""
    return (np.asarray(hgb) + np.asarray(rf)) / 2.0


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), LOGIT_CLIP, 1.0 - LOGIT_CLIP)
    return np.log(p / (1.0 - p))


def combine_logit(hgb, rf):
    """Elementwise mean of the two logit-transformed proba vectors. AUC is
    rank-invariant so the blended logit vector is used directly as the score."""
    return (_logit(hgb) + _logit(rf)) / 2.0


COMBINERS = {RANK: combine_rank, PROB: combine_prob, LOGIT: combine_logit}


def run_one(name, train_csv, test_csv, stats):
    """Returns (preds, meta). preds maps config_name -> {row_id -> score}.

    base  = seed-0 base-08 HGB proba on the FULL train (byte-identical to 08).
    rank  = rank-average(base HGB proba, RandomForest proba)   [== round61].
    prob  = prob-average(...).
    logit = logit-average(...).
    All three blends fuse the SAME two proba vectors — only the combine
    function differs.
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
    preds = {}
    n_fits = 0

    # ---- BASE: seed-0, FULL train, base-08 (byte-identical to shipped 08). ----
    cat_mask = [train[c].dtype == object for c in features]
    hgb_proba = fit_hgb(train, test, features, cat_mask, l2, msl_val, BASE_SEED)
    preds[BASE] = dict(zip(row_ids, hgb_proba.tolist()))
    n_fits += 1

    # ---- RF proba (fit ONCE, reused by all three combine spaces). ----
    rf_proba = fit_rf(train, test, features, BASE_SEED)
    n_fits += 1

    # ---- THREE combine spaces over the SAME (hgb, rf) proba vectors. ----
    for cfg in BLENDS:
        score = COMBINERS[cfg](hgb_proba, rf_proba)
        preds[cfg] = dict(zip(row_ids, np.asarray(score, dtype=float).tolist()))

    st = stats[name]
    meta = {
        "n_train": st["n_train"],
        "n_object_cols": st["n_object_cols"],
        "l2": l2,
        "msl": msl_val,
        "n_features": len(features),
        "n_cat": sum(cat_mask),
        "gate_d": gate_d_fires(st["n_train"], st["n_object_cols"]),
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
    anchor61 = round61_firing_anchor()
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
            preds, meta = run_one(name, train_csv, test_csv, stats)
            total_fits += meta["n_fits"]
            rec.update({
                "n_train": meta["n_train"],
                "n_object_cols": meta["n_object_cols"],
                "gate_d": meta["gate_d"],
                "l2": meta["l2"],
                "msl": meta["msl"],
            })
            for cfg in ALL_CONFIGS:
                pub, prv = score_split(preds[cfg], sol)
                rec[f"{cfg}_pub"] = pub
                rec[f"{cfg}_prv"] = prv
                if math.isnan(pub) or math.isnan(prv):
                    single_class_skips.append((name, cfg))
            print(f"[OK] {name} n_tr={meta['n_train']} obj={meta['n_object_cols']} "
                  f"gateD={meta['gate_d']} feats={meta['n_features']} "
                  f"cat={meta['n_cat']} l2={meta['l2']} msl={meta['msl']} "
                  f"fits={meta['n_fits']} | "
                  f"base pub={rec['base_pub']:.6f} prv={rec['base_prv']:.6f}  "
                  f"rank pub={rec['rank_pub']:.6f}  "
                  f"prob pub={rec['prob_pub']:.6f}  "
                  f"logit pub={rec['logit_pub']:.6f}")
        except Exception as e:
            exceptions.append((name, repr(e)))
            rec.update({"n_train": stats.get(name, {}).get("n_train", ""),
                        "n_object_cols": stats.get(name, {}).get("n_object_cols", ""),
                        "gate_d": gate_d_fires(
                            stats.get(name, {}).get("n_train", 10 ** 9),
                            stats.get(name, {}).get("n_object_cols", 0)),
                        "l2": float("nan"), "msl": float("nan")})
            for cfg in ALL_CONFIGS:
                rec[f"{cfg}_pub"] = float("nan")
                rec[f"{cfg}_prv"] = float("nan")
            print(f"[ERROR] {name}: {e!r}")
        rows.append(rec)

    is_firing = lambda r: bool(r.get("gate_d"))
    firing_rows = [r for r in rows if is_firing(r)]

    # ---- per-combine-space delta vs base (raw, per-dataset) ----
    def delta(rec, cfg, split):
        b = rec.get(f"{BASE}_{split}")
        c = rec.get(f"{cfg}_{split}")
        if b is None or c is None or (isinstance(b, float) and math.isnan(b)) \
                or (isinstance(c, float) and math.isnan(c)):
            return float("nan")
        return c - b

    def gated_mean_delta(cfg, split):
        """Gated delta averaged over ALL scored datasets: (blend-base) on firing
        datasets, 0 on non-firing. == candidate B's effect on the 16-set
        benchmark == sum_over_firing(blend-base)/N_scored."""
        contrib = []
        for r in rows:
            b = r.get(f"{BASE}_{split}")
            if b is None or (isinstance(b, float) and math.isnan(b)):
                continue  # unscored dataset
            if is_firing(r):
                dd = delta(r, cfg, split)
                if math.isnan(dd):
                    continue
                contrib.append(dd)
            else:
                contrib.append(0.0)
        return (sum(contrib) / len(contrib)) if contrib else float("nan")

    def firing_mean_delta(cfg, split):
        """Mean of (blend-base) over the firing set only (no zero padding)."""
        vals = [delta(r, cfg, split) for r in firing_rows]
        vals = [v for v in vals if not math.isnan(v)]
        return sum(vals) / len(vals) if vals else float("nan")

    def firing_wlt(cfg, split, eps=WLT_EPS):
        w = l = t = 0
        for r in firing_rows:
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

    # ---- results.csv ----
    csv_path = os.path.join(OUT_DIR, "results.csv")
    fieldnames = ["dataset", "n_train", "n_object_cols", "gate_d_fires",
                  "base_public", "base_private",
                  "rank_public", "rank_private",
                  "prob_public", "prob_private",
                  "logit_public", "logit_private"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            out = {
                "dataset": r.get("dataset", ""),
                "n_train": r.get("n_train", ""),
                "n_object_cols": r.get("n_object_cols", ""),
                "gate_d_fires": r.get("gate_d", ""),
                "base_public": r.get("base_pub", ""),
                "base_private": r.get("base_prv", ""),
                "rank_public": r.get("rank_pub", ""),
                "rank_private": r.get("rank_prv", ""),
                "prob_public": r.get("prob_pub", ""),
                "prob_private": r.get("prob_prv", ""),
                "logit_public": r.get("logit_pub", ""),
                "logit_private": r.get("logit_prv", ""),
            }
            w.writerow(out)

    # ---- gate-D' summary numbers per combine space ----
    gated = {}  # cfg -> {"pub":..,"prv":..}
    for cfg in BLENDS:
        gated[cfg] = {
            "gpub": gated_mean_delta(cfg, "pub"),
            "gprv": gated_mean_delta(cfg, "prv"),
            "fpub": firing_mean_delta(cfg, "pub"),
            "fprv": firing_mean_delta(cfg, "prv"),
            "wlt_pub": firing_wlt(cfg, "pub"),
            "wlt_prv": firing_wlt(cfg, "prv"),
        }

    # ---- reproduction anchor for the rank-avg arm ----
    rank_gpub = gated[RANK]["gpub"]
    rank_gprv = gated[RANK]["gprv"]
    # deviation vs round62 desk-eval reported figures
    dev_reported_pub = abs(rank_gpub - ROUND61_ANCHOR_PUB)
    dev_reported_prv = abs(rank_gprv - ROUND61_ANCHOR_PRV)
    max_dev_reported = max(dev_reported_pub, dev_reported_prv)
    # deviation vs anchor reconstructed from round61 results.csv (full precision)
    if anchor61 is not None:
        a_pub, a_prv, a_nfire = anchor61
        dev_recon_pub = abs(rank_gpub - a_pub)
        dev_recon_prv = abs(rank_gprv - a_prv)
        max_dev_recon = max(dev_recon_pub, dev_recon_prv)
    else:
        a_pub = a_prv = float("nan")
        a_nfire = 0
        dev_recon_pub = dev_recon_prv = max_dev_recon = float("nan")

    # partition sanity
    present = {r["dataset"] for r in rows if not (
        isinstance(r.get("base_pub"), float) and math.isnan(r.get("base_pub")))}
    all16_ok = (len(present) == N_DATASETS and not skipped)
    n_firing = sum(1 for r in rows if is_firing(r))

    # ================= SUMMARY =================
    L = []
    L.append("=" * 78)
    L.append("bench_03 round67 — RF x HGB BLEND COMBINE-SPACE ROBUSTNESS "
             "(rank / prob / logit) — ALL 16  [OFFLINE]")
    L.append("=" * 78)
    L.append("")
    L.append("SETUP:")
    L.append("  base  == shipped 08: HGB, early_stopping, cat_mask=object cols,")
    L.append("           seed-0, single fit on the FULL train frame.")
    L.append("  Fit base HGB proba (hgb) and RandomForest proba (rf) ONCE per")
    L.append("  dataset (RF: numeric[median-impute,no scaling] + object[const-")
    L.append("  impute + one-hot(ignore)] Pipeline -> RF(n_estimators=300)).")
    L.append("  THREE combine functions fuse the SAME (hgb, rf):")
    L.append("    rank  = (rankdata(hgb)+rankdata(rf))/2        [== round61]")
    L.append("    prob  = (hgb+rf)/2")
    L.append(f"    logit = (logit(hgb)+logit(rf))/2, logit(p)=log(p/(1-p)),")
    L.append(f"            p clipped to [{LOGIT_CLIP}, 1-{LOGIT_CLIP}]")
    L.append("  Single swept lever = the combine function. Fits are identical")
    L.append("  across the three spaces, so rank reproduces round61 exactly.")
    L.append("")
    L.append("GATE-D' firing predicate: n_train < 5000 AND n_object_cols > 0.")
    L.append(f"  firing datasets ({n_firing}): "
             + ", ".join(r["dataset"] for r in firing_rows))
    L.append("  GATE-D' MEAN DELTA = (blend-base) on firing datasets, 0 on non-")
    L.append("  firing, averaged over ALL 16 (== candidate B's benchmark effect).")
    L.append("")

    # ---- per combine space: gate-D' mean delta ----
    L.append("=== GATE-D' MEAN DELTA per COMBINE SPACE (vs base == shipped 08) ===")
    L.append(f"{'combine':<8} {'gate_dPub':>12} {'gate_dPrv':>12} "
             f"{'WLT_pub':>10} {'WLT_prv':>10}  (firing-only mean)")
    for cfg in BLENDS:
        g = gated[cfg]
        wp = "/".join(map(str, g["wlt_pub"]))
        wv = "/".join(map(str, g["wlt_prv"]))
        L.append(f"{cfg:<8} {g['gpub']:>+12.6f} {g['gprv']:>+12.6f} "
                 f"{wp:>10} {wv:>10}  "
                 f"(fPub={g['fpub']:+.6f} fPrv={g['fprv']:+.6f})")
    L.append("")
    L.append("  gate_dPub/gate_dPrv = gated mean over all 16 (non-firing = 0).")
    L.append("  WLT_* = win/lose/tie over the 5 firing datasets (eps=1e-6).")
    L.append("  firing-only mean = mean(blend-base) over the 5 firing datasets.")

    # ---- per-dataset delta table (firing set) ----
    for split, tag in (("pub", "Public"), ("prv", "Private")):
        L.append("")
        L.append(f"=== FIRING-SET PER-DATASET dAUC ({tag}) — blend vs base ===")
        L.append(f"{'dataset':<10} {'n_train':>8} {'obj':>4} {'base':>10} "
                 f"{'rank_d':>11} {'prob_d':>11} {'logit_d':>11}")
        for r in firing_rows:
            b = r.get(f"{BASE}_{split}")
            bstr = (f"{b:>10.6f}" if isinstance(b, float) and not math.isnan(b)
                    else f"{'nan':>10}")
            dstrs = []
            for cfg in BLENDS:
                dd = delta(r, cfg, split)
                dstrs.append(f"{dd:>+11.6f}" if not math.isnan(dd)
                             else f"{'nan':>11}")
            L.append(f"{r['dataset']:<10} {r.get('n_train',''):>8} "
                     f"{r.get('n_object_cols',''):>4} {bstr} "
                     f"{dstrs[0]} {dstrs[1]} {dstrs[2]}")

    # ---- reproduction anchor ----
    L.append("")
    L.append("=== REPRODUCTION ANCHOR (rank-avg arm reproduces round61) ===")
    L.append(f"  this run rank-avg gate-D' mean delta: "
             f"dPublic {rank_gpub:+.6f} / dPrivate {rank_gprv:+.6f}")
    L.append(f"  round62 desk-eval reported            : "
             f"dPublic {ROUND61_ANCHOR_PUB:+.6f} / dPrivate {ROUND61_ANCHOR_PRV:+.6f}")
    L.append(f"    |dev| vs reported: Public {dev_reported_pub:.2e}, "
             f"Private {dev_reported_prv:.2e}")
    if anchor61 is not None:
        L.append(f"  reconstructed from round61 results.csv ({a_nfire} firing): "
                 f"dPublic {a_pub:+.6f} / dPrivate {a_prv:+.6f}")
        L.append(f"    |dev| vs reconstructed: Public {dev_recon_pub:.2e}, "
                 f"Private {dev_recon_prv:.2e}")
        L.append(f"  >>> max|dev| (rank-avg vs round61 reconstructed) = "
                 f"{max_dev_recon:.2e}")
    else:
        L.append("  round61 results.csv NOT found -> reconstruction unavailable.")
    L.append(f"  >>> max|dev| (rank-avg vs round62 reported figures) = "
             f"{max_dev_reported:.2e}")
    repro_ok = (max_dev_reported <= 1e-4) or (
        anchor61 is not None and max_dev_recon <= 1e-9)
    L.append(f"  REPRODUCTION (rank-avg == round61): "
             f"{'PASS' if repro_ok else 'FAIL'}")

    # ---- verdict: does prob or logit beat rank, or is combine immaterial? ----
    L.append("")
    L.append("=== VERDICT (combine-space: does prob/logit beat rank-avg?) ===")

    def diff_vs_rank(cfg, key):
        return gated[cfg][key] - gated[RANK][key]

    for cfg in (PROB, LOGIT):
        dpub = diff_vs_rank(cfg, "gpub")
        dprv = diff_vs_rank(cfg, "gprv")
        L.append(f"  {cfg} - rank : gate_dPublic {dpub:+.6f}, "
                 f"gate_dPrivate {dprv:+.6f}")

    def clean_win_over_rank(cfg):
        return (diff_vs_rank(cfg, "gpub") > IMMATERIAL_EPS and
                diff_vs_rank(cfg, "gprv") > IMMATERIAL_EPS)

    def immaterial_vs_rank(cfg):
        return (abs(diff_vs_rank(cfg, "gpub")) <= IMMATERIAL_EPS and
                abs(diff_vs_rank(cfg, "gprv")) <= IMMATERIAL_EPS)

    prob_win = clean_win_over_rank(PROB)
    logit_win = clean_win_over_rank(LOGIT)
    prob_imm = immaterial_vs_rank(PROB)
    logit_imm = immaterial_vs_rank(LOGIT)
    L.append("")
    L.append(f"  immaterial threshold |Δ(gate-mean)| <= {IMMATERIAL_EPS:g} on BOTH splits.")
    L.append(f"  prob  vs rank: {'CLEAN WIN' if prob_win else ('IMMATERIAL' if prob_imm else 'MIXED/WORSE')}")
    L.append(f"  logit vs rank: {'CLEAN WIN' if logit_win else ('IMMATERIAL' if logit_imm else 'MIXED/WORSE')}")
    L.append("")
    if prob_win or logit_win:
        winners = [c for c, w in ((PROB, prob_win), (LOGIT, logit_win)) if w]
        L.append(f"  VERDICT: {'/'.join(winners)} CLEAN-WINS over rank-avg on BOTH "
                 "splits -> combine space MATTERS; prefer the winner.")
    elif prob_imm and logit_imm:
        L.append("  VERDICT: COMBINE SPACE IMMATERIAL — prob-avg and logit-avg both")
        L.append("  land within |Δ|<=5e-4 of rank-avg on both splits. Like round58's")
        L.append("  seed-agg-space finding, the RF x HGB fusion function does not")
        L.append("  change the gate-mean delta materially; rank-avg (round61) stands.")
    else:
        L.append("  VERDICT: MIXED — neither prob nor logit cleanly beats rank-avg,")
        L.append("  and at least one is not within the immaterial band. rank-avg")
        L.append("  remains the safe default (no clean-win challenger).")

    # ---- CLEAN RUN marker ----
    clean_run = ((not exceptions) and all16_ok and repro_ok
                 and (not skipped) and (not single_class_skips))
    L.append("")
    L.append(f"CLEAN RUN: {'YES' if clean_run else 'NO'} "
             f"({len(exceptions)} exceptions)  "
             f"[total_fits={total_fits}, datasets_scored={len(present)}/16, "
             f"skipped={len(skipped)}, single_class_skips={len(single_class_skips)}, "
             f"firing_datasets={n_firing}, "
             f"reproduction={'YES' if repro_ok else 'NO'} "
             f"(rank_maxdev_reported={max_dev_reported:.2e})]")
    for nm, msg in exceptions:
        L.append(f"  EXC {nm}: {msg}")
    for nm, cfg in single_class_skips:
        L.append(f"  SINGLE-CLASS {nm}/{cfg}")

    summary = "\n".join(L)
    print("\n" + summary)
    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
        f.write(summary + "\n")
    print(f"\n[WROTE] {csv_path}")
    print(f"[WROTE] {os.path.join(OUT_DIR, 'summary.txt')}")
    print(f"FINAL_MARKER CLEAN_RUN={'YES' if clean_run else 'NO'} "
          f"SCORED={len(present)}/16 EXC={len(exceptions)} "
          f"RANK_MAXDEV={max_dev_reported:.2e} "
          f"RANK_gPub={gated[RANK]['gpub']:+.6f} RANK_gPrv={gated[RANK]['gprv']:+.6f} "
          f"PROB_gPub={gated[PROB]['gpub']:+.6f} PROB_gPrv={gated[PROB]['gprv']:+.6f} "
          f"LOGIT_gPub={gated[LOGIT]['gpub']:+.6f} LOGIT_gPrv={gated[LOGIT]['gprv']:+.6f}")
    print("HARNESS_DONE")


if __name__ == "__main__":
    main()
