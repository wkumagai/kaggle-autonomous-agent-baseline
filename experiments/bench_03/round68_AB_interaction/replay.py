#!/usr/bin/env python
"""
bench_03 round68 — CANDIDATE-A x CANDIDATE-B ORTHOGONAL-COMPOSITION INTERACTION
(do the two standing ship-candidate levers STACK or INTERFERE where both fire?)
ALL 16.
OFFLINE ONLY. No subprocess, no LLM, no Kaggle, no network. Calls sklearn
in-process only. Writes ONLY under this round68 directory; NEVER touches
submissions/.

GOAL (improvement-log angle "AB_interaction")
---------------------------------------------
Two standing ship candidates for this agent, both gated:
  Candidate A = seed-averaging the base HGB: fit base-08 HGB K=10 times with
                random_state 0..9, average predict_proba (prob-mean). Gate-C =
                (n_object_cols > 0).
  Candidate B = cross-family RF blend: rank-average(base-08 HGB proba, RF proba).
                Gate-D' = (n_train < 5000 AND n_object_cols > 0), which fires on
                {train_03, train_05, train_09, train_13, train_15}. Gate-D' is a
                subset of gate-C, so on those 5 datasets BOTH levers fire.

This round asks the orthogonal-composition question: where BOTH fire, do the two
levers STACK (additive/superadditive gain), are they REDUNDANT (they capture the
same signal, so composing them buys little over the better single lever), or do
they INTERFERE (composing is worse than the better single lever)?

Design (4 arms; the ONLY difference between B and AB is seed-avg on the HGB arm):
  base = single HGB seed-0 (byte-identical to shipped 08) -> reproduces round61
         base_public / base_private with max|dev| = 0.
  A    = prob-mean of the base-08 HGB over seeds 0..9 (seed-avg, == round55
         PROBMEAN construction: arithmetic mean of the K proba vectors).
  B    = rankdata-avg(base HGB seed-0 proba, RF seed-0 proba) -> reproduces
         round61 blend_public / blend_private (same seed-0 HGB + same seed-0 RF,
         same (rankdata(a)+rankdata(b))/2 rank-avg).
  AB   = rankdata-avg(A's seed-avg HGB proba, RF seed-0 proba) <- the orthogonal
         composition: seed-avg the HGB arm THEN blend with the SAME RF.

  The SAME RF fit (seed-0) is reused for both B and AB, so the only difference
  between B and AB is whether the HGB arm is seed-0 (B) or seed-avg (AB).

BASE recipe (== shipped 08, identical to round61):
  features = [c for c in train.columns if c not in ("row_id","target")]
  cat_mask = [train[c].dtype == object for c in features]
  n=len(train); ratio = len(features)/n
  l2  = 1.0 if ratio >= 0.010 else 0.0
  msl = 70 if ratio>=0.030 else 50 if ratio>=0.015 else 20
  HistGradientBoostingClassifier(categorical_features=cat_mask, random_state=k,
      max_iter=300, early_stopping=True, l2_regularization=l2,
      min_samples_leaf=msl)

RF view (identical to round61): numeric[median-impute, no scaling] +
  object[constant-impute + OneHotEncoder(handle_unknown='ignore')] in a
  Pipeline -> RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=4).
  Single fit, seed-0, reused for B and AB.

rank_average (identical to round61): (rankdata(a) + rankdata(b)) / 2.0, default
  method (='average'). AUC is rank-invariant so the averaged rank is the score.

REPRODUCTION (MANDATORY — proves the harness is faithful, BIT-IDENTICAL):
  1. base on ALL 16 must match round61 base_public/base_private, max|dev| = 0.
  2. B    on ALL 16 must match round61 blend_public/blend_private, max|dev| ~ 0
     (same seed-0 HGB + RF rank-avg).
  If either check fails, CLEAN RUN = NO.
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
OUT_DIR = os.path.join(BENCH_DIR, "round68_AB_interaction")
STATS_CSV = os.path.join(BENCH_DIR, "dataset_stats.csv")
# Anchor against round61 (base column == base-08 seed-0; blend column == the
# seed-0 HGB rank-avg RF blend). Both are reproduced bit-identically here.
ROUND61_RESULTS = os.path.join(BENCH_DIR, "round61_rf_blend", "results.csv")

L2_GATE_THRESHOLD = 0.010   # shipped-08 ratio gate for l2 (FIXED)
GATED_L2 = 1.0
DEFAULT_MSL = 20
MSL_TIERS = [(0.030, 70), (0.015, 50)]
BASE_SEED = 0
K = 10                       # seed-avg ensemble size (Candidate A)
RF_N_ESTIMATORS = 300
RF_N_JOBS = 4
N_DATASETS = 16
REPRO_TOL = 0.0             # BIT-IDENTICAL: base & B must match round61 exactly

# Gate-D' fires on these 5 (n_train < 5000 AND n_object_cols > 0). Both levers
# fire here. Asserted below against the gate computed from dataset_stats.csv.
GATE_DPRIME_N = 5000
EXPECTED_DPRIME = {"train_03", "train_05", "train_09", "train_13", "train_15"}

BASE = "base"
A = "A"          # seed-avg HGB (prob-mean over seeds 0..9)
B = "B"          # rank-avg(base HGB seed-0, RF seed-0)
AB = "AB"        # rank-avg(seed-avg HGB, RF seed-0)
ARMS = [A, B, AB]
ALL_CONFIGS = [BASE, A, B, AB]


def load_stats(path=STATS_CSV):
    stats = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            stats[row["name"]] = {
                "n_train": int(row["n_train"]),
                "n_object_cols": int(row["n_object_cols"]),
            }
    return stats


def round61_anchors(path=ROUND61_RESULTS):
    """Read round61's base_public/base_private (anchor for base arm) AND
    blend_public/blend_private (anchor for B arm) for ALL 16 datasets to anchor
    reproduction at full precision. Returns dict
    name -> {"base": (pub, prv), "blend": (pub, prv)} or None if unavailable."""
    if not os.path.exists(path):
        return None
    anchors = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            name = row.get("dataset")
            entry = {}
            try:
                entry["base"] = (float(row["base_public"]),
                                 float(row["base_private"]))
            except (KeyError, ValueError):
                entry["base"] = None
            try:
                entry["blend"] = (float(row["blend_public"]),
                                  float(row["blend_private"]))
            except (KeyError, ValueError):
                entry["blend"] = None
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


def fit_rf(train_frame, test, features, seed):
    """Fit a RandomForestClassifier on a robust numeric+categorical view and
    return P(class==1) on test. Numeric: median-impute (NO scaling — trees are
    scale-invariant). Object: constant-impute + one-hot(handle_unknown='ignore').
    Identical to round61's RF arm. Single fit, seed-0, reused by B and AB."""
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


def rank_average(a, b):
    """Elementwise mean of the ranks of two score vectors (identical to round61).
    AUC is rank-invariant so the averaged rank is used directly as the score."""
    return (rankdata(a) + rankdata(b)) / 2.0


def run_one(name, train_csv, test_csv, stats):
    """Returns (preds, meta). preds maps config_name -> {row_id -> score}.

    base = seed-0 base-08 HGB (byte-identical to shipped 08 / round61 base).
    A    = prob-mean of base-08 HGB over seeds 0..9 (seed-avg).
    B    = rank-avg(base HGB seed-0, RF seed-0) (== round61 blend).
    AB   = rank-avg(A seed-avg HGB, RF seed-0) (orthogonal composition).

    The K HGB seed fits (0..9) are done ONCE; base = seed 0 vector, A = their
    mean. The RF is fit ONCE (seed-0) and reused for B and AB.
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

    # ---- HGB seed cache: fit K=10 base-08 HGB on FULL train, seeds 0..9. ----
    # seed 0 IS the base-08 arm (byte-identical to round61 base); the mean over
    # all K is Candidate A (seed-avg). Both derived from the SAME cached vectors.
    seed_vecs = np.zeros((K, n_test), dtype=np.float64)
    for k in range(K):
        seed_vecs[k] = fit_hgb(train, test, features, cat_mask, l2, msl_val, k)
        n_fits += 1

    base_vec = seed_vecs[BASE_SEED]                 # seed-0 HGB (== shipped 08)
    a_vec = seed_vecs.mean(axis=0)                  # seed-avg HGB (Candidate A)

    # ---- RF: ONE seed-0 fit, reused for BOTH B and AB. ----
    rf_proba = fit_rf(train, test, features, BASE_SEED)
    n_fits += 1

    # ---- blends (identical rank-avg; only the HGB arm differs). ----
    b_vec = rank_average(base_vec, rf_proba)        # == round61 blend
    ab_vec = rank_average(a_vec, rf_proba)          # orthogonal composition

    preds[BASE] = dict(zip(row_ids, base_vec.tolist()))
    preds[A] = dict(zip(row_ids, a_vec.tolist()))
    preds[B] = dict(zip(row_ids, b_vec.tolist()))
    preds[AB] = dict(zip(row_ids, ab_vec.tolist()))

    st = stats[name]
    gate_c = st["n_object_cols"] > 0
    gate_dprime = (st["n_train"] < GATE_DPRIME_N) and (st["n_object_cols"] > 0)
    meta = {
        "n_train": st["n_train"],
        "n_object_cols": st["n_object_cols"],
        "l2": l2,
        "msl": msl_val,
        "n_features": len(features),
        "n_cat": sum(cat_mask),
        "gate_c": gate_c,
        "gate_dprime": gate_dprime,
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
    anchors61 = round61_anchors()
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
                "gate_c": meta["gate_c"],
                "gate_dprime": meta["gate_dprime"],
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
                  f"gateC={meta['gate_c']} gateD'={meta['gate_dprime']} "
                  f"feats={meta['n_features']} cat={meta['n_cat']} "
                  f"l2={meta['l2']} msl={meta['msl']} fits={meta['n_fits']} | "
                  f"base pub={rec['base_pub']:.6f} prv={rec['base_prv']:.6f}  "
                  f"A pub={rec['A_pub']:.6f} prv={rec['A_prv']:.6f}  "
                  f"B pub={rec['B_pub']:.6f} prv={rec['B_prv']:.6f}  "
                  f"AB pub={rec['AB_pub']:.6f} prv={rec['AB_prv']:.6f}")
        except Exception as e:
            exceptions.append((name, repr(e)))
            rec.update({"n_train": stats.get(name, {}).get("n_train", ""),
                        "n_object_cols": stats.get(name, {}).get("n_object_cols", ""),
                        "gate_c": stats.get(name, {}).get("n_object_cols", 0) > 0,
                        "gate_dprime": (stats.get(name, {}).get("n_train", 10**9)
                                        < GATE_DPRIME_N
                                        and stats.get(name, {}).get("n_object_cols", 0) > 0),
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

    def mean_over(vals):
        vals = [v for v in vals if not math.isnan(v)]
        return sum(vals) / len(vals) if vals else float("nan")

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
    is_dprime = lambda r: bool(r.get("gate_dprime"))

    # ---- results.csv ----
    csv_path = os.path.join(OUT_DIR, "results.csv")
    fieldnames = ["dataset", "n_train", "n_object_cols", "gate_c", "gate_dprime",
                  "base_public", "base_private",
                  "A_public", "A_private", "B_public", "B_private",
                  "AB_public", "AB_private",
                  "dA_public", "dA_private", "dB_public", "dB_private",
                  "dAB_public", "dAB_private"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            out = {
                "dataset": r.get("dataset", ""),
                "n_train": r.get("n_train", ""),
                "n_object_cols": r.get("n_object_cols", ""),
                "gate_c": r.get("gate_c", ""),
                "gate_dprime": r.get("gate_dprime", ""),
                "base_public": r.get("base_pub", ""),
                "base_private": r.get("base_prv", ""),
                "A_public": r.get("A_pub", ""),
                "A_private": r.get("A_prv", ""),
                "B_public": r.get("B_pub", ""),
                "B_private": r.get("B_prv", ""),
                "AB_public": r.get("AB_pub", ""),
                "AB_private": r.get("AB_prv", ""),
                "dA_public": delta(r, A, "pub"),
                "dA_private": delta(r, A, "prv"),
                "dB_public": delta(r, B, "pub"),
                "dB_private": delta(r, B, "prv"),
                "dAB_public": delta(r, AB, "pub"),
                "dAB_private": delta(r, AB, "prv"),
            }
            w.writerow(out)

    # ---- REPRODUCTION: base vs round61 base; B vs round61 blend (BIT-IDENTICAL) --
    repro_available = anchors61 is not None
    by_name = {r["dataset"]: r for r in rows}

    def build_repro(arm_key, anchor_key):
        repro = {}
        ok_all = True
        max_dev = 0.0
        for i in range(1, N_DATASETS + 1):
            nm = f"train_{i:02d}"
            r = by_name.get(nm)
            mine = (r.get(f"{arm_key}_pub"), r.get(f"{arm_key}_prv")) if r \
                else (None, None)
            ref = anchors61.get(nm, {}).get(anchor_key) if anchors61 else None
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
    repro_b, repro_b_ok, max_dev_b = build_repro(B, "blend")
    repro_ok = repro_base_ok and repro_b_ok
    max_abs_dev = max(max_dev_base, max_dev_b)

    # ---- partition sanity ----
    present = {r["dataset"] for r in rows if not (
        isinstance(r.get("base_pub"), float) and math.isnan(r.get("base_pub")))}
    all16_ok = (len(present) == N_DATASETS and not skipped)
    n_gate_c = sum(1 for r in rows if r.get("gate_c"))
    fired = [r for r in rows if r.get("gate_dprime")]
    fired_names = sorted(r["dataset"] for r in fired)
    dprime_ok = (set(fired_names) == EXPECTED_DPRIME)

    # ================= SUMMARY =================
    L = []
    L.append("=" * 78)
    L.append("bench_03 round68 — CANDIDATE-A x CANDIDATE-B ORTHOGONAL-COMPOSITION "
             "INTERACTION (ALL 16)  [OFFLINE]")
    L.append("=" * 78)
    L.append("")
    L.append("SETUP (4 arms; only the HGB arm differs between B and AB):")
    L.append("  base = single HGB seed-0 (== shipped 08 / round61 base).")
    L.append("  A    = prob-mean of base-08 HGB over seeds 0..9 (seed-avg).")
    L.append("  B    = rank-avg(base HGB seed-0, RF seed-0) (== round61 blend).")
    L.append("  AB   = rank-avg(A seed-avg HGB, RF seed-0) (orthogonal composition).")
    L.append("  The SAME RF seed-0 fit is reused for B and AB.")
    L.append("")
    L.append("  Candidate A gate-C  = (n_object_cols > 0).")
    L.append("  Candidate B gate-D' = (n_train < 5000 AND n_object_cols > 0).")
    L.append(f"  gate-D' subset of gate-C -> where D' fires BOTH levers fire.")
    L.append(f"  gate-D' fired on {len(fired_names)} datasets: {fired_names}")
    L.append(f"  expected {sorted(EXPECTED_DPRIME)} -> "
             f"{'MATCH' if dprime_ok else 'MISMATCH'}")

    # ---- FOCUS: the 5 gate-D' datasets (where BOTH fire) ----
    L.append("")
    L.append("=" * 78)
    L.append("FOCUS: 5 GATE-D' DATASETS (where BOTH levers fire)")
    L.append("=" * 78)

    # per-dataset interaction table
    L.append("")
    L.append("=== PER-DATASET (gate-D' only): dA, dB, dAB and interaction ===")
    L.append(f"{'dataset':<10} {'split':>7} {'dA':>10} {'dB':>10} {'dAB':>10} "
             f"{'AB-B':>10} {'AB-(dA+dB)':>12} {'max(dA,dB)':>11} {'verdict':>10}")

    def per_verdict(dA, dB, dAB, eps=1e-4):
        if math.isnan(dA) or math.isnan(dB) or math.isnan(dAB):
            return "nan"
        best = max(dA, dB)
        if dAB > best + eps:
            return "STACK"
        if dAB < best - eps:
            return "INTERFERE"
        return "REDUNDANT"

    for r in fired:
        for split, tag in (("pub", "Public"), ("prv", "Private")):
            dA = delta(r, A, split)
            dB = delta(r, B, split)
            dAB = delta(r, AB, split)
            ab_minus_b = (dAB - dB) if not (math.isnan(dAB) or math.isnan(dB)) \
                else float("nan")
            ab_minus_add = (dAB - (dA + dB)) \
                if not (math.isnan(dAB) or math.isnan(dA) or math.isnan(dB)) \
                else float("nan")
            best = max(dA, dB) if not (math.isnan(dA) or math.isnan(dB)) \
                else float("nan")

            def fmt(x, w=10):
                return f"{x:>+{w}.6f}" if not math.isnan(x) else f"{'nan':>{w}}"
            L.append(f"{r['dataset']:<10} {tag:>7} {fmt(dA)} {fmt(dB)} {fmt(dAB)} "
                     f"{fmt(ab_minus_b)} {fmt(ab_minus_add,12)} {fmt(best,11)} "
                     f"{per_verdict(dA, dB, dAB):>10}")

    # mean deltas on the 5 fired datasets (both splits)
    L.append("")
    L.append("=== MEAN Δ vs base on the 5 gate-D' datasets (both splits) ===")
    means = {}
    for arm in ARMS:
        mp = mean_delta(arm, "pub", is_dprime)
        mv = mean_delta(arm, "prv", is_dprime)
        means[arm] = (mp, mv)
        L.append(f"  {arm:<3}: mean dPublic={mp:+.6f}   mean dPrivate={mv:+.6f}")

    # ---- INTERACTION TEST (on 5-dataset means) ----
    L.append("")
    L.append("=== INTERACTION TEST (on the 5-dataset means) ===")
    L.append("  STACK    : mean dAB > max(mean dA, mean dB)   (composition beats "
             "the better single lever; ideally dAB >~ dA+dB)")
    L.append("  REDUNDANT: mean dAB ~= max(mean dA, mean dB)  (levers overlap)")
    L.append("  INTERFERE: mean dAB < max(mean dA, mean dB)   (composition hurts)")
    L.append("")
    EPS_V = 1e-4
    split_verdicts = {}
    for split, tag in (("pub", "Public"), ("prv", "Private")):
        mA = means[A][0] if split == "pub" else means[A][1]
        mB = means[B][0] if split == "pub" else means[B][1]
        mAB = means[AB][0] if split == "pub" else means[AB][1]
        best = max(mA, mB)
        additive = mA + mB
        ab_vs_b = mAB - mB
        ab_vs_best = mAB - best
        ab_vs_add = mAB - additive
        v = per_verdict(mA, mB, mAB, EPS_V)
        split_verdicts[split] = v
        L.append(f"  [{tag}]")
        L.append(f"    mean dA={mA:+.6f}  mean dB={mB:+.6f}  mean dAB={mAB:+.6f}")
        L.append(f"    max(dA,dB)={best:+.6f}   additive(dA+dB)={additive:+.6f}")
        L.append(f"    AB vs B          = {ab_vs_b:+.6f}  "
                 f"(does adding seed-avg into the blend help? "
                 f"{'YES' if ab_vs_b > EPS_V else 'NO' if ab_vs_b < -EPS_V else 'NEUTRAL'})")
        L.append(f"    AB vs max(dA,dB) = {ab_vs_best:+.6f}")
        L.append(f"    AB vs additive   = {ab_vs_add:+.6f}  "
                 f"({'superadditive' if ab_vs_add > EPS_V else 'subadditive' if ab_vs_add < -EPS_V else 'additive'})")
        L.append(f"    -> {v}")
        L.append("")

    # combined verdict = average of the two split means, then classify
    mA_c = mean_over([means[A][0], means[A][1]])
    mB_c = mean_over([means[B][0], means[B][1]])
    mAB_c = mean_over([means[AB][0], means[AB][1]])
    combined_verdict = per_verdict(mA_c, mB_c, mAB_c, EPS_V)
    if split_verdicts["pub"] == split_verdicts["prv"]:
        oneline = split_verdicts["pub"]
    else:
        oneline = (f"{combined_verdict} (combined; Public={split_verdicts['pub']}, "
                   f"Private={split_verdicts['prv']})")
    L.append(f"  COMBINED (avg of both split means): mean dA={mA_c:+.6f}  "
             f"mean dB={mB_c:+.6f}  mean dAB={mAB_c:+.6f}")
    L.append(f"  ONE-LINE VERDICT: {oneline}")

    # ---- UN-GATED all-16 context ----
    L.append("")
    L.append("=" * 78)
    L.append("CONTEXT: UN-GATED ALL-16 VIEW (arm vs base, all 16 datasets)")
    L.append("=" * 78)
    for arm in ARMS:
        mp = mean_delta(arm, "pub")
        mv = mean_delta(arm, "prv")
        wp, lp, tp = wlt(arm, "pub")
        wv, lv, tv = wlt(arm, "prv")
        L.append(f"  {arm:<3}: mean dPublic={mp:+.6f} (W/L/T {wp}/{lp}/{tp})  "
                 f"mean dPrivate={mv:+.6f} (W/L/T {wv}/{lv}/{tv})")
    L.append("")
    L.append(f"=== GATE-C VIEW (n_object_cols>0, {n_gate_c} datasets) ===")
    for arm in ARMS:
        mp = mean_delta(arm, "pub", is_gatec)
        mv = mean_delta(arm, "prv", is_gatec)
        wp, lp, tp = wlt(arm, "pub", is_gatec)
        wv, lv, tv = wlt(arm, "prv", is_gatec)
        L.append(f"  {arm:<3}: mean dPublic={mp:+.6f} (W/L/T {wp}/{lp}/{tp})  "
                 f"mean dPrivate={mv:+.6f} (W/L/T {wv}/{lv}/{tv})")

    # ---- FULL PER-DATASET TABLE (all 16) ----
    for split, tag in (("pub", "Public"), ("prv", "Private")):
        L.append("")
        L.append(f"=== PER-DATASET dAUC ({tag}) — all 16 ===")
        L.append(f"{'dataset':<10} {'gateC':>6} {'gateD':>6} {'base':>10} "
                 f"{'dA':>10} {'dB':>10} {'dAB':>10}")
        for r in rows:
            b = r.get(f"{BASE}_{split}")
            bstr = f"{b:>10.6f}" if isinstance(b, float) and not math.isnan(b) \
                else f"{'nan':>10}"

            def fmt(x):
                return f"{x:>+10.6f}" if not math.isnan(x) else f"{'nan':>10}"
            L.append(f"{r['dataset']:<10} {str(bool(r.get('gate_c'))):>6} "
                     f"{str(bool(r.get('gate_dprime'))):>6} {bstr} "
                     f"{fmt(delta(r, A, split))} {fmt(delta(r, B, split))} "
                     f"{fmt(delta(r, AB, split))}")

    # ---- REPRODUCTION CHECK 1: base vs round61 base ----
    L.append("")
    L.append("=== REPRODUCTION CHECK 1 (base on ALL 16 vs round61 base, tol=0) ===")
    if not repro_available:
        L.append("  round61 results.csv NOT found -> reproduction NOT anchored (FAIL).")
    else:
        for i in range(1, N_DATASETS + 1):
            nm = f"train_{i:02d}"
            rr = repro_base[nm]
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
        L.append(f"  REPRODUCTION 1 (base==round61 base): "
                 f"{'PASS' if repro_base_ok else 'FAIL'}")

    # ---- REPRODUCTION CHECK 2: B vs round61 blend ----
    L.append("")
    L.append("=== REPRODUCTION CHECK 2 (B on ALL 16 vs round61 blend, tol=0) ===")
    if not repro_available:
        L.append("  round61 results.csv NOT found -> reproduction NOT anchored (FAIL).")
    else:
        for i in range(1, N_DATASETS + 1):
            nm = f"train_{i:02d}"
            rr = repro_b[nm]
            mp_, mv_ = rr["mine"]
            rp_, rv_ = rr["ref"] if rr["ref"] else (float("nan"), float("nan"))
            mp_s = f"{mp_:.6f}" if isinstance(mp_, float) and not math.isnan(mp_) else "nan"
            mv_s = f"{mv_:.6f}" if isinstance(mv_, float) and not math.isnan(mv_) else "nan"
            L.append(
                f"  {nm}: Public {mp_s} vs r61bl {rp_:.6f} "
                f"(|d|={rr['devp']:.2e}, {'YES' if rr['okp'] else 'NO'}); "
                f"Private {mv_s} vs r61bl {rv_:.6f} "
                f"(|d|={rr['devv']:.2e}, {'YES' if rr['okv'] else 'NO'})")
        L.append(f"  max |dev| over all 16x2 = {max_dev_b:.2e}")
        L.append(f"  REPRODUCTION 2 (B==round61 blend): "
                 f"{'PASS' if repro_b_ok else 'FAIL'}")

    # ---- CLEAN RUN marker ----
    clean_run = ((not exceptions) and all16_ok and repro_ok and repro_available
                 and (not skipped) and (not single_class_skips) and dprime_ok)
    L.append("")
    L.append(f"CLEAN RUN: {'YES' if clean_run else 'NO'} "
             f"({len(exceptions)} exceptions)  "
             f"[total_fits={total_fits}, datasets_scored={len(present)}/16, "
             f"skipped={len(skipped)}, single_class_skips={len(single_class_skips)}, "
             f"gate_c_datasets={n_gate_c}, gate_dprime_datasets={len(fired_names)}, "
             f"dprime_set_ok={'YES' if dprime_ok else 'NO'}, "
             f"reproduction={'YES' if repro_ok else 'NO'} "
             f"(base_maxdev={max_dev_base:.2e}, B_maxdev={max_dev_b:.2e})]")
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
    print(f"FINAL_MARKER CLEAN_RUN={'YES' if clean_run else 'NO'} "
          f"SCORED={len(present)}/16 EXC={len(exceptions)} "
          f"BASE_MAXDEV={max_dev_base:.2e} B_MAXDEV={max_dev_b:.2e} "
          f"DPRIME_5MEAN_dA_pub={means[A][0]:+.6f} dA_prv={means[A][1]:+.6f} "
          f"dB_pub={means[B][0]:+.6f} dB_prv={means[B][1]:+.6f} "
          f"dAB_pub={means[AB][0]:+.6f} dAB_prv={means[AB][1]:+.6f} "
          f"VERDICT={oneline}")
    print("HARNESS_DONE")


if __name__ == "__main__":
    main()
