#!/usr/bin/env python
"""
bench_03 round70 — AB SYNTHESIS ON THE FULL GATE-C FIRING SET
(if a future "AB synthesis" ship were gated on the BROAD gate-C instead of the
narrow gate-D', does AB regress on the gate-C-only datasets?)
OFFLINE ONLY. No subprocess, no LLM, no Kaggle, no network. Calls sklearn
in-process only. Writes ONLY under this round70 directory; NEVER touches
submissions/.

GOAL (improvement-log angle "AB_gateC")
---------------------------------------
round68 examined the AB orthogonal composition ONLY on the 5 gate-D' overlap
datasets {train_03, train_05, train_09, train_13, train_15}, where BOTH ship
levers fire, and found AB STACKs (subadditively) there.

round70 extends the SAME 4-arm AB analysis to the FULL gate-C firing set (ALL
datasets with n_object_cols > 0). The ship-relevant question: if a future "AB
synthesis" were gated on the broad gate-C (candidate A's gate) instead of the
narrow gate-D' (candidate B's gate), does AB REGRESS on the "gate-C-only"
datasets — those where gate-C fires but gate-D' does NOT (n_train >= 5000 AND
n_object_cols > 0)? Prior rounds (round66) found the RF-blend clean-win region is
n_train in (500, 8173], so we expect AB to potentially regress on larger-n
gate-C-only datasets because the RF arm hurts there.

Design (4 arms; identical to round68 — the ONLY difference between B and AB is
seed-avg on the HGB arm):
  base = single HGB seed-0 (byte-identical to shipped 08) -> reproduces round61
         base_public / base_private with max|dev| = 0.
  A    = prob-mean of the base-08 HGB over seeds 0..9 (seed-avg, arithmetic mean
         of the K proba vectors == round55 PROBMEAN construction). Gate-C.
  B    = rankdata-avg(base HGB seed-0 proba, RF seed-0 proba) -> reproduces
         round61 blend_public / blend_private (same seed-0 HGB + same seed-0 RF,
         same (rankdata(a)+rankdata(b))/2 rank-avg). Gate-D'.
  AB   = rankdata-avg(A's seed-avg HGB proba, RF seed-0 proba) <- the AB
         synthesis: seed-avg the HGB arm THEN blend with the SAME RF.

  The SAME RF fit (seed-0) is reused for both B and AB, so the only difference
  between B and AB is whether the HGB arm is seed-0 (B) or seed-avg (AB).

BASE recipe (== shipped 08, identical to round61 / round68):
  features = [c for c in train.columns if c not in ("row_id","target")]
  cat_mask = [train[c].dtype == object for c in features]
  n=len(train); ratio = len(features)/n
  l2  = 1.0 if ratio >= 0.010 else 0.0
  msl = 70 if ratio>=0.030 else 50 if ratio>=0.015 else 20
  HistGradientBoostingClassifier(categorical_features=cat_mask, random_state=k,
      max_iter=300, early_stopping=True, l2_regularization=l2,
      min_samples_leaf=msl)

RF view (identical to round61 / round68): numeric[median-impute, no scaling] +
  object[constant-impute + OneHotEncoder(handle_unknown='ignore')] in a
  Pipeline -> RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=4).
  Single fit, seed-0, reused for B and AB.

rank_average (identical to round61 / round68): (rankdata(a) + rankdata(b)) / 2.0,
  default method (='average'). AUC is rank-invariant so the averaged rank is the
  score.

GATES:
  gate-C  = (n_object_cols > 0)                       — candidate A's gate (broad).
  gate-D' = (n_train < 5000 AND n_object_cols > 0)    — candidate B's gate (subset
            of gate-C; fires on {train_03, train_05, train_09, train_13,
            train_15}).
  gate-C-only = gate-C AND NOT gate-D' (n_train >= 5000 AND n_object_cols > 0)
            — the NEW region this round scrutinizes.

REPRODUCTION (MANDATORY — proves the harness is faithful, BIT-IDENTICAL):
  1. base on ALL 16 must match round61 base_public/base_private, max|dev| = 0.
  2. B    on ALL 16 must match round61 blend_public/blend_private, max|dev| ~ 0
     (same seed-0 HGB + RF rank-avg).
  If either check fails, CLEAN RUN = NO and the summary says why.
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
OUT_DIR = os.path.join(BENCH_DIR, "round70_AB_gateC")
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
# Reproduction tolerances (per the round70 contract):
#   base MUST be bit-identical to round61 base  -> tol = 0.
#   B    MUST match round61 blend at max|dev| ~ 0, tolerance 1e-6. The RF arm's
#        proba can drift at the ~1e-8 AUC level across numpy/pandas builds (float
#        accumulation in the RF/one-hot path), which is why the contract allows a
#        1e-6 band for B rather than exact bit-identity.
REPRO_TOL_BASE = 0.0
REPRO_TOL_B = 1e-6

# Gate-D' fires on these 5 (n_train < 5000 AND n_object_cols > 0). Subset of
# gate-C. Asserted below against the gate computed from dataset_stats.csv.
GATE_DPRIME_N = 5000
EXPECTED_DPRIME = {"train_03", "train_05", "train_09", "train_13", "train_15"}
# gate-C fires on these 11 (n_object_cols > 0). Asserted below.
EXPECTED_GATE_C = {"train_01", "train_02", "train_03", "train_05", "train_06",
                   "train_07", "train_08", "train_09", "train_12", "train_13",
                   "train_14", "train_15"}
# gate-C-only = gate-C AND NOT gate-D' (n_train >= 5000 AND n_object_cols > 0).
EXPECTED_GATE_C_ONLY = EXPECTED_GATE_C - EXPECTED_DPRIME

BASE = "base"
A = "A"          # seed-avg HGB (prob-mean over seeds 0..9)
B = "B"          # rank-avg(base HGB seed-0, RF seed-0)
AB = "AB"        # rank-avg(seed-avg HGB, RF seed-0)
ARMS = [A, B, AB]
ALL_CONFIGS = [BASE, A, B, AB]


def read_frame(path):
    """Load a CSV and normalise string columns back to numpy `object` dtype.

    pandas >= 3.0 infers text columns as the new pandas StringDtype rather than
    `object`, which silently breaks the shipped-08 recipe's categorical detection
    (`cat_mask = [train[c].dtype == object]`) — StringDtype != object, so the mask
    would come out empty and HGB would try to parse strings as floats. round61 /
    round68 ran under an older pandas where these columns were `object`. Restoring
    `object` here reproduces the ORIGINAL cat_mask (and the numeric/categorical
    split used by the RF view) bit-identically, so base==round61 base and
    B==round61 blend still hold at max|dev|=0. Numeric (int64/float64) columns are
    untouched. This is a pure load-time compatibility shim; the recipe below is
    unchanged from round68."""
    df = pd.read_csv(path)
    for c in df.columns:
        if isinstance(df[c].dtype, pd.StringDtype):
            df[c] = df[c].astype(object)
    return df


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
    AB   = rank-avg(A seed-avg HGB, RF seed-0) (AB synthesis).

    The K HGB seed fits (0..9) are done ONCE; base = seed 0 vector, A = their
    mean. The RF is fit ONCE (seed-0) and reused for B and AB.
    """
    train = read_frame(train_csv)
    test = read_frame(test_csv)

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
    ab_vec = rank_average(a_vec, rf_proba)          # AB synthesis

    preds[BASE] = dict(zip(row_ids, base_vec.tolist()))
    preds[A] = dict(zip(row_ids, a_vec.tolist()))
    preds[B] = dict(zip(row_ids, b_vec.tolist()))
    preds[AB] = dict(zip(row_ids, ab_vec.tolist()))

    st = stats[name]
    gate_c = st["n_object_cols"] > 0
    gate_dprime = (st["n_train"] < GATE_DPRIME_N) and (st["n_object_cols"] > 0)
    gate_c_only = gate_c and not gate_dprime
    meta = {
        "n_train": st["n_train"],
        "n_object_cols": st["n_object_cols"],
        "l2": l2,
        "msl": msl_val,
        "n_features": len(features),
        "n_cat": sum(cat_mask),
        "gate_c": gate_c,
        "gate_dprime": gate_dprime,
        "gate_c_only": gate_c_only,
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
                "gate_c_only": meta["gate_c_only"],
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
                  f"gateC_only={meta['gate_c_only']} "
                  f"feats={meta['n_features']} cat={meta['n_cat']} "
                  f"l2={meta['l2']} msl={meta['msl']} fits={meta['n_fits']} | "
                  f"base pub={rec['base_pub']:.6f} prv={rec['base_prv']:.6f}  "
                  f"A pub={rec['A_pub']:.6f} prv={rec['A_prv']:.6f}  "
                  f"B pub={rec['B_pub']:.6f} prv={rec['B_prv']:.6f}  "
                  f"AB pub={rec['AB_pub']:.6f} prv={rec['AB_prv']:.6f}")
        except Exception as e:
            exceptions.append((name, repr(e)))
            st = stats.get(name, {})
            gc = st.get("n_object_cols", 0) > 0
            gd = (st.get("n_train", 10**9) < GATE_DPRIME_N
                  and st.get("n_object_cols", 0) > 0)
            rec.update({"n_train": st.get("n_train", ""),
                        "n_object_cols": st.get("n_object_cols", ""),
                        "gate_c": gc,
                        "gate_dprime": gd,
                        "gate_c_only": gc and not gd,
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
    is_gatec_only = lambda r: bool(r.get("gate_c_only"))

    # ---- results.csv ----
    csv_path = os.path.join(OUT_DIR, "results.csv")
    fieldnames = ["dataset", "n_train", "n_object_cols", "gate_c", "gate_dprime",
                  "gate_c_only",
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
                "gate_c_only": r.get("gate_c_only", ""),
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

    def build_repro(arm_key, anchor_key, tol):
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
                okp = devp <= tol
                okv = devv <= tol
                max_dev = max(max_dev, devp, devv)
            repro[nm] = {"mine": mine, "ref": ref, "okp": okp, "okv": okv,
                         "devp": devp, "devv": devv}
            if not (okp and okv):
                ok_all = False
        return repro, ok_all, max_dev

    repro_base, repro_base_ok, max_dev_base = build_repro(BASE, "base",
                                                          REPRO_TOL_BASE)
    repro_b, repro_b_ok, max_dev_b = build_repro(B, "blend", REPRO_TOL_B)
    repro_ok = repro_base_ok and repro_b_ok
    max_abs_dev = max(max_dev_base, max_dev_b)

    # ---- partition sanity ----
    present = {r["dataset"] for r in rows if not (
        isinstance(r.get("base_pub"), float) and math.isnan(r.get("base_pub")))}
    all16_ok = (len(present) == N_DATASETS and not skipped)
    gate_c_rows = [r for r in rows if r.get("gate_c")]
    gate_c_names = sorted(r["dataset"] for r in gate_c_rows)
    n_gate_c = len(gate_c_rows)
    fired = [r for r in rows if r.get("gate_dprime")]
    fired_names = sorted(r["dataset"] for r in fired)
    dprime_ok = (set(fired_names) == EXPECTED_DPRIME)
    gate_c_only_rows = [r for r in rows if r.get("gate_c_only")]
    gate_c_only_names = sorted(r["dataset"] for r in gate_c_only_rows)
    gate_c_ok = (set(gate_c_names) == EXPECTED_GATE_C)
    gate_c_only_ok = (set(gate_c_only_names) == EXPECTED_GATE_C_ONLY)

    # ================= SUMMARY =================
    L = []
    L.append("=" * 78)
    L.append("bench_03 round70 — AB SYNTHESIS ON THE FULL GATE-C FIRING SET "
             "(ALL 16)  [OFFLINE]")
    L.append("=" * 78)
    L.append("")
    L.append("SETUP (4 arms; only the HGB arm differs between B and AB):")
    L.append("  base = single HGB seed-0 (== shipped 08 / round61 base).")
    L.append("  A    = prob-mean of base-08 HGB over seeds 0..9 (seed-avg).")
    L.append("  B    = rank-avg(base HGB seed-0, RF seed-0) (== round61 blend).")
    L.append("  AB   = rank-avg(A seed-avg HGB, RF seed-0) (AB synthesis).")
    L.append("  The SAME RF seed-0 fit is reused for B and AB.")
    L.append("")
    L.append("  Candidate A gate-C      = (n_object_cols > 0)  [broad].")
    L.append("  Candidate B gate-D'     = (n_train < 5000 AND n_object_cols > 0).")
    L.append("  gate-C-only             = gate-C AND NOT gate-D' "
             "(n_train >= 5000 AND n_object_cols > 0).")
    L.append("")

    # ---- GATE PARTITION ----
    L.append("=" * 78)
    L.append("GATE-C FIRING SET PARTITION")
    L.append("=" * 78)
    L.append(f"  gate-C fired on {len(gate_c_names)} datasets: {gate_c_names}")
    L.append(f"    expected {sorted(EXPECTED_GATE_C)} -> "
             f"{'MATCH' if gate_c_ok else 'MISMATCH'}")
    L.append(f"  gate-D' overlap (n_train<5000), {len(fired_names)} datasets: "
             f"{fired_names}")
    L.append(f"    expected {sorted(EXPECTED_DPRIME)} -> "
             f"{'MATCH' if dprime_ok else 'MISMATCH'}")
    L.append(f"  gate-C-only (n_train>=5000), {len(gate_c_only_names)} datasets: "
             f"{gate_c_only_names}")
    L.append(f"    expected {sorted(EXPECTED_GATE_C_ONLY)} -> "
             f"{'MATCH' if gate_c_only_ok else 'MISMATCH'}")

    # ---- PER-DATASET dAB TABLE over the full gate-C set ----
    def fmt(x, w=11):
        return f"{x:>+{w}.6f}" if not math.isnan(x) else f"{'nan':>{w}}"

    def clean_flag(dAB, eps=1e-6):
        if math.isnan(dAB):
            return "nan"
        return "OK" if dAB >= -eps else "REGRESS"

    for split, tag in (("pub", "Public"), ("prv", "Private")):
        L.append("")
        L.append("=" * 78)
        L.append(f"PER-DATASET dA / dB / dAB on the GATE-C set ({tag})")
        L.append("=" * 78)
        L.append(f"{'dataset':<10} {'n_train':>8} {'subset':>10} {'base':>10} "
                 f"{'dA':>11} {'dB':>11} {'dAB':>11} {'dAB-flag':>9}")
        # gate-D' overlap first, then gate-C-only, each sorted by dataset
        ordered = ([r for r in gate_c_rows if r.get("gate_dprime")]
                   + [r for r in gate_c_rows if r.get("gate_c_only")])
        for r in ordered:
            subset = "gateD'" if r.get("gate_dprime") else "gateC-only"
            b = r.get(f"{BASE}_{split}")
            bstr = f"{b:>10.6f}" if isinstance(b, float) and not math.isnan(b) \
                else f"{'nan':>10}"
            dA = delta(r, A, split)
            dB = delta(r, B, split)
            dAB = delta(r, AB, split)
            L.append(f"{r['dataset']:<10} {r.get('n_train',''):>8} {subset:>10} "
                     f"{bstr} {fmt(dA)} {fmt(dB)} {fmt(dAB)} "
                     f"{clean_flag(dAB):>9}")

    # ---- CLEAN-WIN VERDICT for AB (no-regression) on gate-C, split by subset ----
    L.append("")
    L.append("=" * 78)
    L.append("CLEAN-WIN VERDICT for AB (dAB >= 0 == no regression vs base)")
    L.append("=" * 78)

    def regressions(subset_fn, split, eps=1e-6):
        out = []
        for r in rows:
            if not subset_fn(r):
                continue
            dAB = delta(r, AB, split)
            if math.isnan(dAB):
                continue
            if dAB < -eps:
                out.append((r["dataset"], dAB))
        return out

    verdict_lines = {}
    for split, tag in (("pub", "Public"), ("prv", "Private")):
        L.append("")
        L.append(f"[{tag}]")
        for label, fn, expected_set in (
                ("FULL gate-C", is_gatec, EXPECTED_GATE_C),
                ("gate-D' overlap", is_dprime, EXPECTED_DPRIME),
                ("gate-C-only", is_gatec_only, EXPECTED_GATE_C_ONLY)):
            regs = regressions(fn, split)
            wA, lA, tA = wlt(A, split, fn)
            wB, lB, tB = wlt(B, split, fn)
            wAB, lAB, tAB = wlt(AB, split, fn)
            n_sub = len([r for r in rows if fn(r)])
            if regs:
                reg_str = ", ".join(f"{nm}({d:+.6f})" for nm, d in regs)
                clean = f"REGRESSES on {len(regs)}/{n_sub}: {reg_str}"
            else:
                clean = f"CLEAN (no regression on {n_sub} datasets)"
            L.append(f"  {label:<16} AB: {clean}")
            L.append(f"  {label:<16}     AB W/L/T = {wAB}/{lAB}/{tAB}  "
                     f"(dA W/L/T {wA}/{lA}/{tA}, dB W/L/T {wB}/{lB}/{tB})")
            verdict_lines[(split, label)] = (len(regs), n_sub)

    # ---- MEAN Δ over full gate-C and gate-C-only ----
    L.append("")
    L.append("=" * 78)
    L.append("MEAN Δ vs base — full gate-C set and gate-C-only subset")
    L.append("=" * 78)
    means_full = {}
    means_conly = {}
    for split, tag in (("pub", "Public"), ("prv", "Private")):
        L.append("")
        L.append(f"[{tag}]")
        L.append(f"  FULL gate-C ({n_gate_c} datasets):")
        for arm in ARMS:
            m = mean_delta(arm, split, is_gatec)
            means_full[(arm, split)] = m
            L.append(f"    mean d{arm:<3} = {m:+.6f}")
        L.append(f"  gate-C-only ({len(gate_c_only_names)} datasets):")
        for arm in ARMS:
            m = mean_delta(arm, split, is_gatec_only)
            means_conly[(arm, split)] = m
            L.append(f"    mean d{arm:<3} = {m:+.6f}")
        L.append(f"  gate-D' overlap ({len(fired_names)} datasets):")
        for arm in ARMS:
            m = mean_delta(arm, split, is_dprime)
            L.append(f"    mean d{arm:<3} = {m:+.6f}")

    # ---- FULL PER-DATASET TABLE (all 16, context) ----
    for split, tag in (("pub", "Public"), ("prv", "Private")):
        L.append("")
        L.append(f"=== PER-DATASET dAUC ({tag}) — all 16 (context) ===")
        L.append(f"{'dataset':<10} {'gateC':>6} {'gateD':>6} {'Conly':>6} "
                 f"{'base':>10} {'dA':>11} {'dB':>11} {'dAB':>11}")
        for r in rows:
            b = r.get(f"{BASE}_{split}")
            bstr = f"{b:>10.6f}" if isinstance(b, float) and not math.isnan(b) \
                else f"{'nan':>10}"
            L.append(f"{r['dataset']:<10} {str(bool(r.get('gate_c'))):>6} "
                     f"{str(bool(r.get('gate_dprime'))):>6} "
                     f"{str(bool(r.get('gate_c_only'))):>6} {bstr} "
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
    L.append("=== REPRODUCTION CHECK 2 (B on ALL 16 vs round61 blend, tol=1e-6) ===")
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

    # ---- CONCLUSION ----
    # gate-C-only regressions decide whether the broad gate stays clean.
    conly_regs_pub = regressions(is_gatec_only, "pub")
    conly_regs_prv = regressions(is_gatec_only, "prv")
    dprime_regs_pub = regressions(is_dprime, "pub")
    dprime_regs_prv = regressions(is_dprime, "prv")
    gatec_clean = (not conly_regs_pub and not conly_regs_prv
                   and not dprime_regs_pub and not dprime_regs_prv)

    L.append("")
    L.append("=" * 78)
    L.append("CONCLUSION")
    L.append("=" * 78)
    if gatec_clean:
        conclusion = ("Gating AB on the BROAD gate-C stays CLEAN: dAB >= 0 on "
                      "every gate-C dataset (both splits), including the "
                      f"{len(gate_c_only_names)} gate-C-only datasets — a broad "
                      "gate-C AB ship would NOT regress.")
    else:
        reg_names = sorted(set(nm for nm, _ in
                               (conly_regs_pub + conly_regs_prv
                                + dprime_regs_pub + dprime_regs_prv)))
        conly_names_reg = sorted(set(nm for nm, _ in
                                     (conly_regs_pub + conly_regs_prv)))
        if conly_names_reg:
            conclusion = ("Gating AB on the BROAD gate-C REGRESSES: AB drops "
                          f"below base on gate-C-only dataset(s) {conly_names_reg} "
                          "— an AB ship MUST keep the narrow gate-D' restriction "
                          "(n_train < 5000).")
        else:
            conclusion = ("Gating AB on the BROAD gate-C shows regressions on "
                          f"{reg_names} (within gate-D' overlap, not gate-C-only) "
                          "— review before broadening; the narrow gate-D' "
                          "restriction is the safe ship gate.")
    L.append(f"  {conclusion}")

    # ---- CLEAN RUN marker ----
    clean_run = ((not exceptions) and all16_ok and repro_ok and repro_available
                 and (not skipped) and (not single_class_skips)
                 and dprime_ok and gate_c_ok and gate_c_only_ok)
    L.append("")
    L.append(f"CLEAN RUN = {'YES' if clean_run else 'NO'}  "
             f"[reproduction base_maxdev={max_dev_base:.2e}, "
             f"B_maxdev={max_dev_b:.2e}; "
             f"datasets_fit={len(present)}/16, gate_c={n_gate_c} "
             f"(overlap={len(fired_names)}, C-only={len(gate_c_only_names)}); "
             f"exceptions={len(exceptions)}, skipped={len(skipped)}, "
             f"single_class_skips={len(single_class_skips)}, "
             f"total_fits={total_fits}]")
    if not clean_run:
        why = []
        if exceptions:
            why.append(f"{len(exceptions)} exception(s)")
        if not all16_ok:
            why.append("not all 16 fit")
        if not repro_available:
            why.append("round61 anchor missing")
        if not repro_base_ok:
            why.append(f"base!=round61 (maxdev={max_dev_base:.2e})")
        if not repro_b_ok:
            why.append(f"B!=round61 blend (maxdev={max_dev_b:.2e})")
        if not dprime_ok:
            why.append("gate-D' set mismatch")
        if not gate_c_ok:
            why.append("gate-C set mismatch")
        if not gate_c_only_ok:
            why.append("gate-C-only set mismatch")
        if single_class_skips:
            why.append(f"{len(single_class_skips)} single-class skip(s)")
        L.append(f"  WHY NOT CLEAN: {'; '.join(why)}")
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
          f"GATEC={n_gate_c} CONLY={len(gate_c_only_names)} "
          f"CONLY_dAB_pub={means_conly[(AB,'pub')]:+.6f} "
          f"CONLY_dAB_prv={means_conly[(AB,'prv')]:+.6f} "
          f"GATEC_CLEAN={'YES' if gatec_clean else 'NO'}")
    print("HARNESS_DONE")


if __name__ == "__main__":
    main()
