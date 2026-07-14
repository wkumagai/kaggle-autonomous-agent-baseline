#!/usr/bin/env python
"""
bench_03 round42 — GATE C vs GATE D vs GATE G, K (seeds-averaged) SWEEP.
OFFLINE ONLY. No subprocess, no LLM, no Kaggle, no network. Calls sklearn
in-process only. Writes ONLY under this round42 directory; never touches
submissions/.

GOAL (improvement-log angle "f")
--------------------------------
"Does raising K interact with gate D / gate G differently than with gate C?"

Gate C (round40) only fires on the 12 categorical datasets (n_object_cols>0);
the 4 pure-numeric datasets train_04/10/11/16 are left at base. Gate D (round35)
ALSO admits the pure-numeric LARGE-n datasets train_04/10/11 (n>=4000) while still
excluding the small-n pure-numeric train_16. Gate G (round36) is a training-time
self-verification gate: fire iff the K-seed-averaged internal-holdout AUC exceeds
the seed-0 internal-holdout AUC (lift>0). round41 found seed-avg at K=5 helps
train_04/10/11 but poisons train_16.

The NOVEL region gate C never touches is the pure-numeric datasets
train_04/10/11. The open question: as K increases on gate D's and gate G's firing
sets, do the extra pure-numeric datasets keep improving CLEANLY (zero regression),
or does an interaction appear (regression, or accelerating gain)? And is there any
K where gate D or gate G cleanly BEATS gate C (zero regression, higher mean)?

  Swept axis:  K (seeds averaged) in {5, 8, 10}.
  Compared:    gate C, gate D, gate G — at each K.

base recipe reproduced (== shipped 08, git HEAD:submissions/08_ratio_tiered_msl)
-------------------------------------------------------------------------------
  features = [c for c in train.columns if c not in ("row_id","target")]
  cat_mask = [train[c].dtype == object for c in features]
  n=len(train); ratio = len(features)/n
  l2  = 1.0 if ratio >= 0.010 else 0.0
  msl = 70 if ratio>=0.030 else 50 if ratio>=0.015 else 20
  HistGradientBoostingClassifier(categorical_features=cat_mask, random_state=s,
      max_iter=300, early_stopping=True, l2_regularization=l2,
      min_samples_leaf=msl)   # validation_fraction NOT set -> sklearn default 0.1
  pred = predict_proba(test)[:, class==1]
BASE column = seed-0, validation_fraction UNSET (byte-identical to shipped 08 and
to round40/35/36's base). This is the ΔAUC reference for every gate/K.

SEED-AVG FOR ALL 16 DATASETS (unlike round40): every dataset — including the 4
pure-numeric ones — is fit for seeds 0..MAX_K-1 (validation_fraction UNSET), and
the K-average is available for K in {5,8,10}. A dataset only USES its K-average
where a gate fires; otherwise it uses the exact seed-0 base (delta 0). Because
every seed uses the true base-08 default holdout, gate-C K5 reproduces round40,
gate-D K5 reproduces round35, gate-G K5 reproduces round36 exactly.

GATES
-----
  gate C: fire iff n_object_cols > 0                       (K-independent mask; 12)
  gate D: fire iff NOT(n_object_cols==0 AND n_train<4000)  (K-independent mask; 15)
  gate G: fire iff internal-holdout lift(K) > 0            (K-DEPENDENT mask)

Gate G's internal-holdout lift (round36 logic, reused verbatim): fix ONE
deterministic stratified holdout (train_test_split, test_size=0.25,
random_state=0, stratify=y) carved from TRAINING rows; fit shipped-08 HGB with
model seeds 0..K-1 on the fit part, predict on that SAME holdout;
seed0_auc = AUC(holdout, seed-0 pred), avgK_auc = AUC(holdout, mean of K preds);
lift(K) = avgK_auc - seed0_auc. Fire iff lift(K) > 0. The firing set can SHIFT
with K — reported at each K as part of the interaction story. No test labels, no
n_train, no n_object_cols enter gate G.

EFFICIENCY: per dataset, fit seeds 0..MAX_K-1 on ALL training rows (test preds,
cached) AND seeds 0..MAX_K-1 on the fixed internal holdout (holdout preds,
cached). Total fits = 16 * (MAX_K + MAX_K) = 16 * 20 = 320. seed-0 test fit IS the
base column. Every K averages the first K cached seeds; every gate reuses the same
cached predictions.

REPRODUCTION anchors (assert tol<5e-6; recomputed here, NOT hardcode-trusted):
  gate-C K5 == round40 (+0.00363 Pub / +0.00316 Prv)
  gate-D K5 == round35 (+0.00473 Pub / +0.00453 Prv)
  gate-G K5 == round36 (+0.00426 Pub / +0.00419 Prv)
Full-precision references are re-derived from the prior rounds' results.csv (not
from 5-decimal printouts), so each check is a genuine cross-round reproduction. If
a reference differs, the recomputed value is reported (never fudged).

INVARIANT: for each gate and K, every NON-fired dataset is byte-identical to base
(applied delta exactly 0 on both splits). Asserted and reported.

Adoption: a (gate,K) is a CLEAN IMPROVEMENT over base-08 iff mean ΔAUC > 0 on BOTH
splits AND zero regressions on BOTH splits. To BEAT gate C at the same K it must
additionally exceed gate C's mean on BOTH splits with zero regressions.
"""
import os
import csv
import math
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

REPO = "/Users/kumacmini/kaggle-autonomous-agent-baseline-auto"
DATA_DIR = os.path.join(REPO, "data")
BENCH_DIR = os.path.join(REPO, "experiments", "bench_03")
OUT_DIR = os.path.join(BENCH_DIR, "round42_gateDG_ksweep")
STATS_CSV = os.path.join(BENCH_DIR, "dataset_stats.csv")
ROUND40_RESULTS = os.path.join(BENCH_DIR, "round40_gateC_ksweep", "results.csv")
ROUND35_RESULTS = os.path.join(BENCH_DIR, "round35_gate_purenumeric", "results.csv")
ROUND36_RESULTS = os.path.join(BENCH_DIR, "round36_mechanism_gate", "results.csv")

# shipped-08 base recipe constants (byte-identical to git HEAD:submissions/08).
L2_GATE_THRESHOLD = 0.010
GATED_L2 = 1.0
DEFAULT_MSL = 20
MSL_TIERS = [(0.030, 70), (0.015, 50)]
BASE_SEED = 0
N_DATASETS = 16

# gate D pure-numeric small-n cutoff (obj==0 AND n_train<PN_SMALLN_CAP -> excluded).
PN_SMALLN_CAP = 4000

# gate G internal-holdout config (training-data only; round36 verbatim).
HOLDOUT_FRAC = 0.25
SPLIT_SEED_G = 0
GATE_G_EPS = 0.0        # fire iff lift(K) > GATE_G_EPS

# sweep axis: K only. Every seed fit with validation_fraction UNSET (base-08).
KS = [5, 8, 10]
MAX_K = max(KS)

BASE = "base"
GATES = ["C", "D", "G"]  # C, D from stats; G from training-time holdout lift(K).

REPRO_TOL = 5e-6
# coarse context targets (printed, NOT asserted — assertion is vs full-precision
# references re-derived from the prior rounds' results.csv below).
REF_COARSE = {
    "C": (0.00363, 0.00316),   # round40 gate C, K5
    "D": (0.00473, 0.00453),   # round35 gate D, K5
    "G": (0.00426, 0.00419),   # round36 gate G, K5 (eps=0, 13 datasets)
}

# expected K-independent firing sets (verified at run time).
CATEGORICAL = {"train_01", "train_02", "train_03", "train_05", "train_06",
               "train_07", "train_08", "train_09", "train_12", "train_13",
               "train_14", "train_15"}
PURE_NUMERIC = {"train_04", "train_10", "train_11", "train_16"}
ALL_NAMES = {f"train_{i:02d}" for i in range(1, N_DATASETS + 1)}
EXPECTED_FIRE_C = set(CATEGORICAL)                  # 12
EXPECTED_FIRE_D = ALL_NAMES - {"train_16"}          # 15
# gate G at K5 is expected to exclude train_06, train_13, train_16 (round36).
EXPECTED_FIRE_G_K5 = ALL_NAMES - {"train_06", "train_13", "train_16"}   # 13

# the pure-numeric datasets are the interaction-focus region gate C never touches.
PN_LARGE = ["train_04", "train_10", "train_11"]     # obj=0, n>=4000
PN_SMALL = "train_16"                               # obj=0, n<4000 (sole regressor)
PN_FOCUS = PN_LARGE + [PN_SMALL]


def load_stats(path=STATS_CSV):
    stats = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            stats[row["name"]] = {
                "n_train": int(row["n_train"]),
                "n_object_cols": int(row["n_object_cols"]),
            }
    return stats


def _mean_cols(path, col_pub, col_prv):
    """Mean of two full-precision delta columns over rows of a prior results.csv,
    ignoring blank/nan cells. Returns (mean_pub, mean_prv) or None if missing."""
    if not os.path.exists(path):
        return None
    accp, accv = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            for col, acc in ((col_pub, accp), (col_prv, accv)):
                v = row.get(col, "")
                if v is None or v == "":
                    continue
                try:
                    fv = float(v)
                except ValueError:
                    continue
                if not math.isnan(fv):
                    acc.append(fv)
    if not accp or not accv:
        return None
    return sum(accp) / len(accp), sum(accv) / len(accv)


def ref_gateC_K5():
    """round40 gate C, K5: mean cand_K5_d_pub / cand_K5_d_prv over all datasets."""
    return _mean_cols(ROUND40_RESULTS, "cand_K5_d_pub", "cand_K5_d_prv")


def ref_gateD_K5():
    """round35 gate D, K5: mean cand_D_exceptpn_d_pub / _d_prv over all datasets."""
    return _mean_cols(ROUND35_RESULTS, "cand_D_exceptpn_d_pub",
                      "cand_D_exceptpn_d_prv")


def ref_gateG_K5():
    """round36 gate G, K5 (eps=0): mean over all datasets of (d_pub if lift>eps
    else 0), (d_prv if lift>eps else 0). Re-derived at full precision from
    round36's results.csv (lift, d_pub, d_prv columns)."""
    if not os.path.exists(ROUND36_RESULTS):
        return None
    dp, dv = [], []
    with open(ROUND36_RESULTS, newline="") as f:
        for row in csv.DictReader(f):
            try:
                lift = float(row["lift"])
                dpub = float(row["d_pub"])
                dprv = float(row["d_prv"])
            except (ValueError, KeyError, TypeError):
                continue
            fires = lift > GATE_G_EPS
            dp.append(dpub if fires else 0.0)
            dv.append(dprv if fires else 0.0)
    if not dp or not dv:
        return None
    return sum(dp) / len(dp), sum(dv) / len(dv)


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


def _pos_proba(clf, X):
    proba = clf.predict_proba(X)
    classes = list(clf.classes_)
    pos_idx = classes.index(1) if 1 in classes else 1
    return proba[:, pos_idx]


def fit_one_seed(train, test, features, cat_mask, l2, msl_val, seed):
    """Fit ONE shipped-08 HGB (validation_fraction UNSET) on ALL training rows;
    return the positive-class probability vector aligned to test row order
    (byte-identical to shipped 08 when seed==0)."""
    clf = HistGradientBoostingClassifier(
        categorical_features=cat_mask, random_state=seed, max_iter=300,
        early_stopping=True, l2_regularization=l2, min_samples_leaf=msl_val)
    clf.fit(train[features], train["target"])
    return _pos_proba(clf, test[features])


def fit_holdout(X_fit, y_fit, X_hold, cat_mask, l2, msl_val, seed):
    """Fit shipped-08 HGB (random_state=seed) on a TRAINING sub-split; return the
    positive-class probability vector on the held-out TRAINING rows."""
    clf = HistGradientBoostingClassifier(
        categorical_features=cat_mask, random_state=seed, max_iter=300,
        early_stopping=True, l2_regularization=l2, min_samples_leaf=msl_val)
    clf.fit(X_fit, y_fit)
    return _pos_proba(clf, X_hold)


def score_split(pred_map, sol):
    sol = sol.copy()
    sol["pred"] = sol["row_id"].map(pred_map)
    if sol["pred"].isna().any():
        raise ValueError(f"{int(sol['pred'].isna().sum())} row_ids unmatched")
    pub = sol[sol["Usage"] == "Public"]
    prv = sol[sol["Usage"] == "Private"]
    return (auc_or_nan(pub["target"], pub["pred"]),
            auc_or_nan(prv["target"], prv["pred"]))


def run_one(name, train_csv, test_csv, sol, stats):
    """Reproduce shipped-08 base + K-averaged test predictions (K in {5,8,10}) for
    EVERY dataset, plus the training-time internal-holdout lift(K) that gate G
    fires on. Returns (rec, n_fits).

    rec holds: base_pub/base_prv, and for each K: avg{K}_pub/avg{K}_prv,
    d{K}_pub/d{K}_prv (avg-base), lift{K}, and fires_G_{K} (lift{K}>eps). Gate C/D
    firing flags (K-independent) are read from stats."""
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)

    features = [c for c in train.columns if c not in ("row_id", "target")]
    cat_mask = [train[c].dtype == object for c in features]
    n = len(train)
    ratio = len(features) / n
    l2 = GATED_L2 if ratio >= L2_GATE_THRESHOLD else 0.0
    msl_val = msl_for_ratio(ratio)

    row_ids = test["row_id"].tolist()

    # --- test predictions: seeds 0..MAX_K-1 on ALL rows (cached) ---
    seed_vecs = [fit_one_seed(train, test, features, cat_mask, l2, msl_val, s)
                 for s in range(MAX_K)]
    n_fits = MAX_K
    base_vec = seed_vecs[BASE_SEED]
    base_map = dict(zip(row_ids, base_vec.tolist()))
    base_pub, base_prv = score_split(base_map, sol)

    # --- internal-holdout predictions: fixed split (seed 0), model seeds 0..MAX_K-1 ---
    y = train["target"].values
    X = train[features]
    Xf, Xh, yf, yh = train_test_split(
        X, y, test_size=HOLDOUT_FRAC, random_state=SPLIT_SEED_G, stratify=y)
    hold_vecs = [fit_holdout(Xf, yf, Xh, cat_mask, l2, msl_val, s)
                 for s in range(MAX_K)]
    n_fits += MAX_K
    seed0_hold_auc = auc_or_nan(yh, hold_vecs[BASE_SEED])

    st = stats[name]
    rec = {
        "dataset": name,
        "n_train": st["n_train"],
        "n_object_cols": st["n_object_cols"],
        "msl": msl_val,
        "base_pub": base_pub, "base_prv": base_prv,
        "seed0_hold_auc": seed0_hold_auc,
        "base_is_seed0": bool(np.array_equal(seed_vecs[BASE_SEED], base_vec)),
    }
    # gate C / gate D masks (K-independent, from stats).
    rec["fires_C"] = st["n_object_cols"] > 0
    rec["fires_D"] = not (st["n_object_cols"] == 0
                          and st["n_train"] < PN_SMALLN_CAP)

    for K in KS:
        avg_vec = np.mean(np.vstack(seed_vecs[:K]), axis=0)
        avg_map = dict(zip(row_ids, avg_vec.tolist()))
        avg_pub, avg_prv = score_split(avg_map, sol)
        rec[f"avg{K}_pub"] = avg_pub
        rec[f"avg{K}_prv"] = avg_prv
        rec[f"d{K}_pub"] = avg_pub - base_pub
        rec[f"d{K}_prv"] = avg_prv - base_prv
        # gate G firing signal at THIS K (internal-holdout averaging lift).
        avgK_hold = np.mean(np.vstack(hold_vecs[:K]), axis=0)
        avgK_hold_auc = auc_or_nan(yh, avgK_hold)
        liftK = avgK_hold_auc - seed0_hold_auc
        rec[f"lift{K}"] = liftK
        rec[f"fires_G_{K}"] = bool(liftK > GATE_G_EPS)

    return rec, n_fits


# ---- gate firing lookup: returns bool for (rec, gate, K) ----
def fires(rec, gate, K):
    if gate == "C":
        return bool(rec["fires_C"])
    if gate == "D":
        return bool(rec["fires_D"])
    if gate == "G":
        return bool(rec[f"fires_G_{K}"])
    raise ValueError(gate)


def applied_delta(rec, gate, K, split):
    """Delta vs base applied under a gate at K: (avg{K}-base) if fires, else 0.0."""
    if fires(rec, gate, K):
        return rec[f"d{K}_{split}"]
    return 0.0


def main():
    warnings.filterwarnings("ignore")
    os.makedirs(OUT_DIR, exist_ok=True)

    stats = load_stats()
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
        try:
            rec, n_fits = run_one(name, train_csv, test_csv, sol, stats)
            total_fits += n_fits
            rows.append(rec)
            gflags = "".join("1" if rec[f"fires_G_{K}"] else "0" for K in KS)
            print(f"[OK] {name} n_tr={rec['n_train']} obj={rec['n_object_cols']} "
                  f"C={int(rec['fires_C'])} D={int(rec['fires_D'])} "
                  f"G[{','.join(str(k) for k in KS)}]={gflags} "
                  f"fits={n_fits} base pub={rec['base_pub']:.6f} "
                  f"prv={rec['base_prv']:.6f} "
                  f"lift5={rec['lift5']:+.5f}")
        except Exception as e:
            exceptions.append((name, repr(e)))
            print(f"[ERROR] {name}: {e!r}")

    # ---- aggregation helpers (over precomputed per-dataset records) ----
    def mean_delta(gate, K, split):
        vals = [applied_delta(r, gate, K, split) for r in rows]
        return sum(vals) / len(vals) if vals else float("nan")

    def wlt(gate, K, split, eps=1e-6):
        w = l = t = 0
        for r in rows:
            dd = applied_delta(r, gate, K, split)
            if dd > eps:
                w += 1
            elif dd < -eps:
                l += 1
            else:
                t += 1
        return w, l, t

    def regressions(gate, K, split, eps=1e-6):
        return [(r["dataset"], applied_delta(r, gate, K, split)) for r in rows
                if applied_delta(r, gate, K, split) < -eps]

    def firing_set(gate, K):
        return {r["dataset"] for r in rows if fires(r, gate, K)}

    # ---- reproduction references (full precision, from prior results.csv) ----
    refs = {"C": ref_gateC_K5(), "D": ref_gateD_K5(), "G": ref_gateG_K5()}

    # ---- INVARIANT: non-firing datasets contribute delta exactly 0 (both splits),
    #      for every gate and K. ----
    invariant_violations = []
    for gate in GATES:
        for K in KS:
            for r in rows:
                if not fires(r, gate, K):
                    dp = applied_delta(r, gate, K, "pub")
                    dv = applied_delta(r, gate, K, "prv")
                    if dp != 0.0 or dv != 0.0:
                        invariant_violations.append((gate, K, r["dataset"], dp, dv))

    # ---- firing-set checks ----
    fireC = firing_set("C", KS[0])          # K-independent
    fireD = firing_set("D", KS[0])          # K-independent
    fireG = {K: firing_set("G", K) for K in KS}
    fireC_ok = (fireC == EXPECTED_FIRE_C)
    fireD_ok = (fireD == EXPECTED_FIRE_D)
    fireG_k5_ok = (fireG[KS[0]] == EXPECTED_FIRE_G_K5)
    base_seed0_ok = bool(rows) and all(r["base_is_seed0"] for r in rows)

    # ---- sweep metrics ----
    sweep = {}
    for gate in GATES:
        for K in KS:
            sweep[(gate, K)] = {
                "mp": mean_delta(gate, K, "pub"),
                "mv": mean_delta(gate, K, "prv"),
                "pub_wlt": wlt(gate, K, "pub"),
                "prv_wlt": wlt(gate, K, "prv"),
                "regs_pub": regressions(gate, K, "pub"),
                "regs_prv": regressions(gate, K, "prv"),
                "nfire": len(firing_set(gate, K)),
            }

    # ---- reproduction check (recomputed K5 vs full-precision references) ----
    repro = {}
    repro_ok = True
    repro_available = all(refs[g] is not None for g in GATES)
    for gate in GATES:
        mp, mv = sweep[(gate, 5)]["mp"], sweep[(gate, 5)]["mv"]
        ref = refs[gate]
        if ref is None:
            repro[gate] = {"mp": mp, "mv": mv, "rp": None, "rv": None,
                           "okp": False, "okv": False}
            repro_ok = False
            continue
        rp, rv = ref
        okp = abs(mp - rp) < REPRO_TOL
        okv = abs(mv - rv) < REPRO_TOL
        repro[gate] = {"mp": mp, "mv": mv, "rp": rp, "rv": rv,
                       "okp": okp, "okv": okv}
        repro_ok = repro_ok and okp and okv

    # =========================== results.csv ===========================
    csv_path = os.path.join(OUT_DIR, "results.csv")
    fieldnames = ["dataset", "n_train", "n_object_cols", "msl", "base_is_seed0",
                  "fires_C", "fires_D", "seed0_hold_auc", "base_pub", "base_prv"]
    for K in KS:
        fieldnames += [f"lift{K}", f"fires_G_{K}",
                       f"avg{K}_pub", f"d{K}_pub", f"avg{K}_prv", f"d{K}_prv"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            out = {k: r.get(k, "") for k in
                   ["dataset", "n_train", "n_object_cols", "msl", "base_is_seed0",
                    "fires_C", "fires_D", "seed0_hold_auc", "base_pub", "base_prv"]}
            for K in KS:
                out[f"lift{K}"] = r.get(f"lift{K}", "")
                out[f"fires_G_{K}"] = r.get(f"fires_G_{K}", "")
                out[f"avg{K}_pub"] = r.get(f"avg{K}_pub", "")
                out[f"d{K}_pub"] = r.get(f"d{K}_pub", "")
                out[f"avg{K}_prv"] = r.get(f"avg{K}_prv", "")
                out[f"d{K}_prv"] = r.get(f"d{K}_prv", "")
            w.writerow(out)

    # =============================== SUMMARY ===============================
    L = []
    L.append("=" * 78)
    L.append("bench_03 round42 — GATE C vs D vs G, K SWEEP {5,8,10}   [OFFLINE]")
    L.append("=" * 78)
    L.append("")
    L.append("SETUP:")
    L.append("  base == shipped 08 (git HEAD:submissions/08...): HGB, early_stopping,")
    L.append("    validation_fraction UNSET -> sklearn default 0.10. base column =")
    L.append("    seed-0, vf UNSET, for all 16 datasets.")
    L.append("  Seed-avg computed for ALL 16 datasets (incl. pure-numeric); a dataset")
    L.append("    USES its K-average only where a gate fires, else exact seed-0 base.")
    L.append("  gate C: fire iff n_object_cols>0            (12 categorical; K-indep)")
    L.append("  gate D: fire iff NOT(obj==0 AND n_train<4000)  (15, all but train_16)")
    L.append("  gate G: fire iff internal-holdout lift(K)>0    (K-DEPENDENT; round36)")
    L.append(f"  Sweep: K seeds averaged in {KS}; every seed vf UNSET (base-08).")

    # ---- SWEEP TABLE ----
    L.append("")
    L.append("=== SWEEP TABLE (each gate x K vs base == shipped 08) ===")
    L.append(f"{'gate':<5} {'K':>3} {'nFire':>5} {'meanDPub':>10} {'meanDPrv':>10} "
             f"{'Pub W/L/T':>12} {'Prv W/L/T':>12}")
    for gate in GATES:
        for K in KS:
            s = sweep[(gate, K)]
            wp, lp, tp = s["pub_wlt"]
            wv, lv, tv = s["prv_wlt"]
            tag = "  <- repro anchor" if K == 5 else ""
            L.append(f"{gate:<5} {K:>3} {s['nfire']:>5} {s['mp']:>+10.5f} "
                     f"{s['mv']:>+10.5f} {f'{wp}/{lp}/{tp}':>12} "
                     f"{f'{wv}/{lv}/{tv}':>12}{tag}")
        L.append("")

    # ---- REPRODUCTION ----
    L.append("=== REPRODUCTION CHECK (recomputed K5 here vs prior rounds, tol<5e-6) ===")
    if not repro_available:
        L.append("  one or more prior results.csv NOT found -> reproduction NOT")
        L.append("  fully anchored (FAIL).")
    anchor_src = {"C": "round40", "D": "round35", "G": "round36"}
    for gate in GATES:
        rr = repro[gate]
        if rr["rp"] is None:
            L.append(f"  gate {gate} K5: reference ({anchor_src[gate]}) MISSING -> FAIL")
            continue
        L.append(
            f"  gate {gate} K5 vs {anchor_src[gate]}: "
            f"Public {rr['mp']:+.6f} vs {rr['rp']:+.6f} "
            f"(|d|={abs(rr['mp']-rr['rp']):.2e}, {'YES' if rr['okp'] else 'NO'}); "
            f"Private {rr['mv']:+.6f} vs {rr['rv']:+.6f} "
            f"(|d|={abs(rr['mv']-rr['rv']):.2e}, {'YES' if rr['okv'] else 'NO'})")
    L.append(f"  coarse context (printed, not asserted): "
             f"C ~ +{REF_COARSE['C'][0]:.5f}/+{REF_COARSE['C'][1]:.5f}, "
             f"D ~ +{REF_COARSE['D'][0]:.5f}/+{REF_COARSE['D'][1]:.5f}, "
             f"G ~ +{REF_COARSE['G'][0]:.5f}/+{REF_COARSE['G'][1]:.5f}")
    L.append(f"  REPRODUCTION: {'PASS' if repro_ok else 'FAIL'}")

    # ---- FIRING SETS ----
    L.append("")
    L.append("=== FIRING SETS ===")
    L.append(f"  gate C ({len(fireC)}): {', '.join(sorted(fireC))}")
    L.append(f"    expected 12 categorical matched: {'YES' if fireC_ok else 'NO'}")
    L.append(f"  gate D ({len(fireD)}): {', '.join(sorted(fireD))}")
    L.append(f"    expected 15 (all but train_16) matched: {'YES' if fireD_ok else 'NO'}")
    L.append("  gate G (K-dependent):")
    for K in KS:
        fg = fireG[K]
        excl = sorted(ALL_NAMES - fg)
        L.append(f"    K{K} ({len(fg)}): excludes {', '.join(excl) if excl else '(none)'}")
    L.append(f"    gate G K5 == round36 (excl train_06/13/16): "
             f"{'YES' if fireG_k5_ok else 'NO'}")
    # gate G firing-set shift with K
    g_shift = any(fireG[K] != fireG[KS[0]] for K in KS)
    L.append(f"    gate G firing set SHIFTS with K: {'YES' if g_shift else 'NO'}")
    if g_shift:
        for K in KS:
            added = sorted(fireG[K] - fireG[KS[0]])
            dropped = sorted(fireG[KS[0]] - fireG[K])
            L.append(f"      K{K} vs K{KS[0]}: +[{', '.join(added) or '-'}] "
                     f"-[{', '.join(dropped) or '-'}]")

    # ---- INVARIANT ----
    L.append("")
    L.append("=== INVARIANT (non-firing datasets identical to base, delta 0) ===")
    if invariant_violations:
        L.append("  VIOLATED!")
        for gate, K, ds, dp, dv in invariant_violations:
            L.append(f"    gate {gate} K{K} / {ds}: pub d={dp:+.6g} prv d={dv:+.6g}")
    else:
        L.append("  OK: for every gate x K, each non-firing dataset contributes")
        L.append("  applied delta exactly 0 on BOTH splits (uses seed-0 base). PASS.")
    L.append(f"  base column == seed-0 on all {len(rows)} datasets: "
             f"{'YES' if base_seed0_ok else 'NO'}")

    # ---- INTERACTION FOCUS: pure-numeric datasets (gate C never touches) ----
    L.append("")
    L.append("=== INTERACTION FOCUS: pure-numeric train_04/10/11/16 across K ===")
    L.append("  (raw seed-avg delta vs base; gate C NEVER fires on these. gate D")
    L.append("   fires on 04/10/11, excludes 16. gate G fires iff lift(K)>0.)")
    hdr = (f"{'dataset':<10} {'nTr':>6} {'base_pub':>9} {'base_prv':>9}")
    for K in KS:
        hdr += f" {'dPub_K'+str(K):>10} {'dPrv_K'+str(K):>10} {'G_K'+str(K):>5}"
    L.append(hdr)
    rec_by_name = {r["dataset"]: r for r in rows}
    for nm in PN_FOCUS:
        r = rec_by_name.get(nm)
        if r is None:
            L.append(f"  {nm}: (missing)")
            continue
        line = (f"{nm:<10} {r['n_train']:>6} {r['base_pub']:>9.4f} "
                f"{r['base_prv']:>9.4f}")
        for K in KS:
            gfire = "fire" if r[f"fires_G_{K}"] else " -- "
            line += (f" {r[f'd{K}_pub']:>+10.5f} {r[f'd{K}_prv']:>+10.5f} "
                     f"{gfire:>5}")
        L.append(line)

    # per-K clean-ness of the pure-numeric LARGE-n admits (gate D's & G's extra region)
    L.append("")
    L.append("  Do train_04/10/11 keep improving CLEANLY as K rises? (raw seed-avg")
    L.append("  delta on each; a regression on either split at any K breaks 'clean')")
    pn_clean_all = True
    for nm in PN_LARGE:
        r = rec_by_name.get(nm)
        if r is None:
            pn_clean_all = False
            continue
        parts = []
        nm_clean = True
        for K in KS:
            dp, dv = r[f"d{K}_pub"], r[f"d{K}_prv"]
            reg = (dp < -1e-6) or (dv < -1e-6)
            gain = (dp > 1e-6) and (dv > 1e-6)
            if reg:
                nm_clean = False
            tag = "GAIN" if gain else ("REG" if reg else "mix")
            parts.append(f"K{K}:{dp:+.5f}/{dv:+.5f}[{tag}]")
        pn_clean_all = pn_clean_all and nm_clean
        L.append(f"    {nm:<10} " + "  ".join(parts)
                 + f"   -> {'CLEAN across K' if nm_clean else 'REGRESSES at some K'}")
    L.append(f"    => train_04/10/11 all clean (no regression) at every K: "
             f"{'YES' if pn_clean_all else 'NO'}")
    # train_16 (excluded by D and by G) — show what it WOULD do if admitted.
    r16 = rec_by_name.get(PN_SMALL)
    if r16 is not None:
        parts = []
        for K in KS:
            dp, dv = r16[f"d{K}_pub"], r16[f"d{K}_prv"]
            reg = (dp < -1e-6) or (dv < -1e-6)
            parts.append(f"K{K}:{dp:+.5f}/{dv:+.5f}[{'REG' if reg else 'ok'}]")
        L.append(f"    {PN_SMALL} (excluded by D & G) if admitted: "
                 + "  ".join(parts))

    # gate-D / gate-G cleanliness on their firing sets, per K
    L.append("")
    L.append("  Per-K cleanliness of gate D and gate G firing sets (zero regression):")
    for gate in ("D", "G"):
        for K in KS:
            s = sweep[(gate, K)]
            rp, rv = s["regs_pub"], s["regs_prv"]
            zero = (not rp) and (not rv)
            regstr = ""
            if not zero:
                allr = rp + rv
                regstr = " regs[" + ", ".join(
                    f"{n}({d:+.5f})" for n, d in allr) + "]"
            L.append(f"    gate {gate} K{K}: {'CLEAN (zero regression)' if zero else 'HAS REGRESSION'}"
                     + regstr)

    # ---- PER-DATASET ΔAUC across (gate,K) ----
    for split, tag in (("pub", "Public"), ("prv", "Private")):
        L.append("")
        L.append(f"=== PER-DATASET applied ΔAUC ({tag}) — gate x K "
                 f"(0 where the gate does not fire) ===")
        header = f"{'dataset':<10} {'obj':>4} {'base':>8}"
        for gate in GATES:
            for K in KS:
                header += f" {gate+'K'+str(K):>9}"
        L.append(header)
        for r in rows:
            line = (f"{r['dataset']:<10} {str(r.get('n_object_cols')):>4} "
                    f"{r[f'{BASE}_{split}']:>8.4f}")
            for gate in GATES:
                for K in KS:
                    line += f" {applied_delta(r, gate, K, split):>+9.5f}"
            L.append(line)

    # ---- ADOPTION ANALYSIS ----
    L.append("")
    L.append("=== ADOPTION ANALYSIS ===")
    L.append("Criterion A (clean vs base-08): mean D > 0 AND zero regressions on BOTH")
    L.append("  splits. Criterion B (beats gate C at same K): also strictly greater")
    L.append("  mean than gate C on BOTH splits with zero regressions.")
    L.append("")
    for K in KS:
        cmp, cmv = sweep[("C", K)]["mp"], sweep[("C", K)]["mv"]
        L.append(f"  --- K{K} (gate C ref: pub{cmp:+.5f} prv{cmv:+.5f}) ---")
        for gate in GATES:
            s = sweep[(gate, K)]
            mp, mv = s["mp"], s["mv"]
            rp, rv = s["regs_pub"], s["regs_prv"]
            zero_regs = (not rp) and (not rv)
            clean = (mp > 1e-9) and (mv > 1e-9) and zero_regs
            status = ["clean-vs-08" if clean else "NOT-clean-vs-08"]
            if gate == "C":
                status.append("== gate C baseline")
            else:
                beats_c = (mp > cmp + 1e-9) and (mv > cmv + 1e-9) and zero_regs
                status.append("beats-C(clean,both)" if beats_c
                              else "does-not-cleanly-beat-C")
            regstr = ""
            if not zero_regs:
                allr = rp + rv
                regstr = " regs[" + ", ".join(
                    f"{n}({dd:+.5f})" for n, dd in allr) + "]"
            L.append(f"    gate {gate} K{K}: pub{mp:+.5f} prv{mv:+.5f}  "
                     + "; ".join(status) + regstr)

    # ---- VERDICT (angle f) ----
    L.append("")
    L.append("=== VERDICT (angle f) ===")
    L.append("Q: Does raising K interact with gate D / gate G differently than with")
    L.append("   gate C? Do the extra pure-numeric datasets keep improving cleanly?")
    L.append("")

    # monotonicity of mean per gate
    def mono(gate, split):
        vals = [sweep[(gate, K)][("mp" if split == "pub" else "mv")] for K in KS]
        return all(vals[i + 1] >= vals[i] - 1e-9 for i in range(len(vals) - 1)), vals

    for gate in GATES:
        mp_ok, mps = mono(gate, "pub")
        mv_ok, mvs = mono(gate, "prv")
        L.append(f"  gate {gate}: meanPub over K{KS} = "
                 + ", ".join(f"{v:+.5f}" for v in mps)
                 + f"  (monotone up: {'YES' if mp_ok else 'NO'})")
        L.append(f"  gate {gate}: meanPrv over K{KS} = "
                 + ", ".join(f"{v:+.5f}" for v in mvs)
                 + f"  (monotone up: {'YES' if mv_ok else 'NO'})")

    L.append("")
    L.append(f"  Pure-numeric train_04/10/11 clean (no regression) at every K: "
             f"{'YES' if pn_clean_all else 'NO'}.")
    # does any K let gate D or gate G cleanly beat gate C?
    beat_events = []
    for gate in ("D", "G"):
        for K in KS:
            s = sweep[(gate, K)]
            cmp, cmv = sweep[("C", K)]["mp"], sweep[("C", K)]["mv"]
            zero = (not s["regs_pub"]) and (not s["regs_prv"])
            if zero and (s["mp"] > cmp + 1e-9) and (s["mv"] > cmv + 1e-9):
                beat_events.append((gate, K, s["mp"], s["mv"], cmp, cmv))
    if beat_events:
        L.append("  There IS a K where gate D or gate G CLEANLY beats gate C "
                 "(zero regression, higher mean on both splits):")
        for gate, K, mp, mv, cmp, cmv in beat_events:
            L.append(f"    gate {gate} K{K}: pub{mp:+.5f}(vs C {cmp:+.5f}) "
                     f"prv{mv:+.5f}(vs C {cmv:+.5f})")
    else:
        L.append("  NO K lets gate D or gate G cleanly beat gate C (zero regression "
                 "+ higher mean on both). See per-K regressions above.")

    # interaction characterization for D and G vs C
    L.append("")
    L.append("  INTERACTION: gate C fires only on the 12 categorical datasets, so its")
    L.append("  mean is a pure categorical-region effect. gate D adds pure-numeric")
    L.append("  large-n (train_04/10/11); gate G adds/removes datasets by internal-")
    L.append("  holdout lift(K). The rows above show, per K, whether admitting the")
    L.append("  pure-numeric region accelerates the gain, stays flat, or regresses.")
    # note whether gate G's set shifts (the K-interaction unique to G)
    L.append(f"  gate G's firing set shifts with K: {'YES' if g_shift else 'NO'} "
             f"(gate C and gate D masks are K-independent by construction).")

    # ---- CLEAN RUN marker ----
    clean_run = ((not exceptions) and (not skipped) and (not invariant_violations)
                 and fireC_ok and fireD_ok and fireG_k5_ok and base_seed0_ok
                 and repro_ok and repro_available and len(rows) == N_DATASETS)
    L.append("")
    L.append(f"CLEAN RUN={'YES' if clean_run else 'NO'} "
             f"(total_fits={total_fits}, datasets={len(rows)}/{N_DATASETS}, "
             f"exceptions={len(exceptions)}, skipped={len(skipped)}, "
             f"invariant_violations={len(invariant_violations)}, "
             f"fireC_match={'YES' if fireC_ok else 'NO'}, "
             f"fireD_match={'YES' if fireD_ok else 'NO'}, "
             f"fireG_K5_match={'YES' if fireG_k5_ok else 'NO'}, "
             f"reproduction={'YES' if repro_ok else 'NO'}, "
             f"base_eq_seed0={'YES' if base_seed0_ok else 'NO'})")
    for name, msg in exceptions:
        L.append(f"  EXC {name}: {msg}")

    summary = "\n".join(L)
    print("\n" + summary)
    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
        f.write(summary + "\n")
    print(f"\n[WROTE] {csv_path}")
    print(f"[WROTE] {os.path.join(OUT_DIR, 'summary.txt')}")
    print("HARNESS_DONE")


if __name__ == "__main__":
    main()
