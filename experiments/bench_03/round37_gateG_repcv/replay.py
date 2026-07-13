#!/usr/bin/env python
"""
bench_03 round37 — REPEATED-CV DENOISED FIRING-GATE (angle "g"): does averaging
round36's single-holdout gate-G LIFT estimate over R REPEATED stratified holdout
splits DENOISE it enough to (a) keep the sole regressor train_16 excluded while
(b) recovering train_13 (a REAL +0.00696 test gainer that single-split gate G
wrongly DROPS because its 125-row holdout AUC is the noisiest of all 16)?
OFFLINE ONLY. No subprocess, no LLM, no Kaggle, no network. Calls sklearn
in-process only.

Adapted from experiments/bench_03/round36_mechanism_gate/replay.py. ALL of
round36's machinery is reused VERBATIM: dataset loading, the shipped-08 base
config reproduction, Public/Private AUC scoring joined on row_id to solution.csv,
K=5 seed-averaging (K FIXED at 5) of the TEST predictions, the byte-identical-on-
non-firing INVARIANT, the reference gates C and D (predicates over
n_train/n_object_cols) as reproduction anchors, the CLEAN RUN marker, and the
summary/csv machinery. The base recipe is UNCHANGED (== shipped 08): seed-0 test
pred is byte-identical to shipped 08.

WHAT ROUND37 ADDS vs round36 -----------------------------------------------
round36 gate G fired seed-avg iff `lift = avg_holdout_auc - seed0_holdout_auc`
measured on ONE fixed stratified 25% holdout (split seed 0) was >= eps. For tiny
data that single-holdout lift is extremely noisy: train_16 (poison) lift =
-0.00179 but train_13 (a REAL +0.00696 test gainer) lift = -0.01818 — the LOWEST
of all 16 — purely because n=500 makes its 125-row holdout AUC estimate noisy. So
gate G at eps=0 wrongly DROPS train_13.

  gate G_rep (denoise the lift with repeated CV):
    For repeat r in 0..R_MAX-1 (R_MAX=10): make a stratified train/holdout split
    train_test_split(..., test_size=0.25, random_state=r, stratify=y); fit
    shipped-08 HGB with model random_state s for s in 0..K-1 (K=5) on the fit
    part; predict on THAT repeat's holdout. Compute
      lift_r = auc(mean of the K holdout preds) - auc(seed0 holdout pred).
    Store lift_0..lift_9 per dataset. The denoised signal for a given R is
      lift_R = mean(lift_0 .. lift_{R-1}).
    Gate G_rep(R, eps) fires seed-avg iff lift_R >= eps.

  CACHING: the 10 repeats x K=5 seeds = 50 holdout fits per dataset are computed
  ONCE; every R is derived by SLICING the first R stored per-repeat lifts (NO
  refit per R). Fit budget per dataset = 5 test fits + 50 holdout fits = 55;
  x16 datasets = 880 fits total. Fully deterministic (fixed seeds only).

  REPRODUCTION ANCHOR: repeat r=0 uses split seed 0 and model seeds 0..K-1 — the
  EXACT configuration of round36's single-split gate G. Hence lift_R at R=1 ==
  round36's gate-G lift for every dataset, so G_rep(R=1, eps=0) MUST reproduce
  round36 gate G exactly: fires 13, mean Public +0.00426 / Private +0.00419,
  W/L/T 13/0/3, excludes train_16, drops train_13 & train_06. This is asserted
  (abort-loud -> CLEAN RUN=NO) within 5e-6.

SWEEP: R in {1, 3, 5, 10}. For each R, sweep eps over the observed lift_R values
plus 0.0 (same distinct-threshold grid style as round36 gate G), and always
evaluate the principled eps=0 point explicitly.

Base recipe reproduced (== shipped 08), IDENTICAL to round36/35/34:
  - load train.csv, test.csv with pandas
  - features = [c for c in train.columns if c not in ("row_id","target")]
  - cat_mask  = [train[c].dtype == object for c in features]
  - n = len(train); n_feat = len(features); ratio = n_feat / n
  - l2  = 1.0 if ratio >= 0.010 else 0.0
  - msl = 70 if ratio >= 0.030 else 50 if ratio >= 0.015 else 20
  - HistGradientBoostingClassifier(categorical_features=cat_mask, random_state=s,
        max_iter=300, early_stopping=True, l2_regularization=l2,
        min_samples_leaf=msl)
  - base = seed-0 test prediction (byte-identical to shipped 08).
Score: ROC AUC on Public rows and Private rows separately, joined by row_id to
solution.csv, over all 16 datasets.

INVARIANT: for every candidate gate, any dataset that does NOT fire under it
reuses the EXACT seed-0 test array, so its Public/Private delta vs base is exactly
0 — checked explicitly per gate.

Adoption criterion (unchanged): a gate is a CLEAN IMPROVEMENT over base(08) iff
its mean delta is positive on BOTH splits AND there are ZERO regressions on BOTH
splits (train_16 must NOT fire).

References this run must reproduce (sanity anchors):
  gate C nocap    == round34/35: mean Public +0.00363 / Private +0.00316 (fires 12).
  gate D exceptpn == round35    : mean Public +0.00473 / Private +0.00453 (fires 15).
  gate G_rep R=1 @ eps=0 == round36 gate G: Public +0.00426 / Private +0.00419 (fires 13).
"""
import os
import csv
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

REPO = "/Users/kumacmini/kaggle-autonomous-agent-baseline-auto"
DATA_DIR = os.path.join(REPO, "data")
BENCH_DIR = os.path.join(REPO, "experiments", "bench_03")
OUT_DIR = os.path.join(BENCH_DIR, "round37_gateG_repcv")
STATS_CSV = os.path.join(BENCH_DIR, "dataset_stats.csv")

L2_GATE_THRESHOLD = 0.010   # shipped 08 feature-to-row-ratio gate for l2 (FIXED)
GATED_L2 = 1.0              # l2 applied when the l2-gate fires (FIXED)
DEFAULT_MSL = 20            # sklearn HGB default, used when no msl tier is cleared.
BASE_SEED = 0              # shipped-08 fixed random_state.
MSL_TIERS = [(0.030, 70), (0.015, 50)]   # 08 tiered min_samples_leaf (desc).

# gate D pure-numeric small-n cutoff (reference gate only).
PN_SMALLN_CAP = 4000

# K is FIXED at 5 (round30 established K=5 as the knee; no K sweep here).
K = 5
SEEDS = list(range(K))     # [0,1,2,3,4]

# Repeated-CV mechanism config (training-data only). Stratified on the ~0.5 target.
HOLDOUT_FRAC = 0.25        # external holdout carved from TRAINING rows.
R_MAX = 10                 # number of repeated holdout splits cached per dataset.
R_LIST = [1, 3, 5, 10]     # denoise depths swept in the report.

BASE = "base"
N_DATASETS = 16

# reference anchors this run must reproduce.
REF = {
    "C_nocap":    (0.00363, 0.00316),   # round34/35 nocap
    "D_exceptpn": (0.00473, 0.00453),   # round35 exceptpn
}
# round36 single-split gate G @ eps=0 anchor (must == G_rep R=1 @ eps=0).
REF_G_R1 = (0.00426, 0.00419)
REF_G_R1_NFIRE = 13
REF_G_R1_DROPS = {"train_06", "train_13", "train_16"}

# datasets of special interest.
GAINERS_OF_INTEREST = ["train_02", "train_04", "train_10", "train_11"]
SOLE_REGRESSOR = "train_16"
FOCUS = ["train_13", "train_16"]   # the two datasets round37 is about.


# ---- reference gate predicates over (n_train, n_object_cols) ----
def gate_C_fire(n_train, n_obj):
    """nocap (== round34): fire iff at least one object (categorical) column."""
    return n_obj > 0


def gate_D_fire(n_train, n_obj):
    """except-pure-numeric-small-n (== round35): fire on everything EXCEPT a
    pure-numeric dataset with few training rows (obj==0 AND n_train<cap)."""
    return not (n_obj == 0 and n_train < PN_SMALLN_CAP)


def load_stats(path=STATS_CSV):
    """Return {name -> {"n_train": int, "n_object_cols": int}}."""
    stats = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            stats[row["name"]] = {
                "n_train": int(row["n_train"]),
                "n_object_cols": int(row["n_object_cols"]),
            }
    return stats


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
    """Positive-class (label 1) probability vector for X."""
    proba = clf.predict_proba(X)
    classes = list(clf.classes_)
    pos_idx = classes.index(1) if 1 in classes else 1
    return proba[:, pos_idx]


def fit_one_seed(train, test, features, cat_mask, l2, msl_val, seed):
    """Fit ONE shipped-08 HGB (random_state=seed) on ALL training rows; return the
    positive-class probability vector aligned to test row order (byte-identical to
    shipped 08 when seed==0)."""
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


def repeated_lift_signals(train, features, cat_mask, l2, msl_val):
    """Compute the repeated-CV denoised gate-G_rep signal from the SAME shipped-08
    HGB config. Uses ONLY training rows (holdouts carved from train). Returns
    (dict_of_signals, n_fits).

      For repeat r in 0..R_MAX-1: stratified 25% holdout (random_state=r), fit K
      model seeds 0..K-1 on the fit part, predict on that repeat's holdout, and
      store lift_r = auc(mean of K preds) - auc(seed0 pred). The 10x5=50 holdout
      fits are done ONCE here; the denoised lift_R for any R is derived downstream
      by mean(lift_0..lift_{R-1}) — no refit per R.

      Repeat r=0 (split seed 0, model seeds 0..K-1) is byte-identical to round36's
      single-split gate G, so lift_R at R=1 reproduces round36's gate-G lift.
    """
    y = train["target"].values
    X = train[features]
    per_repeat = []   # lift_r for r=0..R_MAX-1
    for r in range(R_MAX):
        Xf, Xh, yf, yh = train_test_split(
            X, y, test_size=HOLDOUT_FRAC, random_state=r, stratify=y)
        preds = [fit_holdout(Xf, yf, Xh, cat_mask, l2, msl_val, s) for s in SEEDS]
        seed0_auc = auc_or_nan(yh, preds[0])
        avg_auc = auc_or_nan(yh, np.mean(np.vstack(preds), axis=0))
        per_repeat.append(avg_auc - seed0_auc)

    sig = {"lift_repeats": per_repeat}
    # denoised lift_R = mean of first R per-repeat lifts (sliced, not refit).
    for R in R_LIST:
        sig[f"liftR{R}"] = float(np.mean(per_repeat[:R]))
    return sig, R_MAX * K   # 50 fits


def score_split(pred_map, sol):
    """(public_auc, private_auc) for a row_id->prob map vs solution df."""
    sol = sol.copy()
    sol["pred"] = sol["row_id"].map(pred_map)
    if sol["pred"].isna().any():
        raise ValueError(
            f"{int(sol['pred'].isna().sum())} solution row_ids had no prediction")
    pub = sol[sol["Usage"] == "Public"]
    prv = sol[sol["Usage"] == "Private"]
    return auc_or_nan(pub["target"], pub["pred"]), auc_or_nan(prv["target"], prv["pred"])


def run_one(name, train_csv, test_csv, sol, stats):
    """For one dataset: reproduce shipped-08 base (seed-0 test pred) + the K=5
    averaged test pred, score both splits, and compute the repeated-CV lift
    signals. Returns a per-dataset record dict + n_fits."""
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)

    features = [c for c in train.columns if c not in ("row_id", "target")]
    cat_mask = [train[c].dtype == object for c in features]
    n = len(train)
    ratio = len(features) / n

    l2_fired = ratio >= L2_GATE_THRESHOLD
    l2 = GATED_L2 if l2_fired else 0.0
    msl_val = msl_for_ratio(ratio)

    row_ids = test["row_id"].tolist()

    # --- K=5 test predictions (cached once): base = seed 0, avg = mean(0..4) ---
    seed_preds = [
        fit_one_seed(train, test, features, cat_mask, l2, msl_val, s)
        for s in SEEDS
    ]
    n_fits = K
    base_map = dict(zip(row_ids, seed_preds[BASE_SEED].tolist()))
    avg_map = dict(zip(row_ids, np.mean(np.vstack(seed_preds), axis=0).tolist()))
    base_pub, base_prv = score_split(base_map, sol)
    avg_pub, avg_prv = score_split(avg_map, sol)

    # --- repeated-CV lift signals (training-only, 50 holdout fits) ---
    sig, mech_fits = repeated_lift_signals(train, features, cat_mask, l2, msl_val)
    n_fits += mech_fits

    st = stats[name]
    rec = {
        "dataset": name,
        "n_train": st["n_train"],
        "n_object_cols": st["n_object_cols"],
        "l2_fired": l2_fired,
        "msl": msl_val,
        "base_pub": base_pub, "base_prv": base_prv,
        "avg_pub": avg_pub, "avg_prv": avg_prv,
        # delta IF this dataset fires (avg vs base); 0 applied when not firing.
        "d_pub": avg_pub - base_pub,
        "d_prv": avg_prv - base_prv,
    }
    rec.update(sig)
    # ground-truth: does K=5 averaging TRULY help on the real test splits?
    eps = 1e-6
    if rec["d_pub"] > eps and rec["d_prv"] > eps:
        rec["test_gain"] = "GAIN"
    elif rec["d_pub"] < -eps or rec["d_prv"] < -eps:
        rec["test_gain"] = "REG"
    else:
        rec["test_gain"] = "MIX"
    return rec, n_fits


# ---------- gate evaluation over precomputed per-dataset records ----------
def applied_delta(rec, split, fires):
    """Delta vs base applied under a gate: (avg-base) if fires else EXACTLY 0."""
    return rec[f"d_{split}"] if fires else 0.0


def eval_gate(rows, fire_of):
    """fire_of: name -> bool. Return metrics dict computed from precomputed
    per-dataset base/avg scores. Non-firing datasets contribute EXACTLY 0."""
    eps = 1e-6
    m = {"fire": [r["dataset"] for r in rows if fire_of[r["dataset"]]]}
    for split in ("pub", "prv"):
        dvals = [applied_delta(r, split, fire_of[r["dataset"]]) for r in rows]
        w = sum(1 for d in dvals if d > eps)
        l = sum(1 for d in dvals if d < -eps)
        t = len(dvals) - w - l
        m[f"mean_{split}"] = sum(dvals) / len(dvals)
        m[f"wlt_{split}"] = (w, l, t)
        m[f"reg_{split}"] = [
            (r["dataset"], applied_delta(r, split, fire_of[r["dataset"]]))
            for r in rows
            if applied_delta(r, split, fire_of[r["dataset"]]) < -eps
        ]
    m["zero_regs"] = (not m["reg_pub"]) and (not m["reg_prv"])
    m["clean"] = (m["mean_pub"] > 1e-9 and m["mean_prv"] > 1e-9 and m["zero_regs"])
    m["excludes_16"] = SOLE_REGRESSOR not in set(m["fire"])
    m["fires_13"] = "train_13" in set(m["fire"])
    m["gainers_excluded"] = [
        g for g in GAINERS_OF_INTEREST if g not in set(m["fire"])
    ]
    return m


def fire_map_from_signal(rows, signal_key, thr):
    """fire iff rec[signal_key] >= thr."""
    return {r["dataset"]: (r[signal_key] >= thr) for r in rows}


def eps_grid_for(rows, signal_key):
    """Every distinct '>= eps' firing set for a lift signal: use each observed
    value as a threshold, plus 0.0 (principled point), plus one above the max
    (fires none). Deduplicated, ascending."""
    vals = [r[signal_key] for r in rows]
    grid = sorted(set([round(v, 12) for v in vals] + [0.0]))
    grid.append(round(max(vals) + 1.0, 12))   # fire-none sentinel
    return sorted(set(grid))


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
            print(f"[SKIP] {name}: missing train/test/solution")
            skipped.append(name)
            continue
        sol = pd.read_csv(sol_csv)
        try:
            rec, n_fits = run_one(name, train_csv, test_csv, sol, stats)
            total_fits += n_fits
            rows.append(rec)
            print(f"[OK] {name} (n={rec['n_train']}, obj={rec['n_object_cols']}, "
                  f"msl={rec['msl']}, fits={n_fits}): base pub={rec['base_pub']:.6f} "
                  f"prv={rec['base_prv']:.6f} | avg pub={rec['avg_pub']:.6f} "
                  f"prv={rec['avg_prv']:.6f} | liftR1={rec['liftR1']:+.5f} "
                  f"liftR10={rec['liftR10']:+.5f} test={rec['test_gain']}")
        except Exception as e:  # capture but keep going — CLEAN-RUN diagnostic
            exceptions.append((name, repr(e)))
            print(f"[ERROR] {name}: {e!r}")

    # ---- reference gates C, D (from stats; anchors only) ----
    ref_fire = {"C_nocap": {}, "D_exceptpn": {}}
    for r in rows:
        ref_fire["C_nocap"][r["dataset"]] = bool(
            gate_C_fire(r["n_train"], r["n_object_cols"]))
        ref_fire["D_exceptpn"][r["dataset"]] = bool(
            gate_D_fire(r["n_train"], r["n_object_cols"]))
    mC = eval_gate(rows, ref_fire["C_nocap"])
    mD = eval_gate(rows, ref_fire["D_exceptpn"])
    D_fire_set = set(mD["fire"])   # gate D's ideal target firing set (all but 16)

    # ---- gate G_rep sweep: for each R sweep eps over observed lift_R + 0.0 ----
    Grep = {}          # R -> list of metric dicts (full eps sweep)
    Grep_eps0 = {}     # R -> metric dict at the principled eps=0 point
    for R in R_LIST:
        key = f"liftR{R}"
        results = []
        for eps in eps_grid_for(rows, key):
            fm = fire_map_from_signal(rows, key, eps)
            m = eval_gate(rows, fm)
            m["thr"] = eps
            m["R"] = R
            m["matches_D"] = (set(m["fire"]) == D_fire_set)
            results.append(m)
        Grep[R] = results
        # eps=0 exact point.
        fm0 = fire_map_from_signal(rows, key, 0.0)
        m0 = eval_gate(rows, fm0)
        m0["thr"] = 0.0
        m0["R"] = R
        m0["matches_D"] = (set(m0["fire"]) == D_fire_set)
        Grep_eps0[R] = m0

    # ---- REPRODUCTION ANCHOR: G_rep R=1 @ eps=0 == round36 gate G ----------
    a = Grep_eps0[1]
    a_drops = {r["dataset"] for r in rows} - set(a["fire"])
    anchor_ok = (
        len(a["fire"]) == REF_G_R1_NFIRE
        and abs(a["mean_pub"] - REF_G_R1[0]) < 5e-6
        and abs(a["mean_prv"] - REF_G_R1[1]) < 5e-6
        and a["wlt_pub"] == (13, 0, 3)
        and a["wlt_prv"] == (13, 0, 3)
        and a["excludes_16"]
        and a_drops == REF_G_R1_DROPS
    )

    # ---- best CLEAN gate over the whole (R, eps) grid ----------------------
    # CLEAN = mean delta > 0 on BOTH splits AND zero regressions on BOTH splits
    # (=> train_16 excluded). Among clean gates prefer highest (pub+prv), then
    # matches_D, then larger firing set, then smaller R (simpler), then eps==0.
    clean_cands = []
    for R in R_LIST:
        for m in Grep[R]:
            if m["clean"]:
                clean_cands.append(m)
    if clean_cands:
        best_clean = max(
            clean_cands,
            key=lambda m: (m["mean_pub"] + m["mean_prv"], m["matches_D"],
                           len(m["fire"]), -m["R"], m["thr"] == 0.0))
    else:
        best_clean = None

    # ---- INVARIANT: non-firing datasets identical to base (delta exactly 0) --
    invariant_violations = []
    for label, fm in (("C_nocap", ref_fire["C_nocap"]),
                      ("D_exceptpn", ref_fire["D_exceptpn"])):
        for r in rows:
            if not fm[r["dataset"]]:
                dp = applied_delta(r, "pub", False)
                dv = applied_delta(r, "prv", False)
                if dp != 0.0 or dv != 0.0:
                    invariant_violations.append((label, r["dataset"], dp, dv))
    for R in R_LIST:
        for m in Grep[R]:
            fireset = set(m["fire"])
            for r in rows:
                if r["dataset"] not in fireset:
                    if applied_delta(r, "pub", False) != 0.0 or \
                       applied_delta(r, "prv", False) != 0.0:
                        invariant_violations.append(
                            (f"Grep_R{R}@{m['thr']:.5f}", r["dataset"], 0.0, 0.0))

    # ---- reproduction check (C, D vs references) ----
    repro_ok = True
    repro_lines = []
    for lbl, (rp, rv) in REF.items():
        m = mC if lbl == "C_nocap" else mD
        ok_p = abs(m["mean_pub"] - rp) < 5e-6
        ok_v = abs(m["mean_prv"] - rv) < 5e-6
        repro_ok = repro_ok and ok_p and ok_v
        repro_lines.append(
            f"gate {lbl:<11}: Public {m['mean_pub']:+.5f} (ref +{rp:.5f}, "
            f"{'YES' if ok_p else 'NO'}); Private {m['mean_prv']:+.5f} "
            f"(ref +{rv:.5f}, {'YES' if ok_v else 'NO'})")
    # add G_rep R=1 anchor to the reproduction gate.
    repro_lines.append(
        f"gate G_rep R=1  : Public {a['mean_pub']:+.5f} (ref +{REF_G_R1[0]:.5f}), "
        f"Private {a['mean_prv']:+.5f} (ref +{REF_G_R1[1]:.5f}), nFire={len(a['fire'])} "
        f"(ref {REF_G_R1_NFIRE}), drops={sorted(a_drops)} "
        f"(ref {sorted(REF_G_R1_DROPS)}) -> {'YES' if anchor_ok else 'NO'}")
    repro_ok = repro_ok and anchor_ok

    # per-dataset ranks (ascending lift) at R=1 and R=10.
    order_R1 = [r["dataset"] for r in sorted(rows, key=lambda r: r["liftR1"])]
    order_R10 = [r["dataset"] for r in sorted(rows, key=lambda r: r["liftR10"])]
    rank_R1 = {d: order_R1.index(d) + 1 for d in order_R1}
    rank_R10 = {d: order_R10.index(d) + 1 for d in order_R10}

    # ---------------------------- build summary ----------------------------
    S = []

    def line(x=""):
        S.append(x)

    # (0) per-dataset lift_R table (KEY QUESTION 1) ------------------------
    line("=== KEY QUESTION 1: DENOISED gate-G_rep LIFT_R PER DATASET ===")
    line("  lift_R = mean over R repeated stratified-holdout splits of "
         "(avg_holdout_auc - seed0_holdout_auc).")
    line("  rk1/rk10 = rank of this dataset's lift ASCENDING at R=1 / R=10 "
         "(1 = lowest / most-excluded).")
    line("  test_gain = does K=5 averaging TRULY help on the real test splits "
         "(ground truth).  * = focus dataset.")
    hdr = (f"{'dataset':<10} {'nTr':>6} {'obj':>4} {'liftR1':>9} {'liftR3':>9} "
           f"{'liftR5':>9} {'liftR10':>9} {'rk1':>4} {'rk10':>4} "
           f"{'d_pub':>9} {'d_prv':>9} {'truth':>6}")
    line(hdr)
    for r in rows:
        star = " *" if r["dataset"] in FOCUS else ""
        line(f"{r['dataset']:<10} {r['n_train']:>6} {r['n_object_cols']:>4} "
             f"{r['liftR1']:>+9.5f} {r['liftR3']:>+9.5f} {r['liftR5']:>+9.5f} "
             f"{r['liftR10']:>+9.5f} {rank_R1[r['dataset']]:>4} "
             f"{rank_R10[r['dataset']]:>4} {r['d_pub']:>+9.5f} "
             f"{r['d_prv']:>+9.5f} {r['test_gain']:>6}{star}")
    line("")
    for f in FOCUS:
        fr = next(r for r in rows if r["dataset"] == f)
        line(f"  FOCUS {f}: liftR1={fr['liftR1']:+.5f} liftR3={fr['liftR3']:+.5f} "
             f"liftR5={fr['liftR5']:+.5f} liftR10={fr['liftR10']:+.5f} "
             f"| rank {rank_R1[f]}->{rank_R10[f]} (R1->R10) "
             f"| test_gain={fr['test_gain']} "
             f"(d_pub={fr['d_pub']:+.5f}, d_prv={fr['d_prv']:+.5f})")
    t13 = next(r for r in rows if r["dataset"] == "train_13")
    line(f"  Does train_13's denoised lift rise toward/above 0 as R grows? "
         f"liftR1={t13['liftR1']:+.5f} -> liftR10={t13['liftR10']:+.5f} "
         f"(crosses 0 at R=10? {'YES' if t13['liftR10'] >= 0 else 'NO'})")
    t16 = next(r for r in rows if r["dataset"] == "train_16")
    line(f"  Does train_16 stay negative? liftR1={t16['liftR1']:+.5f} -> "
         f"liftR10={t16['liftR10']:+.5f} "
         f"(still <0 at R=10? {'YES' if t16['liftR10'] < 0 else 'NO'})")

    # (1) reference gates C, D ---------------------------------------------
    line("")
    line("=== REFERENCE GATES (anchors; C from n_obj, D from n_obj & n_train) ===")
    for lbl, m in (("C_nocap", mC), ("D_exceptpn", mD)):
        line(f"gate {lbl:<11} fires on ({len(m['fire'])}): {', '.join(m['fire'])}")
        line(f"    mean Public d={m['mean_pub']:+.5f}  mean Private "
             f"d={m['mean_prv']:+.5f}  Public W/L/T={m['wlt_pub'][0]}/"
             f"{m['wlt_pub'][1]}/{m['wlt_pub'][2]}  Private W/L/T="
             f"{m['wlt_prv'][0]}/{m['wlt_prv'][1]}/{m['wlt_prv'][2]}  "
             f"excludes_16={'YES' if m['excludes_16'] else 'NO'}  "
             f"clean={'YES' if m['clean'] else 'NO'}")
    line("")
    line("=== REPRODUCTION CHECK (C==round34/35, D==round35, G_rep R=1==round36 G) ===")
    for x in repro_lines:
        line(x)

    # (2) per-R eps=0 comparison table (KEY QUESTION 2) --------------------
    line("")
    line("=== KEY QUESTION 2: gate G_rep at the principled eps=0 point, per R ===")
    line("  Fires seed-avg iff lift_R >= 0. clean = mean>0 both splits & zero regs.")
    e0hdr = (f"{'R':>3} {'nFire':>5} {'meanPub':>9} {'meanPrv':>9} "
             f"{'Pub W/L/T':>9} {'Prv W/L/T':>9} {'exc16':>5} {'t13?':>5} "
             f"{'clean':>5} {'=D?':>4} {'gainerExc':>12}")
    line(e0hdr)
    for R in R_LIST:
        m = Grep_eps0[R]
        ge = ",".join(g.replace("train_", "") for g in m["gainers_excluded"]) or "-"
        line(f"{R:>3} {len(m['fire']):>5} {m['mean_pub']:>+9.5f} "
             f"{m['mean_prv']:>+9.5f} "
             f"{m['wlt_pub'][0]}/{m['wlt_pub'][1]}/{m['wlt_pub'][2]:<3} "
             f"{m['wlt_prv'][0]}/{m['wlt_prv'][1]}/{m['wlt_prv'][2]:<3} "
             f"{'YES' if m['excludes_16'] else 'no':>5} "
             f"{'YES' if m['fires_13'] else 'no':>5} "
             f"{'YES' if m['clean'] else 'no':>5} "
             f"{'YES' if m['matches_D'] else 'no':>4} {ge:>12}")
    line("  drops @ eps=0 per R (which datasets do NOT fire):")
    for R in R_LIST:
        m = Grep_eps0[R]
        drops = sorted({r["dataset"] for r in rows} - set(m["fire"]))
        line(f"    R={R:<2}: drops {drops}  (train_13 fires? "
             f"{'YES' if m['fires_13'] else 'NO'})")

    # (3) full eps sweeps per R -------------------------------------------
    line("")
    line("=== KEY QUESTION 2 (full): gate G_rep FULL eps SWEEP for each R ===")
    for R in R_LIST:
        line(f"--- R={R} (fire iff lift_R >= eps) ---")
        line(f"{'eps':>10} {'nFire':>5} {'meanPub':>9} {'meanPrv':>9} "
             f"{'Pub W/L/T':>9} {'Prv W/L/T':>9} {'exc16':>5} {'t13?':>5} "
             f"{'clean':>5} {'=D?':>4} {'gainerExc':>12}")
        for m in Grep[R]:
            ge = ",".join(g.replace("train_", "") for g in m["gainers_excluded"]) or "-"
            line(f"{m['thr']:>+10.5f} {len(m['fire']):>5} {m['mean_pub']:>+9.5f} "
                 f"{m['mean_prv']:>+9.5f} "
                 f"{m['wlt_pub'][0]}/{m['wlt_pub'][1]}/{m['wlt_pub'][2]:<3} "
                 f"{m['wlt_prv'][0]}/{m['wlt_prv'][1]}/{m['wlt_prv'][2]:<3} "
                 f"{'YES' if m['excludes_16'] else 'no':>5} "
                 f"{'YES' if m['fires_13'] else 'no':>5} "
                 f"{'YES' if m['clean'] else 'no':>5} "
                 f"{'YES' if m['matches_D'] else 'no':>4} {ge:>12}")

    # (4) INVARIANT --------------------------------------------------------
    line("")
    line("=== INVARIANT (non-firing datasets byte-identical to base, delta 0) ===")
    if invariant_violations:
        line("VIOLATED!")
        for lbl, ds, dp, dv in invariant_violations:
            line(f"  {lbl}/{ds}: pub d={dp:+.6g} prv d={dv:+.6g}")
    else:
        line("OK: for every reference and swept G_rep gate, each non-firing "
             "dataset reuses the exact seed-0 base array (applied delta exactly "
             "0 on both splits). PASS.")

    # (5) VERDICT — KEY QUESTION 3 ----------------------------------------
    line("")
    line("=== KEY QUESTION 3 / VERDICT ===")
    line(f"  round36 single-split gate G (=G_rep R=1 @ eps=0): Public "
         f"{a['mean_pub']:+.5f}  Private {a['mean_prv']:+.5f}  (fires "
         f"{len(a['fire'])}, drops train_13 & train_06, excludes_16=YES)")
    line(f"  gate C (robust ref): Public {mC['mean_pub']:+.5f}  Private "
         f"{mC['mean_prv']:+.5f}  (fires {len(mC['fire'])}, excludes_16="
         f"{'YES' if mC['excludes_16'] else 'NO'})")
    line(f"  gate D (n=1 caveat): Public {mD['mean_pub']:+.5f}  Private "
         f"{mD['mean_prv']:+.5f}  (fires {len(mD['fire'])}, excludes_16="
         f"{'YES' if mD['excludes_16'] else 'NO'})")

    beats_G_R1 = False
    approaches_D = False
    if best_clean is not None:
        beats_G_R1 = (best_clean["mean_pub"] > a["mean_pub"] + 1e-9 and
                      best_clean["mean_prv"] > a["mean_prv"] + 1e-9)
        approaches_D = (best_clean["mean_pub"] >= mD["mean_pub"] - 1e-9 and
                        best_clean["mean_prv"] >= mD["mean_prv"] - 1e-9)
        line(f"  BEST CLEAN gate G_rep over the (R,eps) grid: R={best_clean['R']} "
             f"eps={best_clean['thr']:+.5f} fires ({len(best_clean['fire'])}) "
             f"{', '.join(best_clean['fire'])}")
        line(f"    mean Public d={best_clean['mean_pub']:+.5f}  mean Private "
             f"d={best_clean['mean_prv']:+.5f}  excludes_16="
             f"{'YES' if best_clean['excludes_16'] else 'NO'}  train_13_fires="
             f"{'YES' if best_clean['fires_13'] else 'NO'}  matches_D="
             f"{'YES' if best_clean['matches_D'] else 'NO'}")
    else:
        line("  BEST CLEAN gate G_rep over the (R,eps) grid: NONE "
             "(no (R,eps) is clean = mean>0 both splits & zero regressions).")

    # does ANY R at eps=0 beat round36 G cleanly?
    eps0_clean_beaters = [
        R for R in R_LIST
        if Grep_eps0[R]["clean"]
        and Grep_eps0[R]["mean_pub"] > a["mean_pub"] + 1e-9
        and Grep_eps0[R]["mean_prv"] > a["mean_prv"] + 1e-9
    ]
    t13_fires_any_eps0 = [R for R in R_LIST if Grep_eps0[R]["fires_13"]]
    t13_clean_fires = [
        R for R in R_LIST
        if Grep_eps0[R]["fires_13"] and Grep_eps0[R]["clean"]
    ]

    line("")
    line(f"  train_13 fires at eps=0 for R in: "
         f"{t13_fires_any_eps0 or 'NONE'}  "
         f"(cleanly, i.e. WITHOUT letting train_16 fire: {t13_clean_fires or 'NONE'})")
    line(f"  R@eps=0 that beat round36 G cleanly: {eps0_clean_beaters or 'NONE'}")

    line("")
    angle_success = bool(best_clean is not None and beats_G_R1)
    if angle_success:
        line("VERDICT: ANGLE (g) SUCCEEDS. Repeated-CV denoising of gate G's lift "
             f"estimate yields a CLEAN gate (R={best_clean['R']}, "
             f"eps={best_clean['thr']:+.5f}) that BEATS round36's single-split gate "
             f"G (Public {best_clean['mean_pub']:+.5f} vs {a['mean_pub']:+.5f}, "
             f"Private {best_clean['mean_prv']:+.5f} vs {a['mean_prv']:+.5f}) while "
             "excluding train_16 with ZERO regressions and WITHOUT using "
             "n_train/n_object_cols/test-labels."
             + (f" It also matches/approaches gate D (Public {mD['mean_pub']:+.5f}"
                f"/Private {mD['mean_prv']:+.5f})." if approaches_D else
                f" It does NOT reach gate D (Public {mD['mean_pub']:+.5f}/Private "
                f"{mD['mean_prv']:+.5f})."))
        if not best_clean["fires_13"]:
            line("  NOTE: even the best clean G_rep still does NOT capture train_13 "
                 "— the improvement comes from other datasets, not from recovering "
                 "train_13's gain. train_13 remains uncapturable by a train_16-safe "
                 "eps at these R.")
    else:
        cap = ("train_13's denoised lift never rises above a train_16-safe firing "
               "line: capturing train_13 (a genuine +test gainer) always drags "
               "train_16 (the poison) in with it, so no clean gate can recover it."
               if not t13_clean_fires else
               "train_13 can be made to fire cleanly at some R, but that gate does "
               "not strictly beat round36's single-split gate G on both splits.")
        line("VERDICT: ANGLE (g) FAILS (valid NEGATIVE result). Repeated-CV "
             "denoising does NOT produce a clean gate G_rep that strictly beats "
             f"round36's single-split gate G (+{a['mean_pub']:.5f}/"
             f"+{a['mean_prv']:.5f}) while excluding train_16 with zero "
             f"regressions. {cap} "
             "gate C (fire iff n_object_cols>0) remains the ROBUST ship "
             "recommendation and gate D's edge stays justified only by the "
             "single-dataset (train_16) observation.")

    # (6) CLEAN RUN marker -------------------------------------------------
    line("")
    firesets_ok = (set(mC["fire"]) ==
                   {"train_01", "train_02", "train_03", "train_05", "train_06",
                    "train_07", "train_08", "train_09", "train_12", "train_13",
                    "train_14", "train_15"}) and \
                  (D_fire_set == {f"train_{i:02d}" for i in range(1, 17)}
                   - {"train_16"})
    clean_run = ((not exceptions) and (not invariant_violations)
                 and (not skipped) and repro_ok and firesets_ok
                 and anchor_ok and len(rows) == N_DATASETS)
    if not anchor_ok:
        line("!!! REPRODUCTION ANCHOR FAILED: G_rep R=1 @ eps=0 does NOT match "
             "round36 gate G within 5e-6 — abort-loud, CLEAN RUN=NO. !!!")
    line(f"CLEAN RUN={'YES' if clean_run else 'NO'} "
         f"(total_fits={total_fits}, datasets={len(rows)}/{N_DATASETS}, "
         f"exceptions={len(exceptions)}, skipped={len(skipped)}, "
         f"invariant_violations={len(invariant_violations)}, "
         f"reproductions_ok={'YES' if repro_ok else 'NO'}, "
         f"anchor_R1_ok={'YES' if anchor_ok else 'NO'}, "
         f"ref_firesets_ok={'YES' if firesets_ok else 'NO'})")
    for name, msg in exceptions:
        line(f"  EXC {name}: {msg}")

    summary = "\n".join(S)
    print("\n" + summary)
    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
        f.write(summary + "\n")

    # ---------------------------- results.csv ------------------------------
    csv_path = os.path.join(OUT_DIR, "results.csv")
    fieldnames = ["dataset", "n_train", "n_object_cols", "l2_fired", "msl",
                  "liftR1", "liftR3", "liftR5", "liftR10", "rank_R1", "rank_R10",
                  "base_pub", "base_prv", "avg_pub", "avg_prv",
                  "d_pub", "d_prv", "test_gain",
                  "fires_C", "fires_D",
                  "fires_Grep_R1_eps0", "fires_Grep_R10_eps0"]
    fm_R1 = fire_map_from_signal(rows, "liftR1", 0.0)
    fm_R10 = fire_map_from_signal(rows, "liftR10", 0.0)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({
                "dataset": r["dataset"], "n_train": r["n_train"],
                "n_object_cols": r["n_object_cols"], "l2_fired": r["l2_fired"],
                "msl": r["msl"],
                "liftR1": r["liftR1"], "liftR3": r["liftR3"],
                "liftR5": r["liftR5"], "liftR10": r["liftR10"],
                "rank_R1": rank_R1[r["dataset"]], "rank_R10": rank_R10[r["dataset"]],
                "base_pub": r["base_pub"], "base_prv": r["base_prv"],
                "avg_pub": r["avg_pub"], "avg_prv": r["avg_prv"],
                "d_pub": r["d_pub"], "d_prv": r["d_prv"],
                "test_gain": r["test_gain"],
                "fires_C": ref_fire["C_nocap"][r["dataset"]],
                "fires_D": ref_fire["D_exceptpn"][r["dataset"]],
                "fires_Grep_R1_eps0": fm_R1[r["dataset"]],
                "fires_Grep_R10_eps0": fm_R10[r["dataset"]],
            })

    print(f"\n[WROTE] {csv_path}")
    print(f"[WROTE] {os.path.join(OUT_DIR, 'summary.txt')}")
    print("HARNESS_DONE")


if __name__ == "__main__":
    main()
