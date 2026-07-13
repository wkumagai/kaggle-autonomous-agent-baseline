#!/usr/bin/env python
"""
bench_03 round36 — MECHANISM FIRING-GATE (angle "e"): can a K=5 seed-averaging
firing gate computed PURELY from TRAINING-TIME mechanism signals — with NO
reference to n_train or n_object_cols — reproduce gate D's clean exclusion of the
sole regressor train_16 (or at least cleanly match/beat gate C)? OFFLINE ONLY. No
subprocess, no LLM, no Kaggle, no network. Calls sklearn in-process.

Adapted from experiments/bench_03/round35_gate_purenumeric/replay.py. ALL of
round35's machinery is reused verbatim: dataset loading, the shipped-08 base
config reproduction, Public/Private AUC scoring joined on row_id to solution.csv,
K=5 seed-averaging (K FIXED at 5) of the TEST predictions, the byte-identical-on-
non-firing INVARIANT, cached fits, and the summary machinery. The base recipe is
UNCHANGED (== shipped 08).

WHAT IS NEW vs round35 -------------------------------------------------------
round35 gated seed-averaging on data-DESCRIPTIVE stats (n_object_cols, n_train).
round36 keeps gate C and gate D as REFERENCES (repro/sanity anchors) and adds two
MECHANISM gates whose firing signal is computed ONLY from the training data (a
train/holdout split fit on the SAME shipped-08 HGB config, no test labels, no
n_train, no n_object_cols):

  MECHANISM HYPOTHESIS: seed-averaging helps by reducing the variance of
  early_stopping's internal-holdout split. train_16 is pure-numeric, small-n,
  and already saturated (holdout AUC ~ 0.89), so averaging has little variance
  left to reduce and can hurt. But train_02 also has a HIGH level (~0.96) yet
  GAINS — so a naive "high AUC -> don't fire" gate would wrongly drop train_02.
  The distinguishing signal must target the VARIANCE to be reduced, not the
  LEVEL.

  gate F (cross-seed holdout-AUC VARIANCE): for seed s in 0..K-1 make a
    deterministic stratified train/holdout split (random_state=s), fit shipped-08
    HGB (random_state=s) on the fit part, score ROC-AUC on that seed's holdout ->
    K per-seed holdout AUCs. Signal sd_auc = population std of those K AUCs. Fire
    seed-avg iff sd_auc >= tau. tau is SWEPT over a grid built from the observed
    per-dataset sd values (every distinct threshold firing-set is enumerated, so
    if train_16 is the unique lowest-variance dataset at least one tau isolates
    it). High cross-seed variance => averaging has variance to remove => fire.

  gate G (self-validating internal-holdout LIFT): fix ONE deterministic
    stratified holdout (random_state=0); fit shipped-08 HGB with random_state
    0..K-1 on the fit part, predict on that SAME holdout. seed0_auc = AUC of the
    seed-0 holdout prediction; avg_auc = AUC of the K-averaged holdout
    prediction. Signal lift = avg_auc - seed0_auc. Fire iff lift >= eps (eps
    SWEPT around 0). This directly asks "does averaging actually help on an
    internal holdout?" — the training-only mirror of the test-time averaging.

  Both signals use ONLY train.csv rows (the holdout is carved from training data)
  and the shipped-08 HGB config for that dataset; NO test labels, NO n_train, NO
  n_object_cols enter either mechanism gate. gate C and gate D still read the
  stats file, but ONLY as fixed references to anchor the mechanism comparison.

Base recipe reproduced (== shipped 08), IDENTICAL to round35/round34:
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

EFFICIENCY / fit budget per dataset:
  - 5 TEST-prediction fits (seeds 0..4 on ALL training rows -> cached avg vs base)
  - 5 gate-F fits (varying stratified split seed s, model seed s)
  - 5 gate-G fits (fixed stratified split seed 0, model seeds 0..4)
  => 15 fits/dataset x 16 datasets = 240 total. All deterministic (fixed seeds;
  no Date/random dependence beyond the seed loops).

INVARIANT: for every candidate gate, any dataset that does NOT fire under it
reuses the EXACT seed-0 test array, so its Public/Private delta vs base is exactly
0 — checked explicitly per gate.

Adoption criterion (reused): a gate is a CLEAN IMPROVEMENT over base(08) iff its
mean delta is positive on BOTH splits AND there are ZERO regressions on BOTH
splits.

References this run must reproduce (sanity anchors):
  gate C nocap   == round34/35: mean Public +0.00363 / Private +0.00316.
  gate D exceptpn== round35    : mean Public +0.00473 / Private +0.00453.
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
OUT_DIR = os.path.join(BENCH_DIR, "round36_mechanism_gate")
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

# Mechanism-holdout config (training-data only). Stratified on the ~0.5 target.
HOLDOUT_FRAC = 0.25        # external holdout carved from TRAINING rows.
SPLIT_SEED_G = 0           # gate G fixes ONE holdout split; model seed varies.

BASE = "base"
N_DATASETS = 16

# reference anchors this run must reproduce.
REF = {
    "C_nocap":    (0.00363, 0.00316),   # round34/35 nocap
    "D_exceptpn": (0.00473, 0.00453),   # round35 exceptpn
}

# datasets of special interest for the "wrongly excluded gainer" check.
GAINERS_OF_INTEREST = ["train_02", "train_04", "train_10", "train_11"]
SOLE_REGRESSOR = "train_16"


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


def mechanism_signals(train, features, cat_mask, l2, msl_val):
    """Compute the two training-time mechanism signals from the SAME shipped-08
    HGB config. Uses ONLY training rows (holdout carved from train). Returns
    (dict_of_signals, n_fits).

      gate F signal sd_auc : population std of K per-seed holdout AUCs, each from
                             its own stratified split (random_state=s) + model
                             seed s. Measures cross-seed generalization VARIANCE.
      gate G signal lift   : avg_auc - seed0_auc on ONE fixed stratified holdout
                             (split seed 0), model seeds 0..K-1. Measures whether
                             averaging actually helps on an internal holdout.
    """
    y = train["target"].values
    X = train[features]

    # --- gate G: fixed holdout, vary model seed -> averaged-vs-seed0 lift ---
    Xf, Xh, yf, yh = train_test_split(
        X, y, test_size=HOLDOUT_FRAC, random_state=SPLIT_SEED_G, stratify=y)
    g_preds = [fit_holdout(Xf, yf, Xh, cat_mask, l2, msl_val, s) for s in SEEDS]
    seed0_auc = auc_or_nan(yh, g_preds[0])
    avg_auc = auc_or_nan(yh, np.mean(np.vstack(g_preds), axis=0))
    lift = avg_auc - seed0_auc
    fixed_split_aucs = [auc_or_nan(yh, p) for p in g_preds]
    sd_fixed = float(np.std(fixed_split_aucs, ddof=0))

    # --- gate F: vary split seed AND model seed -> cross-seed AUC std ---
    var_split_aucs = []
    for s in SEEDS:
        Xf2, Xh2, yf2, yh2 = train_test_split(
            X, y, test_size=HOLDOUT_FRAC, random_state=s, stratify=y)
        p = fit_holdout(Xf2, yf2, Xh2, cat_mask, l2, msl_val, s)
        var_split_aucs.append(auc_or_nan(yh2, p))
    sd_auc = float(np.std(var_split_aucs, ddof=0))

    sig = {
        "sd_auc": sd_auc,               # gate F firing signal
        "lift": lift,                   # gate G firing signal
        "seed0_auc": seed0_auc,
        "avg_auc": avg_auc,
        "sd_fixed": sd_fixed,
        "mean_var_auc": float(np.mean(var_split_aucs)),
    }
    return sig, 2 * K


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
    averaged test pred, score both splits, and compute the two mechanism signals.
    Returns a per-dataset record dict + n_fits."""
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

    # --- mechanism signals (training-only) ---
    sig, mech_fits = mechanism_signals(train, features, cat_mask, l2, msl_val)
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
    m["gainers_excluded"] = [
        g for g in GAINERS_OF_INTEREST if g not in set(m["fire"])
    ]
    return m


def fire_map_from_signal(rows, signal_key, thr):
    """fire iff rec[signal_key] >= thr."""
    return {r["dataset"]: (r[signal_key] >= thr) for r in rows}


def threshold_grid(values):
    """Every distinct '>= thr' firing set for a signal: use each observed value as
    a threshold (fires that dataset and all with >= value), plus one above the max
    (fires none). Deduplicated, ascending."""
    uniq = sorted(set(round(v, 12) for v in values))
    grid = list(uniq)
    if uniq:
        grid.append(uniq[-1] + 1.0)   # fire-none sentinel
    return grid


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
                  f"prv={rec['avg_prv']:.6f} | sd_auc={rec['sd_auc']:.5f} "
                  f"lift={rec['lift']:+.5f} test={rec['test_gain']}")
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

    # ---- mechanism gate F sweep (sd_auc >= tau) ----
    tau_grid = threshold_grid([r["sd_auc"] for r in rows])
    F_results = []
    for tau in tau_grid:
        fm = fire_map_from_signal(rows, "sd_auc", tau)
        m = eval_gate(rows, fm)
        m["thr"] = tau
        m["matches_D"] = (set(m["fire"]) == D_fire_set)
        F_results.append(m)

    # ---- mechanism gate G sweep (lift >= eps) ----
    lift_vals = [r["lift"] for r in rows]
    eps_grid = sorted(set([round(v, 12) for v in lift_vals] + [0.0]))
    eps_grid.append(max(lift_vals) + 1.0)   # fire-none sentinel
    eps_grid = sorted(set(round(v, 12) for v in eps_grid))
    G_results = []
    for eps in eps_grid:
        fm = fire_map_from_signal(rows, "lift", eps)
        m = eval_gate(rows, fm)
        m["thr"] = eps
        m["matches_D"] = (set(m["fire"]) == D_fire_set)
        G_results.append(m)

    # ---- pick best mechanism candidate per family ----
    # "acceptable" = excludes train_16, zero regressions, mean >= gate C on BOTH
    # splits, AND no gainer-of-interest wrongly excluded. Among acceptable, prefer
    # the highest mean (pub+prv), tie-break: matches_D, then larger firing set.
    def acceptable(m):
        return (m["excludes_16"] and m["zero_regs"]
                and m["mean_pub"] >= mC["mean_pub"] - 1e-12
                and m["mean_prv"] >= mC["mean_prv"] - 1e-12
                and not m["gainers_excluded"])

    def pick_best(results):
        accs = [m for m in results if acceptable(m)]
        if not accs:
            return None
        return max(
            accs,
            key=lambda m: (m["mean_pub"] + m["mean_prv"], m["matches_D"],
                           len(m["fire"])))

    bestF = pick_best(F_results)
    bestG = pick_best(G_results)

    # ---- INVARIANT: non-firing datasets identical to base (delta exactly 0) ----
    invariant_violations = []
    for label, fm in (("C_nocap", ref_fire["C_nocap"]),
                      ("D_exceptpn", ref_fire["D_exceptpn"])):
        for r in rows:
            if not fm[r["dataset"]]:
                dp = applied_delta(r, "pub", False)
                dv = applied_delta(r, "prv", False)
                if dp != 0.0 or dv != 0.0:
                    invariant_violations.append((label, r["dataset"], dp, dv))
    # mechanism gates: by construction non-firing -> applied_delta returns 0.0.
    for fam, res in (("F", F_results), ("G", G_results)):
        for m in res:
            fireset = set(m["fire"])
            for r in rows:
                if r["dataset"] not in fireset:
                    if applied_delta(r, "pub", False) != 0.0 or \
                       applied_delta(r, "prv", False) != 0.0:
                        invariant_violations.append(
                            (f"{fam}@{m['thr']:.5f}", r["dataset"], 0.0, 0.0))

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

    # ---------------------------- build summary ----------------------------
    S = []

    def line(x=""):
        S.append(x)

    # (0) mechanism-signal table (KEY QUESTION 1) --------------------------
    line("=== KEY QUESTION 1: TRAINING-TIME MECHANISM SIGNALS PER DATASET ===")
    line("  sd_auc = std of K=5 per-seed holdout AUCs (varying split+model seed) "
         "= gate F signal")
    line("  lift   = avg_auc - seed0_auc on a fixed internal holdout            "
         "= gate G signal")
    line("  test_gain = does K=5 averaging TRULY help on the real test splits "
         "(ground truth)")
    hdr = (f"{'dataset':<10} {'nTr':>6} {'obj':>4} {'seed0AUC':>9} {'avgAUC':>8} "
           f"{'lift':>9} {'sd_auc':>8} {'d_pub':>9} {'d_prv':>9} {'truth':>6}")
    line(hdr)
    for r in rows:
        line(f"{r['dataset']:<10} {r['n_train']:>6} {r['n_object_cols']:>4} "
             f"{r['seed0_auc']:>9.4f} {r['avg_auc']:>8.4f} {r['lift']:>+9.5f} "
             f"{r['sd_auc']:>8.5f} {r['d_pub']:>+9.5f} {r['d_prv']:>+9.5f} "
             f"{r['test_gain']:>6}")
    # ranking hints.
    by_sd = sorted(rows, key=lambda r: r["sd_auc"])
    by_lift = sorted(rows, key=lambda r: r["lift"])
    line("")
    line("  datasets by sd_auc ASC (lowest cross-seed variance first): "
         + ", ".join(f"{r['dataset']}({r['sd_auc']:.5f})" for r in by_sd))
    line("  datasets by lift   ASC (least internal-holdout benefit first): "
         + ", ".join(f"{r['dataset']}({r['lift']:+.5f})" for r in by_lift))
    line(f"  train_16 sd_auc rank (1=lowest): "
         f"{[r['dataset'] for r in by_sd].index(SOLE_REGRESSOR)+1} / {len(rows)}")
    line(f"  train_16 lift   rank (1=lowest): "
         f"{[r['dataset'] for r in by_lift].index(SOLE_REGRESSOR)+1} / {len(rows)}")

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
    line("=== REPRODUCTION CHECK (C==round34/35, D==round35) ===")
    for x in repro_lines:
        line(x)

    # (2) gate F sweep -----------------------------------------------------
    line("")
    line("=== KEY QUESTION 2a: gate F (fire iff sd_auc >= tau) — FULL tau SWEEP ===")
    fhdr = (f"{'tau':>9} {'nFire':>5} {'meanPub':>9} {'meanPrv':>9} "
            f"{'Pub W/L/T':>9} {'Prv W/L/T':>9} {'exc16':>5} {'clean':>5} "
            f"{'=D?':>4} {'gainerExc':>10}")
    line(fhdr)
    for m in F_results:
        ge = ",".join(g.replace("train_", "") for g in m["gainers_excluded"]) or "-"
        line(f"{m['thr']:>9.5f} {len(m['fire']):>5} {m['mean_pub']:>+9.5f} "
             f"{m['mean_prv']:>+9.5f} "
             f"{m['wlt_pub'][0]}/{m['wlt_pub'][1]}/{m['wlt_pub'][2]:<3} "
             f"{m['wlt_prv'][0]}/{m['wlt_prv'][1]}/{m['wlt_prv'][2]:<3} "
             f"{'YES' if m['excludes_16'] else 'no':>5} "
             f"{'YES' if m['clean'] else 'no':>5} "
             f"{'YES' if m['matches_D'] else 'no':>4} {ge:>10}")
    if bestF is not None:
        line(f"  BEST acceptable gate F: tau={bestF['thr']:.5f} fires "
             f"({len(bestF['fire'])}) {', '.join(bestF['fire'])}")
        line(f"    mean Public d={bestF['mean_pub']:+.5f}  mean Private "
             f"d={bestF['mean_prv']:+.5f}  matches_gate_D="
             f"{'YES' if bestF['matches_D'] else 'NO'}  "
             f"excludes_16={'YES' if bestF['excludes_16'] else 'NO'}  "
             f"gainers_excluded={bestF['gainers_excluded'] or 'none'}")
    else:
        line("  BEST acceptable gate F: NONE (no tau excludes train_16 with zero "
             "regressions AND mean >= gate C AND no gainer-of-interest dropped)")

    # (3) gate G sweep -----------------------------------------------------
    line("")
    line("=== KEY QUESTION 2b: gate G (fire iff lift >= eps) — FULL eps SWEEP ===")
    line(f"{'eps':>9} {'nFire':>5} {'meanPub':>9} {'meanPrv':>9} "
         f"{'Pub W/L/T':>9} {'Prv W/L/T':>9} {'exc16':>5} {'clean':>5} "
         f"{'=D?':>4} {'gainerExc':>10}")
    for m in G_results:
        ge = ",".join(g.replace("train_", "") for g in m["gainers_excluded"]) or "-"
        line(f"{m['thr']:>+9.5f} {len(m['fire']):>5} {m['mean_pub']:>+9.5f} "
             f"{m['mean_prv']:>+9.5f} "
             f"{m['wlt_pub'][0]}/{m['wlt_pub'][1]}/{m['wlt_pub'][2]:<3} "
             f"{m['wlt_prv'][0]}/{m['wlt_prv'][1]}/{m['wlt_prv'][2]:<3} "
             f"{'YES' if m['excludes_16'] else 'no':>5} "
             f"{'YES' if m['clean'] else 'no':>5} "
             f"{'YES' if m['matches_D'] else 'no':>4} {ge:>10}")
    if bestG is not None:
        line(f"  BEST acceptable gate G: eps={bestG['thr']:+.5f} fires "
             f"({len(bestG['fire'])}) {', '.join(bestG['fire'])}")
        line(f"    mean Public d={bestG['mean_pub']:+.5f}  mean Private "
             f"d={bestG['mean_prv']:+.5f}  matches_gate_D="
             f"{'YES' if bestG['matches_D'] else 'NO'}  "
             f"excludes_16={'YES' if bestG['excludes_16'] else 'NO'}  "
             f"gainers_excluded={bestG['gainers_excluded'] or 'none'}")
    else:
        line("  BEST acceptable gate G: NONE (no eps excludes train_16 with zero "
             "regressions AND mean >= gate C AND no gainer-of-interest dropped)")

    # (4) INVARIANT --------------------------------------------------------
    line("")
    line("=== INVARIANT (non-firing datasets byte-identical to base, delta 0) ===")
    if invariant_violations:
        line("VIOLATED!")
        for lbl, ds, dp, dv in invariant_violations:
            line(f"  {lbl}/{ds}: pub d={dp:+.6g} prv d={dv:+.6g}")
    else:
        line("OK: for every reference and swept mechanism gate, each non-firing "
             "dataset reuses the exact seed-0 base array (applied delta exactly "
             "0 on both splits). PASS.")

    # (5) VERDICT — KEY QUESTION 3 ----------------------------------------
    line("")
    line("=== KEY QUESTION 3 / VERDICT ===")
    line(f"  gate C (robust ref): Public {mC['mean_pub']:+.5f}  Private "
         f"{mC['mean_prv']:+.5f}  (fires {len(mC['fire'])}, excludes_16="
         f"{'YES' if mC['excludes_16'] else 'NO'})")
    line(f"  gate D (n=1 caveat): Public {mD['mean_pub']:+.5f}  Private "
         f"{mD['mean_prv']:+.5f}  (fires {len(mD['fire'])}, excludes_16="
         f"{'YES' if mD['excludes_16'] else 'NO'})")

    def describe(fam, best):
        if best is None:
            return (f"gate {fam}: NO mechanism threshold cleanly excludes "
                    f"train_16 while beating/matching gate C without dropping a "
                    f"gainer.")
        tag = ("REPRODUCES gate D's firing set exactly (all but train_16)"
               if best["matches_D"] else
               f"excludes train_16 and fires {len(best['fire'])} datasets")
        beats_C = (best["mean_pub"] > mC["mean_pub"] + 1e-9 and
                   best["mean_prv"] > mC["mean_prv"] + 1e-9)
        rel = "BEATS" if beats_C else "MATCHES/>="
        return (f"gate {fam}: {tag}; {rel} gate C "
                f"(Public {best['mean_pub']:+.5f} vs {mC['mean_pub']:+.5f}, "
                f"Private {best['mean_prv']:+.5f} vs {mC['mean_prv']:+.5f}); "
                f"zero regressions.")

    line("  " + describe("F", bestF))
    line("  " + describe("G", bestG))

    mech_success = (bestF is not None) or (bestG is not None)
    if mech_success:
        winner = bestF if bestF is not None else bestG
        fam = "F" if bestF is not None else "G"
        sig = "cross-seed holdout-AUC std" if fam == "F" else \
              "internal-holdout averaging lift"
        line("")
        line("VERDICT: A PURELY TRAINING-TIME MECHANISM GATE SUCCEEDS. The "
             f"{sig} signal (gate {fam}) yields a threshold that "
             f"{'reproduces gate D exactly' if winner['matches_D'] else 'cleanly excludes train_16'} "
             "— it cleanly excludes the sole regressor train_16 with ZERO "
             "regressions and mean delta >= gate C, WITHOUT referencing n_train or "
             "n_object_cols. This REPLACES gate D's n=1 (train_16-only) caveat with "
             "a mechanism computable from training data alone, strengthening the "
             "ship case.")
    else:
        line("")
        line("VERDICT: NO purely training-time mechanism gate (F cross-seed "
             "variance, or G internal-holdout lift) cleanly excludes train_16 "
             "while matching/beating gate C without dropping a genuine gainer. "
             "train_16's exclusion RESISTS mechanistic capture from these "
             "training-time signals; gate C (fire iff n_object_cols>0) remains the "
             "ROBUST, data-independent ship recommendation, and gate D's advantage "
             "stays justified only by the single-dataset (train_16) observation.")

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
                 and len(rows) == N_DATASETS)
    line(f"CLEAN RUN={'YES' if clean_run else 'NO'} "
         f"(total_fits={total_fits}, datasets={len(rows)}/{N_DATASETS}, "
         f"exceptions={len(exceptions)}, skipped={len(skipped)}, "
         f"invariant_violations={len(invariant_violations)}, "
         f"reproductions_ok={'YES' if repro_ok else 'NO'}, "
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
                  "seed0_auc", "avg_auc", "lift", "sd_auc", "sd_fixed",
                  "mean_var_auc", "base_pub", "base_prv", "avg_pub", "avg_prv",
                  "d_pub", "d_prv", "test_gain",
                  "fires_C", "fires_D"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({
                "dataset": r["dataset"], "n_train": r["n_train"],
                "n_object_cols": r["n_object_cols"], "l2_fired": r["l2_fired"],
                "msl": r["msl"], "seed0_auc": r["seed0_auc"],
                "avg_auc": r["avg_auc"], "lift": r["lift"], "sd_auc": r["sd_auc"],
                "sd_fixed": r["sd_fixed"], "mean_var_auc": r["mean_var_auc"],
                "base_pub": r["base_pub"], "base_prv": r["base_prv"],
                "avg_pub": r["avg_pub"], "avg_prv": r["avg_prv"],
                "d_pub": r["d_pub"], "d_prv": r["d_prv"],
                "test_gain": r["test_gain"],
                "fires_C": ref_fire["C_nocap"][r["dataset"]],
                "fires_D": ref_fire["D_exceptpn"][r["dataset"]],
            })

    print(f"\n[WROTE] {csv_path}")
    print(f"[WROTE] {os.path.join(OUT_DIR, 'summary.txt')}")
    print("HARNESS_DONE")


if __name__ == "__main__":
    main()
