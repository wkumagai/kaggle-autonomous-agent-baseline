#!/usr/bin/env python
"""
bench_03 round38 — IN-SAMPLE CHARACTERIZATION GATE (angle "h"): is there a
TRAINING-TIME metric (computed WITHOUT test labels) that flags train_16 as
"seed-avg will HURT" while flagging all 15 true gainers (the 12 categorical
datasets + the pure-numeric gainers train_04/10/11) as "seed-avg helps"?

If such a metric cleanly separates train_16 from every gainer, it reproduces
gate D's exclusion (fire on all but train_16 -> +0.00473/+0.00453) using a
MECHANISTIC in-sample signal instead of the single-dataset n_train/n_object
predicate — removing gate D's n=1 dependence.

OFFLINE ONLY. No subprocess, no LLM, no Kaggle, no network. Calls sklearn
in-process only.

Adapted from experiments/bench_03/round37_gateG_repcv/replay.py. ALL of round37's
machinery is reused VERBATIM: dataset loading, the shipped-08 base config
reproduction, Public/Private AUC scoring joined on row_id to solution.csv, K=5
seed-averaging (K FIXED at 5) of the TEST predictions, the byte-identical-on-
non-firing INVARIANT, the reference gates C and D (predicates over
n_train/n_object_cols) as reproduction anchors, the round36 gate-G_rep R=1 lift
anchor (a free byte-identity check on the reused holdout fits), the CLEAN RUN
marker, and the summary/csv machinery. The base recipe is UNCHANGED (== shipped
08): seed-0 test pred is byte-identical to shipped 08.

WHAT ROUND38 ADDS vs round37 ----------------------------------------------
round37 already fits, for each dataset, R_MAX=10 repeated stratified 25%
holdouts x K=5 model seeds = 50 holdout fits, entirely on TRAINING rows (holdout
labels are TRAINING labels, never test). Round38 REUSES those exact 50 holdout
fits (no extra fits) and, from the K=5 holdout prediction arrays P (shape K x
n_hold) and the holdout labels yh, computes five IN-SAMPLE metrics per repeat,
then averages over the 10 repeats to get a per-dataset scalar:

  seed_var      : mean over holdout rows of the variance across the K=5 seed
                  probabilities (population var, ddof=0). Hypothesis: train_16's
                  seeds barely disagree (already saturated) -> LOW variance ->
                  seed-avg gives little benefit and can hurt.       (HIGH fires)
  seed_disagree : mean over holdout rows of the mean pairwise |p_i - p_j| across
                  the K=5 seeds (a robustness twin of seed_var).    (HIGH fires)
  logloss_impr  : log_loss(seed0) - log_loss(avg) on the holdout. Positive =
                  averaging IMPROVES in-sample log-loss.            (HIGH fires)
  brier_impr    : brier(seed0) - brier(avg) on the holdout (brier = mean
                  (p - y)^2). Positive = averaging IMPROVES in-sample Brier.
                                                                    (HIGH fires)
  holdout_auc   : seed0 holdout AUC LEVEL (saturation). train_16 is already
                  ~0.89. Hypothesis: high-AUC -> saturated -> averaging cannot
                  help.                                              (LOW fires)

For each metric we ask: does a single threshold tau exist that FIRES seed-avg on
all 15 true gainers while EXCLUDING train_16?  For a HIGH-fires metric this needs
train_16's value < min over gainers (separation margin = min_gainer - t16 > 0);
for a LOW-fires metric it needs train_16 > max over gainers (margin = t16 -
max_gainer > 0). If a metric cleanly separates train_16, a threshold at the
separation midpoint fires EXACTLY the 15 gainers == gate D's firing set, i.e. it
REPRODUCES gate D (+0.00473/+0.00453) mechanistically. We also full-sweep each
metric's threshold to find the best CLEAN gate (mean>0 both splits, zero
regressions => train_16 excluded) even when full separation fails.

  CACHING: the 10 repeats x K=5 seeds = 50 holdout fits per dataset are computed
  ONCE (same as round37). Fit budget per dataset = 5 test fits + 50 holdout fits
  = 55; x16 datasets = 880 fits total. Fully deterministic (fixed seeds only).

Base recipe reproduced (== shipped 08), IDENTICAL to round37/36/35:
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
  gate G_rep R=1  == round36 G  : mean Public +0.00426 / Private +0.00419 (fires 13).
"""
import os
import csv
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import train_test_split

REPO = "/Users/kumacmini/kaggle-autonomous-agent-baseline-auto"
DATA_DIR = os.path.join(REPO, "data")
BENCH_DIR = os.path.join(REPO, "experiments", "bench_03")
OUT_DIR = os.path.join(BENCH_DIR, "round38_insample_gate")
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
R_LIST = [1, 3, 5, 10]     # denoise depths (kept for the round36 lift anchor).

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
SOLE_REGRESSOR = "train_16"
# The pure-numeric gainers (obj==0 but seed-avg still helps on the real test).
PURE_NUMERIC_GAINERS = ["train_04", "train_10", "train_11"]

# ---- in-sample metric specs: (key, direction, description) ----
#   direction "ge"  : fire seed-avg iff metric >= tau (HIGH indicates help).
#   direction "le"  : fire seed-avg iff metric <= tau (LOW  indicates help).
METRICS = [
    ("seed_var",      "ge", "mean per-row variance across K=5 seed holdout probs"),
    ("seed_disagree", "ge", "mean per-row mean-pairwise |p_i-p_j| across K seeds"),
    ("logloss_impr",  "ge", "log_loss(seed0) - log_loss(avg) on holdout"),
    ("brier_impr",    "ge", "brier(seed0) - brier(avg) on holdout"),
    ("holdout_auc",   "le", "seed0 holdout AUC level (saturation)"),
]


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


def _brier(y_true, p):
    """Brier score for binary y in {0,1}: mean((p - y)^2). Version-independent."""
    y = np.asarray(y_true, dtype=float)
    return float(np.mean((np.asarray(p, dtype=float) - y) ** 2))


def _logloss(y_true, p):
    """Binary log-loss with sklearn clipping; labels fixed to [0,1]."""
    return float(log_loss(y_true, np.asarray(p, dtype=float), labels=[0, 1]))


def _mean_pairwise_disagree(P):
    """Mean over rows of the mean pairwise |p_i - p_j| across the K seed vectors.
    P has shape (K, n_hold)."""
    K_ = P.shape[0]
    tot, cnt = 0.0, 0
    for i in range(K_):
        for j in range(i + 1, K_):
            tot += float(np.abs(P[i] - P[j]).mean())
            cnt += 1
    return tot / cnt if cnt else 0.0


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


def repeated_signals(train, features, cat_mask, l2, msl_val):
    """Compute round37's repeated-CV lift signal AND round38's five in-sample
    metrics from the SAME shipped-08 holdout fits. Uses ONLY training rows.
    Returns (dict_of_signals, n_fits).

      For repeat r in 0..R_MAX-1: stratified 25% holdout (random_state=r), fit K
      model seeds 0..K-1 on the fit part, predict on that repeat's holdout. From
      the K predictions P (shape K x n_hold) and holdout labels yh compute, per
      repeat: the round36 lift (auc(mean) - auc(seed0)) AND seed_var,
      seed_disagree, logloss_impr, brier_impr, holdout_auc. Average each over the
      10 repeats. The 10x5=50 holdout fits are done ONCE.

      Repeat r=0 (split seed 0, model seeds 0..K-1) is byte-identical to round36's
      single-split gate G, so lift_R at R=1 reproduces round36's gate-G lift (a
      free byte-identity check that the reused holdout fits are unchanged).
    """
    y = train["target"].values
    X = train[features]
    per_lift, per_var, per_dis = [], [], []
    per_ll, per_br, per_auc = [], [], []
    for r in range(R_MAX):
        Xf, Xh, yf, yh = train_test_split(
            X, y, test_size=HOLDOUT_FRAC, random_state=r, stratify=y)
        preds = [fit_holdout(Xf, yf, Xh, cat_mask, l2, msl_val, s) for s in SEEDS]
        P = np.vstack(preds)                 # (K, n_hold)
        seed0 = preds[0]
        avg = P.mean(axis=0)
        seed0_auc = auc_or_nan(yh, seed0)
        avg_auc = auc_or_nan(yh, avg)
        per_lift.append(avg_auc - seed0_auc)
        # in-sample metrics (yh are TRAINING holdout labels, NEVER test labels).
        per_var.append(float(np.var(P, axis=0, ddof=0).mean()))
        per_dis.append(_mean_pairwise_disagree(P))
        per_ll.append(_logloss(yh, seed0) - _logloss(yh, avg))
        per_br.append(_brier(yh, seed0) - _brier(yh, avg))
        per_auc.append(seed0_auc)

    sig = {
        "lift_repeats": per_lift,
        "seed_var": float(np.nanmean(per_var)),
        "seed_disagree": float(np.nanmean(per_dis)),
        "logloss_impr": float(np.nanmean(per_ll)),
        "brier_impr": float(np.nanmean(per_br)),
        "holdout_auc": float(np.nanmean(per_auc)),
    }
    for R in R_LIST:
        sig[f"liftR{R}"] = float(np.mean(per_lift[:R]))
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
    averaged test pred, score both splits, and compute the repeated-CV lift and
    in-sample metric signals. Returns a per-dataset record dict + n_fits."""
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

    # --- repeated-CV lift + in-sample metrics (training-only, 50 holdout fits) --
    sig, mech_fits = repeated_signals(train, features, cat_mask, l2, msl_val)
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
    return m


def fire_map_from_lift(rows, signal_key, thr):
    """fire iff rec[signal_key] >= thr (used only for the round36 lift anchor)."""
    return {r["dataset"]: (r[signal_key] >= thr) for r in rows}


def metric_fire_map(rows, key, direction, thr):
    """fire iff metric passes threshold in the metric's direction."""
    if direction == "ge":
        return {r["dataset"]: (r[key] >= thr) for r in rows}
    return {r["dataset"]: (r[key] <= thr) for r in rows}


def metric_grid(rows, key):
    """Every distinct firing threshold for a metric: each observed value, plus a
    below-min sentinel (fires the extreme side) and an above-max sentinel."""
    vals = [r[key] for r in rows]
    lo, hi = min(vals), max(vals)
    grid = [lo - 1.0] + [round(v, 12) for v in vals] + [hi + 1.0]
    return sorted(set(round(g, 12) for g in grid))


def separation(rows, key, direction):
    """train_16 vs the 15 gainers on this metric. Returns a dict with train_16's
    value, the gainers' min/max, the separation margin, and whether a clean
    threshold exists that fires ALL gainers while excluding train_16.

      ge (HIGH fires): clean iff t16 < min(gainers); margin = min(gainers) - t16.
      le (LOW  fires): clean iff t16 > max(gainers); margin = t16 - max(gainers).
    A positive margin => a midpoint threshold reproduces gate D exactly.
    """
    t16 = next(r[key] for r in rows if r["dataset"] == SOLE_REGRESSOR)
    gvals = [r[key] for r in rows if r["dataset"] != SOLE_REGRESSOR]
    gmin, gmax = min(gvals), max(gvals)
    if direction == "ge":
        margin = gmin - t16
        boundary = gmin   # gainers must all be >= their own min
    else:
        margin = t16 - gmax
        boundary = gmax
    return {
        "t16": t16, "gmin": gmin, "gmax": gmax,
        "margin": margin, "clean_sep": margin > 0, "boundary": boundary,
    }


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
                  f"fits={n_fits}): d_pub={rec['d_pub']:+.5f} d_prv={rec['d_prv']:+.5f} "
                  f"test={rec['test_gain']} | var={rec['seed_var']:.2e} "
                  f"llImpr={rec['logloss_impr']:+.2e} hAUC={rec['holdout_auc']:.4f}")
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

    # ---- round36 gate-G_rep R=1 lift anchor (free byte-identity check) -------
    fm_g = fire_map_from_lift(rows, "liftR1", 0.0)
    a = eval_gate(rows, fm_g)
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

    # ---- per-metric separation + full threshold sweep -----------------------
    sep = {}          # key -> separation dict
    sweep = {}        # key -> list of metric dicts over the threshold grid
    best_clean = {}   # key -> best clean gate dict (or None)
    midpoint = {}     # key -> gate at the separation midpoint (only if clean_sep)
    for key, direction, _desc in METRICS:
        sep[key] = separation(rows, key, direction)
        results = []
        for thr in metric_grid(rows, key):
            fm = metric_fire_map(rows, key, direction, thr)
            m = eval_gate(rows, fm)
            m["thr"] = thr
            m["key"] = key
            m["matches_D"] = (set(m["fire"]) == D_fire_set)
            m["fires_all_gainers"] = all(
                r["dataset"] in set(m["fire"])
                for r in rows if r["dataset"] != SOLE_REGRESSOR)
            results.append(m)
        sweep[key] = results
        cleans = [m for m in results if m["clean"]]
        if cleans:
            # prefer: reproduces gate D, then larger firing set, then higher mean.
            best_clean[key] = max(
                cleans, key=lambda m: (m["matches_D"], len(m["fire"]),
                                       m["mean_pub"] + m["mean_prv"]))
        else:
            best_clean[key] = None
        # gate at the separation midpoint (exists cleanly iff clean_sep).
        if sep[key]["clean_sep"]:
            s = sep[key]
            mid = (s["t16"] + s["boundary"]) / 2.0
            fm = metric_fire_map(rows, key, direction, mid)
            mm = eval_gate(rows, fm)
            mm["thr"] = mid
            mm["matches_D"] = (set(mm["fire"]) == D_fire_set)
            midpoint[key] = mm

    # ---- INVARIANT: non-firing datasets identical to base (delta exactly 0) --
    invariant_violations = []
    for label, fm in (("C_nocap", ref_fire["C_nocap"]),
                      ("D_exceptpn", ref_fire["D_exceptpn"])):
        for r in rows:
            if not fm[r["dataset"]]:
                if applied_delta(r, "pub", False) != 0.0 or \
                   applied_delta(r, "prv", False) != 0.0:
                    invariant_violations.append((label, r["dataset"]))
    for key, _dir, _desc in METRICS:
        for m in sweep[key]:
            fireset = set(m["fire"])
            for r in rows:
                if r["dataset"] not in fireset:
                    if applied_delta(r, "pub", False) != 0.0 or \
                       applied_delta(r, "prv", False) != 0.0:
                        invariant_violations.append(
                            (f"{key}@{m['thr']:.5g}", r["dataset"]))

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
    repro_lines.append(
        f"gate G_rep R=1  : Public {a['mean_pub']:+.5f} (ref +{REF_G_R1[0]:.5f}), "
        f"Private {a['mean_prv']:+.5f} (ref +{REF_G_R1[1]:.5f}), nFire={len(a['fire'])} "
        f"(ref {REF_G_R1_NFIRE}), drops={sorted(a_drops)} "
        f"(ref {sorted(REF_G_R1_DROPS)}) -> {'YES' if anchor_ok else 'NO'}")
    repro_ok = repro_ok and anchor_ok

    # ---------------------------- build summary ----------------------------
    S = []

    def line(x=""):
        S.append(x)

    # (0) per-dataset in-sample metric table ------------------------------
    line("=== KEY QUESTION 1: IN-SAMPLE METRICS PER DATASET (no test labels) ===")
    line("  seed_var / seed_disagree : K=5 seed-prediction spread on the internal")
    line("       holdout (HIGH => seeds disagree => averaging should help).")
    line("  ll_impr / br_impr : in-sample log-loss / Brier improvement of avg over")
    line("       seed0 (HIGH/positive => averaging helps in-sample).")
    line("  hAUC : seed0 holdout AUC level (HIGH => saturated => averaging can't help).")
    line("  truth = does K=5 averaging TRULY help on the real test (ground truth).")
    line("  * = train_16 (the sole real-test regressor we need to EXCLUDE).")
    hdr = (f"{'dataset':<10} {'nTr':>6} {'obj':>4} {'seed_var':>10} "
           f"{'seed_dis':>9} {'ll_impr':>10} {'br_impr':>10} {'hAUC':>7} "
           f"{'d_pub':>9} {'d_prv':>9} {'truth':>6}")
    line(hdr)
    for r in rows:
        star = " *" if r["dataset"] == SOLE_REGRESSOR else ""
        line(f"{r['dataset']:<10} {r['n_train']:>6} {r['n_object_cols']:>4} "
             f"{r['seed_var']:>10.3e} {r['seed_disagree']:>9.3e} "
             f"{r['logloss_impr']:>+10.3e} {r['brier_impr']:>+10.3e} "
             f"{r['holdout_auc']:>7.4f} {r['d_pub']:>+9.5f} {r['d_prv']:>+9.5f} "
             f"{r['test_gain']:>6}{star}")

    # (1) SEPARATION analysis (KEY QUESTION 2) ----------------------------
    line("")
    line("=== KEY QUESTION 2: does a metric SEPARATE train_16 from all 15 gainers? ===")
    line("  For a HIGH-fires metric, a clean gate needs train_16 < min(gainers).")
    line("  For a LOW-fires  metric, a clean gate needs train_16 > max(gainers).")
    line("  margin > 0 => a midpoint threshold reproduces gate D exactly "
         "(fires all 15, excludes train_16).")
    sephdr = (f"{'metric':<14} {'dir':>4} {'train_16':>12} {'gainer_min':>12} "
              f"{'gainer_max':>12} {'margin':>12} {'clean_sep':>10}")
    line(sephdr)
    for key, direction, desc in METRICS:
        s = sep[key]
        line(f"{key:<14} {direction:>4} {s['t16']:>12.4e} {s['gmin']:>12.4e} "
             f"{s['gmax']:>12.4e} {s['margin']:>+12.4e} "
             f"{'YES' if s['clean_sep'] else 'no':>10}")
    line("")
    for key, direction, desc in METRICS:
        s = sep[key]
        if direction == "ge":
            rel = (f"train_16={s['t16']:.4e} vs gainer MIN={s['gmin']:.4e} "
                   f"(needs train_16 BELOW gainer min)")
        else:
            rel = (f"train_16={s['t16']:.4e} vs gainer MAX={s['gmax']:.4e} "
                   f"(needs train_16 ABOVE gainer max)")
        verdict = ("CLEAN SEPARATION" if s["clean_sep"]
                   else "NO clean separation")
        line(f"  {key:<14} ({desc}):")
        line(f"      {rel}; margin={s['margin']:+.4e} -> {verdict}")

    # (2) reference gates C, D --------------------------------------------
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

    # (3) best CLEAN gate per metric (KEY QUESTION 3) ---------------------
    line("")
    line("=== KEY QUESTION 3: best CLEAN in-sample gate per metric ===")
    line("  clean = mean>0 on BOTH splits AND zero regressions (=> train_16 NOT firing).")
    line("  A metric REPRODUCES gate D iff its best clean gate fires exactly the 15 gainers.")
    bhdr = (f"{'metric':<14} {'thr':>12} {'nFire':>5} {'meanPub':>9} {'meanPrv':>9} "
            f"{'Pub W/L/T':>9} {'Prv W/L/T':>9} {'exc16':>5} {'allGain':>7} {'=D?':>4}")
    line(bhdr)
    for key, direction, desc in METRICS:
        bc = best_clean[key]
        if bc is None:
            line(f"{key:<14} {'--':>12} {'0':>5}  (no clean gate exists for this metric)")
            continue
        line(f"{key:<14} {bc['thr']:>12.4e} {len(bc['fire']):>5} "
             f"{bc['mean_pub']:>+9.5f} {bc['mean_prv']:>+9.5f} "
             f"{bc['wlt_pub'][0]}/{bc['wlt_pub'][1]}/{bc['wlt_pub'][2]:<3} "
             f"{bc['wlt_prv'][0]}/{bc['wlt_prv'][1]}/{bc['wlt_prv'][2]:<3} "
             f"{'YES' if bc['excludes_16'] else 'no':>5} "
             f"{'YES' if bc['fires_all_gainers'] else 'no':>7} "
             f"{'YES' if bc['matches_D'] else 'no':>4}")
    line("")
    line("  best clean gate firing set per metric:")
    for key, direction, desc in METRICS:
        bc = best_clean[key]
        if bc is None:
            line(f"    {key:<14}: (none)")
        else:
            excl = sorted({r['dataset'] for r in rows} - set(bc['fire']))
            line(f"    {key:<14}: fires {len(bc['fire'])}, excludes {excl}")

    # (3b) midpoint (clean-separation) gates ------------------------------
    line("")
    line("=== SEPARATION-MIDPOINT GATES (only metrics with clean_sep) ===")
    any_mid = False
    for key, direction, desc in METRICS:
        if key in midpoint:
            any_mid = True
            mm = midpoint[key]
            line(f"  {key:<14}: midpoint thr={mm['thr']:.4e} fires "
                 f"({len(mm['fire'])}) -> mean Public {mm['mean_pub']:+.5f} / "
                 f"Private {mm['mean_prv']:+.5f}  excludes_16="
                 f"{'YES' if mm['excludes_16'] else 'NO'}  matches_D="
                 f"{'YES' if mm['matches_D'] else 'NO'}")
    if not any_mid:
        line("  NONE — no in-sample metric cleanly separates train_16 from every "
             "gainer, so no midpoint gate reproduces gate D.")

    # (4) full per-metric threshold sweeps --------------------------------
    line("")
    line("=== KEY QUESTION 3 (full): per-metric FULL threshold sweep ===")
    for key, direction, desc in METRICS:
        op = ">=" if direction == "ge" else "<="
        line(f"--- {key} (fire iff metric {op} thr) : {desc} ---")
        line(f"{'thr':>12} {'nFire':>5} {'meanPub':>9} {'meanPrv':>9} "
             f"{'Pub W/L/T':>9} {'Prv W/L/T':>9} {'exc16':>5} {'allGain':>7} "
             f"{'clean':>5} {'=D?':>4}")
        for m in sweep[key]:
            line(f"{m['thr']:>12.4e} {len(m['fire']):>5} {m['mean_pub']:>+9.5f} "
                 f"{m['mean_prv']:>+9.5f} "
                 f"{m['wlt_pub'][0]}/{m['wlt_pub'][1]}/{m['wlt_pub'][2]:<3} "
                 f"{m['wlt_prv'][0]}/{m['wlt_prv'][1]}/{m['wlt_prv'][2]:<3} "
                 f"{'YES' if m['excludes_16'] else 'no':>5} "
                 f"{'YES' if m['fires_all_gainers'] else 'no':>7} "
                 f"{'YES' if m['clean'] else 'no':>5} "
                 f"{'YES' if m['matches_D'] else 'no':>4}")

    # (5) INVARIANT --------------------------------------------------------
    line("")
    line("=== INVARIANT (non-firing datasets byte-identical to base, delta 0) ===")
    if invariant_violations:
        line("VIOLATED!")
        for lbl, ds in invariant_violations:
            line(f"  {lbl}/{ds}")
    else:
        line("OK: for every reference and swept in-sample gate, each non-firing "
             "dataset reuses the exact seed-0 base array (applied delta exactly "
             "0 on both splits). PASS.")

    # (6) VERDICT ----------------------------------------------------------
    line("")
    line("=== VERDICT (angle h) ===")
    clean_sep_metrics = [k for k, _d, _s in METRICS if sep[k]["clean_sep"]]
    repro_D_metrics = [
        k for k, _d, _s in METRICS
        if best_clean[k] is not None and best_clean[k]["matches_D"]]
    line(f"  gate C (robust ref): Public {mC['mean_pub']:+.5f}  Private "
         f"{mC['mean_prv']:+.5f}  (fires {len(mC['fire'])})")
    line(f"  gate D (n=1 caveat): Public {mD['mean_pub']:+.5f}  Private "
         f"{mD['mean_prv']:+.5f}  (fires {len(mD['fire'])}, excludes only train_16)")
    line(f"  in-sample metrics that CLEANLY separate train_16 from all 15 gainers: "
         f"{clean_sep_metrics or 'NONE'}")
    line(f"  in-sample metrics whose best clean gate REPRODUCES gate D: "
         f"{repro_D_metrics or 'NONE'}")

    angle_success = bool(repro_D_metrics)
    if angle_success:
        k0 = repro_D_metrics[0]
        line("")
        line(f"VERDICT: ANGLE (h) SUCCEEDS. In-sample metric '{k0}' yields a CLEAN "
             f"gate that reproduces gate D (fires all 15 gainers, excludes train_16, "
             f"Public {mD['mean_pub']:+.5f} / Private {mD['mean_prv']:+.5f}) using a "
             "training-time signal — NO test labels and NO single-dataset "
             "n_train/n_object dependence. train_16's harm is mechanistically "
             f"flagged by its '{k0}' value separating from every gainer.")
    else:
        # characterize how each metric fails.
        near = min(METRICS, key=lambda kd: -sep[kd[0]]["margin"])
        line("")
        line("VERDICT: ANGLE (h) FAILS (valid NEGATIVE result). No in-sample metric "
             "(seed-prediction variance, seed disagreement, in-sample log-loss or "
             "Brier improvement, or holdout-AUC saturation) cleanly separates "
             "train_16 from all 15 real-test gainers, so none reproduces gate D "
             "mechanistically. train_16's real-test harm is NOT accompanied by a "
             "distinctive in-sample signature at K=5: on the internal holdout its "
             "seed-averaging looks like a gainer (its holdout lift even goes "
             "positive, cf. round37), so training-time behaviour does not foresee "
             "its unique test-time regression. gate C (fire iff n_object_cols>0, "
             "+0.00363/+0.00316) remains the ROBUST ship recommendation; gate D's "
             "edge stays justified only by the single-dataset (train_16) "
             "observation, not by any in-sample mechanism.")

    # one-line mechanistic takeaway.
    line("")
    s16 = {k: sep[k] for k, _d, _s in METRICS}
    line("  MECHANISTIC TAKEAWAY: "
         + ("train_16 is in-sample-distinguishable via " + repro_D_metrics[0]
            if angle_success else
            "train_16's seed-averaging is in-sample INDISTINGUISHABLE from the "
            "gainers at K=5 — its holdout seed_var/disagreement/loss-improvement/"
            "AUC all fall inside the gainer cloud, so its test-time harm has no "
            "training-time tell."))

    # (7) CLEAN RUN marker -------------------------------------------------
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
                  "seed_var", "seed_disagree", "logloss_impr", "brier_impr",
                  "holdout_auc", "liftR1", "liftR10",
                  "base_pub", "base_prv", "avg_pub", "avg_prv",
                  "d_pub", "d_prv", "test_gain", "fires_C", "fires_D"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({
                "dataset": r["dataset"], "n_train": r["n_train"],
                "n_object_cols": r["n_object_cols"], "l2_fired": r["l2_fired"],
                "msl": r["msl"],
                "seed_var": r["seed_var"], "seed_disagree": r["seed_disagree"],
                "logloss_impr": r["logloss_impr"], "brier_impr": r["brier_impr"],
                "holdout_auc": r["holdout_auc"],
                "liftR1": r["liftR1"], "liftR10": r["liftR10"],
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
