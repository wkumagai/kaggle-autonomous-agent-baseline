#!/usr/bin/env python
"""
bench_03 round46 — GATE-C SEED-AVERAGING: AGGREGATION FUNCTION x K INTERACTION.
OFFLINE ONLY. No subprocess, no LLM, no Kaggle, no network. Calls sklearn
in-process only. Writes ONLY under this round46 directory; never touches
submissions/.

GOAL (offline exploration angle: "aggregation function x K interaction")
------------------------------------------------------------------------
round45 (round45_agg_function) fixed K=10 (seeds 0..9), gate C, base-08, and
swept three aggregators {mean, median, trim10} over the K per-seed proba
vectors. It concluded arithmetic MEAN is the ship aggregator: median loses on
Private, trim10 ties within noise. The open cross-section round45 could NOT
answer: does that aggregator ranking INTERACT with K? At small K a trimmed mean
drops proportionally more mass (drop 1 of 5 = 20% vs drop 1 of 20 = 5%); at
large K robust aggregators are more stable. So this round sweeps the full
aggregator x K grid.

  Fixed axis:  base model = base-08 (shipped 08 HGB recipe, vf UNSET -> 0.10).
  Fixed axis:  gate = gate C (fires iff n_object_cols>0; the 12 categorical
               datasets fire, the 4 pure-numeric datasets stay single-seed).
  Fixed axis:  holdout = base-08 default (validation_fraction UNSET -> 0.10).
  Swept axis 1: AGGREGATION in {mean, median, trim10} over the K per-seed proba
               vectors (applied per test-row).
  Swept axis 2: K in {5, 10, 20}; for each K, seeds = {0,1,...,K-1}.

base-08 DEFAULT recipe reproduced (== shipped 08), identical to round40/43/44/45:
  features = [c for c in train.columns if c not in ("row_id","target")]
  cat_mask = [train[c].dtype == object for c in features]
  n=len(train); ratio = len(features)/n
  l2  = 1.0 if ratio >= 0.010 else 0.0
  msl = 70 if ratio>=0.030 else 50 if ratio>=0.015 else 20
  HistGradientBoostingClassifier(categorical_features=cat_mask, random_state=s,
      max_iter=300, early_stopping=True, l2_regularization=l2,
      min_samples_leaf=msl)   # validation_fraction NOT set -> default 0.1
  pred = predict_proba(test)[:, class==1]
BASE column = seed-0 with validation_fraction UNSET (byte-identical to shipped
08, round40/43/44/45 base). This is the delta reference for every (agg, K).

AGGREGATOR DEFINITIONS (per test-row, over the K per-seed proba values):
  mean   = arithmetic mean of the K values.
  median = per-row median of the K values.
  trim10 = "drop one each end" trimmed mean: drop the SINGLE min AND SINGLE max,
           average the middle (K-2). This is EXACTLY round45's K=10 trim10
           definition, and it is held FIXED at every K (drop 1 min + 1 max) so
           the aggregator is comparable across K -- deliberately NOT a strict
           percentage that would round to 0 trims at K=5.

Gate C firing set (obj>0): train_01,02,03,05,06,07,08,09,12,13,14,15  (12).
Non-firing (obj=0): train_04,10,11,16 -> every (agg, K) == base on these (delta
exactly 0) -- the obj=0 invariant, checked below.

EFFICIENCY / caching: for each fired dataset, seeds 0..19 are each fit EXACTLY
ONCE (validation_fraction UNSET) and their predict_proba cached; every (agg, K)
result is then computed purely from the SAME cache (NO refits). For a given K
the aggregator sees seeds {0..K-1} (a prefix of the cache). seed-0 IS the base
column on fired datasets (reused, no double fit). Non-fired (obj=0) datasets fit
seed-0 once as base and reuse it for every (agg, K).

Expected fits: 12 fired x 20 seeds + 4 obj0 x 1 seed-0 = 244.

REPRODUCTION anchor (recomputed here, NOT hardcode-trusted): the (mean, K=10)
cell's mean dAUC (Public/Private) must match round44/round45 SET_A mean dAUC
(itself == round43 K10) to < 5e-6. round44's SET_A_d_pub/SET_A_d_prv columns are
re-derived at full precision from round44's results.csv, so the check is a
genuine cross-round reproduction (target ~ Public +0.004343 / Private +0.003984).

HEADLINE OUTPUT: for each K in {5,10,20}, does median or trim10 CLEANLY beat mean
at that K -- i.e. a higher mean dAUC on BOTH Public AND Private with ZERO
per-dataset regressions vs mean at the SAME K? And does the mean-vs-robust
ranking CHANGE with K? If no robust aggregator cleanly beats mean at any K,
`mean` stays the ship aggregator for gate-C seed-averaging at every K.
"""
import os
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
OUT_DIR = os.path.join(BENCH_DIR, "round46_agg_x_k")
STATS_CSV = os.path.join(BENCH_DIR, "dataset_stats.csv")
ROUND44_RESULTS = os.path.join(BENCH_DIR, "round44_seedset_determinism", "results.csv")

L2_GATE_THRESHOLD = 0.010   # shipped-08 ratio gate for l2 (FIXED)
GATED_L2 = 1.0
DEFAULT_MSL = 20
MSL_TIERS = [(0.030, 70), (0.015, 50)]
MIN_OBJECT_COLS = 0         # gate C fires iff n_object_cols > 0
BASE_SEED = 0
N_DATASETS = 16

# ---- the two swept axes ----
AGG_NAMES = ["mean", "median", "trim10"]     # aggregation over K per-seed vectors
K_VALUES = [5, 10, 20]                        # K sweep; seeds = 0..K-1
MAX_K = max(K_VALUES)                         # 20 -> fit seeds 0..19 once each
ALL_SEEDS = list(range(0, MAX_K))             # {0..19} fit exactly once per fired ds


def agg_mean(stack):
    return np.mean(stack, axis=0)


def agg_median(stack):
    return np.median(stack, axis=0)


def agg_trim10(stack):
    # "drop one each end" trimmed mean: per test-row (per column of the K x n_test
    # stack), drop the SINGLE min AND SINGLE max, mean the middle (K-2). Held
    # fixed at every K so it stays comparable (never rounds to 0 trims).
    s = np.sort(stack, axis=0)
    return np.mean(s[1:-1, :], axis=0)


AGG_FUNCS = {"mean": agg_mean, "median": agg_median, "trim10": agg_trim10}

# every (agg, K) config, plus a stable label helper.
CONFIGS = [(agg, K) for agg in AGG_NAMES for K in K_VALUES]


def cfg_label(agg, K):
    return f"{agg}_K{K}"


CONFIG_LABELS = [cfg_label(a, k) for (a, k) in CONFIGS]

REF_AGG, REF_K = "mean", 10          # (mean, K=10) must reproduce round44 SET_A.
REF_LABEL = cfg_label(REF_AGG, REF_K)
REPRO_TOL = 5e-6
REF43_K10_PUB, REF43_K10_PRV = 0.00434, 0.00398   # coarse context only

EXPECTED_FIRE = {"train_01", "train_02", "train_03", "train_05", "train_06",
                 "train_07", "train_08", "train_09", "train_12", "train_13",
                 "train_14", "train_15"}
OBJ0_NAMES = {"train_04", "train_10", "train_11", "train_16"}

# expected fits: 12 fired x 20 seeds + 4 obj0 x 1 seed-0
EXPECTED_TOTAL_FITS = len(EXPECTED_FIRE) * MAX_K + len(OBJ0_NAMES) * 1


def load_stats(path=STATS_CSV):
    stats = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            stats[row["name"]] = {
                "n_train": int(row["n_train"]),
                "n_object_cols": int(row["n_object_cols"]),
            }
    return stats


def round44_setA_ref_means():
    """Re-derive round44's SET_A mean dAUC at full precision from round44's
    results.csv (SET_A == K=10, seeds 0..9, mean aggregator, == round43 K10), to
    anchor the (mean, K=10) cell's reproduction check. Returns dict keyed by
    split ('pub'/'prv') -> mean delta, ignoring blank/nan cells."""
    if not os.path.exists(ROUND44_RESULTS):
        return None
    cols = {"pub": "SET_A_d_pub", "prv": "SET_A_d_prv"}
    acc = {s: [] for s in cols}
    with open(ROUND44_RESULTS, newline="") as f:
        for row in csv.DictReader(f):
            for s, col in cols.items():
                v = row.get(col, "")
                if v is None or v == "":
                    continue
                try:
                    fv = float(v)
                except ValueError:
                    continue
                if not math.isnan(fv):
                    acc[s].append(fv)
    return {s: (sum(vals) / len(vals)) if vals else float("nan")
            for s, vals in acc.items()}


def gate_fires(n_object_cols):
    return n_object_cols > MIN_OBJECT_COLS


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


def fit_one_seed(train, test, features, cat_mask, l2, msl_val, seed):
    """Fit ONE shipped-08 HGB with validation_fraction left UNSET (sklearn
    default 0.1, byte-identical to shipped 08 / base-08). All other
    hyperparameters are byte-identical to shipped 08; only random_state (the
    averaged seed) varies."""
    clf = HistGradientBoostingClassifier(
        categorical_features=cat_mask,
        random_state=seed,
        max_iter=300,
        early_stopping=True,
        l2_regularization=l2,
        min_samples_leaf=msl_val,
    )
    clf.fit(train[features], train["target"])
    proba = clf.predict_proba(test[features])
    classes = list(clf.classes_)
    pos_idx = classes.index(1) if 1 in classes else 1
    return proba[:, pos_idx]


def run_one(name, train_csv, test_csv, stats):
    """Returns (preds, meta). preds maps config_label -> {row_id -> prob} for
    BASE and every (agg, K).

    base = seed-0 with validation_fraction UNSET (== shipped 08). A dataset that
    fires (obj>0) is fit ONCE per seed in {0..19} (validation_fraction UNSET) and
    cached; seed-0 IS the base column (reused, no double fit); every (agg, K)
    combines the first K cached vectors (seeds 0..K-1) per test-row (no refits).
    A dataset that does not fire (obj=0) reuses the exact seed-0 base for every
    (agg, K) (byte-identical)."""
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)

    features = [c for c in train.columns if c not in ("row_id", "target")]
    cat_mask = [train[c].dtype == object for c in features]

    n = len(train)
    ratio = len(features) / n
    l2 = GATED_L2 if ratio >= L2_GATE_THRESHOLD else 0.0
    msl_val = msl_for_ratio(ratio)

    st = stats[name]
    n_train_stat = st["n_train"]
    n_obj_stat = st["n_object_cols"]
    fires = gate_fires(n_obj_stat)

    row_ids = test["row_id"].tolist()

    # base = seed-0, validation_fraction UNSET (byte-identical to shipped 08).
    base_vec = fit_one_seed(train, test, features, cat_mask, l2, msl_val, BASE_SEED)
    base_map = dict(zip(row_ids, base_vec.tolist()))
    preds = {"base": base_map}
    n_fits = 1
    base_is_seed0 = None
    obj0_identical = None   # for non-firing: every (agg,K) byte-identical to base?

    if not fires:
        # obj=0 -> every (agg, K) identical to base (same seed-0 map object).
        for lbl in CONFIG_LABELS:
            preds[lbl] = base_map
        obj0_identical = True
    else:
        # fit each seed in {0..19} ONCE (vf UNSET); reuse base_vec for seed 0.
        seed_cache = {BASE_SEED: base_vec}
        for s in ALL_SEEDS:
            if s == BASE_SEED:
                continue
            seed_cache[s] = fit_one_seed(train, test, features, cat_mask, l2, msl_val, s)
            n_fits += 1
        # sanity: cached seed-0 is literally the base vector (reused object).
        base_is_seed0 = bool(np.array_equal(seed_cache[BASE_SEED], base_vec))
        # build every (agg, K) purely from the cache -- no refits.
        for agg, K in CONFIGS:
            seeds_K = list(range(0, K))                     # seeds 0..K-1
            stack = np.vstack([seed_cache[s] for s in seeds_K])   # (K, n_test)
            agg_vec = AGG_FUNCS[agg](stack)
            preds[cfg_label(agg, K)] = dict(zip(row_ids, agg_vec.tolist()))

    meta = {
        "n_train": n_train_stat,
        "n_object_cols": n_obj_stat,
        "fires": bool(fires),
        "l2": l2,
        "msl": msl_val,
        "n_fits": n_fits,
        "base_is_seed0": base_is_seed0,
        "obj0_identical": obj0_identical,
    }
    return preds, meta


def score_split(pred_map, sol):
    sol = sol.copy()
    sol["pred"] = sol["row_id"].map(pred_map)
    if sol["pred"].isna().any():
        raise ValueError(f"{int(sol['pred'].isna().sum())} row_ids unmatched")
    pub = sol[sol["Usage"] == "Public"]
    prv = sol[sol["Usage"] == "Private"]
    return auc_or_nan(pub["target"], pub["pred"]), auc_or_nan(prv["target"], prv["pred"])


def main():
    warnings.filterwarnings("ignore")
    os.makedirs(OUT_DIR, exist_ok=True)

    stats = load_stats()
    ref44 = round44_setA_ref_means()
    rows = []
    exceptions = []
    skipped = []
    total_fits = 0
    base_seed0_flags = []

    ALL_CONFIGS = ["base"] + CONFIG_LABELS

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
                "fires": meta["fires"],
                "msl": meta["msl"],
                "base_is_seed0": meta["base_is_seed0"],
                "obj0_identical": meta["obj0_identical"],
            })
            if meta["base_is_seed0"] is not None:
                base_seed0_flags.append(meta["base_is_seed0"])
            for cfg in ALL_CONFIGS:
                pub, prv = score_split(preds[cfg], sol)
                rec[f"{cfg}_pub"] = pub
                rec[f"{cfg}_prv"] = prv
            print(f"[OK] {name} n_tr={meta['n_train']} obj={meta['n_object_cols']} "
                  f"fires={meta['fires']} msl={meta['msl']} fits={meta['n_fits']} "
                  f"base pub={rec['base_pub']:.6f} prv={rec['base_prv']:.6f}")
        except Exception as e:
            exceptions.append((name, repr(e)))
            rec.update({"n_train": stats.get(name, {}).get("n_train", ""),
                        "n_object_cols": stats.get(name, {}).get("n_object_cols", ""),
                        "fires": False, "msl": float("nan"),
                        "base_is_seed0": None, "obj0_identical": None})
            for cfg in ALL_CONFIGS:
                rec[f"{cfg}_pub"] = float("nan")
                rec[f"{cfg}_prv"] = float("nan")
            print(f"[ERROR] {name}: {e!r}")
        rows.append(rec)

    # ---- delta helpers (all vs base == shipped 08) ----
    def delta(rec, lbl, split):
        b = rec.get(f"base_{split}")
        c = rec.get(f"{lbl}_{split}")
        if b is None or c is None or math.isnan(b) or math.isnan(c):
            return float("nan")
        return c - b

    def mean_delta(lbl, split):
        vals = [delta(r, lbl, split) for r in rows]
        vals = [v for v in vals if not math.isnan(v)]
        return sum(vals) / len(vals) if vals else float("nan")

    def wlt(lbl, split, eps=1e-6):
        w = l = t = 0
        for r in rows:
            dd = delta(r, lbl, split)
            if math.isnan(dd):
                continue
            if dd > eps:
                w += 1
            elif dd < -eps:
                l += 1
            else:
                t += 1
        return w, l, t

    # ---- per-(agg,K) delta VS MEAN AT THE SAME K (does a robust agg beat mean?) ----
    def delta_vs_mean(rec, agg, K, split):
        m = rec.get(f"{cfg_label('mean', K)}_{split}")
        c = rec.get(f"{cfg_label(agg, K)}_{split}")
        if m is None or c is None or math.isnan(m) or math.isnan(c):
            return float("nan")
        return c - m

    def wlt_vs_mean(agg, K, split, eps=1e-6):
        w = l = t = 0
        for r in rows:
            dd = delta_vs_mean(r, agg, K, split)
            if math.isnan(dd):
                continue
            if dd > eps:
                w += 1
            elif dd < -eps:
                l += 1
            else:
                t += 1
        return w, l, t

    def regressions_vs_mean(agg, K, split, eps=1e-6):
        out = []
        for r in rows:
            dd = delta_vs_mean(r, agg, K, split)
            if not math.isnan(dd) and dd < -eps:
                out.append((r["dataset"], dd))
        return out

    # ---- results.csv (per-dataset per-(agg,K) dAUC on both splits) ----
    csv_path = os.path.join(OUT_DIR, "results.csv")
    base_cols = ["dataset", "n_train", "n_object_cols", "fires", "msl",
                 "base_is_seed0", "obj0_identical", "base_pub", "base_prv"]
    fieldnames = list(base_cols)
    for lbl in CONFIG_LABELS:
        fieldnames += [f"{lbl}_pub", f"{lbl}_d_pub", f"{lbl}_prv", f"{lbl}_d_prv"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            out = {k: r.get(k, "") for k in base_cols}
            for lbl in CONFIG_LABELS:
                out[f"{lbl}_pub"] = r.get(f"{lbl}_pub", "")
                out[f"{lbl}_prv"] = r.get(f"{lbl}_prv", "")
                out[f"{lbl}_d_pub"] = delta(r, lbl, "pub")
                out[f"{lbl}_d_prv"] = delta(r, lbl, "prv")
            w.writerow(out)

    # ---- INVARIANT: non-firing (obj=0) datasets must be byte-identical to base
    #      for EVERY (agg, K) (delta exactly 0). ----
    invariant_violations = []
    for r in rows:
        if r.get("fires"):
            continue
        for lbl in CONFIG_LABELS:
            dp = delta(r, lbl, "pub")
            dv = delta(r, lbl, "prv")
            dp = 0.0 if math.isnan(dp) else dp
            dv = 0.0 if math.isnan(dv) else dv
            if dp != 0.0 or dv != 0.0:
                invariant_violations.append((lbl, r["dataset"], dp, dv))
    obj0_identical_ok = all(
        (r.get("obj0_identical") is True) for r in rows if not r.get("fires")
    ) and any(not r.get("fires") for r in rows)

    # ---- firing-set check ----
    fired = {r["dataset"] for r in rows if r.get("fires")}
    fire_ok = (fired == EXPECTED_FIRE)
    obj0_excluded = not (OBJ0_NAMES & fired)

    # ---- base==seed0 confirmation (per fired dataset) ----
    base_seed0_ok = bool(base_seed0_flags) and all(base_seed0_flags)

    # ---- sweep (per (agg, K)) means vs base ----
    sweep = {}
    for agg, K in CONFIGS:
        lbl = cfg_label(agg, K)
        mp, mv = mean_delta(lbl, "pub"), mean_delta(lbl, "prv")
        wp, lp, tp = wlt(lbl, "pub")
        wv, lv, tv = wlt(lbl, "prv")
        sweep[(agg, K)] = {"mp": mp, "mv": mv, "pub_wlt": (wp, lp, tp),
                           "prv_wlt": (wv, lv, tv)}

    # ---- reproduction check ((mean,K10) recomputed vs round44 SET_A full prec) ----
    repro = {}
    repro_ok = True
    if ref44 is None:
        repro_available = False
    else:
        repro_available = True
        mp, mv = sweep[(REF_AGG, REF_K)]["mp"], sweep[(REF_AGG, REF_K)]["mv"]
        rp, rv = ref44.get("pub"), ref44.get("prv")
        okp = (rp is not None) and (not math.isnan(rp)) and abs(mp - rp) < REPRO_TOL
        okv = (rv is not None) and (not math.isnan(rv)) and abs(mv - rv) < REPRO_TOL
        repro = {"mp": mp, "mv": mv, "rp": rp, "rv": rv, "okp": okp, "okv": okv}
        repro_ok = okp and okv

    # ---- HEAD-TO-HEAD vs mean AT EACH K (does median/trim10 cleanly beat mean?) ----
    head = {}   # keyed (agg, K)
    for K in K_VALUES:
        for agg in ["median", "trim10"]:
            dmp = mean_delta(cfg_label(agg, K), "pub") - mean_delta(cfg_label("mean", K), "pub")
            dmv = mean_delta(cfg_label(agg, K), "prv") - mean_delta(cfg_label("mean", K), "prv")
            wp, lp, tp = wlt_vs_mean(agg, K, "pub")
            wv, lv, tv = wlt_vs_mean(agg, K, "prv")
            regs_p = regressions_vs_mean(agg, K, "pub")
            regs_v = regressions_vs_mean(agg, K, "prv")
            clean_beat = (dmp > 1e-9) and (dmv > 1e-9) and (not regs_p) and (not regs_v)
            head[(agg, K)] = {"dmp": dmp, "dmv": dmv,
                              "pub_wlt": (wp, lp, tp), "prv_wlt": (wv, lv, tv),
                              "regs_pub": regs_p, "regs_prv": regs_v,
                              "clean_beat": clean_beat}

    # ================= SUMMARY =================
    L = []
    L.append("=" * 78)
    L.append("bench_03 round46 — GATE-C SEED-AVG: AGGREGATION FUNCTION x K "
             "INTERACTION  [OFFLINE]")
    L.append("=" * 78)
    L.append("")
    L.append("SETUP:")
    L.append("  base == shipped 08 (git HEAD:submissions/08_ratio_tiered_msl/agent/"
             "prompts/system.md): HGB, early_stopping, validation_fraction UNSET ->")
    L.append("    sklearn default 0.10. base column = seed-0, vf UNSET, for all 16")
    L.append("    datasets.")
    L.append("  Gate C: fires iff n_object_cols>0 (12 categorical datasets); pure-")
    L.append("    numeric (obj=0) datasets stay single seed-0 == base (delta 0) for")
    L.append("    every (agg, K).")
    L.append(f"  Swept axis 1 (aggregator): {{{','.join(AGG_NAMES)}}}.")
    L.append(f"  Swept axis 2 (K): {{{','.join(str(k) for k in K_VALUES)}}}; "
             f"for each K, seeds = {{0..K-1}}.")
    L.append(f"  Fit ONCE per fired dataset: seeds {{0..{MAX_K-1}}} (K={MAX_K}); "
             f"every (agg,K) built from the cache prefix, no refits.")
    L.append("  Aggregators (per test-row over the K per-seed proba values):")
    L.append("    mean   = arithmetic mean of the K values")
    L.append("    median = per-test-row median of the K values")
    L.append("    trim10 = drop SINGLE min AND SINGLE max, mean middle (K-2)")
    L.append("             (held fixed 'drop one each end' at EVERY K -> "
             "comparable, never 0 trims)")

    # ---- SWEEP TABLE (agg x K, vs base) ----
    L.append("")
    L.append("=== SWEEP TABLE: aggregator x K, each cell vs base == shipped 08 ===")
    L.append(f"{'agg':<8} {'K':>3} {'meanDPub':>10} {'meanDPrv':>10} "
             f"{'Pub W/L/T':>12} {'Prv W/L/T':>12}")
    for agg in AGG_NAMES:
        for K in K_VALUES:
            s = sweep[(agg, K)]
            wp, lp, tp = s["pub_wlt"]
            wv, lv, tv = s["prv_wlt"]
            tag = "  <- anchor (== round44 SET_A)" if (agg == REF_AGG and K == REF_K) else ""
            L.append(f"{agg:<8} {K:>3} {s['mp']:>+10.5f} {s['mv']:>+10.5f} "
                     f"{f'{wp}/{lp}/{tp}':>12} {f'{wv}/{lv}/{tv}':>12}{tag}")

    # ---- SWEEP MATRIX: meanDPub / meanDPrv by aggregator (rows) x K (cols) ----
    L.append("")
    L.append("=== MATRIX meanDPub  (rows=agg, cols=K) ===")
    L.append(f"{'agg':<8}" + "".join(f"{('K'+str(k)):>12}" for k in K_VALUES))
    for agg in AGG_NAMES:
        L.append(f"{agg:<8}" + "".join(f"{sweep[(agg,k)]['mp']:>+12.5f}" for k in K_VALUES))
    L.append("")
    L.append("=== MATRIX meanDPrv  (rows=agg, cols=K) ===")
    L.append(f"{'agg':<8}" + "".join(f"{('K'+str(k)):>12}" for k in K_VALUES))
    for agg in AGG_NAMES:
        L.append(f"{agg:<8}" + "".join(f"{sweep[(agg,k)]['mv']:>+12.5f}" for k in K_VALUES))

    # ---- REPRODUCTION ----
    L.append("")
    L.append("=== REPRODUCTION CHECK ((mean,K=10) recomputed here vs round44 SET_A, tol<5e-6) ===")
    if not repro_available:
        L.append("  round44 results.csv NOT found -> reproduction NOT anchored (FAIL).")
    else:
        L.append(
            f"  (mean,K10): Public {repro['mp']:+.6f} vs round44 SET_A {repro['rp']:+.6f} "
            f"(|d|={abs(repro['mp']-repro['rp']):.2e}, {'YES' if repro['okp'] else 'NO'}); "
            f"Private {repro['mv']:+.6f} vs round44 SET_A {repro['rv']:+.6f} "
            f"(|d|={abs(repro['mv']-repro['rv']):.2e}, {'YES' if repro['okv'] else 'NO'})")
        L.append(f"  round43 coarse context: K10 ~ +{REF43_K10_PUB:.5f}/+{REF43_K10_PRV:.5f}"
                 f" (printed, not asserted)")
        L.append(f"  REPRODUCTION: {'PASS' if repro_ok else 'FAIL'}")

    # ---- FIRING SET / INVARIANT ----
    L.append("")
    L.append("=== GATE-C FIRING SET (fires iff n_object_cols>0) ===")
    L.append(f"  fires on ({len(fired)}): {', '.join(sorted(fired))}")
    L.append(f"  expected 12 categorical matched: {'YES' if fire_ok else 'NO'}"
             + ("" if fire_ok else f" (got {sorted(fired)})"))
    L.append(f"  obj=0 datasets {sorted(OBJ0_NAMES)} excluded: "
             f"{'YES' if obj0_excluded else 'NO'}")
    L.append(f"  base column == seed-0 on all {len(base_seed0_flags)} fired datasets: "
             f"{'YES' if base_seed0_ok else 'NO'}")

    L.append("")
    L.append("=== INVARIANT (obj=0 datasets byte-identical to base for every (agg,K)) ===")
    if invariant_violations:
        L.append("  VIOLATED!")
        for lbl, ds, dp, dv in invariant_violations:
            L.append(f"    {lbl}/{ds}: pub d={dp:+.6g} prv d={dv:+.6g}")
    else:
        L.append(f"  OK: each of the {len(OBJ0_NAMES)} obj=0 datasets "
                 f"{sorted(OBJ0_NAMES)} is byte-identical to base (delta exactly 0) "
                 f"across all {len(CONFIG_LABELS)} (agg,K) configs. PASS.")
        L.append(f"  (obj0 predictions share the seed-0 base object for every "
                 f"(agg,K): {'YES' if obj0_identical_ok else 'NO'})")

    # ---- PER-DATASET Public / Private dAUC across (agg,K) ----
    for split, tag in (("pub", "Public"), ("prv", "Private")):
        L.append("")
        L.append(f"=== PER-DATASET dAUC vs base across (agg,K) ({tag}) ===")
        header = f"{'dataset':<10} {'obj':>4} {'base':>8}"
        for agg in AGG_NAMES:
            for K in K_VALUES:
                header += f" {cfg_label(agg, K):>11}d"
        L.append(header)
        for r in rows:
            line = (f"{r['dataset']:<10} {str(r.get('n_object_cols')):>4} "
                    f"{r[f'base_{split}']:>8.4f}")
            for agg in AGG_NAMES:
                for K in K_VALUES:
                    line += f" {delta(r, cfg_label(agg, K), split):>+12.5f}"
            L.append(line)

    # ---- HEAD-TO-HEAD vs mean AT EACH K ----
    L.append("")
    L.append("=== HEAD-TO-HEAD: median / trim10 vs mean AT EACH K "
             "(does a robust agg beat mean?) ===")
    L.append("  (per-dataset delta = agg_AUC - mean_AUC at the SAME K; W/L/T over "
             "all 16 datasets, obj0 are exact ties)")
    for K in K_VALUES:
        L.append(f"  --- K={K} ---")
        for agg in ["median", "trim10"]:
            h = head[(agg, K)]
            wp, lp, tp = h["pub_wlt"]
            wv, lv, tv = h["prv_wlt"]
            L.append(f"    {agg} vs mean: meanDPub diff {h['dmp']:+.6f}, "
                     f"meanDPrv diff {h['dmv']:+.6f}")
            L.append(f"      Public  W/L/T vs mean = {wp}/{lp}/{tp}; "
                     f"Private W/L/T vs mean = {wv}/{lv}/{tv}")
            if h["regs_pub"] or h["regs_prv"]:
                allr = [("Pub", n, d) for n, d in h["regs_pub"]] + \
                       [("Prv", n, d) for n, d in h["regs_prv"]]
                L.append("      losses vs mean: " +
                         ", ".join(f"{sp}:{n}({d:+.5f})" for sp, n, d in allr))
            L.append(f"      CLEAN BEAT vs mean (higher meanD on BOTH splits, zero "
                     f"per-dataset losses): {'YES' if h['clean_beat'] else 'NO'}")

    # ---- VERDICT ----
    L.append("")
    L.append("=== VERDICT ===")
    L.append("KEY QUESTION: for gate-C seed-averaging, does the aggregator ranking "
             "interact with K? For each K in {5,10,20}, does any robust aggregator "
             "(median / trim10) CLEANLY beat mean (higher mean dAUC on BOTH Public "
             "AND Private with zero per-dataset regressions vs mean at the same K)? "
             "Does the mean-vs-robust ranking change with K?")
    L.append("")
    if repro_available and repro_ok:
        L.append(f"  (mean,K10) reproduces round44 SET_A (== round43 K10) exactly "
                 f"(|d|<{REPRO_TOL:.0e} on both splits) -- correctness anchor holds.")
    else:
        L.append("  WARNING: (mean,K10) did NOT reproduce round44 SET_A within "
                 "tolerance -- treat the aggregator comparison with caution.")

    # best aggregator per K on each split, and clean-beat status per K.
    L.append("")
    L.append("  Best aggregator per K (by mean dAUC vs base):")
    ranking_changes = False
    ref_best_pub = ref_best_prv = None
    for K in K_VALUES:
        best_pub = max(AGG_NAMES, key=lambda a: sweep[(a, K)]["mp"])
        best_prv = max(AGG_NAMES, key=lambda a: sweep[(a, K)]["mv"])
        if ref_best_pub is None:
            ref_best_pub, ref_best_prv = best_pub, best_prv
        else:
            if best_pub != ref_best_pub or best_prv != ref_best_prv:
                ranking_changes = True
        L.append(f"    K={K:>2}: best Public = {best_pub:<7} "
                 f"(Pub{sweep[(best_pub,K)]['mp']:+.5f}); "
                 f"best Private = {best_prv:<7} "
                 f"(Prv{sweep[(best_prv,K)]['mv']:+.5f})")

    L.append("")
    L.append("  CLEAN-BEAT of mean by a robust aggregator, per K:")
    any_clean_beat_any_K = False
    clean_beat_ks = []
    for K in K_VALUES:
        winners = [a for a in ["median", "trim10"] if head[(a, K)]["clean_beat"]]
        if winners:
            any_clean_beat_any_K = True
            clean_beat_ks.append((K, winners))
            L.append(f"    K={K:>2}: {', '.join(winners)} CLEANLY beat(s) mean.")
        else:
            L.append(f"    K={K:>2}: NO robust aggregator cleanly beats mean "
                     f"(mean stays best).")

    L.append("")
    # does mean stay best at every K? (mean not strictly beaten cleanly anywhere)
    mean_best_every_k = not any_clean_beat_any_K
    if mean_best_every_k:
        L.append("  ANSWER: mean STAYS best at every K in {5,10,20} -- no robust "
                 "aggregator (median/trim10) cleanly beats mean at ANY K (none is "
                 "higher on BOTH splits with zero per-dataset regressions vs mean).")
    else:
        parts = "; ".join(f"K={k}: {', '.join(w)}" for k, w in clean_beat_ks)
        L.append(f"  ANSWER: a robust aggregator CLEANLY beats mean at some K "
                 f"({parts}). mean is NOT best at every K.")
    if ranking_changes:
        L.append("  RANKING vs K: the best-aggregator ranking CHANGES across K "
                 "(aggregator x K interaction present).")
    else:
        L.append("  RANKING vs K: the best-aggregator ranking is STABLE across K "
                 "(no material aggregator x K interaction in the winner).")
    L.append("  SHIP RECOMMENDATION: mean remains the gate-C seed-averaging "
             "aggregator unless a robust aggregator cleanly beats it above.")

    # ---- CLEAN RUN marker ----
    fits_ok = (total_fits == EXPECTED_TOTAL_FITS)
    clean_run = ((not exceptions) and (not invariant_violations) and fire_ok
                 and obj0_excluded and repro_ok and repro_available
                 and base_seed0_ok and obj0_identical_ok and (not skipped)
                 and fits_ok)
    L.append("")
    L.append(f"CLEAN RUN={'YES' if clean_run else 'NO'} "
             f"(total_fits={total_fits}, expected_fits={EXPECTED_TOTAL_FITS}, "
             f"fits_match={'YES' if fits_ok else 'NO'}, "
             f"exceptions={len(exceptions)}, "
             f"skipped={len(skipped)}, "
             f"invariant_violations={len(invariant_violations)}, "
             f"firing_set_match={'YES' if fire_ok else 'NO'}, "
             f"obj0_excluded={'YES' if obj0_excluded else 'NO'}, "
             f"reproduction={'YES' if repro_ok else 'NO'}, "
             f"base_eq_seed0={'YES' if base_seed0_ok else 'NO'}, "
             f"obj0_identical={'YES' if obj0_identical_ok else 'NO'})")
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
