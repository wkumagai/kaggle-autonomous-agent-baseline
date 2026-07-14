#!/usr/bin/env python
"""
bench_03 round45 — GATE-C SEED-AVERAGING AGGREGATION FUNCTION, K fixed at 10,
seeds fixed at {0..9}, holdout fixed at base-08 default. OFFLINE ONLY. No
subprocess, no LLM, no Kaggle, no network. Calls sklearn in-process only.
Writes ONLY under this round45 directory; never touches submissions/.

GOAL (offline exploration angle "(n): aggregation function")
--------------------------------------------------------------
Every prior seed-averaging round (round29..round44) combined the K per-seed
predict_proba vectors with the ARITHMETIC MEAN. round43/round44 established
that gate-C seed-averaging with K=10 (seeds 0..9) gains +0.00434 Public /
+0.00398 Private mean ΔAUC vs base-08 -- but always via `mean`. The untested
question: is `mean` the best way to aggregate the K per-seed proba vectors,
or does a ROBUST aggregator (median, or a trimmed mean) beat it?

  Fixed axis:  base model = base-08 (shipped 08 HGB recipe, vf UNSET).
  Fixed axis:  gate = gate C (fires iff n_object_cols>0; the 12 categorical
               datasets fire, the 4 pure-numeric datasets stay single-seed).
  Fixed axis:  K = 10, seeds = {0,1,...,9} (== round43/round44 SET_A exactly).
  Fixed axis:  holdout = base-08 default (validation_fraction UNSET -> 0.10).
  Swept axis:  the AGGREGATION over the K=10 per-seed proba vectors:
                 (a) mean    -- arithmetic mean (reproduces round43/round44 K10)
                 (b) median  -- per-row median of the 10 proba values
                 (c) trim10  -- 10%-trimmed mean: per row drop the min AND max
                               of the 10 proba values, mean the middle 8.

base-08 DEFAULT recipe reproduced (== shipped 08), identical to round40/43/44:
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
08, round40/43/44 base). This is the delta reference for every aggregator.

CANDIDATES: 3 aggregators, each over the SAME cached K=10 per-seed proba
vectors (seeds 0..9). Gate C fires (n_object_cols>0) -> the aggregator is
applied per test-row over the 10 proba vectors. Does NOT fire (obj=0) -> exact
seed-0 base (byte-identical, delta 0) for ALL 3 aggregators.

Gate C firing set (obj>0): train_01,02,03,05,06,07,08,09,12,13,14,15  (12).
Non-firing (obj=0): train_04,10,11,16 -> every aggregator == base on these
(delta exactly 0) -- the obj=0 invariant, checked below.

EFFICIENCY / caching: for each fired dataset, seeds 0..9 are each fit EXACTLY
ONCE (validation_fraction UNSET) and their predict_proba cached; all three
aggregators are then computed purely from the SAME cache (NO refits per
aggregator). seed-0 IS the base column on fired datasets (reused, no double
fit). Non-fired datasets fit seed-0 once as base and reuse it for every
aggregator.

REPRODUCTION anchor (recomputed here, NOT hardcode-trusted): the MEAN
aggregator's mean ΔAUC (Public/Private) must match round44's SET_A mean ΔAUC
(itself == round43 K10) to < 5e-6. round44's SET_A_d_pub/SET_A_d_prv columns
are re-derived at full precision from round44's results.csv, so the check is a
genuine cross-round reproduction, not a comparison against a printed digest.
round43's coarse target (K10: +0.00434/+0.00398) is printed for context.

HEADLINE OUTPUT: does median or trim10 CLEANLY beat mean -- i.e. a higher mean
ΔAUC on BOTH Public AND Private with ZERO per-dataset regressions vs mean? If
so, on which datasets. If neither cleanly beats mean, `mean` stays the ship
aggregator for the gate-C x K=10 recommendation.
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
OUT_DIR = os.path.join(BENCH_DIR, "round45_agg_function")
STATS_CSV = os.path.join(BENCH_DIR, "dataset_stats.csv")
ROUND44_RESULTS = os.path.join(BENCH_DIR, "round44_seedset_determinism", "results.csv")

L2_GATE_THRESHOLD = 0.010   # shipped-08 ratio gate for l2 (FIXED)
GATED_L2 = 1.0
DEFAULT_MSL = 20
MSL_TIERS = [(0.030, 70), (0.015, 50)]
MIN_OBJECT_COLS = 0         # gate C fires iff n_object_cols > 0
BASE_SEED = 0
N_DATASETS = 16
K = 10                      # FIXED -- this round sweeps the aggregator, not K.
SEEDS = list(range(0, K))   # FIXED {0..9} == round43/round44 SET_A.

# aggregators swept over the K=10 per-seed proba vectors (all applied per row).
AGG_NAMES = ["mean", "median", "trim10"]


def agg_mean(stack):
    return np.mean(stack, axis=0)


def agg_median(stack):
    return np.median(stack, axis=0)


def agg_trim10(stack):
    # 10%-trimmed mean: per test-row (per column of the K x n_test stack), drop
    # the single min AND single max of the K=10 values, mean the middle 8.
    s = np.sort(stack, axis=0)
    return np.mean(s[1:-1, :], axis=0)


AGG_FUNCS = {"mean": agg_mean, "median": agg_median, "trim10": agg_trim10}

REF_AGG = "mean"            # must reproduce round44 SET_A (== round43 K10).
REPRO_TOL = 5e-6
REF43_K10_PUB, REF43_K10_PRV = 0.00434, 0.00398   # coarse context only

EXPECTED_FIRE = {"train_01", "train_02", "train_03", "train_05", "train_06",
                 "train_07", "train_08", "train_09", "train_12", "train_13",
                 "train_14", "train_15"}
OBJ0_NAMES = {"train_04", "train_10", "train_11", "train_16"}


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
    """Re-derive round44's SET_A mean ΔAUC at full precision from round44's
    results.csv (SET_A == K=10, seeds 0..9, mean aggregator, == round43 K10),
    to anchor the MEAN aggregator's reproduction check. Returns dict keyed by
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
    hyperparameters are byte-identical to shipped 08; only random_state
    (the averaged seed) varies."""
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
    """Returns (preds, meta). preds maps config_name -> {row_id -> prob} for
    BASE and every aggregator.

    base = seed-0 with validation_fraction UNSET (== shipped 08). A dataset
    that fires (obj>0) is fit ONCE per seed in {0..9} (validation_fraction
    UNSET) and cached; seed-0 IS the base column (reused, no double fit); each
    aggregator combines the SAME 10 cached vectors per test-row (no refits). A
    dataset that does not fire (obj=0) reuses the exact seed-0 base for every
    aggregator (byte-identical)."""
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
    obj0_identical = None   # for non-firing: every agg byte-identical to base?

    if not fires:
        # obj=0 -> every aggregator identical to base (same seed-0 map object).
        for agg in AGG_NAMES:
            preds[agg] = base_map
        # byte-identical invariant holds trivially (same dict); flag True.
        obj0_identical = True
    else:
        # fit each seed in {0..9} ONCE (vf UNSET); reuse base_vec for seed 0.
        seed_cache = {BASE_SEED: base_vec}
        for s in SEEDS:
            if s == BASE_SEED:
                continue
            seed_cache[s] = fit_one_seed(train, test, features, cat_mask, l2, msl_val, s)
            n_fits += 1
        # sanity: cached seed-0 is literally the base vector (reused object).
        base_is_seed0 = bool(np.array_equal(seed_cache[BASE_SEED], base_vec))
        stack = np.vstack([seed_cache[s] for s in SEEDS])   # shape (K, n_test)
        for agg in AGG_NAMES:
            agg_vec = AGG_FUNCS[agg](stack)
            preds[agg] = dict(zip(row_ids, agg_vec.tolist()))

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

    ALL_CONFIGS = ["base"] + AGG_NAMES

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
    def delta(rec, cfg, split):
        b = rec.get(f"base_{split}")
        c = rec.get(f"{cfg}_{split}")
        if b is None or c is None or math.isnan(b) or math.isnan(c):
            return float("nan")
        return c - b

    def mean_delta(cfg, split):
        vals = [delta(r, cfg, split) for r in rows]
        vals = [v for v in vals if not math.isnan(v)]
        return sum(vals) / len(vals) if vals else float("nan")

    def wlt(cfg, split, eps=1e-6):
        w = l = t = 0
        for r in rows:
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

    def regressions(cfg, split, eps=1e-6):
        return [(r["dataset"], delta(r, cfg, split)) for r in rows
                if not math.isnan(delta(r, cfg, split)) and delta(r, cfg, split) < -eps]

    # per-aggregator delta VS MEAN (does a robust agg beat mean on a dataset?)
    def delta_vs_mean(rec, cfg, split):
        m = rec.get(f"mean_{split}")
        c = rec.get(f"{cfg}_{split}")
        if m is None or c is None or math.isnan(m) or math.isnan(c):
            return float("nan")
        return c - m

    def wlt_vs_mean(cfg, split, eps=1e-6):
        w = l = t = 0
        for r in rows:
            dd = delta_vs_mean(r, cfg, split)
            if math.isnan(dd):
                continue
            if dd > eps:
                w += 1
            elif dd < -eps:
                l += 1
            else:
                t += 1
        return w, l, t

    def regressions_vs_mean(cfg, split, eps=1e-6):
        return [(r["dataset"], delta_vs_mean(r, cfg, split)) for r in rows
                if not math.isnan(delta_vs_mean(r, cfg, split))
                and delta_vs_mean(r, cfg, split) < -eps]

    # ---- results.csv ----
    csv_path = os.path.join(OUT_DIR, "results.csv")
    fieldnames = ["dataset", "n_train", "n_object_cols", "fires", "msl",
                  "base_is_seed0", "obj0_identical", "base_pub", "base_prv"]
    for agg in AGG_NAMES:
        fieldnames += [f"{agg}_pub", f"{agg}_d_pub", f"{agg}_prv", f"{agg}_d_prv"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            out = {k: r.get(k, "") for k in
                   ["dataset", "n_train", "n_object_cols", "fires", "msl",
                    "base_is_seed0", "obj0_identical", "base_pub", "base_prv"]}
            for agg in AGG_NAMES:
                out[f"{agg}_pub"] = r.get(f"{agg}_pub", "")
                out[f"{agg}_prv"] = r.get(f"{agg}_prv", "")
                out[f"{agg}_d_pub"] = delta(r, agg, "pub")
                out[f"{agg}_d_prv"] = delta(r, agg, "prv")
            w.writerow(out)

    # ---- INVARIANT: non-firing (obj=0) datasets must be byte-identical to
    #      base for EVERY aggregator (delta exactly 0). ----
    invariant_violations = []
    for r in rows:
        if r.get("fires"):
            continue
        for agg in AGG_NAMES:
            dp = delta(r, agg, "pub")
            dv = delta(r, agg, "prv")
            dp = 0.0 if math.isnan(dp) else dp
            dv = 0.0 if math.isnan(dv) else dv
            if dp != 0.0 or dv != 0.0:
                invariant_violations.append((agg, r["dataset"], dp, dv))
    obj0_identical_ok = all(
        (r.get("obj0_identical") is True) for r in rows if not r.get("fires")
    ) and any(not r.get("fires") for r in rows)

    # ---- firing-set check ----
    fired = {r["dataset"] for r in rows if r.get("fires")}
    fire_ok = (fired == EXPECTED_FIRE)
    obj0_excluded = not (OBJ0_NAMES & fired)

    # ---- base==seed0 confirmation (per fired dataset) ----
    base_seed0_ok = bool(base_seed0_flags) and all(base_seed0_flags)

    # ---- sweep (per-aggregator) means vs base ----
    sweep = {}
    for agg in AGG_NAMES:
        mp, mv = mean_delta(agg, "pub"), mean_delta(agg, "prv")
        wp, lp, tp = wlt(agg, "pub")
        wv, lv, tv = wlt(agg, "prv")
        sweep[agg] = {"mp": mp, "mv": mv, "pub_wlt": (wp, lp, tp),
                      "prv_wlt": (wv, lv, tv),
                      "regs_pub": regressions(agg, "pub"),
                      "regs_prv": regressions(agg, "prv")}

    # ---- reproduction check (mean recomputed vs round44 SET_A full precision) ----
    repro = {}
    repro_ok = True
    if ref44 is None:
        repro_available = False
    else:
        repro_available = True
        mp, mv = sweep[REF_AGG]["mp"], sweep[REF_AGG]["mv"]
        rp, rv = ref44.get("pub"), ref44.get("prv")
        okp = (rp is not None) and (not math.isnan(rp)) and abs(mp - rp) < REPRO_TOL
        okv = (rv is not None) and (not math.isnan(rv)) and abs(mv - rv) < REPRO_TOL
        repro[REF_AGG] = {"mp": mp, "mv": mv, "rp": rp, "rv": rv,
                          "okp": okp, "okv": okv}
        repro_ok = okp and okv

    # ---- HEAD-TO-HEAD vs mean (does median / trim10 cleanly beat mean?) ----
    head = {}
    for agg in ["median", "trim10"]:
        dmp = mean_delta(agg, "pub") - mean_delta("mean", "pub")   # meanΔ diff
        dmv = mean_delta(agg, "prv") - mean_delta("mean", "prv")
        wp, lp, tp = wlt_vs_mean(agg, "pub")
        wv, lv, tv = wlt_vs_mean(agg, "prv")
        regs_p = regressions_vs_mean(agg, "pub")
        regs_v = regressions_vs_mean(agg, "prv")
        clean_beat = (dmp > 1e-9) and (dmv > 1e-9) and (not regs_p) and (not regs_v)
        head[agg] = {"dmp": dmp, "dmv": dmv,
                     "pub_wlt": (wp, lp, tp), "prv_wlt": (wv, lv, tv),
                     "regs_pub": regs_p, "regs_prv": regs_v,
                     "clean_beat": clean_beat}

    # ================= SUMMARY =================
    L = []
    L.append("=" * 78)
    L.append("bench_03 round45 — GATE-C SEED-AVG AGGREGATION FUNCTION (K=10 fixed, "
             "seeds {0..9}, holdout base-08 default 0.10)  [OFFLINE]")
    L.append("=" * 78)
    L.append("")
    L.append("SETUP:")
    L.append("  base == shipped 08 (git HEAD:submissions/08_ratio_tiered_msl/agent/"
             "prompts/system.md): HGB, early_stopping, validation_fraction UNSET ->")
    L.append("    sklearn default 0.10. base column = seed-0, vf UNSET, for all 16")
    L.append("    datasets.")
    L.append("  Gate C: fires iff n_object_cols>0 (12 categorical datasets); pure-")
    L.append("    numeric (obj=0) datasets stay single seed-0 == base (delta 0) for")
    L.append("    every aggregator.")
    L.append(f"  Seeds FIXED: {{{','.join(str(s) for s in SEEDS)}}} (K={K}, "
             f"== round43/round44 SET_A).")
    L.append("  Sweep: the AGGREGATION over the 10 per-seed proba vectors:")
    L.append("    mean   = arithmetic mean (reproduces round43/round44 K10)")
    L.append("    median = per-test-row median of the 10 proba values")
    L.append("    trim10 = 10%-trimmed mean: per row drop min AND max, mean middle 8")

    # ---- SWEEP TABLE (vs base) ----
    L.append("")
    L.append("=== SWEEP TABLE (each aggregator vs base == shipped 08, K=10 seeds 0..9) ===")
    L.append(f"{'agg':<8} {'meanDPub':>10} {'meanDPrv':>10} "
             f"{'Pub W/L/T':>12} {'Prv W/L/T':>12}")
    for agg in AGG_NAMES:
        s = sweep[agg]
        wp, lp, tp = s["pub_wlt"]
        wv, lv, tv = s["prv_wlt"]
        tag = "  <- anchor (must == round44 SET_A)" if agg == REF_AGG else ""
        L.append(f"{agg:<8} {s['mp']:>+10.5f} {s['mv']:>+10.5f} "
                 f"{f'{wp}/{lp}/{tp}':>12} {f'{wv}/{lv}/{tv}':>12}{tag}")

    # ---- REPRODUCTION ----
    L.append("")
    L.append("=== REPRODUCTION CHECK (mean recomputed here vs round44 SET_A K10, tol<5e-6) ===")
    if not repro_available:
        L.append("  round44 results.csv NOT found -> reproduction NOT anchored (FAIL).")
    else:
        rr = repro[REF_AGG]
        L.append(
            f"  mean: Public {rr['mp']:+.6f} vs round44 SET_A {rr['rp']:+.6f} "
            f"(|d|={abs(rr['mp']-rr['rp']):.2e}, {'YES' if rr['okp'] else 'NO'}); "
            f"Private {rr['mv']:+.6f} vs round44 SET_A {rr['rv']:+.6f} "
            f"(|d|={abs(rr['mv']-rr['rv']):.2e}, {'YES' if rr['okv'] else 'NO'})")
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
    L.append("=== INVARIANT (obj=0 datasets byte-identical to base for every aggregator) ===")
    if invariant_violations:
        L.append("  VIOLATED!")
        for agg, ds, dp, dv in invariant_violations:
            L.append(f"    {agg}/{ds}: pub d={dp:+.6g} prv d={dv:+.6g}")
    else:
        L.append(f"  OK: each of the {len(OBJ0_NAMES)} obj=0 datasets "
                 f"{sorted(OBJ0_NAMES)} is byte-identical to base (delta exactly 0) "
                 f"across all {len(AGG_NAMES)} aggregators. PASS.")
        L.append(f"  (obj0 predictions share the seed-0 base object for every "
                 f"aggregator: {'YES' if obj0_identical_ok else 'NO'})")

    # ---- PER-DATASET Public / Private ΔAUC across aggregators ----
    for split, tag in (("pub", "Public"), ("prv", "Private")):
        L.append("")
        L.append(f"=== PER-DATASET ΔAUC vs base across aggregators ({tag}) — "
                 f"base + {{{','.join(AGG_NAMES)}}} ===")
        header = f"{'dataset':<10} {'obj':>4} {'base':>8}"
        for agg in AGG_NAMES:
            header += f" {agg:>8} {'d':>9}"
        L.append(header)
        for r in rows:
            line = (f"{r['dataset']:<10} {str(r.get('n_object_cols')):>4} "
                    f"{r[f'base_{split}']:>8.4f}")
            for agg in AGG_NAMES:
                line += (f" {r[f'{agg}_{split}']:>8.4f} "
                         f"{delta(r, agg, split):>+9.5f}")
            L.append(line)

    # ---- HEAD-TO-HEAD vs mean ----
    L.append("")
    L.append("=== HEAD-TO-HEAD: median / trim10 vs mean (does a robust agg beat mean?) ===")
    L.append("  (per-dataset delta = agg_AUC - mean_AUC; W/L/T counts fired+obj0; "
             "obj0 are exact ties)")
    for agg in ["median", "trim10"]:
        h = head[agg]
        wp, lp, tp = h["pub_wlt"]
        wv, lv, tv = h["prv_wlt"]
        L.append(f"  {agg} vs mean: meanΔPub diff {h['dmp']:+.6f}, "
                 f"meanΔPrv diff {h['dmv']:+.6f}")
        L.append(f"    Public  W/L/T vs mean = {wp}/{lp}/{tp}; "
                 f"Private W/L/T vs mean = {wv}/{lv}/{tv}")
        if h["regs_pub"] or h["regs_prv"]:
            allr = [("Pub", n, d) for n, d in h["regs_pub"]] + \
                   [("Prv", n, d) for n, d in h["regs_prv"]]
            L.append("    losses vs mean: " +
                     ", ".join(f"{sp}:{n}({d:+.5f})" for sp, n, d in allr))
        L.append(f"    CLEAN BEAT vs mean (higher meanΔ on BOTH splits, zero "
                 f"per-dataset losses): {'YES' if h['clean_beat'] else 'NO'}")

    # ---- VERDICT ----
    best_pub_agg = max(AGG_NAMES, key=lambda a: sweep[a]["mp"])
    best_prv_agg = max(AGG_NAMES, key=lambda a: sweep[a]["mv"])
    any_clean_beat = any(head[a]["clean_beat"] for a in ["median", "trim10"])
    L.append("")
    L.append("=== VERDICT ===")
    L.append("KEY QUESTION: for gate-C K=10 seed-averaging (seeds 0..9), is the "
             "arithmetic MEAN the best aggregator, or does a robust aggregator "
             "(median / 10%-trimmed mean) beat it?")
    L.append("")
    if repro_available and repro_ok:
        L.append(f"  mean reproduces round44 SET_A (== round43 K10) exactly "
                 f"(|d|<{REPRO_TOL:.0e} on both splits) -- correctness anchor holds.")
    else:
        L.append("  WARNING: mean did NOT reproduce round44 SET_A within tolerance "
                 "-- treat the aggregator comparison with caution.")
    L.append(f"  Mean ΔAUC vs base: " +
             ", ".join(f"{a}=Pub{sweep[a]['mp']:+.5f}/Prv{sweep[a]['mv']:+.5f}"
                       for a in AGG_NAMES))
    L.append(f"  Best mean ΔPub aggregator: {best_pub_agg}; best mean ΔPrv "
             f"aggregator: {best_prv_agg}.")
    if any_clean_beat:
        winners = [a for a in ["median", "trim10"] if head[a]["clean_beat"]]
        L.append(f"  ANSWER: {', '.join(winners)} CLEANLY beat(s) mean (higher mean "
                 f"ΔAUC on BOTH splits AND zero per-dataset regressions vs mean). A "
                 f"robust aggregator improves the gate-C K=10 recommendation.")
    else:
        L.append("  ANSWER: neither median nor trim10 cleanly beats mean (no "
                 "aggregator is higher on BOTH splits with zero per-dataset "
                 "regressions vs mean). MEAN remains the ship aggregator for the "
                 "gate-C x K=10 seed-averaging recommendation.")

    # ---- CLEAN RUN marker ----
    clean_run = ((not exceptions) and (not invariant_violations) and fire_ok
                 and obj0_excluded and repro_ok and repro_available
                 and base_seed0_ok and obj0_identical_ok and (not skipped))
    L.append("")
    L.append(f"CLEAN RUN={'YES' if clean_run else 'NO'} "
             f"(total_fits={total_fits}, exceptions={len(exceptions)}, "
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
