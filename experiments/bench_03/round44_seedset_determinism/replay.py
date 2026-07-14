#!/usr/bin/env python
"""
bench_03 round44 — GATE-C SEED-SET DETERMINISM, K fixed at 10, holdout fixed at
base default. OFFLINE ONLY. No subprocess, no LLM, no Kaggle, no network. Calls
sklearn in-process only. Writes ONLY under this round44 directory; never
touches submissions/.

GOAL (offline exploration angle "(m): seed-selection determinism")
--------------------------------------------------------------
round43 (round43_gateC_khigh) established that gate-C seed-averaging with
K=10 (seeds 0..9) gains +0.00434 Public / +0.00398 Private mean ΔAUC vs
base-08, and is the round40/round43 "knee" pick. That result is anchored on
ONE specific 10-seed set: {0,...,9}. round44 asks whether that gain is an
artifact of picking seeds 0..9 specifically, or whether it is stable across
*which* 10 seeds are averaged — i.e. it characterizes the seed-set noise
floor of the ship recommendation (gate C x K=10).

  Fixed axis:  K = 10 (ten models averaged per fired dataset).
  Fixed axis:  holdout_fraction = 0.10 (validation_fraction UNSET, base-08
               default) for EVERY fit, exactly as round40/round43.
  Swept axis:  WHICH 10 seeds are averaged (5 seed SETS, see below).

base-08 DEFAULT holdout (identical to round40/round43)
-------------------------------------------------------
base-08 (git HEAD:submissions/08_ratio_tiered_msl/agent/prompts/system.md,
the actual shipped path -- NOT "08_2gate", which does not exist in this
repo; verified via `git ls-files submissions/08*`) fits
    HistGradientBoostingClassifier(..., early_stopping=True, ...)
and NEVER sets validation_fraction, so it uses sklearn's DEFAULT
validation_fraction = 0.1. round44 therefore fits EVERY seed with
validation_fraction UNSET (the base-08 default) -- there is no hf change
anywhere in this round.

BASE recipe reproduced (== shipped 08), identical to round40/round43:
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
08, round40's base, and round43's base). This is the delta reference for
every seed-set candidate.

CANDIDATES: 5 seed SETS, each K=10, gate C fires (n_object_cols>0) ->
mean of predict_proba over the 10 random_state values in that set, each fit
with validation_fraction UNSET; does NOT fire (obj=0) -> exact seed-0 base
(byte-identical, delta 0) for every set.

  SET_A (anchor): seeds {0,1,2,3,4,5,6,7,8,9}   -- == round43's K10 exactly.
  SET_B:          seeds {10,11,...,19}
  SET_C:          seeds {20,21,...,29}
  SET_D:          seeds {100,101,...,109}
  SET_E (spread): seeds {3,17,42,58,71,99,123,200,314,777}

Gate C firing set (obj>0): train_01,02,03,05,06,07,08,09,12,13,14,15  (12).
Non-firing (obj=0): train_04,10,11,16 -> every SET == base on these
(delta 0) for ALL 5 sets -- this is the obj=0 invariant, checked below.

EFFICIENCY / caching: the union of all seeds referenced by any of the 5 sets
is computed once; for each of the 12 fired datasets, each unique seed in
that union is fit EXACTLY ONCE (validation_fraction UNSET) and its
predict_proba is cached; every SET's mean is then formed purely by
averaging cached vectors for its 10 member seeds -- NO refits, and no seed is
ever fit twice for the same dataset even though it may appear in >1 set
(seed 3 in SET_A and SET_E; seed 17 in SET_B and SET_E). seed-0 IS the base
column on fired datasets (reused, no double fit). Non-fired datasets fit
seed-0 once as base and reuse it for every set.

REPRODUCTION anchor (recomputed here, NOT hardcode-trusted): SET_A's mean
ΔAUC (Public/Private) must match round43's K10 mean ΔAUC to < 5e-6. The
reference means are re-derived at full precision from round43's results.csv
(cand_K10_d_pub / cand_K10_d_prv columns), so the check is a genuine
cross-round reproduction, not a comparison against a 5-decimal printout.
round43's coarse target (K10: +0.00434/+0.00398) is printed for context.

HEADLINE OUTPUT: the SEED-SET SPREAD -- max-min across the 5 sets' mean
ΔPub and mean ΔPrv (the overall noise floor of the gate-C x K=10 ship
recommendation), plus the max per-dataset spread across sets. If the spread
is small relative to the +0.0043 base gain, the K10 gate-C recommendation is
seed-set-robust; if the spread is comparable to or larger than the gain, the
round40/round43 result is largely a specific-seed-set artifact.
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
OUT_DIR = os.path.join(BENCH_DIR, "round44_seedset_determinism")
STATS_CSV = os.path.join(BENCH_DIR, "dataset_stats.csv")
ROUND43_RESULTS = os.path.join(BENCH_DIR, "round43_gateC_khigh", "results.csv")

L2_GATE_THRESHOLD = 0.010   # shipped-08 ratio gate for l2 (FIXED)
GATED_L2 = 1.0
DEFAULT_MSL = 20
MSL_TIERS = [(0.030, 70), (0.015, 50)]
MIN_OBJECT_COLS = 0         # gate C fires iff n_object_cols > 0
BASE_SEED = 0
N_DATASETS = 16
K = 10                      # FIXED -- this round sweeps WHICH seeds, not K.

# seed SETS to compare (all K=10, gate C, holdout fixed at base-08 default).
SEED_SETS = {
    "SET_A": list(range(0, 10)),
    "SET_B": list(range(10, 20)),
    "SET_C": list(range(20, 30)),
    "SET_D": list(range(100, 110)),
    "SET_E": [3, 17, 42, 58, 71, 99, 123, 200, 314, 777],
}
SET_NAMES = list(SEED_SETS.keys())
for _sname, _seeds in SEED_SETS.items():
    assert len(_seeds) == K, f"{_sname} must have exactly K={K} seeds"

# union of every seed referenced by any set -> fit each ONCE per fired dataset.
ALL_SEEDS = sorted(set(s for seeds in SEED_SETS.values() for s in seeds))

REF_SET = "SET_A"           # must reproduce round43's K10 exactly.
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


def round43_ref_means():
    """Re-derive round43's K10 mean ΔAUC at full precision from round43's
    results.csv, to anchor SET_A's reproduction check. Returns dict keyed by
    split ('pub'/'prv') -> mean delta, ignoring blank/nan cells."""
    if not os.path.exists(ROUND43_RESULTS):
        return None
    cols = {"pub": "cand_K10_d_pub", "prv": "cand_K10_d_prv"}
    acc = {s: [] for s in cols}
    with open(ROUND43_RESULTS, newline="") as f:
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
    BASE and every SET candidate.

    base = seed-0 with validation_fraction UNSET (== shipped 08). A dataset
    that fires (obj>0) is fit ONCE per unique seed in ALL_SEEDS
    (validation_fraction UNSET) and cached; seed-0 IS the base column
    (reused, no double fit); each SET averages its 10 member seeds from the
    cache (no refits, even for seeds shared across sets, e.g. 3 and 17). A
    dataset that does not fire (obj=0) reuses the exact seed-0 base for
    every SET (byte-identical)."""
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

    if not fires:
        # obj=0 -> every SET identical to base.
        for sname in SET_NAMES:
            preds[sname] = base_map
    else:
        # fit each unique seed in ALL_SEEDS ONCE (vf UNSET); reuse base_vec
        # for seed 0 instead of refitting it.
        seed_cache = {BASE_SEED: base_vec}
        for s in ALL_SEEDS:
            if s == BASE_SEED:
                continue
            seed_cache[s] = fit_one_seed(train, test, features, cat_mask, l2, msl_val, s)
            n_fits += 1
        # sanity: cached seed-0 is literally the base vector (reused object).
        base_is_seed0 = bool(np.array_equal(seed_cache[BASE_SEED], base_vec))
        for sname, seeds in SEED_SETS.items():
            avg = np.mean(np.vstack([seed_cache[s] for s in seeds]), axis=0)
            preds[sname] = dict(zip(row_ids, avg.tolist()))

    meta = {
        "n_train": n_train_stat,
        "n_object_cols": n_obj_stat,
        "fires": bool(fires),
        "l2": l2,
        "msl": msl_val,
        "n_fits": n_fits,
        "base_is_seed0": base_is_seed0,
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
    ref43 = round43_ref_means()
    rows = []
    exceptions = []
    skipped = []
    total_fits = 0
    base_seed0_flags = []

    ALL_CONFIGS = ["base"] + SET_NAMES

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
                        "fires": False, "msl": float("nan"), "base_is_seed0": None})
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

    # ---- results.csv ----
    csv_path = os.path.join(OUT_DIR, "results.csv")
    fieldnames = ["dataset", "n_train", "n_object_cols", "fires", "msl",
                  "base_is_seed0", "base_pub", "base_prv"]
    for sname in SET_NAMES:
        fieldnames += [f"{sname}_pub", f"{sname}_d_pub", f"{sname}_prv", f"{sname}_d_prv"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            out = {k: r.get(k, "") for k in
                   ["dataset", "n_train", "n_object_cols", "fires", "msl",
                    "base_is_seed0", "base_pub", "base_prv"]}
            for sname in SET_NAMES:
                out[f"{sname}_pub"] = r.get(f"{sname}_pub", "")
                out[f"{sname}_prv"] = r.get(f"{sname}_prv", "")
                out[f"{sname}_d_pub"] = delta(r, sname, "pub")
                out[f"{sname}_d_prv"] = delta(r, sname, "prv")
            w.writerow(out)

    # ---- INVARIANT: non-firing (obj=0) datasets must be byte-identical to
    #      base for EVERY seed SET (delta exactly 0). ----
    invariant_violations = []
    for r in rows:
        if r.get("fires"):
            continue
        for sname in SET_NAMES:
            dp = delta(r, sname, "pub")
            dv = delta(r, sname, "prv")
            dp = 0.0 if math.isnan(dp) else dp
            dv = 0.0 if math.isnan(dv) else dv
            if dp != 0.0 or dv != 0.0:
                invariant_violations.append((sname, r["dataset"], dp, dv))

    # ---- firing-set check ----
    fired = {r["dataset"] for r in rows if r.get("fires")}
    fire_ok = (fired == EXPECTED_FIRE)
    obj0_excluded = not (OBJ0_NAMES & fired)

    # ---- base==seed0 confirmation (per fired dataset) ----
    base_seed0_ok = bool(base_seed0_flags) and all(base_seed0_flags)

    # ---- sweep (per-set) means ----
    sweep = {}
    for sname in SET_NAMES:
        mp, mv = mean_delta(sname, "pub"), mean_delta(sname, "prv")
        wp, lp, tp = wlt(sname, "pub")
        wv, lv, tv = wlt(sname, "prv")
        sweep[sname] = {"mp": mp, "mv": mv, "pub_wlt": (wp, lp, tp),
                         "prv_wlt": (wv, lv, tv),
                         "regs_pub": regressions(sname, "pub"),
                         "regs_prv": regressions(sname, "prv")}

    # ---- reproduction check (SET_A recomputed vs round43 full-precision K10) ----
    repro = {}
    repro_ok = True
    if ref43 is None:
        repro_available = False
    else:
        repro_available = True
        mp, mv = sweep[REF_SET]["mp"], sweep[REF_SET]["mv"]
        rp, rv = ref43.get("pub"), ref43.get("prv")
        okp = (rp is not None) and (not math.isnan(rp)) and abs(mp - rp) < REPRO_TOL
        okv = (rv is not None) and (not math.isnan(rv)) and abs(mv - rv) < REPRO_TOL
        repro[REF_SET] = {"mp": mp, "mv": mv, "rp": rp, "rv": rv,
                           "okp": okp, "okv": okv}
        repro_ok = okp and okv

    # ---- SEED-SET SPREAD (headline) ----
    mean_pubs = [sweep[s]["mp"] for s in SET_NAMES]
    mean_prvs = [sweep[s]["mv"] for s in SET_NAMES]
    spread_pub = max(mean_pubs) - min(mean_pubs)
    spread_prv = max(mean_prvs) - min(mean_prvs)
    best_set_pub = max(SET_NAMES, key=lambda s: sweep[s]["mp"])
    worst_set_pub = min(SET_NAMES, key=lambda s: sweep[s]["mp"])
    best_set_prv = max(SET_NAMES, key=lambda s: sweep[s]["mv"])
    worst_set_prv = min(SET_NAMES, key=lambda s: sweep[s]["mv"])

    # per-dataset spread across sets (fired datasets only; obj=0 spread is 0
    # by the invariant and included for completeness).
    per_dataset_spread = {}
    for r in rows:
        ds = r["dataset"]
        dp_vals = [delta(r, s, "pub") for s in SET_NAMES]
        dv_vals = [delta(r, s, "prv") for s in SET_NAMES]
        dp_vals = [v for v in dp_vals if not math.isnan(v)]
        dv_vals = [v for v in dv_vals if not math.isnan(v)]
        sp = (max(dp_vals) - min(dp_vals)) if dp_vals else float("nan")
        sv = (max(dv_vals) - min(dv_vals)) if dv_vals else float("nan")
        per_dataset_spread[ds] = (sp, sv)
    fired_spreads_pub = [per_dataset_spread[d][0] for d in per_dataset_spread
                          if d in fired and not math.isnan(per_dataset_spread[d][0])]
    fired_spreads_prv = [per_dataset_spread[d][1] for d in per_dataset_spread
                          if d in fired and not math.isnan(per_dataset_spread[d][1])]
    max_ds_spread_pub = max(fired_spreads_pub) if fired_spreads_pub else float("nan")
    max_ds_spread_prv = max(fired_spreads_prv) if fired_spreads_prv else float("nan")
    max_ds_spread_pub_name = (max(fired, key=lambda d: per_dataset_spread[d][0])
                               if fired_spreads_pub else None)
    max_ds_spread_prv_name = (max(fired, key=lambda d: per_dataset_spread[d][1])
                               if fired_spreads_prv else None)

    # ================= SUMMARY =================
    L = []
    L.append("=" * 78)
    L.append("bench_03 round44 — GATE-C SEED-SET DETERMINISM (K=10 fixed, holdout "
             "fixed at base-08 default 0.10)  [OFFLINE]")
    L.append("=" * 78)
    L.append("")
    L.append("SETUP:")
    L.append("  base == shipped 08 (git HEAD:submissions/08_ratio_tiered_msl/agent/"
             "prompts/system.md): HGB, early_stopping, validation_fraction UNSET ->")
    L.append("    sklearn default 0.10. base column = seed-0, vf UNSET, for all 16")
    L.append("    datasets.")
    L.append("  Gate C: fires iff n_object_cols>0 (12 categorical datasets); pure-")
    L.append("    numeric (obj=0) datasets stay single seed-0 == base (delta 0) for")
    L.append("    every seed SET.")
    L.append("  Sweep: WHICH 10 seeds are averaged (K fixed at 10); holdout FIXED")
    L.append("    (no hf change, vf UNSET for every fit).")
    L.append("  Seed sets:")
    for sname in SET_NAMES:
        L.append(f"    {sname}: {{{','.join(str(s) for s in SEED_SETS[sname])}}}"
                  + ("  <- anchor, == round43 K10" if sname == REF_SET else ""))
    L.append(f"  Union of unique seeds fit per fired dataset: {len(ALL_SEEDS)} "
             f"({min(ALL_SEEDS)}..{max(ALL_SEEDS)}, non-contiguous).")

    # ---- SWEEP TABLE ----
    L.append("")
    L.append("=== SWEEP TABLE (each seed SET vs base == shipped 08, K=10) ===")
    L.append(f"{'setting':<8} {'meanDPub':>10} {'meanDPrv':>10} "
             f"{'Pub W/L/T':>12} {'Prv W/L/T':>12}")
    for sname in SET_NAMES:
        s = sweep[sname]
        wp, lp, tp = s["pub_wlt"]
        wv, lv, tv = s["prv_wlt"]
        tag = "  <- anchor (must == round43 K10)" if sname == REF_SET else ""
        L.append(f"{sname:<8} {s['mp']:>+10.5f} {s['mv']:>+10.5f} "
                 f"{f'{wp}/{lp}/{tp}':>12} {f'{wv}/{lv}/{tv}':>12}{tag}")

    # ---- REPRODUCTION ----
    L.append("")
    L.append("=== REPRODUCTION CHECK (SET_A recomputed here vs round43 K10, tol<5e-6) ===")
    if not repro_available:
        L.append("  round43 results.csv NOT found -> reproduction NOT anchored (FAIL).")
    else:
        rr = repro[REF_SET]
        L.append(
            f"  {REF_SET}: Public {rr['mp']:+.6f} vs round43 K10 {rr['rp']:+.6f} "
            f"(|d|={abs(rr['mp']-rr['rp']):.2e}, {'YES' if rr['okp'] else 'NO'}); "
            f"Private {rr['mv']:+.6f} vs round43 K10 {rr['rv']:+.6f} "
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
    L.append("=== INVARIANT (obj=0 datasets identical to base for every seed SET) ===")
    if invariant_violations:
        L.append("  VIOLATED!")
        for sname, ds, dp, dv in invariant_violations:
            L.append(f"    {sname}/{ds}: pub d={dp:+.6g} prv d={dv:+.6g}")
    else:
        L.append(f"  OK: each of the {len(OBJ0_NAMES)} obj=0 datasets "
                 f"{sorted(OBJ0_NAMES)} is byte-identical to base (delta 0) "
                 f"across all {len(SET_NAMES)} seed sets. PASS.")

    # ---- PER-DATASET Public / Private ΔAUC across seed SETS ----
    for split, tag in (("pub", "Public"), ("prv", "Private")):
        L.append("")
        L.append(f"=== PER-DATASET ΔAUC across seed SETS ({tag}) — base + "
                 f"{{{','.join(SET_NAMES)}}} ===")
        header = f"{'dataset':<10} {'obj':>4} {'base':>8}"
        for sname in SET_NAMES:
            header += f" {sname:>8} {'d':>9}"
        header += f" {'spread':>8}"
        L.append(header)
        for r in rows:
            line = (f"{r['dataset']:<10} {str(r.get('n_object_cols')):>4} "
                    f"{r[f'base_{split}']:>8.4f}")
            for sname in SET_NAMES:
                line += (f" {r[f'{sname}_{split}']:>8.4f} "
                         f"{delta(r, sname, split):>+9.5f}")
            sp = per_dataset_spread[r["dataset"]][0 if split == "pub" else 1]
            line += f" {sp:>8.5f}" if not math.isnan(sp) else f" {'--':>8}"
            L.append(line)

    # ---- SEED-SET SPREAD (headline) ----
    L.append("")
    L.append("=== SEED-SET SPREAD (headline: is the K10 gate-C gain seed-set-robust?) ===")
    L.append(f"  mean ΔPub across sets: " +
             ", ".join(f"{sname}={sweep[sname]['mp']:+.5f}" for sname in SET_NAMES))
    L.append(f"  mean ΔPrv across sets: " +
             ", ".join(f"{sname}={sweep[sname]['mv']:+.5f}" for sname in SET_NAMES))
    L.append(f"  Public  spread (max-min of mean ΔPub): {spread_pub:.6f} "
             f"(max={best_set_pub} {sweep[best_set_pub]['mp']:+.5f}, "
             f"min={worst_set_pub} {sweep[worst_set_pub]['mp']:+.5f})")
    L.append(f"  Private spread (max-min of mean ΔPrv): {spread_prv:.6f} "
             f"(max={best_set_prv} {sweep[best_set_prv]['mv']:+.5f}, "
             f"min={worst_set_prv} {sweep[worst_set_prv]['mv']:+.5f})")
    L.append(f"  Max per-dataset spread across sets: Public {max_ds_spread_pub:.5f} "
             f"({max_ds_spread_pub_name}), Private {max_ds_spread_prv:.5f} "
             f"({max_ds_spread_prv_name})")
    ref_mp, ref_mv = sweep[REF_SET]["mp"], sweep[REF_SET]["mv"]
    ratio_pub = (spread_pub / ref_mp) if abs(ref_mp) > 1e-12 else float("nan")
    ratio_prv = (spread_prv / ref_mv) if abs(ref_mv) > 1e-12 else float("nan")
    L.append(f"  Spread as a fraction of the SET_A (round43 K10) gain: "
             f"Public {ratio_pub*100:.1f}%, Private {ratio_prv*100:.1f}%")

    # ---- ADOPTION-STYLE CHECK: does every set stay a clean win vs base? ----
    L.append("")
    L.append("=== PER-SET CLEAN-WIN CHECK (mean D > 0 AND zero regressions, both splits) ===")
    all_clean = True
    for sname in SET_NAMES:
        s = sweep[sname]
        zero_regs = (not s["regs_pub"]) and (not s["regs_prv"])
        clean = (s["mp"] > 1e-9) and (s["mv"] > 1e-9) and zero_regs
        if not clean:
            all_clean = False
        regstr = ""
        if not zero_regs:
            allr = s["regs_pub"] + s["regs_prv"]
            regstr = " regs[" + ", ".join(f"{n}({d:+.5f})" for n, d in allr) + "]"
        L.append(f"  {sname}: pub{s['mp']:+.5f} prv{s['mv']:+.5f}  "
                 f"{'clean-win' if clean else 'NOT-clean'}{regstr}")
    L.append(f"  ALL {len(SET_NAMES)} seed sets clean wins vs base: "
             f"{'YES' if all_clean else 'NO'}")

    # ---- VERDICT ----
    L.append("")
    L.append("=== VERDICT ===")
    L.append("KEY QUESTION: is the K10 gate-C seed-averaging gain (+0.0043 Pub / "
             "+0.0040 Prv vs base-08, from round40/round43) an artifact of the "
             "specific seed set {0..9}, or stable across which 10 seeds are chosen?")
    L.append("")
    if repro_available and repro_ok:
        L.append(f"  SET_A reproduces round43's K10 exactly (|d|<{REPRO_TOL:.0e} on both "
                 f"splits) -- correctness anchor holds.")
    else:
        L.append("  WARNING: SET_A did NOT reproduce round43's K10 within tolerance "
                 "-- treat downstream spread numbers with caution.")
    L.append(f"  Every one of the 5 seed sets is a clean win vs base-08: "
             f"{'YES' if all_clean else 'NO'} (mean D>0 both splits, zero regressions).")
    L.append(f"  Seed-set spread: Public {spread_pub:.6f} ({ratio_pub*100:.1f}% of the "
             f"SET_A gain), Private {spread_prv:.6f} ({ratio_prv*100:.1f}% of the SET_A "
             f"gain).")
    if (not math.isnan(ratio_pub) and not math.isnan(ratio_prv)
            and ratio_pub < 0.5 and ratio_prv < 0.5 and all_clean):
        L.append("  ANSWER: the gain is STABLE across seed sets -- spread is a minority "
                 "of the base gain and every set remains a clean win. The K10 gate-C "
                 "ship recommendation is NOT seed-set-dependent within the sets tested.")
    elif all_clean:
        L.append("  ANSWER: the gain direction is robust (every set wins cleanly) but "
                 "the MAGNITUDE varies non-trivially across seed sets -- the exact "
                 "reported gain (+0.0043/+0.0040) should be read as one draw from a "
                 "noisy distribution, not an exact constant.")
    else:
        L.append("  ANSWER: seed-set choice materially changes the outcome -- at least "
                 "one seed set is NOT a clean win vs base-08. The K10 gate-C gain is "
                 "seed-set-DEPENDENT, not a robust property of averaging 10 seeds.")

    # ---- CLEAN RUN marker ----
    clean_run = ((not exceptions) and (not invariant_violations) and fire_ok
                 and obj0_excluded and repro_ok and repro_available
                 and base_seed0_ok and (not skipped))
    L.append("")
    L.append(f"CLEAN RUN={'YES' if clean_run else 'NO'} "
             f"(total_fits={total_fits}, exceptions={len(exceptions)}, "
             f"skipped={len(skipped)}, "
             f"invariant_violations={len(invariant_violations)}, "
             f"firing_set_match={'YES' if fire_ok else 'NO'}, "
             f"obj0_excluded={'YES' if obj0_excluded else 'NO'}, "
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
