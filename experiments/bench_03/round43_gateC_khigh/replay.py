#!/usr/bin/env python
"""
bench_03 round43 — GATE-C K (seeds-averaged) HIGH SWEEP, holdout FIXED at base default.
OFFLINE ONLY. No subprocess, no LLM, no Kaggle, no network. Calls sklearn
in-process only. Writes ONLY under this round43 directory; never touches
submissions/.

GOAL (offline exploration angle: extend the K sweep beyond 10)
--------------------------------------------------------------
round40 (round40_gateC_ksweep) swept K (number of seed-averaged models) over
{5,6,7,8,10} on GATE C at a FIXED base-08-default holdout, and found K=10 is the
argmax (+0.00434 Pub / +0.00398 Prv vs base-08) with the mean-ΔAUC curve NOT
plateauing by K=10 (the K8->K10 step was still positive on both splits). round43
extends the sweep to

  Swept axis:  K (seeds averaged) in {10, 12, 15, 20}.
  Fixed axis:  holdout_fraction = 0.10 (validation_fraction UNSET, base-08 default).

at the SAME fixed setup, to locate where the K curve PLATEAUS. This refines the
pending seed-averaging ship recommendation: if gate-C seed-avg is approved, what
K should be used? round40 said "at least 10, curve still rising"; round43 asks
"how much more is left above 10, and where does the per-seed gain go flat?"

base-08 DEFAULT holdout (identical to round40)
----------------------------------------------
base-08 (git HEAD:submissions/08_ratio_tiered_msl/agent/prompts/system.md) fits
    HistGradientBoostingClassifier(..., early_stopping=True, ...)
and NEVER sets validation_fraction, so it uses sklearn's DEFAULT
validation_fraction = 0.1. round43 therefore fits EVERY seed with
validation_fraction UNSET (the base-08 default) — there is no hf change anywhere
in this round, exactly as in round40.

BASE recipe reproduced (== shipped 08), identical to round40:
  features = [c for c in train.columns if c not in ("row_id","target")]
  cat_mask = [train[c].dtype == object for c in features]
  n=len(train); ratio = len(features)/n
  l2  = 1.0 if ratio >= 0.010 else 0.0
  msl = 70 if ratio>=0.030 else 50 if ratio>=0.015 else 20
  HistGradientBoostingClassifier(categorical_features=cat_mask, random_state=s,
      max_iter=300, early_stopping=True, l2_regularization=l2,
      min_samples_leaf=msl)   # validation_fraction NOT set -> default 0.1
  pred = predict_proba(test)[:, class==1]
BASE column = seed-0 with validation_fraction UNSET (byte-identical to shipped 08
and to round40's base). This is the ΔAUC reference for every candidate K.

CANDIDATES: for each K in {10,12,15,20}:
  cand_K{K} = gate C fires (n_object_cols>0) -> mean of predict_proba over
              random_state 0..K-1, each fit with validation_fraction UNSET;
              does NOT fire (obj=0) -> exact seed-0 base (byte-identical, delta 0).
Gate C firing set (obj>0): train_01,02,03,05,06,07,08,09,12,13,14,15  (12).
Non-firing (obj=0): train_04,10,11,16 -> every candidate == base on these (delta 0).

EFFICIENCY / caching: for each of the 12 fired datasets, fit seeds 0..19 ONCE
(validation_fraction UNSET) and cache the 20 predict_proba vectors; each K averages
the first K. seed-0 (unset vf) IS the base column on fired datasets (reused, no
double fit). Non-fired datasets fit seed-0 once as base. Total fits = 12*20 + 4 =
244 (== "16 + 12*20" minus the 12 seed-0 base fits reused as seeds[0]).

REPRODUCTION anchor (recomputed here, NOT hardcode-trusted): the K=10 mean deltas
from THIS round must match round40's K10 to < 5e-6. The reference means are
re-derived at full precision from round40's results.csv (cand_K10_d_pub /
cand_K10_d_prv columns), so the check is a genuine cross-round reproduction, not a
comparison against a 5-decimal printout. round40's coarse target (K10:
+0.00434/+0.00398) is printed for context.

Adoption: a K is a CLEAN IMPROVEMENT over base-08 iff mean ΔAUC > 0 on BOTH splits
AND zero regressions on BOTH splits. To beat round40's pick (K10) it must
additionally exceed K10's mean on BOTH splits with zero regressions. The verdict
names the argmax K, whether the K curve has PLATEAUED by K=20 (is K10->K20 still
monotone, and how large per ADDITIONAL SEED), and the effort/gain tradeoff (K =
number of models trained per fired dataset).
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
OUT_DIR = os.path.join(BENCH_DIR, "round43_gateC_khigh")
STATS_CSV = os.path.join(BENCH_DIR, "dataset_stats.csv")
ROUND40_RESULTS = os.path.join(BENCH_DIR, "round40_gateC_ksweep", "results.csv")

L2_GATE_THRESHOLD = 0.010   # shipped-08 ratio gate for l2 (FIXED)
GATED_L2 = 1.0
DEFAULT_MSL = 20
MSL_TIERS = [(0.030, 70), (0.015, 50)]
MIN_OBJECT_COLS = 0         # gate C fires iff n_object_cols > 0
BASE_SEED = 0
N_DATASETS = 16

# sweep axis: K only. holdout FIXED (validation_fraction UNSET = base-08 default).
KS = [10, 12, 15, 20]
MAX_K = max(KS)             # fit seeds 0..MAX_K-1 per fired dataset

BASE = "base"

# round40 K=10 is the reproduction anchor and round43's internal reference (round40
# argmax / current pending seed-avg ship pick). Coarse round40 context (printed,
# NOT asserted — assertion is vs round40's full-precision cand_K10_d_*).
REF40_K10_PUB, REF40_K10_PRV = 0.00434, 0.00398
REPRO_TOL = 5e-6
REF_K = KS[0]               # 10 = round40 argmax, internal "beat" reference

EXPECTED_FIRE = {"train_01", "train_02", "train_03", "train_05", "train_06",
                 "train_07", "train_08", "train_09", "train_12", "train_13",
                 "train_14", "train_15"}
OBJ0_NAMES = {"train_04", "train_10", "train_11", "train_16"}


def cand_name(K):
    return f"cand_K{K}"


CANDIDATES = [cand_name(K) for K in KS]


def load_stats(path=STATS_CSV):
    stats = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            stats[row["name"]] = {
                "n_train": int(row["n_train"]),
                "n_object_cols": int(row["n_object_cols"]),
            }
    return stats


def round40_ref_means():
    """Re-derive round40's K10 mean ΔAUC at full precision from round40's
    results.csv, to anchor reproduction. Returns dict keyed by split
    ('pub'/'prv') -> mean delta, ignoring blank/nan cells."""
    if not os.path.exists(ROUND40_RESULTS):
        return None
    cols = {"pub": "cand_K10_d_pub", "prv": "cand_K10_d_prv"}
    acc = {s: [] for s in cols}
    with open(ROUND40_RESULTS, newline="") as f:
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
    """Fit ONE shipped-08 HGB with validation_fraction left UNSET (sklearn default
    0.1, byte-identical to shipped 08 / base-08). All other hyperparameters are
    byte-identical to shipped 08."""
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
    """Returns (preds, meta). preds maps config_name -> {row_id -> prob} for BASE
    and every K candidate.

    base = seed-0 with validation_fraction UNSET (== shipped 08). A dataset that
    fires (obj>0) is fit for seeds 0..MAX_K-1 ONCE (validation_fraction UNSET) and
    cached; seed-0 IS the base column (reused, no double fit); each candidate
    cand_K{K} averages the first K cached seeds. A dataset that does not fire
    (obj=0) reuses the exact seed-0 base for every candidate (byte-identical)."""
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
    preds = {BASE: base_map}
    n_fits = 1
    base_is_seed0 = None

    if not fires:
        # obj=0 -> every candidate identical to base.
        for c in CANDIDATES:
            preds[c] = base_map
    else:
        # fit seeds 0..MAX_K-1 once (vf UNSET); seed-0 IS base_vec (reuse it).
        seed_vecs = [base_vec]
        for s in range(1, MAX_K):
            seed_vecs.append(
                fit_one_seed(train, test, features, cat_mask, l2, msl_val, s))
            n_fits += 1
        # sanity: seeds[0] is literally the base vector (reused object).
        base_is_seed0 = bool(np.array_equal(seed_vecs[BASE_SEED], base_vec))
        for K in KS:
            avg = np.mean(np.vstack(seed_vecs[:K]), axis=0)
            preds[cand_name(K)] = dict(zip(row_ids, avg.tolist()))

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
    ref40 = round40_ref_means()
    rows = []
    exceptions = []
    skipped = []
    total_fits = 0
    base_seed0_flags = []

    ALL_CONFIGS = [BASE] + CANDIDATES

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
        b = rec.get(f"{BASE}_{split}")
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
    for cfg in CANDIDATES:
        fieldnames += [f"{cfg}_pub", f"{cfg}_d_pub", f"{cfg}_prv", f"{cfg}_d_prv"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            out = {k: r.get(k, "") for k in
                   ["dataset", "n_train", "n_object_cols", "fires", "msl",
                    "base_is_seed0", "base_pub", "base_prv"]}
            for cfg in CANDIDATES:
                out[f"{cfg}_pub"] = r.get(f"{cfg}_pub", "")
                out[f"{cfg}_prv"] = r.get(f"{cfg}_prv", "")
                out[f"{cfg}_d_pub"] = delta(r, cfg, "pub")
                out[f"{cfg}_d_prv"] = delta(r, cfg, "prv")
            w.writerow(out)

    # ---- INVARIANT: non-firing (obj=0) datasets must be byte-identical to base
    #      for EVERY candidate (delta exactly 0). ----
    invariant_violations = []
    for r in rows:
        if r.get("fires"):
            continue
        for cfg in CANDIDATES:
            dp = delta(r, cfg, "pub")
            dv = delta(r, cfg, "prv")
            dp = 0.0 if math.isnan(dp) else dp
            dv = 0.0 if math.isnan(dv) else dv
            if dp != 0.0 or dv != 0.0:
                invariant_violations.append((cfg, r["dataset"], dp, dv))

    # ---- firing-set check ----
    fired = {r["dataset"] for r in rows if r.get("fires")}
    fire_ok = (fired == EXPECTED_FIRE)
    obj0_excluded = not (OBJ0_NAMES & fired)

    # ---- base==seed0 confirmation (per fired dataset) ----
    base_seed0_ok = bool(base_seed0_flags) and all(base_seed0_flags)

    # ---- sweep means ----
    sweep = {}
    for K in KS:
        cfg = cand_name(K)
        mp, mv = mean_delta(cfg, "pub"), mean_delta(cfg, "prv")
        wp, lp, tp = wlt(cfg, "pub")
        wv, lv, tv = wlt(cfg, "prv")
        sweep[K] = {"mp": mp, "mv": mv, "pub_wlt": (wp, lp, tp),
                    "prv_wlt": (wv, lv, tv),
                    "regs_pub": regressions(cfg, "pub"),
                    "regs_prv": regressions(cfg, "prv")}

    # ---- reproduction check (recomputed K10 vs round40 full-precision) ----
    repro = {}
    repro_ok = True
    if ref40 is None:
        repro_available = False
    else:
        repro_available = True
        mp, mv = sweep[REF_K]["mp"], sweep[REF_K]["mv"]
        rp, rv = ref40.get("pub"), ref40.get("prv")
        okp = (rp is not None) and (not math.isnan(rp)) and abs(mp - rp) < REPRO_TOL
        okv = (rv is not None) and (not math.isnan(rv)) and abs(mv - rv) < REPRO_TOL
        repro[REF_K] = {"mp": mp, "mv": mv, "rp": rp, "rv": rv,
                        "okp": okp, "okv": okv}
        repro_ok = okp and okv

    # ================= SUMMARY =================
    L = []
    L.append("=" * 78)
    L.append("bench_03 round43 — GATE-C K HIGH SWEEP (holdout fixed at base-08 default 0.10)"
             "  [OFFLINE]")
    L.append("=" * 78)
    L.append("")
    L.append("SETUP:")
    L.append("  base == shipped 08 (git HEAD:submissions/08...): HGB, early_stopping,")
    L.append("    validation_fraction UNSET -> sklearn default 0.10. base column =")
    L.append("    seed-0, vf UNSET, for all 16 datasets.")
    L.append("  Gate C: fires iff n_object_cols>0 (12 categorical datasets); pure-")
    L.append("    numeric (obj=0) datasets stay single seed-0 == base (delta 0).")
    L.append("  Sweep: K seeds averaged in {10,12,15,20}; holdout FIXED (no hf change).")
    L.append("    Every seed fit with validation_fraction UNSET (the true base-08")
    L.append("    default) — so K10 == round40 K10 (argmax of the {5..10} sweep).")

    # ---- SWEEP TABLE ----
    L.append("")
    L.append("=== SWEEP TABLE (each K vs base == shipped 08) ===")
    L.append(f"{'setting':<10} {'meanDPub':>10} {'meanDPrv':>10} "
             f"{'Pub W/L/T':>12} {'Prv W/L/T':>12}")
    for K in KS:
        s = sweep[K]
        wp, lp, tp = s["pub_wlt"]
        wv, lv, tv = s["prv_wlt"]
        tag = ""
        if K == 10:
            tag = "  <- round40 argmax (repro anchor)"
        L.append(f"{'K'+str(K):<10} {s['mp']:>+10.5f} {s['mv']:>+10.5f} "
                 f"{f'{wp}/{lp}/{tp}':>12} {f'{wv}/{lv}/{tv}':>12}{tag}")

    # ---- REPRODUCTION ----
    L.append("")
    L.append("=== REPRODUCTION CHECK (recomputed K10 here vs round40, tol<5e-6) ===")
    if not repro_available:
        L.append("  round40 results.csv NOT found -> reproduction NOT anchored (FAIL).")
    else:
        rr = repro[REF_K]
        L.append(
            f"  K{REF_K}: Public {rr['mp']:+.6f} vs round40 {rr['rp']:+.6f} "
            f"(|d|={abs(rr['mp']-rr['rp']):.2e}, {'YES' if rr['okp'] else 'NO'}); "
            f"Private {rr['mv']:+.6f} vs round40 {rr['rv']:+.6f} "
            f"(|d|={abs(rr['mv']-rr['rv']):.2e}, {'YES' if rr['okv'] else 'NO'})")
        L.append(f"  round40 coarse context: K10 ~ +{REF40_K10_PUB:.5f}/+{REF40_K10_PRV:.5f}"
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
    L.append("=== INVARIANT (obj=0 datasets identical to base for every K) ===")
    if invariant_violations:
        L.append("  VIOLATED!")
        for cfg, ds, dp, dv in invariant_violations:
            L.append(f"    {cfg}/{ds}: pub d={dp:+.6g} prv d={dv:+.6g}")
    else:
        L.append(f"  OK: each of the {len(OBJ0_NAMES)} obj=0 datasets "
                 f"{sorted(OBJ0_NAMES)} is byte-identical to base (delta 0) "
                 f"across all {len(CANDIDATES)} K. PASS.")

    # ---- PER-DATASET Public / Private ΔAUC across K ----
    for split, tag in (("pub", "Public"), ("prv", "Private")):
        L.append("")
        L.append(f"=== PER-DATASET ΔAUC across K ({tag}) — base + K"
                 f"{{10,12,15,20}} ===")
        header = f"{'dataset':<10} {'obj':>4} {'base':>8}"
        for K in KS:
            header += f" {'K'+str(K):>8} {'d':>9}"
        L.append(header)
        for r in rows:
            line = (f"{r['dataset']:<10} {str(r.get('n_object_cols')):>4} "
                    f"{r[f'{BASE}_{split}']:>8.4f}")
            for K in KS:
                cfg = cand_name(K)
                line += (f" {r[f'{cfg}_{split}']:>8.4f} "
                         f"{delta(r, cfg, split):>+9.5f}")
            L.append(line)

    # ---- MONOTONICITY / PLATEAU analysis ----
    # step = mean-delta change from previous swept K; per-seed = step / (K - K_prev).
    L.append("")
    L.append("=== MONOTONICITY & PLATEAU ANALYSIS (mean ΔAUC vs K) ===")
    L.append(f"  {'K':>4} {'meanDPub':>10} {'meanDPrv':>10} "
             f"{'stepPub':>10} {'stepPrv':>10} {'+seeds':>7} "
             f"{'perSeedPub':>11} {'perSeedPrv':>11}")
    prev = None
    prevK = None
    mono_pub = mono_prv = True
    per_seed = {}
    for K in KS:
        mp, mv = sweep[K]["mp"], sweep[K]["mv"]
        if prev is None:
            sp = sv = float("nan")
            nseed = 0
            psp = psv = float("nan")
        else:
            sp, sv = mp - prev[0], mv - prev[1]
            nseed = K - prevK
            psp, psv = sp / nseed, sv / nseed
            per_seed[K] = (psp, psv)
            if sp < -1e-9:
                mono_pub = False
            if sv < -1e-9:
                mono_prv = False
        sp_s = "   --" if math.isnan(sp) else f"{sp:>+10.5f}"
        sv_s = "   --" if math.isnan(sv) else f"{sv:>+10.5f}"
        ns_s = "  --" if nseed == 0 else f"{nseed:>7d}"
        psp_s = "     --" if math.isnan(psp) else f"{psp:>+11.6f}"
        psv_s = "     --" if math.isnan(psv) else f"{psv:>+11.6f}"
        L.append(f"  {'K'+str(K):>4} {mp:>+10.5f} {mv:>+10.5f} {sp_s:>10} {sv_s:>10} "
                 f"{ns_s:>7} {psp_s:>11} {psv_s:>11}")
        prev = (mp, mv)
        prevK = K
    L.append(f"  monotone non-decreasing in K: Public {'YES' if mono_pub else 'NO'}, "
             f"Private {'YES' if mono_prv else 'NO'}")

    # argmax K by mean(pub+prv)
    best_sum_K = max(KS, key=lambda K: sweep[K]["mp"] + sweep[K]["mv"])
    L.append(f"  argmax mean(Pub+Prv): K{best_sum_K} "
             f"(Pub{sweep[best_sum_K]['mp']:+.5f} Prv{sweep[best_sum_K]['mv']:+.5f})")

    # plateau: compare last per-seed step to first per-seed step.
    steps_seed = {}  # K -> avg per-seed (pub+prv)/2 step
    for idx in range(1, len(KS)):
        K = KS[idx]
        psp, psv = per_seed[K]
        steps_seed[K] = (psp + psv) / 2.0
    first_seed_step = steps_seed[KS[1]] if len(KS) > 1 else 0.0
    last_seed_step = steps_seed[KS[-1]] if len(KS) > 1 else 0.0

    # plateau K = smallest K whose incoming per-seed step falls <= 20% of the first
    # per-seed step (clearly diminishing). If never, plateau not reached by max K.
    plateau_K = None
    if first_seed_step > 1e-12:
        for idx in range(1, len(KS)):
            K = KS[idx]
            if steps_seed[K] <= 0.20 * first_seed_step:
                plateau_K = KS[idx - 1]
                break

    # ---- ADOPTION ANALYSIS ----
    L.append("")
    L.append("=== ADOPTION ANALYSIS ===")
    L.append("Criterion A (clean vs base-08): mean D > 0 AND zero regressions on BOTH")
    L.append(f"  splits. Criterion B (beats round40 pick K{REF_K}): also strictly greater")
    L.append(f"  mean than K{REF_K} on BOTH splits with zero regressions.")
    L.append("")
    krmp, krmv = sweep[REF_K]["mp"], sweep[REF_K]["mv"]
    beaters = []
    clean_list = []
    for K in KS:
        s = sweep[K]
        mp, mv = s["mp"], s["mv"]
        rp, rv = s["regs_pub"], s["regs_prv"]
        zero_regs = (not rp) and (not rv)
        clean_vs_base = (mp > 1e-9) and (mv > 1e-9) and zero_regs
        beats_ref = (mp > krmp + 1e-9) and (mv > krmv + 1e-9)
        is_ref = (K == REF_K)
        status = ["clean-vs-08" if clean_vs_base else "NOT-clean-vs-08"]
        if is_ref:
            status.append(f"== round40 pick K{REF_K}")
        else:
            status.append(f"beats-K{REF_K}(both)" if beats_ref
                          else f"does-not-beat-K{REF_K}")
        regstr = ""
        if not zero_regs:
            allr = rp + rv
            regstr = " regs[" + ", ".join(f"{n}({d:+.5f})" for n, d in allr) + "]"
        L.append(f"  K{K}: pub{mp:+.5f} prv{mv:+.5f}  " + "; ".join(status) + regstr)
        if clean_vs_base:
            clean_list.append(K)
        if (not is_ref) and clean_vs_base and beats_ref:
            beaters.append((K, mp, mv))

    # ---- VERDICT ----
    L.append("")
    L.append("=== VERDICT ===")
    L.append(f"KEY QUESTION: does the K curve keep rising above K=10 (round40's argmax) "
             f"toward K=20, or has it PLATEAUED — and where?")
    L.append("")
    # K10->K20 gain
    g20p = sweep[20]["mp"] - krmp
    g20v = sweep[20]["mv"] - krmv
    L.append(f"  K{REF_K}->K20 gain: Public {g20p:+.5f}, Private {g20v:+.5f} "
             f"(K{REF_K} pub{krmp:+.5f}/prv{krmv:+.5f} -> K20 pub{sweep[20]['mp']:+.5f}"
             f"/prv{sweep[20]['mv']:+.5f}).")
    L.append(f"  That is +10 seeds; per-added-seed over K{REF_K}->K20: "
             f"Public {g20p/10:+.6f}, Private {g20v/10:+.6f}.")
    L.append(f"  Monotone non-decreasing in K: Public {'YES' if mono_pub else 'NO'}, "
             f"Private {'YES' if mono_prv else 'NO'}.")
    # per-step
    L.append("  Incremental gain per swept step (avg of Pub+Prv step, and per-seed):")
    for idx in range(1, len(KS)):
        K = KS[idx]
        psp, psv = per_seed[K]
        step_avg = (sweep[K]["mp"] - sweep[KS[idx - 1]]["mp"]
                    + sweep[K]["mv"] - sweep[KS[idx - 1]]["mv"]) / 2.0
        L.append(f"    K{KS[idx-1]}->K{K} (+{K-KS[idx-1]} seeds): step {step_avg:+.5f}, "
                 f"per-seed {(psp+psv)/2.0:+.6f}")
    L.append(f"  argmax mean(Pub+Prv) over swept K: K{best_sum_K}.")

    # plateau statement
    L.append("")
    if plateau_K is not None:
        L.append(f"  PLATEAU: per-seed gain falls to <=20% of the K{KS[0]}->K{KS[1]} "
                 f"per-seed step by K{plateau_K} -> curve effectively FLAT beyond "
                 f"~K{plateau_K}.")
    else:
        L.append(f"  PLATEAU: per-seed gain has NOT dropped below 20% of the "
                 f"K{KS[0]}->K{KS[1]} per-seed step by K{KS[-1]} -> curve is STILL "
                 f"RISING at K{KS[-1]} (no clear plateau within the swept range).")
    L.append(f"  first per-seed step (K{KS[0]}->K{KS[1]}) = {first_seed_step:+.6f}; "
             f"last per-seed step (K{KS[-2]}->K{KS[-1]}) = {last_seed_step:+.6f} "
             f"({(100.0*last_seed_step/first_seed_step) if abs(first_seed_step)>1e-12 else float('nan'):.0f}% "
             f"of first).")

    # regressions note
    any_reg_K = [K for K in KS if sweep[K]["regs_pub"] or sweep[K]["regs_prv"]]
    if any_reg_K:
        L.append(f"  NOTE: K with a regression on some dataset: "
                 f"{', '.join('K'+str(K) for K in any_reg_K)}.")
    else:
        L.append("  No swept K regresses on ANY dataset on either split "
                 "(all 12 fired datasets win-or-tie; 4 obj=0 tie exactly).")

    # recommendation
    L.append("")
    if beaters:
        beaters.sort(key=lambda x: (x[1] + x[2]), reverse=True)
        bK, bmp, bmv = beaters[0]
        L.append(f"  ANSWER: Above K{REF_K}, the curve is "
                 f"{'still climbing' if (mono_pub and mono_prv) else 'not fully monotone'}"
                 f"; K cleanly beating round40's K{REF_K} on both splits: "
                 f"{', '.join('K'+str(k) for k,_,_ in beaters)}.")
        if bK == best_sum_K:
            L.append(f"  Best mean = argmax = K{bK} "
                     f"(pub{bmp:+.5f}/prv{bmv:+.5f}).")
        else:
            L.append(f"  Best mean = K{best_sum_K}; top clean beater = K{bK}.")
        if plateau_K is not None:
            L.append(f"  For the gate-C seed-avg family, the knee/plateau is ~K{plateau_K}: "
                     f"beyond it per-seed gain is <20% of the initial step. K{best_sum_K} "
                     f"squeezes marginally more at ~{best_sum_K}/{REF_K}x the cost of K{REF_K}.")
        else:
            L.append(f"  For the gate-C seed-avg family, the curve has NOT plateaued by "
                     f"K{KS[-1]}: K{best_sum_K} is the argmax and still the top pick within "
                     f"the swept range, at K = models trained per fired dataset.")
    else:
        L.append(f"  ANSWER: NO swept K > K{REF_K} cleanly beats round40's pick K{REF_K} on "
                 f"BOTH splits with zero regressions. K{REF_K} stays the recommended gate-C "
                 f"seed-avg K; the K{REF_K}->K20 gain ({g20p:+.5f}/{g20v:+.5f}) is not a "
                 f"clean improvement.")

    L.append("  (Effort note: K = number of HGB models trained per fired dataset. "
             "K20 doubles K10's training cost; pick the knee/plateau, not the max.)")

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
