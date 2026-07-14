#!/usr/bin/env python
"""
bench_03 round41 — PURE-NUMERIC (obj=0) ORTHOGONAL SINGLE-KNOB SWEEP.
OFFLINE ONLY. No subprocess, no LLM, no Kaggle, no network. Calls sklearn
in-process only. Writes ONLY under this round41 directory; never touches
submissions/.

GOAL (improvement-log angle "k")
--------------------------------
Gate C / seed-averaging (rounds 34..40) fires only on datasets with
n_object_cols > 0. The 4 PURE-NUMERIC datasets (n_object_cols == 0) —
train_04, train_10, train_11, train_16 — are the untouched region: they always
run base-08's single seed-0 model. Question: holding base-08 fixed everywhere,
is there a NON-ensemble, ORTHOGONAL single knob (one hyperparameter change on
the seed-0 model) that CLEANLY improves any obj=0 dataset with ZERO regression,
while leaving all 12 categorical (obj>0) datasets BYTE-IDENTICAL to base-08?

Design:
  base column (= shipped 08, seed-0, validation_fraction UNSET) for all 16.
  For each candidate knob, apply the knob ONLY to obj=0 datasets; obj>0
  datasets are ALWAYS the exact base-08 seed-0 map (byte-identical, delta 0,
  asserted). Each candidate is a single seed-0 model (NON-ensemble), so this is
  orthogonal to the gate-C seed-avg work.

  Candidate A (cand_A_lr005):  learning_rate = 0.05 (base default 0.1),
                               early_stopping on. obj=0 only.
  Candidate B (cand_B_mf070):  max_features = 0.7 (base default 1.0),
                               random_state=0. obj=0 only.
  Candidate C (cand_C_l2force): l2_regularization = 1.0 FORCED on obj=0
                               (base gives 04/10/11 l2=0 via the ratio gate;
                               train_16 already has l2=1.0 since its ratio
                               21/1809=0.0116 >= 0.010, so C is a no-op there).
  Candidate D (cand_D_msl50):  min_samples_leaf = 50 on obj=0 (base gives 20).

BASE recipe reproduced (== shipped 08), identical to round40:
  features = [c for c in train.columns if c not in ("row_id","target")]
  cat_mask = [train[c].dtype == object for c in features]
  n=len(train); ratio = len(features)/n
  l2  = 1.0 if ratio >= 0.010 else 0.0
  msl = 70 if ratio>=0.030 else 50 if ratio>=0.015 else 20
  HistGradientBoostingClassifier(categorical_features=cat_mask, random_state=0,
      max_iter=300, early_stopping=True, l2_regularization=l2,
      min_samples_leaf=msl)   # validation_fraction NOT set -> default 0.1
  pred = predict_proba(test)[:, class==1]
BASE column = seed-0 with validation_fraction UNSET (byte-identical to shipped 08
and to round40's base). This is the ΔAUC reference for every candidate.

REPRODUCTION anchors (recomputed here, NOT hardcode-trusted): the base column on
the 4 obj=0 datasets must match round40's base column to < 5e-6. round40's
base_pub/base_prv are read from its results.csv; fixed known values are also
printed for context:
  train_04 Pub 0.8236 / Prv 0.8250, train_10 0.8368 / 0.8477,
  train_11 0.8281 / 0.8167, train_16 0.8922 / 0.8933 (Private VERIFIED from
  round40 summary.txt line 74 = 0.89325... — NOT 0.8250).

Adoption: a candidate is an ADOPT for the obj=0 group iff, on the obj=0 datasets
it touches, it improves at least one dataset (ΔAUC>0 on BOTH splits for that
dataset OR a clean group mean gain) with ZERO regression anywhere on either
split (obj=0 and obj>0). Any negative ΔAUC on any dataset/split => REJECT.
"""
import os

# keep the run polite / modest on CPU; HGB is deterministic w.r.t. random_state
# regardless of thread count, so this does not affect reproduction.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

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
OUT_DIR = os.path.join(BENCH_DIR, "round41_purenumeric_knob")
STATS_CSV = os.path.join(BENCH_DIR, "dataset_stats.csv")
ROUND40_RESULTS = os.path.join(BENCH_DIR, "round40_gateC_ksweep", "results.csv")

L2_GATE_THRESHOLD = 0.010   # shipped-08 ratio gate for l2 (FIXED)
GATED_L2 = 1.0
DEFAULT_MSL = 20
MSL_TIERS = [(0.030, 70), (0.015, 50)]
BASE_SEED = 0
N_DATASETS = 16
REPRO_TOL = 5e-6

BASE = "base"

# the 4 pure-numeric datasets (n_object_cols == 0)
OBJ0_NAMES = {"train_04", "train_10", "train_11", "train_16"}
# the 12 categorical datasets (must stay byte-identical to base under every cand)
EXPECTED_OBJPOS = {"train_01", "train_02", "train_03", "train_05", "train_06",
                   "train_07", "train_08", "train_09", "train_12", "train_13",
                   "train_14", "train_15"}

# known round40 base anchors (printed for context; the live assertion is vs
# round40 results.csv full precision, tol < 5e-6).
KNOWN_BASE = {
    "train_04": (0.8236, 0.8250),
    "train_10": (0.8368, 0.8477),
    "train_11": (0.8281, 0.8167),
    "train_16": (0.8922, 0.8933),   # Private VERIFIED = 0.89325 (round40 line 74)
}

# candidate knob overrides (applied ONLY to obj=0 datasets).
#   each entry: cand_name -> dict of HGB kwargs that OVERRIDE the base recipe.
CANDIDATES = {
    "cand_A_lr005":   {"learning_rate": 0.05},
    "cand_B_mf070":   {"max_features": 0.7},
    "cand_C_l2force": {"l2_regularization": 1.0},
    "cand_D_msl50":   {"min_samples_leaf": 50},
}
CAND_NAMES = list(CANDIDATES.keys())
ALL_CONFIGS = [BASE] + CAND_NAMES


def load_stats(path=STATS_CSV):
    stats = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            stats[row["name"]] = {
                "n_train": int(row["n_train"]),
                "n_object_cols": int(row["n_object_cols"]),
            }
    return stats


def round40_base_anchors(path=ROUND40_RESULTS):
    """Read round40's base_pub/base_prv for the obj=0 datasets to anchor
    reproduction at full precision. Returns dict name -> (pub, prv) or None."""
    if not os.path.exists(path):
        return None
    anchors = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            name = row.get("dataset")
            if name in OBJ0_NAMES:
                try:
                    anchors[name] = (float(row["base_pub"]), float(row["base_prv"]))
                except (KeyError, ValueError):
                    pass
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


def fit_hgb(train, test, features, cat_mask, l2, msl_val, seed, overrides=None):
    """Fit ONE shipped-08 HGB. validation_fraction left UNSET (sklearn default
    0.1, byte-identical to shipped 08). `overrides` replaces individual kwargs
    for the candidate knobs; when None this is the exact base-08 model."""
    kwargs = dict(
        categorical_features=cat_mask,
        random_state=seed,
        max_iter=300,
        early_stopping=True,
        l2_regularization=l2,
        min_samples_leaf=msl_val,
    )
    if overrides:
        kwargs.update(overrides)
    clf = HistGradientBoostingClassifier(**kwargs)
    clf.fit(train[features], train["target"])
    proba = clf.predict_proba(test[features])
    classes = list(clf.classes_)
    pos_idx = classes.index(1) if 1 in classes else 1
    return proba[:, pos_idx]


def run_one(name, train_csv, test_csv, stats):
    """Returns (preds, meta). preds maps config_name -> {row_id -> prob}.

    base = seed-0, validation_fraction UNSET (== shipped 08). obj>0 datasets:
    every candidate reuses the exact base map (byte-identical, delta 0). obj=0
    datasets: each candidate fits ONE seed-0 model with its single knob override.
    """
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)

    features = [c for c in train.columns if c not in ("row_id", "target")]
    cat_mask = [train[c].dtype == object for c in features]

    n = len(train)
    ratio = len(features) / n
    l2 = GATED_L2 if ratio >= L2_GATE_THRESHOLD else 0.0
    msl_val = msl_for_ratio(ratio)

    st = stats[name]
    n_obj_stat = st["n_object_cols"]
    is_obj0 = (n_obj_stat == 0)

    row_ids = test["row_id"].tolist()

    # base = seed-0, vf UNSET (byte-identical to shipped 08).
    base_vec = fit_hgb(train, test, features, cat_mask, l2, msl_val, BASE_SEED)
    base_map = dict(zip(row_ids, base_vec.tolist()))
    preds = {BASE: base_map}
    n_fits = 1

    if not is_obj0:
        # obj>0 -> every candidate identical to base (reuse the SAME object).
        for c in CAND_NAMES:
            preds[c] = base_map
    else:
        # obj=0 -> fit one seed-0 model per candidate knob.
        for c in CAND_NAMES:
            vec = fit_hgb(train, test, features, cat_mask, l2, msl_val,
                          BASE_SEED, overrides=CANDIDATES[c])
            preds[c] = dict(zip(row_ids, vec.tolist()))
            n_fits += 1

    meta = {
        "n_train": st["n_train"],
        "n_object_cols": n_obj_stat,
        "is_obj0": bool(is_obj0),
        "l2": l2,
        "msl": msl_val,
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
    anchors40 = round40_base_anchors()
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
        rec = {"dataset": name}
        try:
            preds, meta = run_one(name, train_csv, test_csv, stats)
            total_fits += meta["n_fits"]
            rec.update({
                "n_train": meta["n_train"],
                "n_object_cols": meta["n_object_cols"],
                "is_obj0": meta["is_obj0"],
                "l2": meta["l2"],
                "msl": meta["msl"],
            })
            for cfg in ALL_CONFIGS:
                pub, prv = score_split(preds[cfg], sol)
                rec[f"{cfg}_pub"] = pub
                rec[f"{cfg}_prv"] = prv
            print(f"[OK] {name} n_tr={meta['n_train']} obj={meta['n_object_cols']} "
                  f"obj0={meta['is_obj0']} l2={meta['l2']} msl={meta['msl']} "
                  f"fits={meta['n_fits']} base pub={rec['base_pub']:.6f} "
                  f"prv={rec['base_prv']:.6f}")
        except Exception as e:
            exceptions.append((name, repr(e)))
            rec.update({"n_train": stats.get(name, {}).get("n_train", ""),
                        "n_object_cols": stats.get(name, {}).get("n_object_cols", ""),
                        "is_obj0": False, "l2": float("nan"), "msl": float("nan")})
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
    fieldnames = ["dataset", "n_train", "n_object_cols", "is_obj0", "l2", "msl",
                  "base_pub", "base_prv"]
    for cfg in CAND_NAMES:
        fieldnames += [f"{cfg}_pub", f"{cfg}_d_pub", f"{cfg}_prv", f"{cfg}_d_prv"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            out = {k: r.get(k, "") for k in
                   ["dataset", "n_train", "n_object_cols", "is_obj0", "l2", "msl",
                    "base_pub", "base_prv"]}
            for cfg in CAND_NAMES:
                out[f"{cfg}_pub"] = r.get(f"{cfg}_pub", "")
                out[f"{cfg}_prv"] = r.get(f"{cfg}_prv", "")
                out[f"{cfg}_d_pub"] = delta(r, cfg, "pub")
                out[f"{cfg}_d_prv"] = delta(r, cfg, "prv")
            w.writerow(out)

    # ---- INVARIANT: obj>0 datasets byte-identical to base for EVERY candidate ----
    invariant_violations = []
    for r in rows:
        if r.get("is_obj0"):
            continue
        for cfg in CAND_NAMES:
            dp = delta(r, cfg, "pub")
            dv = delta(r, cfg, "prv")
            dp = 0.0 if math.isnan(dp) else dp
            dv = 0.0 if math.isnan(dv) else dv
            if dp != 0.0 or dv != 0.0:
                invariant_violations.append((cfg, r["dataset"], dp, dv))

    # ---- obj=0 partition check ----
    obj0_found = {r["dataset"] for r in rows if r.get("is_obj0")}
    objpos_found = {r["dataset"] for r in rows if (not r.get("is_obj0"))
                    and r["dataset"] not in skipped}
    obj0_ok = (obj0_found == OBJ0_NAMES)
    objpos_ok = (objpos_found == EXPECTED_OBJPOS)

    # ---- REPRODUCTION: base on obj=0 matches round40 (tol<5e-6) ----
    repro = {}
    repro_ok = True
    repro_available = anchors40 is not None
    by_name = {r["dataset"]: r for r in rows}
    for nm in sorted(OBJ0_NAMES):
        r = by_name.get(nm)
        mine = (r.get("base_pub"), r.get("base_prv")) if r else (None, None)
        ref = anchors40.get(nm) if anchors40 else None
        if ref is None or mine[0] is None or mine[1] is None:
            okp = okv = False
        else:
            okp = abs(mine[0] - ref[0]) < REPRO_TOL
            okv = abs(mine[1] - ref[1]) < REPRO_TOL
        repro[nm] = {"mine": mine, "ref": ref, "okp": okp, "okv": okv}
        if not (okp and okv):
            repro_ok = False

    # ---- per-candidate stats ----
    cand_stats = {}
    for cfg in CAND_NAMES:
        cand_stats[cfg] = {
            "mp": mean_delta(cfg, "pub"),
            "mv": mean_delta(cfg, "prv"),
            "pub_wlt": wlt(cfg, "pub"),
            "prv_wlt": wlt(cfg, "prv"),
            "regs_pub": regressions(cfg, "pub"),
            "regs_prv": regressions(cfg, "prv"),
        }

    # ================= SUMMARY =================
    L = []
    L.append("=" * 78)
    L.append("bench_03 round41 — PURE-NUMERIC (obj=0) ORTHOGONAL SINGLE-KNOB "
             "SWEEP  [OFFLINE]")
    L.append("=" * 78)
    L.append("")
    L.append("SETUP:")
    L.append("  base == shipped 08 (git HEAD:submissions/08...): HGB, early_stopping,")
    L.append("    validation_fraction UNSET -> sklearn default 0.10. base column =")
    L.append("    seed-0 for all 16 datasets.")
    L.append("  Candidates apply ONE orthogonal knob to the seed-0 model, ONLY on the")
    L.append(f"    4 obj=0 (pure-numeric) datasets {sorted(OBJ0_NAMES)};")
    L.append("    the 12 obj>0 datasets always == base-08 (byte-identical, delta 0).")
    L.append("  Knobs:")
    L.append("    A cand_A_lr005   : learning_rate 0.1 -> 0.05 (early_stopping on)")
    L.append("    B cand_B_mf070   : max_features 1.0 -> 0.7 (random_state=0)")
    L.append("    C cand_C_l2force : l2_regularization forced to 1.0 "
             "(base=0 on 04/10/11; no-op on 16)")
    L.append("    D cand_D_msl50   : min_samples_leaf 20 -> 50")

    # ---- SWEEP TABLE ----
    L.append("")
    L.append("=== SWEEP TABLE (each candidate vs base == shipped 08, over all 16) ===")
    L.append(f"{'candidate':<16} {'meanDPub':>10} {'meanDPrv':>10} "
             f"{'Pub W/L/T':>12} {'Prv W/L/T':>12}")
    for cfg in CAND_NAMES:
        s = cand_stats[cfg]
        wp, lp, tp = s["pub_wlt"]
        wv, lv, tv = s["prv_wlt"]
        L.append(f"{cfg:<16} {s['mp']:>+10.5f} {s['mv']:>+10.5f} "
                 f"{f'{wp}/{lp}/{tp}':>12} {f'{wv}/{lv}/{tv}':>12}")

    # ---- REPRODUCTION ----
    L.append("")
    L.append("=== REPRODUCTION CHECK (base on obj=0 vs round40, tol<5e-6) ===")
    if not repro_available:
        L.append("  round40 results.csv NOT found -> reproduction NOT anchored (FAIL).")
    else:
        for nm in sorted(OBJ0_NAMES):
            rr = repro[nm]
            mp_, mv_ = rr["mine"]
            rp_, rv_ = rr["ref"] if rr["ref"] else (float("nan"), float("nan"))
            kp, kv = KNOWN_BASE[nm]
            L.append(
                f"  {nm}: Public {mp_:.6f} vs r40 {rp_:.6f} "
                f"(|d|={abs(mp_-rp_):.2e}, {'YES' if rr['okp'] else 'NO'}); "
                f"Private {mv_:.6f} vs r40 {rv_:.6f} "
                f"(|d|={abs(mv_-rv_):.2e}, {'YES' if rr['okv'] else 'NO'})  "
                f"[known ~{kp:.4f}/{kv:.4f}]")
        L.append(f"  REPRODUCTION: {'PASS' if repro_ok else 'FAIL'}")

    # ---- PARTITION / INVARIANT ----
    L.append("")
    L.append("=== PARTITION CHECK ===")
    L.append(f"  obj=0 (pure-numeric) matched {sorted(OBJ0_NAMES)}: "
             f"{'YES' if obj0_ok else 'NO'}"
             + ("" if obj0_ok else f" (got {sorted(obj0_found)})"))
    L.append(f"  obj>0 (12 categorical) matched: {'YES' if objpos_ok else 'NO'}"
             + ("" if objpos_ok else f" (got {sorted(objpos_found)})"))

    L.append("")
    L.append("=== INVARIANT (obj>0 datasets byte-identical to base for every "
             "candidate) ===")
    if invariant_violations:
        L.append("  VIOLATED!")
        for cfg, ds, dp, dv in invariant_violations:
            L.append(f"    {cfg}/{ds}: pub d={dp:+.6g} prv d={dv:+.6g}")
    else:
        L.append(f"  OK: each of the 12 obj>0 datasets is byte-identical to base "
                 f"(delta exactly 0) across all {len(CAND_NAMES)} candidates. PASS.")

    # ---- PER-obj=0-DATASET DELTAS ----
    for split, tag in (("pub", "Public"), ("prv", "Private")):
        L.append("")
        L.append(f"=== PER obj=0 DATASET ΔAUC ({tag}) — base + 4 candidates ===")
        header = f"{'dataset':<10} {'base':>8}"
        for cfg in CAND_NAMES:
            short = cfg.replace("cand_", "")
            header += f" {short:>14} {'d':>9}"
        L.append(header)
        for r in rows:
            if not r.get("is_obj0"):
                continue
            line = f"{r['dataset']:<10} {r[f'{BASE}_{split}']:>8.4f}"
            for cfg in CAND_NAMES:
                line += (f" {r[f'{cfg}_{split}']:>14.4f} "
                         f"{delta(r, cfg, split):>+9.5f}")
            L.append(line)

    # ---- obj=0 GROUP MEAN per candidate ----
    L.append("")
    L.append("=== obj=0 GROUP MEAN ΔAUC (over the 4 pure-numeric datasets only) ===")
    L.append(f"  {'candidate':<16} {'meanDPub':>10} {'meanDPrv':>10} "
             f"{'obj0 Pub W/L/T':>16} {'obj0 Prv W/L/T':>16}")
    obj0_rows = [r for r in rows if r.get("is_obj0")]
    for cfg in CAND_NAMES:
        dps = [delta(r, cfg, "pub") for r in obj0_rows]
        dvs = [delta(r, cfg, "prv") for r in obj0_rows]
        dps = [x for x in dps if not math.isnan(x)]
        dvs = [x for x in dvs if not math.isnan(x)]
        mp = sum(dps) / len(dps) if dps else float("nan")
        mv = sum(dvs) / len(dvs) if dvs else float("nan")

        def _wlt(vals, eps=1e-6):
            w = sum(1 for v in vals if v > eps)
            l = sum(1 for v in vals if v < -eps)
            t = len(vals) - w - l
            return w, l, t
        wp, lp, tp = _wlt(dps)
        wv, lv, tv = _wlt(dvs)
        L.append(f"  {cfg:<16} {mp:>+10.5f} {mv:>+10.5f} "
                 f"{f'{wp}/{lp}/{tp}':>16} {f'{wv}/{lv}/{tv}':>16}")

    # ---- ADOPTION ANALYSIS ----
    L.append("")
    L.append("=== ADOPTION ANALYSIS ===")
    L.append("A candidate is ADOPT iff on the obj=0 group it improves >=1 dataset")
    L.append("  (ΔAUC>0 on BOTH splits for that dataset) with ZERO regression")
    L.append("  anywhere (obj=0 or obj>0, either split). Any negative ΔAUC => REJECT.")
    L.append("")
    adopts = []
    for cfg in CAND_NAMES:
        s = cand_stats[cfg]
        rp, rv = s["regs_pub"], s["regs_prv"]
        zero_regs = (not rp) and (not rv)
        # datasets cleanly improved on BOTH splits
        clean_improved = []
        for r in obj0_rows:
            dp = delta(r, cfg, "pub")
            dv = delta(r, cfg, "prv")
            if (not math.isnan(dp) and not math.isnan(dv)
                    and dp > 1e-6 and dv > 1e-6):
                clean_improved.append((r["dataset"], dp, dv))
        is_adopt = zero_regs and bool(clean_improved)
        regstr = ""
        if not zero_regs:
            allr = rp + rv
            regstr = " REGRESSIONS[" + ", ".join(
                f"{n}({d:+.5f})" for n, d in allr) + "]"
        impstr = ""
        if clean_improved:
            impstr = " improves{" + ", ".join(
                f"{n}(+{dp:.5f}/+{dv:.5f})" for n, dp, dv in clean_improved) + "}"
        L.append(f"  {cfg}: {'ADOPT' if is_adopt else 'REJECT'}"
                 f"  zero_regressions={'YES' if zero_regs else 'NO'}"
                 + impstr + regstr)
        if is_adopt:
            adopts.append((cfg, clean_improved))

    # ---- VERDICT ----
    L.append("")
    L.append("=== VERDICT ===")
    L.append("KEY QUESTION: is there an orthogonal single knob that cleanly improves")
    L.append("  any obj=0 (pure-numeric) dataset with ZERO regression anywhere, while")
    L.append("  leaving all 12 obj>0 datasets byte-identical to base-08?")
    L.append("")
    if adopts:
        for cfg, imp in adopts:
            imps = ", ".join(f"{n} (+{dp:.5f} Pub / +{dv:.5f} Prv)"
                             for n, dp, dv in imp)
            L.append(f"  ADOPT: {cfg} cleanly improves {imps} with zero regression "
                     f"anywhere.")
    else:
        L.append("  NO candidate is a clean ADOPT: every knob either regresses at "
                 "least one obj=0 dataset/split or fails to improve any dataset on "
                 "BOTH splits. base-08 stays best on the pure-numeric region.")

    ship = ("ADOPT: " + ", ".join(cfg for cfg, _ in adopts)) if adopts else \
           "REJECT (no clean single-knob improvement on the obj=0 group)"
    L.append("")
    L.append(f"SHIP VERDICT: {ship}")

    # ---- CLEAN RUN marker ----
    clean_run = ((not exceptions) and (not invariant_violations) and obj0_ok
                 and objpos_ok and repro_ok and repro_available and (not skipped))
    L.append("")
    L.append(f"CLEAN RUN: {'YES' if clean_run else 'NO'} "
             f"({len(exceptions)} exceptions)  "
             f"[total_fits={total_fits}, skipped={len(skipped)}, "
             f"invariant_violations={len(invariant_violations)}, "
             f"obj0_partition={'YES' if obj0_ok else 'NO'}, "
             f"objpos_partition={'YES' if objpos_ok else 'NO'}, "
             f"reproduction={'YES' if repro_ok else 'NO'}]")
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
