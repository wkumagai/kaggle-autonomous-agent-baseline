#!/usr/bin/env python
"""
bench_03 round34 — SEED-AVERAGING FIRING-GATE ENDPOINT TEST (no n_train cap).
OFFLINE ONLY. No subprocess, no LLM, no Kaggle, no network. Calls sklearn
in-process.

Adapted from experiments/bench_03/round33_gate_n_sweep/replay.py (dataset
loading, the shipped-08 base config reproduction, Public/Private AUC scoring
joined on row_id to solution.csv, K=5 seed-averaging, the byte-identical-on-
non-firing INVARIANT check, cached-fit efficiency, and the summary machinery
are all reused). The ONLY change vs round33 is the set of gates compared:
round34 adds the ENDPOINT gate (no n_train cap, obj>0 only) alongside the two
best prior thresholds, so all three sit in one side-by-side run.

Base recipe reproduced (== shipped 08), IDENTICAL to round33/round32:
  - load train.csv, test.csv with pandas
  - features = [c for c in train.columns if c not in ("row_id","target")]
  - cat_mask  = [train[c].dtype == object for c in features]
  - n = len(train); n_feat = len(features); ratio = n_feat / n
  - L2 GATE:  l2  = 1.0 if ratio >= 0.010 else 0.0
  - MSL GATE (08 tiered): msl = 70 if ratio >= 0.030
                          else 50 if ratio >= 0.015 else 20
  - HistGradientBoostingClassifier(categorical_features=cat_mask, random_state=s,
        max_iter=300, early_stopping=True, l2_regularization=l2,
        min_samples_leaf=msl)
  - fit(train[features], train["target"])
  - pred = clf.predict_proba(test[features])[:, pos_class_1]
Score: ROC AUC on Public rows and Private rows separately, joined by row_id to
solution.csv, over all 16 datasets. base = each dataset's single seed-0
prediction (byte-identical to shipped 08). The base recipe is UNCHANGED.

THE ONLY CHANGE — three seed-averaging firing gates compared in one run
(K FIXED at 5):
  gate(N): n_train < N  AND  n_object_cols > 0
    A: N=4000   (== round32 current best)
    B: N=15000  (== round33 best)
    C: no cap / obj>0 only (N = infinity)  <-- the ENDPOINT under test
  n_train and n_object_cols are read per dataset from
  experiments/bench_03/dataset_stats.csv. All three compared against the SAME
  base == shipped-08. The `n_object_cols > 0` requirement is common to all
  three, so train_16 / train_04 / train_10 / train_11 (obj=0) are always
  excluded.

Expected firing sets (verified at run time and logged per config):
  A N=4000  -> {train_03, train_05, train_09, train_13, train_15}    (5)
  B N=15000 -> {01,02,03,05,06,07,08,09,13,14,15}                    (11)
  C nocap   -> B + train_12 (n=49432, obj=3) -> all 12 categorical   (12)

KEY QUESTION: gate C is the only one that seed-averages train_12. Does
train_12 GAIN or REGRESS under K=5 seed-avg? If C beats B on BOTH splits with
zero regressions, no-cap is the definitive recommended gate. If train_12
regresses, B (N=15000) — or the last clean point — is the ceiling.

EFFICIENCY: each dataset that fires under the WIDEST gate (C = obj>0 only,
i.e. every categorical dataset) is fit K=5 times ONCE and cached; all three
gates reuse those cached seed predictions. A dataset uses the cached K=5
average under a gate iff it fires under that gate, else the exact seed-0 base
(byte-identical). Non-categorical datasets are fit seed-0 only. Total fits =
12 categorical x 5 + 4 non-categorical x 1 = 64 (only ~5 more than round33's
60: the extra is train_12's K=5).

IMPLEMENTATION INVARIANT: for a given gate, any dataset that does NOT fire
under it reuses the EXACT seed-0 array, so its delta MUST be exactly 0 on both
splits — checked explicitly per gate.

Adoption criterion (reused): a gate is a CLEAN IMPROVEMENT over base(08) iff
its mean delta is positive on BOTH splits AND there are ZERO regressions on
BOTH splits.

References this run must reproduce:
  A N=4000  == round32: mean Public +0.00229 / Private +0.00215, zero regs.
  B N=15000 == round33: mean Public +0.00355 / Private +0.00303, zero regs.
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
OUT_DIR = os.path.join(BENCH_DIR, "round34_gate_nocap")
STATS_CSV = os.path.join(BENCH_DIR, "dataset_stats.csv")

L2_GATE_THRESHOLD = 0.010   # shipped 08 feature-to-row-ratio gate for l2 (FIXED)
GATED_L2 = 1.0              # l2 applied when the l2-gate fires (FIXED)
DEFAULT_MSL = 20            # sklearn HGB default, used when no msl tier is cleared.
BASE_SEED = 0              # shipped-08 fixed random_state.

MIN_OBJECT_COLS = 0        # require strictly more than this many object columns

# round34 gates: (label, n_train_cap). No cap is represented by +inf.
#   A: N=4000 (round32), B: N=15000 (round33), C: no cap / obj>0 only.
GATES = [
    ("4000", 4000),
    ("15000", 15000),
    ("nocap", float("inf")),
]
# widest cap decides which datasets get the cached K=5 fit (obj>0 & n<inf).
MAX_CAP = max(cap for _, cap in GATES)   # +inf

# K is FIXED at 5 (round30 established K=5 as the knee; no K sweep here).
K = 5
SEEDS = list(range(K))     # [0,1,2,3,4]

# 08 tiered min_samples_leaf, IDENTICAL across base and cand (descending order).
MSL_TIERS = [(0.030, 70), (0.015, 50)]

BASE = "base"


def cand_name(label):
    """Candidate column name for a given gate label."""
    return f"cand_{label}"


CANDIDATES = [cand_name(lbl) for lbl, _ in GATES]

N_DATASETS = 16

# reference points this run must reproduce.
REF = {
    "4000": (0.00229, 0.00215),    # round32
    "15000": (0.00355, 0.00303),   # round33
}

# expected firing sets per gate (verified at run time).
EXPECTED_FIRE = {
    "4000": {"train_03", "train_05", "train_09", "train_13", "train_15"},
    "15000": {"train_01", "train_02", "train_03", "train_05", "train_06",
              "train_07", "train_08", "train_09", "train_13", "train_14",
              "train_15"},
    "nocap": {"train_01", "train_02", "train_03", "train_05", "train_06",
              "train_07", "train_08", "train_09", "train_12", "train_13",
              "train_14", "train_15"},
}


def load_stats(path=STATS_CSV):
    """Return {dataset_name -> {"n_train": int, "n_object_cols": int}} from
    dataset_stats.csv. The gate reads n_train and n_object_cols from here."""
    stats = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stats[row["name"]] = {
                "n_train": int(row["n_train"]),
                "n_object_cols": int(row["n_object_cols"]),
            }
    return stats


def gate_fires(n_train, n_object_cols, cap):
    """round34 firing rule for cap: n_train < cap AND >=1 object col.
    cap == +inf means "no n_train cap" (obj>0 only)."""
    return (n_train < cap) and (n_object_cols > MIN_OBJECT_COLS)


def msl_for_ratio(ratio, tiers=MSL_TIERS):
    """First tier (threshold, msl) whose threshold the ratio clears wins;
    tiers must be given in descending-threshold order. Else DEFAULT_MSL."""
    for thr, val in tiers:
        if ratio >= thr:
            return val
    return DEFAULT_MSL


def auc_or_nan(y_true, y_score):
    """ROC AUC, or NaN if the subset has a single class (undefined)."""
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def fit_one_seed(train, test, features, cat_mask, l2, msl_val, seed):
    """Fit ONE shipped-08 HGB with the given random_state=seed; return the
    positive-class probability vector aligned to test row order. All other
    hyperparameters are byte-identical to shipped 08."""
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
    """Reproduce shipped-08 base + derive all gate candidates for one dataset.
    Returns (preds, l2, l2_fired, fires_by_gate, msl_val, n_fits, n_train_stat,
    n_obj_stat, ever_fires) where:
      preds maps config_name -> {row_id -> prob} for BASE and every cand.
      base = seed-0 prediction (== shipped 08).
      fires_by_gate maps gate label -> bool.
      ever_fires = fires under the WIDEST gate (obj>0) -> decides K=5 vs seed-0.

    A dataset that ever fires is fit seeds 0..4 ONCE and cached; each cand uses
    the K=5 mean if it fires under that gate, else the exact seed-0 array. A
    dataset that never fires (obj=0) is fit seed-0 only and every cand reuses
    that exact seed-0 array (byte-identical to base)."""
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)

    features = [c for c in train.columns if c not in ("row_id", "target")]
    cat_mask = [train[c].dtype == object for c in features]

    n = len(train)
    n_feat = len(features)
    ratio = n_feat / n

    # base config gates (UNCHANGED from shipped 08 / round33).
    l2_fired = ratio >= L2_GATE_THRESHOLD
    l2 = GATED_L2 if l2_fired else 0.0
    msl_val = msl_for_ratio(ratio)

    # round34 seed-averaging gates read stats (n_train, n_object_cols).
    st = stats[name]
    n_train_stat = st["n_train"]
    n_obj_stat = st["n_object_cols"]
    fires_by_gate = {
        lbl: gate_fires(n_train_stat, n_obj_stat, cap) for lbl, cap in GATES
    }
    ever_fires = gate_fires(n_train_stat, n_obj_stat, MAX_CAP)

    row_ids = test["row_id"].tolist()

    if ever_fires:
        # Fit K=5 ONCE and cache; reused for every gate that fires.
        seed_preds = [
            fit_one_seed(train, test, features, cat_mask, l2, msl_val, s)
            for s in SEEDS
        ]
        n_fits = len(SEEDS)
        base_vec = seed_preds[BASE_SEED]  # seed-0 == base
        avg_vec = np.mean(np.vstack(seed_preds), axis=0)
        base_map = dict(zip(row_ids, base_vec.tolist()))
        avg_map = dict(zip(row_ids, avg_vec.tolist()))
        preds = {BASE: base_map}
        for lbl, _ in GATES:
            # fires under gate -> cached K=5 average; else -> exact seed-0 base.
            preds[cand_name(lbl)] = avg_map if fires_by_gate[lbl] else base_map
    else:
        # obj=0 -> never fires -> seed-0 only; every cand == base.
        base_vec = fit_one_seed(train, test, features, cat_mask, l2, msl_val,
                                BASE_SEED)
        n_fits = 1
        base_map = dict(zip(row_ids, base_vec.tolist()))
        preds = {BASE: base_map}
        for lbl, _ in GATES:
            preds[cand_name(lbl)] = base_map  # byte-identical to base

    return (preds, l2, l2_fired, fires_by_gate, msl_val, n_fits, n_train_stat,
            n_obj_stat, ever_fires)


def score_split(pred_map, sol):
    """Given pred_map (row_id->prob) and solution df, return (public_auc, private_auc)."""
    sol = sol.copy()
    sol["pred"] = sol["row_id"].map(pred_map)
    if sol["pred"].isna().any():
        n_missing = int(sol["pred"].isna().sum())
        raise ValueError(f"{n_missing} solution row_ids had no matching prediction")
    pub = sol[sol["Usage"] == "Public"]
    prv = sol[sol["Usage"] == "Private"]
    pub_auc = auc_or_nan(pub["target"], pub["pred"])
    prv_auc = auc_or_nan(prv["target"], prv["pred"])
    return pub_auc, prv_auc


def main():
    warnings.filterwarnings("ignore")
    os.makedirs(OUT_DIR, exist_ok=True)

    stats = load_stats()

    rows = []            # per-dataset results
    exceptions = []      # (dataset, message)
    skipped = []
    total_fits = 0

    ALL_CONFIGS = [BASE] + CANDIDATES
    GATE_LABELS = [lbl for lbl, _ in GATES]

    for i in range(1, N_DATASETS + 1):
        name = f"train_{i:02d}"
        d = os.path.join(DATA_DIR, name)
        train_csv = os.path.join(d, "train.csv")
        test_csv = os.path.join(d, "test.csv")
        sol_csv = os.path.join(d, "solution.csv")

        if not (os.path.exists(train_csv) and os.path.exists(test_csv) and os.path.exists(sol_csv)):
            print(f"[SKIP] {name}: missing train/test/solution")
            skipped.append(name)
            continue

        sol = pd.read_csv(sol_csv)
        rec = {"dataset": name}
        try:
            (preds, l2, l2_fired, fires_by_gate, msl_val, n_fits, n_train_stat,
             n_obj_stat, ever_fires) = run_one(name, train_csv, test_csv, stats)
            total_fits += n_fits
            rec["l2_fired"] = l2_fired
            rec["ever_fires"] = bool(ever_fires)
            rec["n_train"] = n_train_stat
            rec["n_object_cols"] = n_obj_stat
            for lbl in GATE_LABELS:
                rec[f"fires_{lbl}"] = bool(fires_by_gate[lbl])
            for cfg in ALL_CONFIGS:
                pub, prv = score_split(preds[cfg], sol)
                rec[f"{cfg}_pub"] = pub
                rec[f"{cfg}_prv"] = prv
            rec["msl"] = msl_val
            fire_flags = "".join(
                "1" if fires_by_gate[lbl] else "0" for lbl in GATE_LABELS)
            print(f"[OK] {name} (n_train={n_train_stat}, n_obj={n_obj_stat}, "
                  f"ever_fires={ever_fires}, fires[4k/15k/nocap]={fire_flags}, "
                  f"l2={l2}, msl={msl_val}, fits={n_fits}): "
                  f"base pub={rec['base_pub']:.6f} prv={rec['base_prv']:.6f}")
        except Exception as e:  # capture but keep going — CLEAN-RUN diagnostic
            exceptions.append((name, repr(e)))
            rec["l2_fired"] = False
            rec["ever_fires"] = False
            rec["n_train"] = stats.get(name, {}).get("n_train", "")
            rec["n_object_cols"] = stats.get(name, {}).get("n_object_cols", "")
            for lbl in GATE_LABELS:
                rec[f"fires_{lbl}"] = False
            for cfg in ALL_CONFIGS:
                rec[f"{cfg}_pub"] = float("nan")
                rec[f"{cfg}_prv"] = float("nan")
            rec["msl"] = float("nan")
            print(f"[ERROR] {name}: {e!r}")
        rows.append(rec)

    # ---- deltas & helpers (all vs base == shipped 08) ----
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
        """Datasets where cfg is strictly worse than base(08) on this split."""
        out = []
        for r in rows:
            dd = delta(r, cfg, split)
            if not math.isnan(dd) and dd < -eps:
                out.append((r["dataset"], dd))
        return out

    def differing_datasets(cfg, eps=1e-9):
        """Datasets where cfg differs from base on EITHER split."""
        out = []
        for r in rows:
            dp = delta(r, cfg, "pub")
            dv = delta(r, cfg, "prv")
            dp = 0.0 if math.isnan(dp) else dp
            dv = 0.0 if math.isnan(dv) else dv
            if abs(dp) > eps or abs(dv) > eps:
                out.append(r["dataset"])
        return out

    def firing_list_for(lbl):
        return [r["dataset"] for r in rows if r.get(f"fires_{lbl}")]

    # ---- write results CSV ----
    csv_path = os.path.join(OUT_DIR, "results.csv")
    fieldnames = ["dataset", "n_train", "n_object_cols", "l2_fired", "msl",
                  "ever_fires"]
    for lbl in GATE_LABELS:
        fieldnames.append(f"fires_{lbl}")
    fieldnames += ["base_pub", "base_prv"]
    for cfg in CANDIDATES:
        fieldnames += [f"{cfg}_pub", f"{cfg}_d_pub", f"{cfg}_prv", f"{cfg}_d_prv"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            out = {
                "dataset": r["dataset"],
                "n_train": r.get("n_train", ""),
                "n_object_cols": r.get("n_object_cols", ""),
                "l2_fired": r.get("l2_fired", ""),
                "msl": r.get("msl", ""),
                "ever_fires": r.get("ever_fires", ""),
                "base_pub": r.get("base_pub", ""),
                "base_prv": r.get("base_prv", ""),
            }
            for lbl in GATE_LABELS:
                out[f"fires_{lbl}"] = r.get(f"fires_{lbl}", "")
            for cfg in CANDIDATES:
                out[f"{cfg}_pub"] = r.get(f"{cfg}_pub", "")
                out[f"{cfg}_prv"] = r.get(f"{cfg}_prv", "")
                out[f"{cfg}_d_pub"] = delta(r, cfg, "pub")
                out[f"{cfg}_d_prv"] = delta(r, cfg, "prv")
            w.writerow(out)

    # ---- INVARIANT check: for each gate, cand on any dataset that does NOT
    #      fire under it must be byte-identical to base (delta 0). ----
    invariant_violations = []
    for lbl in GATE_LABELS:
        cfg = cand_name(lbl)
        for r in rows:
            if not r.get(f"fires_{lbl}"):
                dp = delta(r, cfg, "pub")
                dv = delta(r, cfg, "prv")
                dp = 0.0 if math.isnan(dp) else dp
                dv = 0.0 if math.isnan(dv) else dv
                if dp != 0.0 or dv != 0.0:
                    invariant_violations.append((cfg, r["dataset"], dp, dv))

    summary_lines = []

    # ---- per-dataset tables (Public, Private) ----
    for split, tag in (("pub", "Public"), ("prv", "Private")):
        if split == "prv":
            summary_lines.append("")
        summary_lines.append(f"=== PER-DATASET ({tag}) — base + 3 gates ===")
        header = (f"{'dataset':<10} {'nTr':>6} {'obj':>4} {'msl':>4} "
                  f"{BASE:>8}")
        for lbl in GATE_LABELS:
            header += f" {'cand_'+lbl:>11} {'d_'+lbl:>10}"
        summary_lines.append(header)
        for r in rows:
            line = (f"{r['dataset']:<10} {str(r.get('n_train')):>6} "
                    f"{str(r.get('n_object_cols')):>4} {str(r.get('msl')):>4} "
                    f"{r[f'{BASE}_{split}']:>8.4f}")
            for lbl in GATE_LABELS:
                cfg = cand_name(lbl)
                line += (f" {r[f'{cfg}_{split}']:>11.4f} "
                         f"{delta(r, cfg, split):>+10.5f}")
            summary_lines.append(line)

    # ---- gate firings per gate ----
    summary_lines.append("")
    summary_lines.append("=== SEED-AVERAGING FIRING SETS PER GATE "
                         "(gate: n_train<cap AND n_object_cols>0, K=5) ===")
    fire_match_all = True
    for lbl, cap in GATES:
        fl = firing_list_for(lbl)
        fset = set(fl)
        expected = EXPECTED_FIRE[lbl]
        ok = (fset == expected)
        fire_match_all = fire_match_all and ok
        cap_str = "inf" if cap == float("inf") else str(cap)
        summary_lines.append(
            f"gate {lbl:<6} (cap={cap_str}) fires on ({len(fl)}): "
            f"{', '.join(fl) if fl else '(none)'}")
        summary_lines.append(
            f"          expected {sorted(expected)} matched: "
            f"{'YES' if ok else 'NO'}"
            + ("" if ok else f" (got {sorted(fset)})"))
    # obj=0 datasets must always be excluded.
    obj0_names = {"train_04", "train_10", "train_11", "train_16"}
    obj0_excluded = all(
        not (obj0_names & set(firing_list_for(lbl))) for lbl in GATE_LABELS)
    summary_lines.append(
        f"obj=0 datasets {sorted(obj0_names)} excluded from ALL gates: "
        f"{'YES' if obj0_excluded else 'NO'}")

    # ---- INVARIANT report ----
    summary_lines.append("")
    summary_lines.append(
        "=== INVARIANT (per gate: cand on datasets not firing under the gate "
        "is identical to base, delta 0) ===")
    if invariant_violations:
        summary_lines.append("VIOLATED! cand differs from base on a non-firing "
                             "dataset:")
        for cfg, ds, dp, dv in invariant_violations:
            summary_lines.append(
                f"  {cfg}/{ds}: pub d={dp:+.6g} prv d={dv:+.6g}")
    else:
        for lbl in GATE_LABELS:
            n_nonfire = len([r for r in rows if not r.get(f"fires_{lbl}")])
            summary_lines.append(
                f"OK gate {lbl}: each of the {n_nonfire} datasets not firing "
                f"under it is byte-identical to base (delta exactly 0).")
        summary_lines.append("Required base-reproduction check. PASS.")

    # ---- which datasets actually differed, per gate ----
    summary_lines.append("")
    summary_lines.append("=== DATASETS THAT ACTUALLY DIFFER (candidate vs base) ===")
    for lbl in GATE_LABELS:
        cfg = cand_name(lbl)
        diff = differing_datasets(cfg)
        summary_lines.append(
            f"{cfg}: ({len(diff)}) {', '.join(diff) if diff else '(none)'}")

    # ---- per-candidate summary vs base(08) ----
    summary_lines.append("")
    summary_lines.append("=== SUMMARY (each gate vs base == shipped 08) ===")
    for lbl in GATE_LABELS:
        cfg = cand_name(lbl)
        mp = mean_delta(cfg, "pub")
        mv = mean_delta(cfg, "prv")
        wp, lp, tp = wlt(cfg, "pub")
        wv, lv, tv = wlt(cfg, "prv")
        summary_lines.append(
            f"gate {lbl:<6} {cfg}: mean Public d={mp:+.5f}  mean Private "
            f"d={mv:+.5f}  Public W/L/T={wp}/{lp}/{tp}  "
            f"Private W/L/T={wv}/{lv}/{tv}")

    # ---- reproduction checks (A==round32, B==round33) ----
    summary_lines.append("")
    summary_lines.append("=== REPRODUCTION CHECKS (A must match round32, "
                         "B must match round33) ===")
    repro_ok = True
    for lbl, (rp, rv) in REF.items():
        cfg = cand_name(lbl)
        mp = mean_delta(cfg, "pub")
        mv = mean_delta(cfg, "prv")
        ok_p = abs(mp - rp) < 5e-6
        ok_v = abs(mv - rv) < 5e-6
        repro_ok = repro_ok and ok_p and ok_v
        ref_round = "round32" if lbl == "4000" else "round33"
        summary_lines.append(
            f"gate {lbl:<6} vs {ref_round}: Public {mp:+.5f} (ref +{rp:.5f}, "
            f"{'YES' if ok_p else 'NO'}); Private {mv:+.5f} (ref +{rv:.5f}, "
            f"{'YES' if ok_v else 'NO'})")

    # ---- KEY QUESTION: does train_12 gain or regress under seed-avg? ----
    summary_lines.append("")
    summary_lines.append("=== KEY QUESTION: train_12 under seed-avg (only gate "
                         "C 'nocap' fires it) ===")
    r12 = next((x for x in rows if x["dataset"] == "train_12"), None)
    t12_regresses = False
    if r12 is None:
        summary_lines.append("  train_12: (missing)")
    else:
        cfgC = cand_name("nocap")
        dp = delta(r12, cfgC, "pub")
        dv = delta(r12, cfgC, "prv")
        summary_lines.append(
            f"  train_12 (n={r12.get('n_train')}, obj={r12.get('n_object_cols')}) "
            f"under gate C: pub {r12[f'{BASE}_pub']:.4f}->{r12[f'{cfgC}_pub']:.4f} "
            f"({dp:+.5f})   prv {r12[f'{BASE}_prv']:.4f}->{r12[f'{cfgC}_prv']:.4f} "
            f"({dv:+.5f})")
        t12_regresses = (dp < -1e-6) or (dv < -1e-6)
        if t12_regresses:
            which = []
            if dp < -1e-6:
                which.append("Public")
            if dv < -1e-6:
                which.append("Private")
            summary_lines.append(
                f"  -> train_12 REGRESSES on {', '.join(which)} under seed-avg. "
                f"No-cap admits a losing dataset.")
        elif (dp > 1e-6) or (dv > 1e-6):
            summary_lines.append(
                "  -> train_12 GAINS under seed-avg (no regression on either "
                "split).")
        else:
            summary_lines.append(
                "  -> train_12 is neutral under seed-avg (no change on either "
                "split).")

    # ---- per-candidate differing-dataset deltas + regressions ----
    summary_lines.append("")
    summary_lines.append("=== PER-GATE DETAIL (datasets differing from base; "
                         "all other deltas are exactly 0) ===")
    for lbl in GATE_LABELS:
        cfg = cand_name(lbl)
        summary_lines.append(f"--- gate {lbl} ({cfg}) vs {BASE} ---")
        diff = set(differing_datasets(cfg))
        for r in rows:
            if r["dataset"] not in diff:
                continue
            dp = delta(r, cfg, "pub")
            dv = delta(r, cfg, "prv")
            summary_lines.append(
                f"  {r['dataset']:<10} "
                f"pub {r[f'{BASE}_pub']:.4f}->{r[f'{cfg}_pub']:.4f} ({dp:+.5f})   "
                f"prv {r[f'{BASE}_prv']:.4f}->{r[f'{cfg}_prv']:.4f} ({dv:+.5f})")
        if not diff:
            summary_lines.append("  (no datasets differ)")
        rp = regressions(cfg, "pub")
        rv = regressions(cfg, "prv")
        pub_str = ", ".join(f"{n}({d:+.5f})" for n, d in rp) if rp else "(none)"
        prv_str = ", ".join(f"{n}({d:+.5f})" for n, d in rv) if rv else "(none)"
        summary_lines.append(f"  Public regressions:  {pub_str}")
        summary_lines.append(f"  Private regressions: {prv_str}")

    # ---- verdict ----
    summary_lines.append("")
    summary_lines.append("=== VERDICT (adoption vs base == shipped 08) ===")
    summary_lines.append(
        "Criterion: CLEAN IMPROVEMENT over 08 iff mean delta positive AND "
        "zero regressions on BOTH splits.")
    for lbl in GATE_LABELS:
        cfg = cand_name(lbl)
        mp = mean_delta(cfg, "pub")
        mv = mean_delta(cfg, "prv")
        rp = regressions(cfg, "pub")
        rv = regressions(cfg, "prv")
        mean_pos = (mp > 1e-9) and (mv > 1e-9)
        zero_regs = (not rp) and (not rv)
        clean = mean_pos and zero_regs
        if clean:
            verdict = (f"CLEAN-IMPROVEMENT "
                       f"(mean pub={mp:+.5f}, prv={mv:+.5f}; no regressions)")
        else:
            reasons = []
            if not (mp > 1e-9):
                reasons.append(f"non-positive mean Public ({mp:+.5f})")
            if not (mv > 1e-9):
                reasons.append(f"non-positive mean Private ({mv:+.5f})")
            if rp:
                reasons.append("Public regs [" +
                               ", ".join(f"{n}({d:+.5f})" for n, d in rp) + "]")
            if rv:
                reasons.append("Private regs [" +
                               ", ".join(f"{n}({d:+.5f})" for n, d in rv) + "]")
            verdict = "NOT-CLEAN (" + "; ".join(reasons) + ")"
        summary_lines.append(f"gate {lbl:<6} {cfg}: {verdict}")

    # ---- endpoint decision: does C (nocap) beat B (15000) on BOTH splits
    #      with zero regressions? ----
    summary_lines.append("")
    summary_lines.append("=== ENDPOINT DECISION (gate C nocap vs gate B "
                         "N=15000) ===")
    cfgB, cfgC = cand_name("15000"), cand_name("nocap")
    mpB, mvB = mean_delta(cfgB, "pub"), mean_delta(cfgB, "prv")
    mpC, mvC = mean_delta(cfgC, "pub"), mean_delta(cfgC, "prv")
    rpC, rvC = regressions(cfgC, "pub"), regressions(cfgC, "prv")
    C_zero_regs = (not rpC) and (not rvC)
    C_beats_B = (mpC > mpB + 1e-9) and (mvC > mvB + 1e-9)
    summary_lines.append(
        f"gate B N=15000: Public {mpB:+.5f}  Private {mvB:+.5f}")
    summary_lines.append(
        f"gate C nocap:   Public {mpC:+.5f} ({mpC-mpB:+.5f} vs B)  "
        f"Private {mvC:+.5f} ({mvC-mvB:+.5f} vs B)  "
        f"zero_regressions={'YES' if C_zero_regs else 'NO'}")
    if C_beats_B and C_zero_regs:
        concl = ("gate C (no cap / obj>0 only) BEATS gate B on BOTH splits "
                 "with zero regressions -> NO-CAP is the definitive recommended "
                 "gate.")
    elif t12_regresses:
        concl = ("train_12 REGRESSES under seed-avg, so no-cap admits a loss -> "
                 "N=15000 (gate B) is the CEILING (last clean point). Keep the "
                 "cap at 15000.")
    elif not C_zero_regs:
        regs = rpC + rvC
        reg_str = ", ".join(f"{n}({d:+.5f})" for n, d in regs)
        concl = (f"gate C has regression(s) [{reg_str}] -> not clean. "
                 f"N=15000 (gate B) remains the ceiling.")
    else:
        concl = ("gate C is clean but does NOT beat gate B on both splits -> "
                 "N=15000 (gate B) remains the recommended gate; no-cap adds no "
                 "net benefit.")
    summary_lines.append("CONCLUSION: " + concl)

    # ---- clean-run line ----
    summary_lines.append("")
    fire_ok = fire_match_all and obj0_excluded
    clean_run = ((not exceptions) and (not invariant_violations)
                 and fire_ok and repro_ok)
    summary_lines.append(
        f"CLEAN RUN={'YES' if clean_run else 'NO'} "
        f"(total_fits={total_fits}, exceptions={len(exceptions)}, "
        f"skipped={len(skipped)}, invariant_violations={len(invariant_violations)}, "
        f"firing_sets_match={'YES' if fire_ok else 'NO'}, "
        f"reproductions_ok={'YES' if repro_ok else 'NO'})")
    for name, msg in exceptions:
        summary_lines.append(f"  EXC {name}: {msg}")

    summary = "\n".join(summary_lines)
    print("\n" + summary)

    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
        f.write(summary + "\n")

    print(f"\n[WROTE] {csv_path}")
    print(f"[WROTE] {os.path.join(OUT_DIR, 'summary.txt')}")
    print("HARNESS_DONE")


if __name__ == "__main__":
    main()
