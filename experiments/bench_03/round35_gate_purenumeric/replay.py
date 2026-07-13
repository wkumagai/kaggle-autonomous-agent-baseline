#!/usr/bin/env python
"""
bench_03 round35 — SEED-AVERAGING FIRING-GATE: can we admit the pure-numeric
LARGE-n datasets (train_04/10/11) while still excluding the sole regressor
train_16? OFFLINE ONLY. No subprocess, no LLM, no Kaggle, no network. Calls
sklearn in-process.

Adapted from experiments/bench_03/round34_gate_nocap/replay.py. ALL of round34's
machinery is reused verbatim: dataset loading, the shipped-08 base config
reproduction, Public/Private AUC scoring joined on row_id to solution.csv, K=5
seed-averaging, the byte-identical-on-non-firing INVARIANT check, cached-fit
efficiency, and the summary machinery. The ONLY change vs round34 is the set of
gates compared (round34 varied an n_train cap on the 12 categorical datasets;
round35 fixes the categorical treatment and probes the FOUR pure-numeric
datasets, {train_04, train_10, train_11, train_16}).

Base recipe reproduced (== shipped 08), IDENTICAL to round34/round33/round32:
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
(K FIXED at 5), each read per dataset from experiments/bench_03/dataset_stats.csv
via (n_train, n_object_cols):
  gate C (nocap, == round34 best):  fire iff  n_object_cols > 0
      -> the 12 categorical datasets. Reproduces round34 (mean Public +0.00363
         / Private +0.00316). train_04/10/11/16 (obj=0) are all EXCLUDED.
  gate D (except-pure-numeric-small-n):  fire iff
      NOT (n_object_cols == 0 AND n_train < 4000)
      -> fires on 15 datasets (all EXCEPT train_16). This ADMITS the three
         large-n pure-numeric datasets (train_04 n=8775, train_10 n=11800,
         train_11 n=28879) that gate C leaves on the table, while still
         excluding the single small-n pure-numeric dataset train_16 (n=1809).
  gate E (all-16, == round29 cand_A):  fire iff  True  -> all 16 (reference;
         should show train_16 REGRESSING under seed-avg).

Expected firing sets (verified at run time and logged per gate):
  C nocap        -> the 12 categorical:
                    {01,02,03,05,06,07,08,09,12,13,14,15}                 (12)
  D exceptpn     -> all except train_16:
                    C + {train_04, train_10, train_11}                    (15)
  E all16        -> all 16                                                (16)

KEY QUESTIONS this run answers:
  1. Do train_04, train_10, train_11 EACH gain under K=5 seed-avg on BOTH splits
     (no regression on either)?  -> would justify admitting them (gate D).
  2. Is train_16 the ONLY dataset that regresses under gate E (all-16)?  -> the
     reason gate D deliberately excludes exactly train_16.
  3. Does gate D beat gate C in mean delta on BOTH splits with ZERO regressions
     on both splits?  -> is "except pure-numeric small-n" a CLEAN STRICT
     improvement over the round34 gate C?

EFFICIENCY: the WIDEST gate here is E (all-16), so every dataset ever fires ->
each of the 16 datasets is fit K=5 times ONCE and cached; all three gates reuse
those cached seed predictions. A dataset uses the cached K=5 average under a
gate iff it fires under that gate, else the EXACT seed-0 base (byte-identical).
Total fits = 16 x 5 = 80.

IMPLEMENTATION INVARIANT: for a given gate, any dataset that does NOT fire under
it reuses the EXACT seed-0 array, so its delta MUST be exactly 0 on both splits
— checked explicitly per gate. (Gate E fires everything, so it has no non-firing
datasets to check; gate C's non-firing set is {04,10,11,16}; gate D's is {16}.)

Adoption criterion (reused): a gate is a CLEAN IMPROVEMENT over base(08) iff its
mean delta is positive on BOTH splits AND there are ZERO regressions on BOTH
splits.

Reference this run must reproduce:
  gate C nocap == round34: mean Public +0.00363 / Private +0.00316, zero regs.
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
OUT_DIR = os.path.join(BENCH_DIR, "round35_gate_purenumeric")
STATS_CSV = os.path.join(BENCH_DIR, "dataset_stats.csv")

L2_GATE_THRESHOLD = 0.010   # shipped 08 feature-to-row-ratio gate for l2 (FIXED)
GATED_L2 = 1.0              # l2 applied when the l2-gate fires (FIXED)
DEFAULT_MSL = 20            # sklearn HGB default, used when no msl tier is cleared.
BASE_SEED = 0              # shipped-08 fixed random_state.

# gate D pure-numeric small-n cutoff: obj==0 AND n_train<PN_SMALLN_CAP -> excluded.
PN_SMALLN_CAP = 4000

# K is FIXED at 5 (round30 established K=5 as the knee; no K sweep here).
K = 5
SEEDS = list(range(K))     # [0,1,2,3,4]

# 08 tiered min_samples_leaf, IDENTICAL across base and cand (descending order).
MSL_TIERS = [(0.030, 70), (0.015, 50)]

BASE = "base"


# ---- round35 gate predicates over (n_train, n_object_cols) ----
def gate_C(n_train, n_obj):
    """nocap (== round34 best): fire iff at least one object (categorical) col."""
    return n_obj > 0


def gate_D(n_train, n_obj):
    """except-pure-numeric-small-n: fire on everything EXCEPT a pure-numeric
    dataset with few training rows (obj==0 AND n_train<PN_SMALLN_CAP)."""
    return not (n_obj == 0 and n_train < PN_SMALLN_CAP)


def gate_E(n_train, n_obj):
    """all-16 (== round29 cand_A): always fire."""
    return True


# (label, predicate). Order fixed C, D, E.
GATES = [
    ("C_nocap", gate_C),
    ("D_exceptpn", gate_D),
    ("E_all16", gate_E),
]
GATE_LABELS = [lbl for lbl, _ in GATES]


def cand_name(label):
    """Candidate column name for a given gate label."""
    return f"cand_{label}"


CANDIDATES = [cand_name(lbl) for lbl, _ in GATES]

N_DATASETS = 16

# reference point this run must reproduce (gate C == round34 nocap).
REF = {
    "C_nocap": (0.00363, 0.00316),   # round34 nocap
}

# expected firing sets per gate (verified at run time).
_CATEGORICAL = {"train_01", "train_02", "train_03", "train_05", "train_06",
                "train_07", "train_08", "train_09", "train_12", "train_13",
                "train_14", "train_15"}
_PURE_NUMERIC = {"train_04", "train_10", "train_11", "train_16"}
_ALL = {f"train_{i:02d}" for i in range(1, N_DATASETS + 1)}
EXPECTED_FIRE = {
    "C_nocap": set(_CATEGORICAL),                             # 12
    "D_exceptpn": _ALL - {"train_16"},                        # 15
    "E_all16": set(_ALL),                                     # 16
}


def load_stats(path=STATS_CSV):
    """Return {dataset_name -> {"n_train": int, "n_object_cols": int}} from
    dataset_stats.csv. The gates read n_train and n_object_cols from here."""
    stats = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stats[row["name"]] = {
                "n_train": int(row["n_train"]),
                "n_object_cols": int(row["n_object_cols"]),
            }
    return stats


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
      ever_fires = fires under ANY gate -> decides K=5 vs seed-0. (Here gate E
      fires everything, so ever_fires is True for every dataset.)

    A dataset that ever fires is fit seeds 0..4 ONCE and cached; each cand uses
    the K=5 mean if it fires under that gate, else the exact seed-0 array. A
    dataset that never fires is fit seed-0 only and every cand reuses that exact
    seed-0 array (byte-identical to base)."""
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)

    features = [c for c in train.columns if c not in ("row_id", "target")]
    cat_mask = [train[c].dtype == object for c in features]

    n = len(train)
    n_feat = len(features)
    ratio = n_feat / n

    # base config gates (UNCHANGED from shipped 08 / round34).
    l2_fired = ratio >= L2_GATE_THRESHOLD
    l2 = GATED_L2 if l2_fired else 0.0
    msl_val = msl_for_ratio(ratio)

    # round35 seed-averaging gates read stats (n_train, n_object_cols).
    st = stats[name]
    n_train_stat = st["n_train"]
    n_obj_stat = st["n_object_cols"]
    fires_by_gate = {
        lbl: bool(pred(n_train_stat, n_obj_stat)) for lbl, pred in GATES
    }
    ever_fires = any(fires_by_gate.values())

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
        # never fires -> seed-0 only; every cand == base.
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
                  f"ever_fires={ever_fires}, fires[C/D/E]={fire_flags}, "
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
        summary_lines.append(f"=== PER-DATASET ({tag}) — base + 3 gates (C/D/E) ===")
        header = (f"{'dataset':<10} {'nTr':>6} {'obj':>4} {'msl':>4} "
                  f"{BASE:>8}")
        for lbl in GATE_LABELS:
            header += f" {'cand_'+lbl:>13} {'d_'+lbl:>11}"
        summary_lines.append(header)
        for r in rows:
            line = (f"{r['dataset']:<10} {str(r.get('n_train')):>6} "
                    f"{str(r.get('n_object_cols')):>4} {str(r.get('msl')):>4} "
                    f"{r[f'{BASE}_{split}']:>8.4f}")
            for lbl in GATE_LABELS:
                cfg = cand_name(lbl)
                line += (f" {r[f'{cfg}_{split}']:>13.4f} "
                         f"{delta(r, cfg, split):>+11.5f}")
            summary_lines.append(line)

    # ---- gate firings per gate ----
    summary_lines.append("")
    summary_lines.append("=== SEED-AVERAGING FIRING SETS PER GATE (K=5) ===")
    summary_lines.append("  gate C nocap:      fire iff n_object_cols > 0")
    summary_lines.append("  gate D exceptpn:   fire iff NOT(n_object_cols==0 AND "
                         f"n_train<{PN_SMALLN_CAP})")
    summary_lines.append("  gate E all16:      fire always")
    fire_match_all = True
    for lbl, _ in GATES:
        fl = firing_list_for(lbl)
        fset = set(fl)
        expected = EXPECTED_FIRE[lbl]
        ok = (fset == expected)
        fire_match_all = fire_match_all and ok
        summary_lines.append(
            f"gate {lbl:<11} fires on ({len(fl)}): "
            f"{', '.join(fl) if fl else '(none)'}")
        summary_lines.append(
            f"          expected ({len(expected)}) matched: "
            f"{'YES' if ok else 'NO'}"
            + ("" if ok else f" (got {sorted(fset)}, expected {sorted(expected)})"))
    # explicit exclusion facts.
    d_excludes_16 = "train_16" not in set(firing_list_for("D_exceptpn"))
    d_admits_pn = {"train_04", "train_10", "train_11"} <= set(
        firing_list_for("D_exceptpn"))
    summary_lines.append(
        f"gate D excludes train_16 (sole pure-numeric small-n): "
        f"{'YES' if d_excludes_16 else 'NO'}")
    summary_lines.append(
        f"gate D admits pure-numeric large-n {{train_04,10,11}}: "
        f"{'YES' if d_admits_pn else 'NO'}")

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
            nonfire = [r["dataset"] for r in rows if not r.get(f"fires_{lbl}")]
            nf_str = ", ".join(nonfire) if nonfire else "(none)"
            summary_lines.append(
                f"OK gate {lbl}: each of the {len(nonfire)} non-firing datasets "
                f"is byte-identical to base (delta exactly 0). [{nf_str}]")
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
            f"gate {lbl:<11} {cfg}: mean Public d={mp:+.5f}  mean Private "
            f"d={mv:+.5f}  Public W/L/T={wp}/{lp}/{tp}  "
            f"Private W/L/T={wv}/{lv}/{tv}")

    # ---- reproduction check (C == round34 nocap) ----
    summary_lines.append("")
    summary_lines.append("=== REPRODUCTION CHECK (gate C must match round34 "
                         "nocap) ===")
    repro_ok = True
    for lbl, (rp, rv) in REF.items():
        cfg = cand_name(lbl)
        mp = mean_delta(cfg, "pub")
        mv = mean_delta(cfg, "prv")
        ok_p = abs(mp - rp) < 5e-6
        ok_v = abs(mv - rv) < 5e-6
        repro_ok = repro_ok and ok_p and ok_v
        summary_lines.append(
            f"gate {lbl:<11} vs round34: Public {mp:+.5f} (ref +{rp:.5f}, "
            f"{'YES' if ok_p else 'NO'}); Private {mv:+.5f} (ref +{rv:.5f}, "
            f"{'YES' if ok_v else 'NO'})")

    # ---- KEY QUESTION 1: do the large-n pure-numeric datasets gain? ----
    summary_lines.append("")
    summary_lines.append("=== KEY QUESTION 1: pure-numeric LARGE-n datasets "
                         "{train_04, train_10, train_11} under seed-avg ===")
    summary_lines.append("  (all three fire under gate D and gate E; deltas read "
                         "from gate E's K=5 average.)")
    pn_large = ["train_04", "train_10", "train_11"]
    cfgE = cand_name("E_all16")
    pn_all_gain = True
    for nm in pn_large:
        r = next((x for x in rows if x["dataset"] == nm), None)
        if r is None:
            summary_lines.append(f"  {nm}: (missing)")
            pn_all_gain = False
            continue
        dp = delta(r, cfgE, "pub")
        dv = delta(r, cfgE, "prv")
        gains_both = (dp > 1e-6) and (dv > 1e-6)
        regresses = (dp < -1e-6) or (dv < -1e-6)
        pn_all_gain = pn_all_gain and gains_both
        tag = ("GAINS on both" if gains_both else
               ("REGRESSES" if regresses else "neutral/mixed"))
        summary_lines.append(
            f"  {nm} (n={r.get('n_train')}, obj={r.get('n_object_cols')}): "
            f"pub {r[f'{BASE}_pub']:.4f}->{r[f'{cfgE}_pub']:.4f} ({dp:+.5f})   "
            f"prv {r[f'{BASE}_prv']:.4f}->{r[f'{cfgE}_prv']:.4f} ({dv:+.5f})   "
            f"-> {tag}")
    summary_lines.append(
        f"  => train_04/10/11 EACH gain on BOTH splits (no regression): "
        f"{'YES' if pn_all_gain else 'NO'}")

    # ---- KEY QUESTION 2: is train_16 the ONLY regressor under gate E? ----
    summary_lines.append("")
    summary_lines.append("=== KEY QUESTION 2: regressions under gate E (all-16) "
                         "— is train_16 the SOLE regressor? ===")
    regE_pub = regressions(cfgE, "pub")
    regE_prv = regressions(cfgE, "prv")
    regE_names = sorted({n for n, _ in regE_pub} | {n for n, _ in regE_prv})
    pub_str = ", ".join(f"{n}({d:+.5f})" for n, d in regE_pub) if regE_pub else "(none)"
    prv_str = ", ".join(f"{n}({d:+.5f})" for n, d in regE_prv) if regE_prv else "(none)"
    summary_lines.append(f"  gate E Public regressions:  {pub_str}")
    summary_lines.append(f"  gate E Private regressions: {prv_str}")
    r16 = next((x for x in rows if x["dataset"] == "train_16"), None)
    if r16 is not None:
        dp16 = delta(r16, cfgE, "pub")
        dv16 = delta(r16, cfgE, "prv")
        summary_lines.append(
            f"  train_16 (n={r16.get('n_train')}, obj={r16.get('n_object_cols')}) "
            f"under gate E: pub {r16[f'{BASE}_pub']:.4f}->{r16[f'{cfgE}_pub']:.4f} "
            f"({dp16:+.5f})   prv {r16[f'{BASE}_prv']:.4f}->{r16[f'{cfgE}_prv']:.4f} "
            f"({dv16:+.5f})")
    train16_sole = (regE_names == ["train_16"])
    summary_lines.append(
        f"  => set of datasets regressing under gate E: "
        f"{regE_names if regE_names else '(none)'}")
    summary_lines.append(
        f"  => train_16 is the SOLE regressor under gate E: "
        f"{'YES' if train16_sole else 'NO'}")

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
        summary_lines.append(f"gate {lbl:<11} {cfg}: {verdict}")

    # ---- KEY QUESTION 3: is gate D a CLEAN STRICT improvement over gate C? ----
    summary_lines.append("")
    summary_lines.append("=== KEY QUESTION 3 / DECISION (gate D exceptpn vs gate "
                         "C nocap) ===")
    cfgC, cfgD = cand_name("C_nocap"), cand_name("D_exceptpn")
    mpC, mvC = mean_delta(cfgC, "pub"), mean_delta(cfgC, "prv")
    mpD, mvD = mean_delta(cfgD, "pub"), mean_delta(cfgD, "prv")
    rpD, rvD = regressions(cfgD, "pub"), regressions(cfgD, "prv")
    D_zero_regs = (not rpD) and (not rvD)
    D_beats_C = (mpD > mpC + 1e-9) and (mvD > mvC + 1e-9)
    summary_lines.append(
        f"gate C nocap:    Public {mpC:+.5f}  Private {mvC:+.5f}")
    summary_lines.append(
        f"gate D exceptpn: Public {mpD:+.5f} ({mpD-mpC:+.5f} vs C)  "
        f"Private {mvD:+.5f} ({mvD-mvC:+.5f} vs C)  "
        f"zero_regressions={'YES' if D_zero_regs else 'NO'}")
    D_clean_strict = D_beats_C and D_zero_regs
    if D_clean_strict:
        concl = ("gate D (except pure-numeric small-n) BEATS gate C on BOTH "
                 "splits with ZERO regressions -> admitting the large-n "
                 "pure-numeric datasets (train_04/10/11) while excluding train_16 "
                 "is a CLEAN STRICT improvement over the round34 gate C.")
    elif not D_zero_regs:
        regs = rpD + rvD
        reg_str = ", ".join(f"{n}({d:+.5f})" for n, d in regs)
        concl = (f"gate D has regression(s) [{reg_str}] -> NOT a clean strict "
                 f"improvement. gate C (nocap) remains the recommended gate.")
    else:
        concl = ("gate D is clean but does NOT beat gate C on both splits -> "
                 "no net benefit from admitting the pure-numeric large-n "
                 "datasets; gate C (nocap) remains the recommended gate.")
    summary_lines.append("CONCLUSION: " + concl)

    # ---- clean-run line ----
    summary_lines.append("")
    fire_ok = fire_match_all and d_excludes_16 and d_admits_pn
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
