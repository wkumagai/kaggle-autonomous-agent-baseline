#!/usr/bin/env python
"""
bench_03 round81 — LOWER the L2 GATE THRESHOLD so that datasets sitting just
BELOW the shipped 0.010 cut also receive l2=1.0. OFFLINE ONLY. No subprocess, no
LLM, no Kaggle. Calls sklearn in-process.

Adapted verbatim in structure from
experiments/bench_03/round80_tiny_no_es/replay.py.

Base recipe reproduced (verified vs
  `git show HEAD:submissions/08_ratio_tiered_msl/agent/prompts/system.md`):
  - load train.csv, test.csv with pandas
  - features = [c for c in train.columns if c not in ("row_id","target")]
  - cat_mask  = [train[c].dtype == object for c in features]
  - n = len(train); n_feat = len(features); ratio = n_feat / n
  - L2 GATE:  l2  = 1.0 if ratio >= 0.010 else 0.0                  [THE LEVER]
  - MSL TIERS: msl = 70 if ratio >= 0.030 else (50 if ratio >= 0.015 else 20)
               [FIXED — shipped 08]
  - HistGradientBoostingClassifier(categorical_features=cat_mask, random_state=0,
        max_iter=300, early_stopping=True, l2_regularization=l2,
        min_samples_leaf=msl)
  - fit(train[features], train["target"])
  - pred = clf.predict_proba(test[features])[:, pos_class_1]
Score: ROC AUC on Public rows and Private rows separately, joined by row_id to
solution.csv, over all 16 datasets.

The QUESTION (round81 angle): round18 ESTABLISHED the l2 gate at threshold 0.010
and round19 swept only the l2 MAGNITUDE L at that FIXED threshold. The gate
THRESHOLD itself has never been swept downward. train_05 has
ratio = 9/1060 = 0.00849, sitting just BELOW the 0.010 cut, so it receives
l2 = 0.0 — yet it is the 3rd-weakest dataset in the benchmark (private AUC
0.6461), and round80 showed it overfits readily (removing early stopping in
favor of fixed iterations made it monotonically worse on BOTH splits:
-0.0197/-0.0132). That is direct evidence train_05 wants regularization the
current gate never gives it.

The untested lever: lower ONLY the gate threshold T, so more (lower-ratio)
datasets clear it and receive the SAME l2=1.0 the gated datasets already get.

  l2 = GATED_L2 (1.0) if ratio >= T else 0.0     (T in {0.010, 0.008, 0.005, 0.002})

ONE LEVER ONLY. The l2 MAGNITUDE (1.0), the msl tiers (70/50/20), max_iter=300,
early_stopping=True, the categorical mask and random_state=0 are all taken from
08 UNCHANGED and apply identically in every config.

Configs:
  base : T=0.010  == shipped 08
  T008 : T=0.008
  T005 : T=0.005
  T002 : T=0.002

Expected NEWLY-firing datasets (per dataset_stats.csv ratio = n_features/n_train;
recomputed in code from the actual data, never hardcoded):
  T=0.008: train_05 (0.00849) ONLY
  T=0.005: + train_03 (0.00514)
  T=0.002: + train_10 (0.00229), train_14 (0.00207)
Already firing at every T (ratio >= 0.010, UNCHANGED vs base, must show delta 0):
  train_09 (0.0162), train_13 (0.0180), train_15 (0.0600), train_16 (0.0116)

INVARIANT: every candidate differs from base ONLY on the datasets that NEWLY
fire under its threshold (ratio in [T, 0.010)). Every other dataset — both the
already-firing ones (ratio >= 0.010) and the still-not-firing ones (ratio < T) —
is byte-identical to base and must show delta EXACTLY 0.00000 on both splits.
At T=0.008 that means 15 of 16 datasets must be exactly 0.00000. If any of them
moves, the harness leaks and the scores are meaningless. The harness checks and
reports this mechanically.

Adoption criterion: a candidate is a CLEAN IMPROVEMENT over base(08) iff its
mean delta is positive on BOTH splits AND there are ZERO regressions on BOTH
splits (no dataset worse on either split). A single regression on either split
=> not clean.
"""
import os
import csv
import json
import math
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

REPO = "/Users/kumacmini/kaggle-autonomous-agent-baseline-auto"
DATA_DIR = os.path.join(REPO, "data")
OUT_DIR = os.path.join(REPO, "experiments", "bench_03", "round81_l2_gate_threshold")

BASE_L2_GATE_THRESHOLD = 0.010  # shipped 08 feature-to-row-ratio gate for l2 [THE LEVER]
GATED_L2 = 1.0              # l2 applied when the l2-gate fires (FIXED — not the lever)
DEFAULT_MSL = 20            # sklearn HGB default, used when no msl tier is cleared.
MSL_TIERS = [(0.030, 70), (0.015, 50)]  # shipped 08 tiers (FIXED — not the lever)

BASE_MAX_ITER = 300         # shipped 08 (FIXED)
BASE_EARLY_STOPPING = True  # shipped 08 (FIXED)

# Each config: (name, l2_gate_threshold). base is 08's 0.010 exactly; the
# candidates lower ONLY this threshold, so strictly more datasets clear the gate
# and receive the SAME GATED_L2=1.0. Nothing else varies.
THRESHOLD_VALUES = [("base", 0.010), ("T008", 0.008), ("T005", 0.005), ("T002", 0.002)]
CONFIGS = THRESHOLD_VALUES
BASE = "base"
CANDIDATES = ["T008", "T005", "T002"]
THRESHOLD_OF = dict(THRESHOLD_VALUES)

N_DATASETS = 16


def msl_for_ratio(ratio):
    """First tier (threshold, msl) whose threshold the ratio clears wins;
    tiers are in descending-threshold order. Else DEFAULT_MSL. (Shipped 08.)"""
    for thr, val in MSL_TIERS:
        if ratio >= thr:
            return val
    return DEFAULT_MSL


def auc_or_nan(y_true, y_score):
    """ROC AUC, or NaN if the subset has a single class (undefined)."""
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def run_one(train_csv, test_csv, l2_gate_threshold):
    """Reproduce the shipped 08 recipe for one dataset, applying THE LEVER: the
    l2 gate fires at `l2_gate_threshold` instead of 08's fixed 0.010. Everything
    else (l2 magnitude, msl tiers, cat mask, max_iter, early_stopping,
    random_state) is 08 UNCHANGED. Returns
    (pred_map, l2, l2_fired, msl_val, n, ratio, newly_fired)."""
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)

    features = [c for c in train.columns if c not in ("row_id", "target")]
    cat_mask = [train[c].dtype == object for c in features]

    n = len(train)
    n_feat = len(features)
    ratio = n_feat / n

    # ---- THE LEVER (the only thing that varies across configs) ----
    l2_fired = ratio >= l2_gate_threshold
    l2 = GATED_L2 if l2_fired else 0.0
    # newly_fired: fires under this config but NOT under shipped 08's 0.010 cut.
    newly_fired = l2_fired and not (ratio >= BASE_L2_GATE_THRESHOLD)

    msl_val = msl_for_ratio(ratio)

    clf = HistGradientBoostingClassifier(
        categorical_features=cat_mask,
        random_state=0,
        max_iter=BASE_MAX_ITER,
        early_stopping=BASE_EARLY_STOPPING,
        l2_regularization=l2,
        min_samples_leaf=msl_val,
    )
    clf.fit(train[features], train["target"])
    proba = clf.predict_proba(test[features])

    # positive-class column: match recipe's [:, 1] but be robust to class order.
    classes = list(clf.classes_)
    pos_idx = classes.index(1) if 1 in classes else 1
    pred = proba[:, pos_idx]

    return (dict(zip(test["row_id"].tolist(), pred.tolist())),
            l2, l2_fired, msl_val, n, ratio, newly_fired)


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

    rows = []            # per-dataset results
    exceptions = []      # (dataset, config, message)
    skipped = []
    n_fits_ok = 0        # successful fits — CLEAN-RUN accounting

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
        for cfg_name, l2_gate_threshold in CONFIGS:
            try:
                (pred_map, l2, l2_fired, msl_val, n, ratio,
                 newly_fired) = run_one(train_csv, test_csv, l2_gate_threshold)
                pub, prv = score_split(pred_map, sol)
                rec[f"{cfg_name}_pub"] = pub
                rec[f"{cfg_name}_prv"] = prv
                rec[f"{cfg_name}_l2"] = l2
                rec[f"{cfg_name}_l2_fired"] = l2_fired
                rec[f"{cfg_name}_newly_fired"] = newly_fired
                # n, ratio and the msl tier are config-independent (all FIXED
                # across configs; only the l2 gate threshold moves).
                rec["n_train"] = n
                rec["ratio"] = ratio
                rec["msl"] = msl_val
                rec["base_l2_fired"] = ratio >= BASE_L2_GATE_THRESHOLD
                n_fits_ok += 1
                print(f"[OK] {name} {cfg_name} (n={n}, ratio={ratio:.5f}, T={l2_gate_threshold}, "
                      f"l2={l2}, l2_fired={l2_fired}, newly_fired={newly_fired}, "
                      f"msl={msl_val}): pub={pub:.6f} prv={prv:.6f}")
            except Exception as e:  # capture but keep going — CLEAN-RUN diagnostic
                exceptions.append((name, cfg_name, repr(e)))
                rec[f"{cfg_name}_pub"] = float("nan")
                rec[f"{cfg_name}_prv"] = float("nan")
                print(f"[ERROR] {name} {cfg_name}: {e!r}")
        rows.append(rec)

    # ---- write results CSV + JSON (raw per-dataset numbers) ----
    csv_path = os.path.join(OUT_DIR, "results.csv")
    fieldnames = ["dataset", "n_train", "ratio", "base_l2_fired", "msl"]
    for cfg, _ in CONFIGS:
        fieldnames += [f"{cfg}_pub", f"{cfg}_prv", f"{cfg}_l2",
                       f"{cfg}_l2_fired", f"{cfg}_newly_fired"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    json_path = os.path.join(OUT_DIR, "results.json")
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2, default=str)

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

    def newly_firing(cfg):
        """Datasets that NEWLY clear the gate under cfg (ratio in [T, 0.010)) —
        the ONLY datasets allowed to move vs base."""
        return [r["dataset"] for r in rows if r.get(f"{cfg}_newly_fired")]

    # union of every dataset that newly fires under ANY candidate
    touched_any = sorted({d for cfg in CANDIDATES for d in newly_firing(cfg)})
    base_fired_list = [r["dataset"] for r in rows if r.get("base_l2_fired")]

    summary_lines = []

    # ---- per-dataset tables (Public, Private) ----
    for split, tag in (("pub", "Public"), ("prv", "Private")):
        if split == "prv":
            summary_lines.append("")
        summary_lines.append(f"=== PER-DATASET ({tag}) ===")
        header = (f"{'dataset':<10} {'n':>6} {'ratio':>8} {'l2@08':>6} {'msl':>4} "
                  f"{BASE:>9}")
        for cfg in CANDIDATES:
            header += f" {cfg:>9} {'d'+cfg:>10}"
        summary_lines.append(header)
        for r in rows:
            line = (f"{r['dataset']:<10} {r.get('n_train'):>6} "
                    f"{r.get('ratio'):>8.5f} "
                    f"{str(bool(r.get('base_l2_fired'))):>6} {str(r.get('msl')):>4} "
                    f"{r[f'{BASE}_{split}']:>9.4f}")
            for cfg in CANDIDATES:
                line += (f" {r[f'{cfg}_{split}']:>9.4f} "
                         f"{delta(r, cfg, split):>+10.5f}")
            summary_lines.append(line)

    # ---- gate firings ----
    summary_lines.append("")
    summary_lines.append("=== GATE FIRINGS ===")
    summary_lines.append(
        f"L2 GATE @ base T={BASE_L2_GATE_THRESHOLD} (shipped 08) fires on "
        f"({len(base_fired_list)}): "
        f"{', '.join(base_fired_list) if base_fired_list else '(none)'}")
    for cfg in CANDIDATES:
        nf = newly_firing(cfg)
        fired = [r["dataset"] for r in rows if r.get(f"{cfg}_l2_fired")]
        summary_lines.append(
            f"L2 GATE @ T={THRESHOLD_OF[cfg]} ({cfg}) fires on ({len(fired)}): "
            f"{', '.join(fired)}   |  NEWLY firing vs 08 ({len(nf)}): "
            f"{', '.join(nf) if nf else '(none)'}")
    summary_lines.append(
        f"Datasets touched by the lever under ANY candidate ({len(touched_any)}): "
        f"{', '.join(touched_any) if touched_any else '(none)'}")
    summary_lines.append(
        f"(all configs share the l2 MAGNITUDE ({GATED_L2}), the msl tiers "
        f"(70/50/20), max_iter={BASE_MAX_ITER}, early_stopping={BASE_EARLY_STOPPING}, "
        f"the categorical mask and random_state=0; they differ ONLY in the l2 gate "
        f"THRESHOLD -> only newly-firing datasets can move. The "
        f"{len(base_fired_list)} already-firing datasets and every dataset still "
        f"below T are identical -> delta 0)")

    # ---- INVARIANT CHECK: only newly-firing datasets may differ ----
    summary_lines.append("")
    summary_lines.append("=== INVARIANT CHECK (every dataset that does NOT newly "
                         "fire must be byte-identical to base: delta EXACTLY "
                         "0.00000 on both splits) ===")
    invariant_ok = True
    for cfg in CANDIDATES:
        diff = differing_datasets(cfg)
        nf = newly_firing(cfg)
        violators = [d for d in diff if d not in nf]
        n_identical = len([r for r in rows if r["dataset"] not in diff])
        n_expected_identical = len(rows) - len(nf)
        ok = not violators
        invariant_ok = invariant_ok and ok
        summary_lines.append(
            f"{cfg} (T={THRESHOLD_OF[cfg]}): differs from base on: "
            f"{', '.join(diff) if diff else '(none)'}  "
            f"(expected subset of newly-firing: "
            f"{', '.join(nf) if nf else '(none)'})  |  "
            f"identical (delta exactly 0 on both splits): {n_identical}/{len(rows)} "
            f"(expected {n_expected_identical}/{len(rows)})  "
            f"-> {'OK' if ok else 'VIOLATION: ' + ', '.join(violators)}")
    # explicit machine check of the non-newly-firing datasets, both splits, exact zero
    invariant_max_abs = 0.0
    for cfg in CANDIDATES:
        nf = set(newly_firing(cfg))
        for r in rows:
            if r["dataset"] in nf:
                continue
            for split in ("pub", "prv"):
                dd = delta(r, cfg, split)
                if not math.isnan(dd):
                    invariant_max_abs = max(invariant_max_abs, abs(dd))
    summary_lines.append(
        f"max |delta| over all NON-newly-firing (dataset x candidate x split) "
        f"cells = {invariant_max_abs:.10g} (must be exactly 0)")
    invariant_ok = invariant_ok and (invariant_max_abs == 0.0)
    summary_lines.append(
        f"INVARIANT={'HOLDS' if invariant_ok else 'VIOLATED — HARNESS LEAKS, '
                    'SCORES ARE NOT TRUSTWORTHY'} "
        f"(only datasets whose ratio lands in [T, {BASE_L2_GATE_THRESHOLD}) are "
        f"affected by the lever, as designed)")

    # ---- THE SWEEP: gate threshold vs the datasets it newly touches ----
    summary_lines.append("")
    summary_lines.append("=== NEWLY-GATED DATASET SWEEP (the datasets the lever touches) ===")
    for r in rows:
        if r["dataset"] not in touched_any:
            continue
        summary_lines.append(
            f"--- {r['dataset']} (n={r.get('n_train')}, ratio={r.get('ratio'):.5f}, "
            f"msl={r.get('msl')}) ---")
        summary_lines.append(f"  {'config':>8} {'T':>7} {'l2':>5} {'Public':>9} "
                             f"{'dPub':>10} {'Private':>9} {'dPrv':>10}")
        for cfg, thr in CONFIGS:
            dp = delta(r, cfg, "pub")
            dv = delta(r, cfg, "prv")
            tag = "  <- base (shipped 08)" if cfg == BASE else ""
            summary_lines.append(
                f"  {cfg:>8} {thr:>7} {str(r.get(f'{cfg}_l2')):>5} "
                f"{r[f'{cfg}_pub']:>9.5f} {dp:>+10.5f} "
                f"{r[f'{cfg}_prv']:>9.5f} {dv:>+10.5f}{tag}")
        # curve / peak read-out per split
        for split, tag in (("pub", "Public"), ("prv", "Private")):
            vals = [(c, r[f"{c}_{split}"]) for c, _ in CONFIGS]
            best_cfg, best_auc = max(vals, key=lambda t: t[1])
            seq = " -> ".join(f"{c}:{v:.5f}" for c, v in vals)
            summary_lines.append(
                f"  {tag} curve: {seq}   BEST={best_cfg} ({best_auc:.5f})")
        # adjacent-setting spread: the jitter yardstick for this dataset
        for split, tag in (("pub", "Public"), ("prv", "Private")):
            cand_vals = [r[f"{c}_{split}"] for c in CANDIDATES]
            adj = [abs(cand_vals[j + 1] - cand_vals[j]) for j in range(len(cand_vals) - 1)]
            summary_lines.append(
                f"  {tag} adjacent-T spread (|T008-T005|, |T005-T002|): "
                + ", ".join(f"{a:.5f}" for a in adj)
                + f"   max={max(adj):.5f}  [yardstick: a 'gain' smaller than this "
                  f"is fit-jitter, not signal]")

    # ---- per-candidate summary vs base(08) ----
    summary_lines.append("")
    summary_lines.append("=== SUMMARY (candidates vs base == shipped 08, "
                         "mean over all 16 datasets) ===")
    for cfg in CANDIDATES:
        mp = mean_delta(cfg, "pub")
        mv = mean_delta(cfg, "prv")
        wp, lp, tp = wlt(cfg, "pub")
        wv, lv, tv = wlt(cfg, "prv")
        summary_lines.append(
            f"{cfg} (l2 gate T={THRESHOLD_OF[cfg]}): "
            f"mean Public d={mp:+.5f}  mean Private d={mv:+.5f}  "
            f"Public W/L/T={wp}/{lp}/{tp}  Private W/L/T={wv}/{lv}/{tv}")

    # ---- per-candidate differing-dataset deltas + regressions ----
    summary_lines.append("")
    summary_lines.append("=== PER-CANDIDATE DETAIL (datasets differing from base; "
                         "all other deltas are exactly 0) ===")
    for cfg in CANDIDATES:
        summary_lines.append(f"--- {cfg} (T={THRESHOLD_OF[cfg]}) vs {BASE} "
                             f"(T={BASE_L2_GATE_THRESHOLD}) ---")
        diff = set(differing_datasets(cfg))
        for r in rows:
            if r["dataset"] not in diff:
                continue
            dp = delta(r, cfg, "pub")
            dv = delta(r, cfg, "prv")
            summary_lines.append(
                f"  {r['dataset']:<10} (n={r.get('n_train')}, "
                f"ratio={r.get('ratio'):.5f}, "
                f"l2 {r.get(f'{BASE}_l2')}->{r.get(f'{cfg}_l2')})  "
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
    clean_names = []
    for cfg in CANDIDATES:
        mp = mean_delta(cfg, "pub")
        mv = mean_delta(cfg, "prv")
        rp = regressions(cfg, "pub")
        rv = regressions(cfg, "prv")
        mean_pos = (mp > 1e-9) and (mv > 1e-9)
        zero_regs = (not rp) and (not rv)
        clean = mean_pos and zero_regs
        if clean:
            clean_names.append(cfg)
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
        summary_lines.append(
            f"{cfg} (l2 gate T={THRESHOLD_OF[cfg]}): {verdict}")
    summary_lines.append("")
    if clean_names:
        # best clean candidate = largest mean Public delta (tie-break Private)
        best = max(clean_names,
                   key=lambda c: (mean_delta(c, "pub"), mean_delta(c, "prv")))
        summary_lines.append(
            f"OVERALL: clean improvement over 08 found: {', '.join(clean_names)}; "
            f"best = {best} (T={THRESHOLD_OF[best]}, "
            f"mean pub={mean_delta(best, 'pub'):+.5f}, "
            f"prv={mean_delta(best, 'prv'):+.5f}) "
            f"(orchestrator decides adoption; compare the gain against the "
            f"adjacent-T spread above before believing it).")
    else:
        summary_lines.append(
            "OVERALL: NO clean improvement over 08; base (shipped 08, l2 gate "
            f"T={BASE_L2_GATE_THRESHOLD}) remains best. Lowering the l2 gate "
            "threshold so that below-cut datasets also receive l2=1.0 did not "
            "cleanly help.")

    # ---- clean-run line ----
    n_fits_expected = len(rows) * len(CONFIGS)
    summary_lines.append("")
    summary_lines.append(
        f"CLEAN RUN={'YES' if not exceptions else 'NO'} "
        f"(fits ok={n_fits_ok}/{n_fits_expected} "
        f"[{len(rows)} datasets x {len(CONFIGS)} configs], "
        f"exceptions={len(exceptions)}, skipped={len(skipped)})")
    for name, cfg, msg in exceptions:
        summary_lines.append(f"  EXC {name}/{cfg}: {msg}")

    summary = "\n".join(summary_lines)
    print("\n" + summary)

    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
        f.write(summary + "\n")

    print(f"\n[WROTE] {csv_path}")
    print(f"[WROTE] {json_path}")
    print(f"[WROTE] {os.path.join(OUT_DIR, 'summary.txt')}")
    print("HARNESS_DONE")


if __name__ == "__main__":
    main()
