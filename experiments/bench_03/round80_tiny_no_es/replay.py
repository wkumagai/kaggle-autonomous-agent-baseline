#!/usr/bin/env python
"""
bench_03 round80 — REMOVE early stopping on small-n datasets and regularize with
a fixed, modest iteration count instead. OFFLINE ONLY. No subprocess, no LLM,
no Kaggle. Calls sklearn in-process.

Adapted verbatim in structure from
experiments/bench_03/round79_msl_toptier/replay.py.

Base recipe reproduced (verified vs
  `git show HEAD:submissions/08_ratio_tiered_msl/agent/prompts/system.md`):
  - load train.csv, test.csv with pandas
  - features = [c for c in train.columns if c not in ("row_id","target")]
  - cat_mask  = [train[c].dtype == object for c in features]
  - n = len(train); n_feat = len(features); ratio = n_feat / n
  - L2 GATE:  l2  = 1.0 if ratio >= 0.010 else 0.0                  [FIXED]
  - MSL TIERS: msl = 70 if ratio >= 0.030 else (50 if ratio >= 0.015 else 20)
               [FIXED — shipped 08]
  - HistGradientBoostingClassifier(categorical_features=cat_mask, random_state=0,
        max_iter=300, early_stopping=True, l2_regularization=l2,
        min_samples_leaf=msl)
  - fit(train[features], train["target"])
  - pred = clf.predict_proba(test[features])[:, pos_class_1]
Score: ROC AUC on Public rows and Private rows separately, joined by row_id to
solution.csv, over all 16 datasets.

The QUESTION (round80 angle): rounds 24 (validation_fraction) and 48
(n_iter_no_change) TUNED early stopping but never REMOVED it. Every config ever
tried on this benchmark has had early_stopping=True. On a tiny dataset that is
doubly costly:
  1. HGB holds out validation_fraction=0.1 of the rows to decide when to stop.
     For n=500 that is ~50 rows — the stopping decision is made on an extremely
     noisy AUC estimate.
  2. Those ~50 rows never reach the trees, so the model fits on ~450 rows.
Round79 explicitly attributed train_15's erratic msl response to this holdout.

The untested complement: for small-n datasets, turn early stopping OFF and let a
fixed, modest max_iter be the regularizer instead — every row is used to fit, and
the iteration count is set a priori rather than chosen from a 50-row signal.

  if n < 2000:  early_stopping=False, max_iter=F     (F in {50, 100, 200})
  else:         early_stopping=True,  max_iter=300   (shipped 08, UNCHANGED)

ONE LEVER ONLY. The l2 gate (ratio>=0.010 -> l2=1.0), the msl tiers
(70/50/20), the categorical mask and random_state=0 are all taken from 08
UNCHANGED and apply identically in every config, including on the gated datasets.

Configs:
  base : early_stopping=True,  max_iter=300 everywhere            == shipped 08
  F50  : n<2000 -> early_stopping=False, max_iter=50
  F100 : n<2000 -> early_stopping=False, max_iter=100
  F200 : n<2000 -> early_stopping=False, max_iter=200

Expected firing (per dataset_stats.csv n_train), the n<2000 gate:
  train_05: n=1060  -> FIRES  (l2 gate NO  [ratio 0.0085], msl 20)
  train_09: n=1109  -> FIRES  (l2 gate YES [ratio 0.0162], msl 50)
  train_13: n=500   -> FIRES  (l2 gate YES [ratio 0.0180], msl 50)
  train_15: n=500   -> FIRES  (l2 gate YES [ratio 0.0600], msl 70)
  train_16: n=1809  -> FIRES  (l2 gate YES [ratio 0.0116], msl 20)
  all other 11 datasets: n >= 2000 -> gate does NOT fire; identical to base.
Three of the five (13/09/05) are the weakest datasets in the whole benchmark
(private AUC 0.6175/0.6241/0.6461), so this lever aims at the actual weak points.

INVARIANT: every candidate differs from base ONLY on the n<2000 datasets. The 11
datasets with n>=2000 are byte-identical to base and must show delta EXACTLY
0.00000 on both splits. If any of them moves, the harness leaks and the scores
are meaningless. The harness checks and reports this mechanically.

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
OUT_DIR = os.path.join(REPO, "experiments", "bench_03", "round80_tiny_no_es")

L2_GATE_THRESHOLD = 0.010   # shipped 08 feature-to-row-ratio gate for l2 (FIXED)
GATED_L2 = 1.0              # l2 applied when the l2-gate fires (FIXED)
DEFAULT_MSL = 20            # sklearn HGB default, used when no msl tier is cleared.
MSL_TIERS = [(0.030, 70), (0.015, 50)]  # shipped 08 tiers (FIXED — not the lever)

BASE_MAX_ITER = 300         # shipped 08 (FIXED)
BASE_EARLY_STOPPING = True  # shipped 08 (FIXED)

N_GATE = 2000               # THE LEVER's gate: n_train < 2000 -> no early stopping

# Each config: (name, fixed_max_iter). fixed_max_iter is None for base (keep 08's
# early_stopping=True / max_iter=300 everywhere); otherwise, datasets with
# n < N_GATE use early_stopping=False with that fixed max_iter, and datasets with
# n >= N_GATE keep 08's exact behavior.
FIXED_ITER_VALUES = [("base", None), ("F50", 50), ("F100", 100), ("F200", 200)]
CONFIGS = FIXED_ITER_VALUES
BASE = "base"
CANDIDATES = ["F50", "F100", "F200"]
FIXED_ITER_OF = dict(FIXED_ITER_VALUES)

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


def run_one(train_csv, test_csv, fixed_max_iter):
    """Reproduce the shipped 08 recipe for one dataset, applying THE LEVER: if
    fixed_max_iter is not None AND n < N_GATE, disable early stopping and use
    max_iter=fixed_max_iter instead. Everything else (l2 gate, msl tiers, cat
    mask, random_state) is 08 UNCHANGED. Returns
    (pred_map, l2, l2_fired, msl_val, n, tiny_fired, es_used, max_iter_used)."""
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)

    features = [c for c in train.columns if c not in ("row_id", "target")]
    cat_mask = [train[c].dtype == object for c in features]

    n = len(train)
    n_feat = len(features)
    ratio = n_feat / n

    l2_fired = ratio >= L2_GATE_THRESHOLD
    l2 = GATED_L2 if l2_fired else 0.0
    msl_val = msl_for_ratio(ratio)

    # ---- THE LEVER (the only thing that varies across configs) ----
    tiny_fired = (fixed_max_iter is not None) and (n < N_GATE)
    if tiny_fired:
        es_used = False
        max_iter_used = fixed_max_iter
    else:
        es_used = BASE_EARLY_STOPPING
        max_iter_used = BASE_MAX_ITER

    clf = HistGradientBoostingClassifier(
        categorical_features=cat_mask,
        random_state=0,
        max_iter=max_iter_used,
        early_stopping=es_used,
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
            l2, l2_fired, msl_val, n, tiny_fired, es_used, max_iter_used)


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
        for cfg_name, fixed_max_iter in CONFIGS:
            try:
                (pred_map, l2, l2_fired, msl_val, n, tiny_fired,
                 es_used, max_iter_used) = run_one(train_csv, test_csv, fixed_max_iter)
                pub, prv = score_split(pred_map, sol)
                rec[f"{cfg_name}_pub"] = pub
                rec[f"{cfg_name}_prv"] = prv
                rec[f"{cfg_name}_es"] = es_used
                rec[f"{cfg_name}_max_iter"] = max_iter_used
                rec[f"{cfg_name}_tiny_fired"] = tiny_fired
                # n, the l2 gate and the msl tier are config-independent
                # (all FIXED across configs).
                rec["n_train"] = n
                rec["l2_fired"] = l2_fired
                rec["msl"] = msl_val
                rec["tiny_gate"] = n < N_GATE
                n_fits_ok += 1
                print(f"[OK] {name} {cfg_name} (n={n}, l2={l2}, msl={msl_val}, "
                      f"tiny_fired={tiny_fired}, es={es_used}, "
                      f"max_iter={max_iter_used}): pub={pub:.6f} prv={prv:.6f}")
            except Exception as e:  # capture but keep going — CLEAN-RUN diagnostic
                exceptions.append((name, cfg_name, repr(e)))
                rec[f"{cfg_name}_pub"] = float("nan")
                rec[f"{cfg_name}_prv"] = float("nan")
                print(f"[ERROR] {name} {cfg_name}: {e!r}")
        rows.append(rec)

    # ---- write results CSV + JSON (raw per-dataset numbers) ----
    csv_path = os.path.join(OUT_DIR, "results.csv")
    fieldnames = ["dataset", "n_train", "tiny_gate", "l2_fired", "msl"]
    for cfg, _ in CONFIGS:
        fieldnames += [f"{cfg}_pub", f"{cfg}_prv", f"{cfg}_es",
                       f"{cfg}_max_iter", f"{cfg}_tiny_fired"]
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

    tiny_list = [r["dataset"] for r in rows if r.get("tiny_gate")]
    big_list = [r["dataset"] for r in rows if not r.get("tiny_gate")]
    l2_fired_list = [r["dataset"] for r in rows if r.get("l2_fired")]

    summary_lines = []

    # ---- per-dataset tables (Public, Private) ----
    for split, tag in (("pub", "Public"), ("prv", "Private")):
        if split == "prv":
            summary_lines.append("")
        summary_lines.append(f"=== PER-DATASET ({tag}) ===")
        header = (f"{'dataset':<10} {'n':>6} {'tiny':>5} {'l2G':>5} {'msl':>4} "
                  f"{BASE:>9}")
        for cfg in CANDIDATES:
            header += f" {cfg:>9} {'d'+cfg:>10}"
        summary_lines.append(header)
        for r in rows:
            line = (f"{r['dataset']:<10} {r.get('n_train'):>6} "
                    f"{str(bool(r.get('tiny_gate'))):>5} "
                    f"{str(bool(r.get('l2_fired'))):>5} {str(r.get('msl')):>4} "
                    f"{r[f'{BASE}_{split}']:>9.4f}")
            for cfg in CANDIDATES:
                line += (f" {r[f'{cfg}_{split}']:>9.4f} "
                         f"{delta(r, cfg, split):>+10.5f}")
            summary_lines.append(line)

    # ---- gate firings ----
    summary_lines.append("")
    summary_lines.append("=== GATE FIRINGS ===")
    summary_lines.append(
        f"TINY GATE (n_train < {N_GATE} — THE LEVER) fires on "
        f"({len(tiny_list)}): {', '.join(tiny_list) if tiny_list else '(none)'}")
    summary_lines.append(
        f"n >= {N_GATE} (08 behavior kept: early_stopping=True, max_iter=300) "
        f"({len(big_list)}): {', '.join(big_list) if big_list else '(none)'}")
    summary_lines.append(
        f"L2 GATE (ratio>=0.010, l2=1.0) fired on ({len(l2_fired_list)}): "
        f"{', '.join(l2_fired_list) if l2_fired_list else '(none)'}   "
        f"[FIXED — identical in all configs]")
    summary_lines.append(
        "(all configs share the l2 gate, the msl tiers (70/50/20), the "
        "categorical mask and random_state=0; they differ ONLY in whether "
        f"n<{N_GATE} datasets use early_stopping=False + fixed max_iter -> the "
        f"{len(tiny_list)} tiny datasets only. All other datasets identical -> delta 0)")

    # ---- INVARIANT CHECK: only tiny-gate datasets may differ ----
    summary_lines.append("")
    summary_lines.append(f"=== INVARIANT CHECK (the {len(big_list)} datasets with "
                         f"n>={N_GATE} must be byte-identical to base: "
                         f"delta EXACTLY 0.00000 on both splits) ===")
    invariant_ok = True
    for cfg in CANDIDATES:
        diff = differing_datasets(cfg)
        violators = [d for d in diff if d not in tiny_list]
        n_identical = len([r for r in rows if r["dataset"] not in diff])
        ok = not violators
        invariant_ok = invariant_ok and ok
        summary_lines.append(
            f"{cfg}: differs from base on: {', '.join(diff) if diff else '(none)'}  "
            f"(expected subset of: {', '.join(tiny_list)})  |  "
            f"identical (delta exactly 0 on both splits): {n_identical}/{len(rows)}  "
            f"-> {'OK' if ok else 'VIOLATION: ' + ', '.join(violators)}")
    # explicit machine check of the big-n datasets, both splits, exact zero
    big_max_abs = 0.0
    for cfg in CANDIDATES:
        for r in rows:
            if r["dataset"] in tiny_list:
                continue
            for split in ("pub", "prv"):
                dd = delta(r, cfg, split)
                if not math.isnan(dd):
                    big_max_abs = max(big_max_abs, abs(dd))
    summary_lines.append(
        f"max |delta| over the {len(big_list)} n>={N_GATE} datasets x "
        f"{len(CANDIDATES)} candidates x 2 splits = {big_max_abs:.10g} "
        f"(must be exactly 0)")
    invariant_ok = invariant_ok and (big_max_abs == 0.0)
    summary_lines.append(
        f"INVARIANT={'HOLDS' if invariant_ok else 'VIOLATED — HARNESS LEAKS, '
                    'SCORES ARE NOT TRUSTWORTHY'} "
        f"(the {len(big_list)} n>={N_GATE} datasets are unaffected by the lever, "
        f"as designed)")

    # ---- THE SWEEP: fixed max_iter (no ES) vs the tiny datasets ----
    summary_lines.append("")
    summary_lines.append("=== TINY-DATASET SWEEP (the datasets the lever touches) ===")
    for r in rows:
        if r["dataset"] not in tiny_list:
            continue
        summary_lines.append(
            f"--- {r['dataset']} (n={r.get('n_train')}, l2_fired="
            f"{bool(r.get('l2_fired'))}, msl={r.get('msl')}) ---")
        summary_lines.append(f"  {'config':>8} {'es':>6} {'iters':>6} {'Public':>9} "
                             f"{'dPub':>10} {'Private':>9} {'dPrv':>10}")
        for cfg, _ in CONFIGS:
            dp = delta(r, cfg, "pub")
            dv = delta(r, cfg, "prv")
            tag = "  <- base (shipped 08)" if cfg == BASE else ""
            summary_lines.append(
                f"  {cfg:>8} {str(r.get(f'{cfg}_es')):>6} "
                f"{str(r.get(f'{cfg}_max_iter')):>6} "
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
                f"  {tag} adjacent-F spread (|F50-F100|, |F100-F200|): "
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
            f"{cfg} (n<{N_GATE}: es=False, max_iter={FIXED_ITER_OF[cfg]}): "
            f"mean Public d={mp:+.5f}  mean Private d={mv:+.5f}  "
            f"Public W/L/T={wp}/{lp}/{tp}  Private W/L/T={wv}/{lv}/{tv}")

    # ---- per-candidate differing-dataset deltas + regressions ----
    summary_lines.append("")
    summary_lines.append("=== PER-CANDIDATE DETAIL (datasets differing from base; "
                         "all other deltas are exactly 0) ===")
    for cfg in CANDIDATES:
        summary_lines.append(f"--- {cfg} vs {BASE} ---")
        diff = set(differing_datasets(cfg))
        for r in rows:
            if r["dataset"] not in diff:
                continue
            dp = delta(r, cfg, "pub")
            dv = delta(r, cfg, "prv")
            summary_lines.append(
                f"  {r['dataset']:<10} (n={r.get('n_train')}, "
                f"es True->{r.get(f'{cfg}_es')}, "
                f"iters {BASE_MAX_ITER}->{r.get(f'{cfg}_max_iter')})  "
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
            f"{cfg} (n<{N_GATE}: es=False, max_iter={FIXED_ITER_OF[cfg]}): {verdict}")
    summary_lines.append("")
    if clean_names:
        # best clean candidate = largest mean Public delta (tie-break Private)
        best = max(clean_names,
                   key=lambda c: (mean_delta(c, "pub"), mean_delta(c, "prv")))
        summary_lines.append(
            f"OVERALL: clean improvement over 08 found: {', '.join(clean_names)}; "
            f"best = {best} (max_iter={FIXED_ITER_OF[best]}, "
            f"mean pub={mean_delta(best, 'pub'):+.5f}, "
            f"prv={mean_delta(best, 'prv'):+.5f}) "
            f"(orchestrator decides adoption; compare the gain against the "
            f"adjacent-F spread above before believing it).")
    else:
        summary_lines.append(
            "OVERALL: NO clean improvement over 08; base (shipped 08, "
            "early_stopping=True everywhere) remains best. Removing early "
            "stopping on tiny datasets in favor of a fixed iteration count did "
            "not cleanly help.")

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
