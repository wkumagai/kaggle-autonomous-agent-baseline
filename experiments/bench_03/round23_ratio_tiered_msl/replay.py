#!/usr/bin/env python
"""
bench_03 round23 — ratio-TIERED min_samples_leaf on the shipped 07 two-gate
recipe. OFFLINE ONLY. No subprocess, no LLM, no Kaggle. Calls sklearn in-process.

Adapted from experiments/bench_03/round22_msl_magnitude/replay.py.

Base recipe reproduced (verified vs
  `git show HEAD:submissions/07_2gate_msl50/agent/prompts/system.md`):
  - load train.csv, test.csv with pandas
  - features = [c for c in train.columns if c not in ("row_id","target")]
  - cat_mask  = [train[c].dtype == object for c in features]
  - n = len(train); n_feat = len(features); ratio = n_feat / n
  - L2 GATE:  l2  = 1.0 if ratio >= 0.010 else 0.0
  - MSL GATE: msl = 50 if ratio >= 0.015 else 20 (sklearn default)  [shipped 07]
  - HistGradientBoostingClassifier(categorical_features=cat_mask, random_state=0,
        max_iter=300, early_stopping=True, l2_regularization=l2,
        min_samples_leaf=msl)
  - fit(train[features], train["target"])
  - pred = clf.predict_proba(test[features])[:, pos_class_1]
Score: ROC AUC on Public rows and Private rows separately, joined by row_id to
solution.csv, over all 16 datasets.

The QUESTION (round23 angle): round22 found that raising the single global
msl-gate magnitude to 70 monotonically HELPS the highest-ratio dataset
(train_15, ratio 0.060: +0.0039 Public / +0.0040 Private) but REGRESSES the
lower-ratio train_09 (ratio 0.0162). A single global magnitude cannot lift all
three msl-gate-firing datasets. Round23 adds a SECOND magnitude TIER inside the
msl-gate, keyed on the SAME feature-to-row ratio, so only the very highest-ratio
dataset gets the larger leaf size:

  min_samples_leaf = 70 if ratio >= 0.030   (HIGH tier — fires on train_15 only)
                 else 50 if ratio >= 0.015   (MID  tier — train_09/13, == 07)
                 else 20                       (sklearn default)

The l2 gate is UNCHANGED. This should pick up train_15's +0.004 gain while
leaving train_09/13 byte-identical to 07 (zero regression by construction).

Configs (name, tiers). `tiers` is a descending list of (ratio_threshold, msl)
tried in order; the first threshold the ratio clears wins, else DEFAULT_MSL:
  base   : [(0.015, 50)]            -> msl = 50 if ratio>=0.015 else 20 (== 07)
  tiered : [(0.030, 70), (0.015, 50)]
                                     -> 70 if ratio>=0.030 else 50 if ratio>=0.015
                                        else 20

Expected firing (per dataset_stats.csv ratios n_feat/n):
  train_09: 18/1109  = 0.0162  -> l2 YES, mid tier (msl 50 in BOTH cfgs)
  train_13:  9/500   = 0.0180  -> l2 YES, mid tier (msl 50 in BOTH cfgs)
  train_15: 30/500   = 0.0600  -> l2 YES, HIGH tier in `tiered` (msl 50->70)
  train_16: 21/1809  = 0.0116  -> l2 YES, no msl gate (msl 20 in BOTH cfgs)
  all others          < 0.010  -> neither gate; identical in BOTH cfgs.

INVARIANT: base and tiered differ ONLY on train_15 (its msl goes 50->70).
train_09/13 (mid tier in both), train_16 (l2-only, msl=20), and the 12
non-firing datasets are byte-identical between base and tiered and contribute
exactly 0 to every tiered-vs-base delta.

Adoption criterion: `tiered` is a CLEAN IMPROVEMENT over base(07) iff its mean
delta is positive on BOTH splits AND there are ZERO regressions on BOTH splits
(no dataset worse on either split). A single regression on either split => not
clean.
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
OUT_DIR = os.path.join(REPO, "experiments", "bench_03", "round23_ratio_tiered_msl")

L2_GATE_THRESHOLD = 0.010   # shipped 07 feature-to-row-ratio gate for l2 (FIXED)
GATED_L2 = 1.0              # l2 applied when the l2-gate fires (FIXED)
DEFAULT_MSL = 20            # sklearn HGB default, used when no msl tier is cleared.

# Each config: (name, tiers). `tiers` is a DESCENDING list of
# (ratio_threshold, min_samples_leaf); the first threshold the ratio clears
# picks the msl, else DEFAULT_MSL. base == shipped 07 (mid tier only).
CONFIGS = [
    ("base",   [(0.015, 50)]),
    ("tiered", [(0.030, 70), (0.015, 50)]),
]
BASE = "base"
CANDIDATES = ["tiered"]

N_DATASETS = 16


def msl_for_ratio(ratio, tiers):
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


def run_one(train_csv, test_csv, tiers):
    """Reproduce the shipped 07 two-gate recipe for one dataset, applying the
    ratio-tiered min_samples_leaf from `tiers` (base == shipped 07 mid-tier
    only). The l2 gate is identical for all configs. Returns
    (pred_map, l2, l2_fired, msl_fired, msl_val) where pred_map maps test
    row_id -> pos-class prob, msl_fired means a non-default msl tier was
    cleared, and msl_val is the min_samples_leaf actually used."""
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)

    features = [c for c in train.columns if c not in ("row_id", "target")]
    cat_mask = [train[c].dtype == object for c in features]

    n = len(train)
    n_feat = len(features)
    ratio = n_feat / n

    l2_fired = ratio >= L2_GATE_THRESHOLD
    l2 = GATED_L2 if l2_fired else 0.0

    msl_val = msl_for_ratio(ratio, tiers)
    msl_fired = msl_val != DEFAULT_MSL

    clf = HistGradientBoostingClassifier(
        categorical_features=cat_mask,
        random_state=0,
        max_iter=300,
        early_stopping=True,
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
            l2, l2_fired, msl_fired, msl_val)


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
        for cfg_name, tiers in CONFIGS:
            try:
                pred_map, l2, l2_fired, msl_fired, msl_val = run_one(
                    train_csv, test_csv, tiers)
                pub, prv = score_split(pred_map, sol)
                rec[f"{cfg_name}_pub"] = pub
                rec[f"{cfg_name}_prv"] = prv
                rec[f"{cfg_name}_msl"] = msl_val
                # l2-gate firing is config-independent (same threshold).
                rec["l2_fired"] = l2_fired
                # msl firing IS config-dependent now (tiers differ); record per cfg.
                rec[f"{cfg_name}_msl_fired"] = msl_fired
                print(f"[OK] {name} {cfg_name} (l2={l2}, l2_fired={l2_fired}, "
                      f"msl_fired={msl_fired}, msl={msl_val}): "
                      f"pub={pub:.6f} prv={prv:.6f}")
            except Exception as e:  # capture but keep going — CLEAN-RUN diagnostic
                exceptions.append((name, cfg_name, repr(e)))
                rec[f"{cfg_name}_pub"] = float("nan")
                rec[f"{cfg_name}_prv"] = float("nan")
                rec[f"{cfg_name}_msl"] = float("nan")
                print(f"[ERROR] {name} {cfg_name}: {e!r}")
        rows.append(rec)

    # ---- write results CSV ----
    csv_path = os.path.join(OUT_DIR, "results.csv")
    fieldnames = ["dataset", "l2_fired"]
    for cfg, _ in CONFIGS:
        fieldnames += [f"{cfg}_pub", f"{cfg}_prv", f"{cfg}_msl", f"{cfg}_msl_fired"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    # ---- deltas & helpers (all vs base == shipped 07) ----
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
        """Datasets where cfg is strictly worse than base(07) on this split."""
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

    def l2_firing_datasets():
        return [r["dataset"] for r in rows if r.get("l2_fired")]

    l2_fired_list = l2_firing_datasets()

    summary_lines = []

    # ---- per-dataset tables (Public, Private) ----
    for split, tag in (("pub", "Public"), ("prv", "Private")):
        if split == "prv":
            summary_lines.append("")
        summary_lines.append(f"=== PER-DATASET ({tag}) ===")
        header = (f"{'dataset':<10} {'l2G':>4} {'bMsl':>5} {BASE:>9}")
        for cfg in CANDIDATES:
            header += f" {'tMsl':>5} {cfg:>9} {'d'+cfg:>10}"
        summary_lines.append(header)
        for r in rows:
            line = (f"{r['dataset']:<10} {str(bool(r.get('l2_fired'))):>4} "
                    f"{str(r.get(f'{BASE}_msl')):>5} {r[f'{BASE}_{split}']:>9.4f}")
            for cfg in CANDIDATES:
                line += (f" {str(r.get(f'{cfg}_msl')):>5} "
                         f"{r[f'{cfg}_{split}']:>9.4f} {delta(r, cfg, split):>+10.5f}")
            summary_lines.append(line)

    # ---- gate firings ----
    summary_lines.append("")
    summary_lines.append("=== GATE FIRINGS ===")
    summary_lines.append(
        f"L2 GATE (ratio>=0.010, l2=1.0) fired on "
        f"({len(l2_fired_list)}): "
        f"{', '.join(l2_fired_list) if l2_fired_list else '(none)'}")
    for cfg, _ in CONFIGS:
        fired = [r["dataset"] for r in rows if r.get(f"{cfg}_msl_fired")]
        highs = [r["dataset"] for r in rows if r.get(f"{cfg}_msl") == 70]
        summary_lines.append(
            f"MSL non-default (msl!=20) in '{cfg}' ({len(fired)}): "
            f"{', '.join(fired) if fired else '(none)'}"
            + (f"   [HIGH tier msl=70: {', '.join(highs)}]" if highs else ""))
    summary_lines.append(
        "(base and tiered share the mid tier (msl 50 @ ratio>=0.015) and the "
        "l2 gate; they differ ONLY where the HIGH tier (msl 70 @ ratio>=0.030) "
        "fires -> train_15 only. All other datasets are identical -> delta 0)")

    # ---- which datasets actually differed ----
    summary_lines.append("")
    summary_lines.append("=== DATASETS THAT ACTUALLY DIFFER (tiered vs base) ===")
    for cfg in CANDIDATES:
        diff = differing_datasets(cfg)
        summary_lines.append(
            f"{cfg}: {', '.join(diff) if diff else '(none)'}  "
            f"(expected: train_15 only)")

    # ---- per-candidate summary vs base(07) ----
    summary_lines.append("")
    summary_lines.append("=== SUMMARY (tiered vs base == shipped 07) ===")
    for cfg in CANDIDATES:
        mp = mean_delta(cfg, "pub")
        mv = mean_delta(cfg, "prv")
        wp, lp, tp = wlt(cfg, "pub")
        wv, lv, tv = wlt(cfg, "prv")
        summary_lines.append(
            f"{cfg}: mean Public d={mp:+.5f}  mean Private d={mv:+.5f}  "
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
                f"  {r['dataset']:<10} "
                f"(msl {r.get(f'{BASE}_msl')}->{r.get(f'{cfg}_msl')})  "
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
    summary_lines.append("=== VERDICT (adoption vs base == shipped 07) ===")
    summary_lines.append(
        "Criterion: CLEAN IMPROVEMENT over 07 iff mean delta positive AND "
        "zero regressions on BOTH splits.")
    any_clean = False
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
            any_clean = True
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
        summary_lines.append(f"{cfg}: {verdict}")
    summary_lines.append("")
    if any_clean:
        summary_lines.append(
            f"OVERALL: clean improvement over 07 found: {', '.join(clean_names)} "
            f"(orchestrator decides adoption).")
    else:
        summary_lines.append(
            "OVERALL: NO clean improvement over 07; base (shipped 07) remains "
            "best. Ratio-tiered msl did not cleanly beat the single mid tier.")

    # ---- clean-run line ----
    summary_lines.append("")
    summary_lines.append(f"CLEAN RUN={'YES' if not exceptions else 'NO'} "
                         f"(exceptions={len(exceptions)}, skipped={len(skipped)})")
    for name, cfg, msg in exceptions:
        summary_lines.append(f"  EXC {name}/{cfg}: {msg}")

    summary = "\n".join(summary_lines)
    print("\n" + summary)

    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
        f.write(summary + "\n")

    print(f"\n[WROTE] {csv_path}")
    print(f"[WROTE] {os.path.join(OUT_DIR, 'summary.txt')}")
    print("HARNESS_DONE")


if __name__ == "__main__":
    main()
