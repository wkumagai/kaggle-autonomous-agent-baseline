#!/usr/bin/env python
"""
bench_03 round79 — TOP-TIER min_samples_leaf magnitude beyond the round22 sweep
ceiling, on the shipped 08 ratio-tiered recipe. OFFLINE ONLY. No subprocess,
no LLM, no Kaggle. Calls sklearn in-process.

Adapted verbatim in structure from
experiments/bench_03/round23_ratio_tiered_msl/replay.py.

Base recipe reproduced (verified vs
  `git show HEAD:submissions/08_ratio_tiered_msl/agent/prompts/system.md`):
  - load train.csv, test.csv with pandas
  - features = [c for c in train.columns if c not in ("row_id","target")]
  - cat_mask  = [train[c].dtype == object for c in features]
  - n = len(train); n_feat = len(features); ratio = n_feat / n
  - L2 GATE:  l2  = 1.0 if ratio >= 0.010 else 0.0                  [FIXED]
  - MSL TIERS: msl = 70 if ratio >= 0.030 else (50 if ratio >= 0.015 else 20)
               [shipped 08]
  - HistGradientBoostingClassifier(categorical_features=cat_mask, random_state=0,
        max_iter=300, early_stopping=True, l2_regularization=l2,
        min_samples_leaf=msl)
  - fit(train[features], train["target"])
  - pred = clf.predict_proba(test[features])[:, pos_class_1]
Score: ROC AUC on Public rows and Private rows separately, joined by row_id to
solution.csv, over all 16 datasets.

The QUESTION (round79 angle): round22's msl magnitude sweep only spanned
40/50/60/70 — 70 was the UPPER END of the swept range, not a demonstrated
optimum. Within that range train_15 (ratio=0.060) improved MONOTONICALLY
(Public: msl50=0.8372 -> msl60 +0.00161 -> msl70 +0.00385). Shipped 08 pinned
the top tier at that untested boundary. This round pushes the TOP tier only,
past the old ceiling:

  min_samples_leaf = X  if ratio >= 0.030   (TOP tier; X in {70(base), 90, 110, 130})
                 else 50 if ratio >= 0.015   (MID tier — UNCHANGED from 08)
                 else 20                       (sklearn default — UNCHANGED)

Does train_15 keep improving past 70, or is there a peak near 70 after which it
reverses? Note train_15 has n=500 rows and HGB's early_stopping holds out 10% of
them, so ~450 rows reach the trees: msl=130 constrains a leaf to >=~29% of the
fitting rows, which is a very coarse tree. A reversal somewhere in this range is
physically plausible; that is exactly what this measures.

ONE LEVER ONLY. The mid tier (50 @ ratio>=0.015), the default (20), and the l2
gate (ratio>=0.010 -> l2=1.0) are all taken from 08 UNCHANGED.

Configs (name, tiers). `tiers` is a descending list of (ratio_threshold, msl)
tried in order; the first threshold the ratio clears wins, else DEFAULT_MSL:
  base   : [(0.030,  70), (0.015, 50)]   == shipped 08
  msl90  : [(0.030,  90), (0.015, 50)]
  msl110 : [(0.030, 110), (0.015, 50)]
  msl130 : [(0.030, 130), (0.015, 50)]

Expected firing (per dataset_stats.csv ratios n_feat/n):
  train_09: 18/1109  = 0.0162  -> l2 YES, MID tier (msl 50 in ALL cfgs)
  train_13:  9/500   = 0.0180  -> l2 YES, MID tier (msl 50 in ALL cfgs)
  train_15: 30/500   = 0.0600  -> l2 YES, TOP tier (msl 70 -> 90/110/130)
  train_16: 21/1809  = 0.0116  -> l2 YES, no msl tier (msl 20 in ALL cfgs)
  all others          < 0.010  -> neither gate; identical in ALL cfgs.

INVARIANT: every candidate differs from base ONLY on train_15. The other 15
datasets are byte-identical to base and must show delta exactly 0.00000 on both
splits. The harness checks and reports this mechanically.

Adoption criterion: a candidate is a CLEAN IMPROVEMENT over base(08) iff its
mean delta is positive on BOTH splits AND there are ZERO regressions on BOTH
splits (no dataset worse on either split). A single regression on either split
=> not clean.
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
OUT_DIR = os.path.join(REPO, "experiments", "bench_03", "round79_msl_toptier")

L2_GATE_THRESHOLD = 0.010   # shipped 08 feature-to-row-ratio gate for l2 (FIXED)
GATED_L2 = 1.0              # l2 applied when the l2-gate fires (FIXED)
DEFAULT_MSL = 20            # sklearn HGB default, used when no msl tier is cleared.
MID_TIER = (0.015, 50)      # shipped 08 mid tier (FIXED — not the lever)
TOP_THRESHOLD = 0.030       # shipped 08 top-tier threshold (FIXED — not the lever)

# Each config: (name, tiers). `tiers` is a DESCENDING list of
# (ratio_threshold, min_samples_leaf); the first threshold the ratio clears
# picks the msl, else DEFAULT_MSL. base == shipped 08. The ONLY thing that
# varies across configs is the TOP-tier magnitude.
TOP_MSL_VALUES = [("base", 70), ("msl90", 90), ("msl110", 110), ("msl130", 130)]
CONFIGS = [(name, [(TOP_THRESHOLD, top), MID_TIER]) for name, top in TOP_MSL_VALUES]
BASE = "base"
CANDIDATES = ["msl90", "msl110", "msl130"]
TOP_MSL_OF = dict(TOP_MSL_VALUES)

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
    """Reproduce the shipped 08 recipe for one dataset, applying the ratio-tiered
    min_samples_leaf from `tiers` (base == shipped 08). The l2 gate is identical
    for all configs. Returns (pred_map, l2, l2_fired, msl_fired, msl_val, top_fired)
    where pred_map maps test row_id -> pos-class prob, msl_fired means a
    non-default msl tier was cleared, msl_val is the min_samples_leaf actually
    used, and top_fired means the TOP tier (ratio>=0.030) was the one that won."""
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
    top_fired = ratio >= TOP_THRESHOLD

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
            l2, l2_fired, msl_fired, msl_val, top_fired)


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
        for cfg_name, tiers in CONFIGS:
            try:
                pred_map, l2, l2_fired, msl_fired, msl_val, top_fired = run_one(
                    train_csv, test_csv, tiers)
                pub, prv = score_split(pred_map, sol)
                rec[f"{cfg_name}_pub"] = pub
                rec[f"{cfg_name}_prv"] = prv
                rec[f"{cfg_name}_msl"] = msl_val
                # l2-gate firing and TOP-tier firing are config-independent
                # (both thresholds are FIXED across configs).
                rec["l2_fired"] = l2_fired
                rec["top_fired"] = top_fired
                rec[f"{cfg_name}_msl_fired"] = msl_fired
                n_fits_ok += 1
                print(f"[OK] {name} {cfg_name} (l2={l2}, l2_fired={l2_fired}, "
                      f"top_fired={top_fired}, msl={msl_val}): "
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
    fieldnames = ["dataset", "l2_fired", "top_fired"]
    for cfg, _ in CONFIGS:
        fieldnames += [f"{cfg}_pub", f"{cfg}_prv", f"{cfg}_msl", f"{cfg}_msl_fired"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

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

    l2_fired_list = [r["dataset"] for r in rows if r.get("l2_fired")]
    top_fired_list = [r["dataset"] for r in rows if r.get("top_fired")]

    summary_lines = []

    # ---- per-dataset tables (Public, Private) ----
    for split, tag in (("pub", "Public"), ("prv", "Private")):
        if split == "prv":
            summary_lines.append("")
        summary_lines.append(f"=== PER-DATASET ({tag}) ===")
        header = f"{'dataset':<10} {'l2G':>4} {'top':>5} {'bMsl':>5} {BASE:>9}"
        for cfg in CANDIDATES:
            header += f" {'msl':>4} {cfg:>9} {'d'+cfg:>10}"
        summary_lines.append(header)
        for r in rows:
            line = (f"{r['dataset']:<10} {str(bool(r.get('l2_fired'))):>4} "
                    f"{str(bool(r.get('top_fired'))):>5} "
                    f"{str(r.get(f'{BASE}_msl')):>5} {r[f'{BASE}_{split}']:>9.4f}")
            for cfg in CANDIDATES:
                line += (f" {str(r.get(f'{cfg}_msl')):>4} "
                         f"{r[f'{cfg}_{split}']:>9.4f} {delta(r, cfg, split):>+10.5f}")
            summary_lines.append(line)

    # ---- gate firings ----
    summary_lines.append("")
    summary_lines.append("=== GATE FIRINGS ===")
    summary_lines.append(
        f"L2 GATE (ratio>=0.010, l2=1.0) fired on ({len(l2_fired_list)}): "
        f"{', '.join(l2_fired_list) if l2_fired_list else '(none)'}")
    summary_lines.append(
        f"TOP TIER (ratio>=0.030 — THE LEVER) fired on ({len(top_fired_list)}): "
        f"{', '.join(top_fired_list) if top_fired_list else '(none)'}")
    for cfg, _ in CONFIGS:
        fired = [r["dataset"] for r in rows if r.get(f"{cfg}_msl_fired")]
        summary_lines.append(
            f"MSL non-default (msl!=20) in '{cfg}' ({len(fired)}): "
            f"{', '.join(fired) if fired else '(none)'}   "
            f"[top-tier msl={TOP_MSL_OF[cfg]}]")
    summary_lines.append(
        "(all configs share the mid tier (msl 50 @ ratio>=0.015), the default "
        "(msl 20), and the l2 gate; they differ ONLY in the TOP-tier magnitude "
        "(@ ratio>=0.030) -> train_15 only. All other datasets identical -> delta 0)")

    # ---- INVARIANT CHECK: only top-tier-firing datasets may differ ----
    summary_lines.append("")
    summary_lines.append("=== INVARIANT CHECK (non-top-tier datasets must be "
                         "byte-identical to base: delta 0.00000) ===")
    invariant_ok = True
    for cfg in CANDIDATES:
        diff = differing_datasets(cfg)
        violators = [d for d in diff if d not in top_fired_list]
        n_identical = len([r for r in rows if r["dataset"] not in diff])
        ok = not violators
        invariant_ok = invariant_ok and ok
        summary_lines.append(
            f"{cfg}: differs from base on: {', '.join(diff) if diff else '(none)'}  "
            f"(expected: {', '.join(top_fired_list)})  |  "
            f"identical (delta exactly 0 on both splits): {n_identical}/{len(rows)}  "
            f"-> {'OK' if ok else 'VIOLATION: ' + ', '.join(violators)}")
    summary_lines.append(
        f"INVARIANT={'HOLDS' if invariant_ok else 'VIOLATED'} "
        f"(the {len(rows) - len(top_fired_list)} non-top-tier datasets are "
        f"unaffected by the lever, as designed)")

    # ---- THE SWEEP: top-tier magnitude vs train_15 ----
    summary_lines.append("")
    summary_lines.append("=== TOP-TIER MSL SWEEP (the datasets the lever touches) ===")
    for r in rows:
        if r["dataset"] not in top_fired_list:
            continue
        summary_lines.append(f"--- {r['dataset']} (ratio>=0.030) ---")
        summary_lines.append(f"  {'msl':>5} {'Public':>9} {'dPub':>10} "
                             f"{'Private':>9} {'dPrv':>10}")
        for cfg, _ in CONFIGS:
            dp = delta(r, cfg, "pub")
            dv = delta(r, cfg, "prv")
            tag = "  <- base (shipped 08)" if cfg == BASE else ""
            summary_lines.append(
                f"  {TOP_MSL_OF[cfg]:>5} {r[f'{cfg}_pub']:>9.5f} {dp:>+10.5f} "
                f"{r[f'{cfg}_prv']:>9.5f} {dv:>+10.5f}{tag}")
        # monotonicity / peak read-out per split
        for split, tag in (("pub", "Public"), ("prv", "Private")):
            vals = [(TOP_MSL_OF[c], r[f"{c}_{split}"]) for c, _ in CONFIGS]
            best_msl, best_auc = max(vals, key=lambda t: t[1])
            seq = " -> ".join(f"{m}:{v:.5f}" for m, v in vals)
            summary_lines.append(
                f"  {tag} curve: {seq}   BEST msl={best_msl} ({best_auc:.5f})")

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
            f"{cfg} (top msl={TOP_MSL_OF[cfg]}): mean Public d={mp:+.5f}  "
            f"mean Private d={mv:+.5f}  "
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
        summary_lines.append(f"{cfg} (top msl={TOP_MSL_OF[cfg]}): {verdict}")
    summary_lines.append("")
    if clean_names:
        # best clean candidate = largest mean Public delta (tie-break Private)
        best = max(clean_names,
                   key=lambda c: (mean_delta(c, "pub"), mean_delta(c, "prv")))
        summary_lines.append(
            f"OVERALL: clean improvement over 08 found: {', '.join(clean_names)}; "
            f"best = {best} (top msl={TOP_MSL_OF[best]}, "
            f"mean pub={mean_delta(best, 'pub'):+.5f}, "
            f"prv={mean_delta(best, 'prv'):+.5f}) "
            f"(orchestrator decides adoption).")
    else:
        summary_lines.append(
            "OVERALL: NO clean improvement over 08; base (shipped 08, top "
            "msl=70) remains best. Raising the top tier past 70 did not cleanly "
            "help -> 70 is at/near the peak, not merely a sweep-range artifact.")

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
    print(f"[WROTE] {os.path.join(OUT_DIR, 'summary.txt')}")
    print("HARNESS_DONE")


if __name__ == "__main__":
    main()
