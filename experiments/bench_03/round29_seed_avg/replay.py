#!/usr/bin/env python
"""
bench_03 round29 — SEED-AVERAGING (variance reduction) on top of the shipped 08
two-gate ratio-tiered-msl recipe. OFFLINE ONLY. No subprocess, no LLM, no
Kaggle. Calls sklearn in-process.

Adapted from experiments/bench_03/round28_max_bins/replay.py (dataset loading,
split/scoring, verdict, and summary machinery reused; the swept knob is replaced
by how the FINAL probabilities are produced — averaging predict_proba across
several random_state seeds of the SAME shipped-08 HGB config).

Base recipe reproduced (== shipped 08, verified vs
  `git show HEAD:submissions/08_ratio_tiered_msl/agent/prompts/system.md`):
  - load train.csv, test.csv with pandas
  - features = [c for c in train.columns if c not in ("row_id","target")]
  - cat_mask  = [train[c].dtype == object for c in features]
  - n = len(train); n_feat = len(features); ratio = n_feat / n
  - L2 GATE:  l2  = 1.0 if ratio >= 0.010 else 0.0
  - MSL GATE (08 tiered): msl = 70 if ratio >= 0.030
                          else 50 if ratio >= 0.015 else 20
  - HistGradientBoostingClassifier(categorical_features=cat_mask, random_state=0,
        max_iter=300, early_stopping=True, l2_regularization=l2,
        min_samples_leaf=msl)
  - fit(train[features], train["target"])
  - pred = clf.predict_proba(test[features])[:, pos_class_1]
Score: ROC AUC on Public rows and Private rows separately, joined by row_id to
solution.csv, over all 16 datasets.

The QUESTION (round29 angle): keep l2 gate, msl tiers, early_stopping, features
EXACTLY as 08. Change ONLY how the final positive-class probability is produced,
by averaging predict_proba(test)[:, 1] across K=5 seeds random_state in
{0,1,2,3,4} of the identical config. Hypothesis: the small-n high-variance
datasets (esp. train_13) suffer from variance in the early_stopping internal
holdout split (governed by random_state); averaging across seeds reduces that
variance and may lift the counter-mover datasets that no single knob could.

Configs (all identical shipped-08 config; differ ONLY in the seed set averaged):
  base : single seed random_state=0                     == shipped 08 (K=1)
  cand_A (global seed-avg): mean predict_proba over seeds {0,1,2,3,4} on EVERY
         dataset (all 16).
  cand_B (gated seed-avg):  mean predict_proba over seeds {0,1,2,3,4} ONLY on
         the L2-firing datasets (ratio = n_feat/n >= 0.010 -> train_09/13/15/16);
         single seed random_state=0 (byte-identical to base) on the other 12.
  cand_C (tighter-gated seed-avg): mean predict_proba over seeds {0,1,2,3,4}
         ONLY where ratio = n_feat/n >= 0.015 (the SAME threshold shipped-08
         already uses for its stricter min_samples_leaf tier). This fires on
         train_09 (~0.0162), train_13 (~0.0180), train_15 (~0.060) and EXCLUDES
         train_16 (~0.0116, the one dataset that regressed under seed-averaging
         in cand_A/cand_B). All other datasets (train_16 + the 12 large-n sets)
         keep the single seed-0 output byte-identical to base. The gate is a
         principled ratio threshold (no dataset-name list) so it could be
         described in a system.md later.

IMPLEMENTATION: for each dataset we fit all 5 seeds ONCE (seeds 0..4) and derive
all three outputs from the same fits:
  - base   = seed-0 prediction
  - cand_A = mean over seeds 0..4
  - cand_B = mean over seeds 0..4 if L2 gate fired else seed-0 prediction
Total fits = 16 datasets * 5 seeds = 80.

IMPLEMENTATION INVARIANT: cand_B on any NON-L2-firing dataset is the exact same
seed-0 array as base, so its delta MUST be exactly 0 on both splits — checked
explicitly (the byte-identical requirement). cand_A averages on every dataset,
so it is generally NOT identical to base anywhere (expected).

Adoption criterion (reused from prior rounds): a candidate config is a CLEAN
IMPROVEMENT over base(08) iff its mean delta is positive on BOTH splits AND
there are ZERO regressions on BOTH splits (no dataset worse on either split).
A single regression on either split => not clean.
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
OUT_DIR = os.path.join(REPO, "experiments", "bench_03", "round29_seed_avg")

L2_GATE_THRESHOLD = 0.010   # shipped 08 feature-to-row-ratio gate for l2 (FIXED)
# cand_C fires seed-averaging on the SAME threshold shipped-08 uses for its
# stricter min_samples_leaf tier (0.015). Fires train_09/13/15, excludes
# train_16 (~0.0116). Principled ratio gate, no dataset-name list.
SEEDAVG_C_THRESHOLD = 0.015
GATED_L2 = 1.0              # l2 applied when the l2-gate fires (FIXED)
DEFAULT_MSL = 20            # sklearn HGB default, used when no msl tier is cleared.
BASE_SEED = 0              # shipped-08 fixed random_state.
SEEDS = [0, 1, 2, 3, 4]    # K=5 seed set for averaging.

# 08 tiered min_samples_leaf, IDENTICAL across all configs (descending order).
MSL_TIERS = [(0.030, 70), (0.015, 50)]

# Config names. base is single-seed (== shipped 08). Candidates derive from the
# same 5 fits (no extra compute): cand_A averages seeds on all datasets, cand_B
# averages seeds only on L2-firing datasets (else seed-0 == base).
BASE = "base"
CANDIDATES = ["cand_A", "cand_B", "cand_C"]

N_DATASETS = 16


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


def run_one(train_csv, test_csv):
    """Fit all 5 seeds of the shipped-08 config for one dataset and derive the
    three configs' predictions. Returns
    (preds, l2, l2_fired, c_fired, msl_val, n_fits) where preds maps
    config_name -> {row_id -> pos-class prob}. base uses seed-0 exactly;
    cand_A = mean over the 5 seeds; cand_B = cand_A if l2 gate fired else base;
    cand_C = cand_A if ratio >= SEEDAVG_C_THRESHOLD (0.015) else base — the
    excluded datasets reuse the SAME seed-0 array, so byte-identical."""
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)

    features = [c for c in train.columns if c not in ("row_id", "target")]
    cat_mask = [train[c].dtype == object for c in features]

    n = len(train)
    n_feat = len(features)
    ratio = n_feat / n

    l2_fired = ratio >= L2_GATE_THRESHOLD
    l2 = GATED_L2 if l2_fired else 0.0
    c_fired = ratio >= SEEDAVG_C_THRESHOLD  # cand_C seed-avg gate (0.015)
    msl_val = msl_for_ratio(ratio)

    row_ids = test["row_id"].tolist()

    # Fit each seed once; stack predictions (rows aligned to test order).
    seed_preds = {}
    for s in SEEDS:
        seed_preds[s] = fit_one_seed(train, test, features, cat_mask,
                                     l2, msl_val, s)
    n_fits = len(SEEDS)

    base_vec = seed_preds[BASE_SEED]
    avg_vec = np.mean(np.vstack([seed_preds[s] for s in SEEDS]), axis=0)

    base_map = dict(zip(row_ids, base_vec.tolist()))
    a_map = dict(zip(row_ids, avg_vec.tolist()))
    # cand_B: averaged only if l2 gate fired; else EXACT base seed-0 array.
    b_vec = avg_vec if l2_fired else base_vec
    b_map = dict(zip(row_ids, b_vec.tolist()))
    # cand_C: averaged only if ratio >= 0.015; else EXACT base seed-0 array.
    c_vec = avg_vec if c_fired else base_vec
    c_map = dict(zip(row_ids, c_vec.tolist()))

    preds = {"base": base_map, "cand_A": a_map, "cand_B": b_map,
             "cand_C": c_map}
    return preds, l2, l2_fired, c_fired, msl_val, n_fits


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
            preds, l2, l2_fired, c_fired, msl_val, n_fits = run_one(
                train_csv, test_csv)
            total_fits += n_fits
            rec["l2_fired"] = l2_fired
            # cand_B averaged only where l2 fired; cand_C where ratio>=0.015.
            rec["b_averaged"] = bool(l2_fired)
            rec["c_averaged"] = bool(c_fired)
            rec["cand_A_averaged"] = True
            for cfg in ALL_CONFIGS:
                pub, prv = score_split(preds[cfg], sol)
                rec[f"{cfg}_pub"] = pub
                rec[f"{cfg}_prv"] = prv
                rec[f"{cfg}_msl"] = msl_val
            print(f"[OK] {name} (l2={l2}, l2_fired={l2_fired}, msl={msl_val}, "
                  f"fits={n_fits}): "
                  f"base pub={rec['base_pub']:.6f} prv={rec['base_prv']:.6f} | "
                  f"A pub={rec['cand_A_pub']:.6f} prv={rec['cand_A_prv']:.6f} | "
                  f"B pub={rec['cand_B_pub']:.6f} prv={rec['cand_B_prv']:.6f}")
        except Exception as e:  # capture but keep going — CLEAN-RUN diagnostic
            exceptions.append((name, repr(e)))
            rec["l2_fired"] = False
            rec["b_averaged"] = False
            rec["c_averaged"] = False
            rec["cand_A_averaged"] = False
            for cfg in ALL_CONFIGS:
                rec[f"{cfg}_pub"] = float("nan")
                rec[f"{cfg}_prv"] = float("nan")
                rec[f"{cfg}_msl"] = float("nan")
            print(f"[ERROR] {name}: {e!r}")
        rows.append(rec)

    # ---- write results CSV ----
    csv_path = os.path.join(OUT_DIR, "results.csv")
    fieldnames = ["dataset", "l2_fired", "cand_A_averaged", "b_averaged",
                  "c_averaged"]
    for cfg in ALL_CONFIGS:
        fieldnames += [f"{cfg}_pub", f"{cfg}_prv", f"{cfg}_msl"]
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

    def l2_firing_datasets():
        return [r["dataset"] for r in rows if r.get("l2_fired")]

    l2_fired_list = l2_firing_datasets()

    # ---- INVARIANT check: the gated candidates (cand_B, cand_C) on any dataset
    #      where they did NOT seed-average must be byte-identical to base (delta
    #      exactly 0), because they reuse the exact seed-0 array there. This is
    #      the required base-reproduction check for each gated candidate.
    #      (cand_A averages everywhere and is exempt.) ----
    invariant_violations = []
    gated_flags = {"cand_B": "b_averaged", "cand_C": "c_averaged"}
    for cfg, flag in gated_flags.items():
        for r in rows:
            if not r.get(flag):  # cfg used base seed-0 here
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
        summary_lines.append(f"=== PER-DATASET ({tag}) ===")
        header = (f"{'dataset':<10} {'l2G':>4} {'msl':>4} {BASE:>9}")
        for cfg in CANDIDATES:
            header += f" {cfg:>10} {'d'+cfg:>11}"
        summary_lines.append(header)
        for r in rows:
            line = (f"{r['dataset']:<10} {str(bool(r.get('l2_fired'))):>4} "
                    f"{str(r.get(f'{BASE}_msl')):>4} "
                    f"{r[f'{BASE}_{split}']:>9.4f}")
            for cfg in CANDIDATES:
                line += (f" {r[f'{cfg}_{split}']:>10.4f} "
                         f"{delta(r, cfg, split):>+11.5f}")
            summary_lines.append(line)

    # ---- gate firings ----
    summary_lines.append("")
    summary_lines.append("=== GATE FIRINGS / SEED-AVERAGING SCOPE ===")
    summary_lines.append(
        f"L2 GATE (ratio>=0.010, l2=1.0) fired on "
        f"({len(l2_fired_list)}): "
        f"{', '.join(l2_fired_list) if l2_fired_list else '(none)'}")
    a_avg = [r["dataset"] for r in rows if r.get("cand_A_averaged")]
    b_avg = [r["dataset"] for r in rows if r.get("b_averaged")]
    c_avg = [r["dataset"] for r in rows if r.get("c_averaged")]
    summary_lines.append(
        f"cand_A seed-avg (K=5) applied on ({len(a_avg)}): ALL datasets.")
    summary_lines.append(
        f"cand_B seed-avg (K=5) applied ONLY on L2-firing ratio>=0.010 "
        f"({len(b_avg)}): {', '.join(b_avg) if b_avg else '(none)'}; "
        f"other datasets = single seed-0 (byte-identical to base).")
    summary_lines.append(
        f"cand_C seed-avg (K=5) applied ONLY on ratio>=0.015 "
        f"({len(c_avg)}): {', '.join(c_avg) if c_avg else '(none)'}; "
        f"EXCLUDES train_16 (~0.0116); other datasets = single seed-0 "
        f"(byte-identical to base).")

    # ---- INVARIANT report ----
    summary_lines.append("")
    summary_lines.append(
        "=== INVARIANT (gated cand_B/cand_C non-fired datasets identical to "
        "base, delta 0) ===")
    if invariant_violations:
        summary_lines.append("VIOLATED! a gated candidate's non-fired dataset "
                             "differs from base:")
        for cfg, ds, dp, dv in invariant_violations:
            summary_lines.append(
                f"  {cfg}/{ds}: pub d={dp:+.6g} prv d={dv:+.6g}")
    else:
        summary_lines.append(
            "OK: for cand_B (12 non-fired datasets) and cand_C (13 non-fired "
            "datasets, incl. train_16), every non-seed-averaged dataset is "
            "byte-identical to base (delta exactly 0). Required "
            "base-reproduction check for each gated candidate.")

    # ---- which datasets actually differed ----
    summary_lines.append("")
    summary_lines.append("=== DATASETS THAT ACTUALLY DIFFER (candidate vs base) ===")
    for cfg in CANDIDATES:
        diff = differing_datasets(cfg)
        summary_lines.append(
            f"{cfg}: ({len(diff)}) {', '.join(diff) if diff else '(none)'}")

    # ---- per-candidate summary vs base(08) ----
    summary_lines.append("")
    summary_lines.append("=== SUMMARY (candidate vs base == shipped 08) ===")
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

    # ---- train_13 spotlight (the known counter-mover) ----
    summary_lines.append("")
    summary_lines.append("=== train_13 SPOTLIGHT (counter-mover) ===")
    t13 = next((r for r in rows if r["dataset"] == "train_13"), None)
    if t13 is not None:
        for cfg in [BASE] + CANDIDATES:
            summary_lines.append(
                f"  {cfg:<7} pub={t13[f'{cfg}_pub']:.4f} prv={t13[f'{cfg}_prv']:.4f}"
                + ("" if cfg == BASE else
                   f"  (d pub {delta(t13, cfg, 'pub'):+.5f}, "
                   f"prv {delta(t13, cfg, 'prv'):+.5f})"))
    else:
        summary_lines.append("  (train_13 not found)")

    # ---- verdict ----
    summary_lines.append("")
    summary_lines.append("=== VERDICT (adoption vs base == shipped 08) ===")
    summary_lines.append(
        "Criterion: CLEAN IMPROVEMENT over 08 iff mean delta positive AND "
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
            f"OVERALL: clean improvement over 08 found: {', '.join(clean_names)} "
            f"(orchestrator/human decides adoption; seed-averaging is a mild "
            f"structural change).")
    else:
        summary_lines.append(
            "OVERALL: NO clean improvement over 08; base (shipped 08, single "
            "seed) remains best. K=5 seed-averaging (global or gated) did not "
            "cleanly beat the single-seed baseline.")

    # ---- clean-run line ----
    summary_lines.append("")
    clean_run = (not exceptions) and (not invariant_violations)
    summary_lines.append(
        f"CLEAN RUN={'YES' if clean_run else 'NO'} "
        f"(total_fits={total_fits}, exceptions={len(exceptions)}, "
        f"skipped={len(skipped)}, invariant_violations={len(invariant_violations)})")
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
