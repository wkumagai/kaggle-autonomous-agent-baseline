#!/usr/bin/env python
"""
bench_03 round30 — SEED-COUNT K-SWEEP on the round29 cand_C seed-averaging
direction. OFFLINE ONLY. No subprocess, no LLM, no Kaggle. Calls sklearn
in-process.

Adapted from experiments/bench_03/round29_seed_avg/replay.py (dataset loading,
the shipped-08 base config reproduction, Public/Private AUC scoring joined on
row_id to solution.csv, seed-averaging, and the summary machinery are all
reused verbatim). The swept knob is the SEED COUNT K applied under the SAME
cand_C gate.

Base recipe reproduced (== shipped 08, verified vs
  `git show HEAD:submissions/08_ratio_tiered_msl/agent/prompts/system.md`):
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
solution.csv, over all 16 datasets.

The QUESTION (round30 angle): round29 found cand_C = averaging predict_proba
across K=5 seeds (random_state 0..4) of the identical shipped-08 config, applied
ONLY on datasets where ratio = n_feat/n >= 0.015 (fires train_09/13/15; leaves
the other 13 datasets byte-identical to seed-0). cand_C was a clean Pareto
improvement over 08. IS K=5 the right seed count? Sweep K in {3, 5, 10} using
the EXACT same cand_C gate:
  - fires (ratio >= 0.015)  -> mean predict_proba over the first K seeds
  - else                    -> single seed-0 (byte-identical to base == 08)
and see whether K=3 already matches K=5 (a simplicity win = fewer models) or
K=10 beats it.

Configs (all identical shipped-08 config; differ ONLY in K, and ALL use the
SAME cand_C ratio>=0.015 gate):
  base : single seed random_state=0                 == shipped 08 (K=1)
  K3   : mean predict_proba over seeds {0,1,2}       on firing datasets else base
  K5   : mean predict_proba over seeds {0,1,2,3,4}   on firing datasets else base
         (== round29 cand_C)
  K10  : mean predict_proba over seeds {0..9}         on firing datasets else base

IMPLEMENTATION: to minimise fits, for each FIRING dataset (only train_09/13/15)
we fit seeds 0..9 ONCE, then derive every K by averaging predict_proba over the
first K seeds (nested prefixes: K3->seeds0-2, K5->0-4, K10->0-9). For NON-firing
datasets only seed-0 is needed (byte-identical base). Total extra fits ~= 3
firing datasets * 10 seeds + 13 non-firing * 1 seed = 43 fits.

IMPLEMENTATION INVARIANT: every K on any NON-firing dataset is the EXACT same
seed-0 array as base, so its delta MUST be exactly 0 on both splits — checked
explicitly. On firing datasets K includes seed-0 in its average so it generally
differs from base (expected). K3/K5/K10 are nested prefixes so K5 average shares
seeds 0-2 with K3 etc. (correctness note only; each K averaged independently).

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
OUT_DIR = os.path.join(REPO, "experiments", "bench_03", "round30_seed_k_sweep")

L2_GATE_THRESHOLD = 0.010   # shipped 08 feature-to-row-ratio gate for l2 (FIXED)
# cand_C gate (round29): fire seed-averaging on the SAME threshold shipped-08
# uses for its stricter min_samples_leaf tier (0.015). Fires train_09/13/15,
# excludes train_16 (~0.0116). Principled ratio gate, no dataset-name list.
SEEDAVG_GATE_THRESHOLD = 0.015
GATED_L2 = 1.0              # l2 applied when the l2-gate fires (FIXED)
DEFAULT_MSL = 20            # sklearn HGB default, used when no msl tier is cleared.
BASE_SEED = 0              # shipped-08 fixed random_state.

# The K values swept, and the maximum seed count we must fit (10) on firing sets.
K_VALUES = [3, 5, 10]
MAX_SEEDS = max(K_VALUES)                 # 10
ALL_SEEDS = list(range(MAX_SEEDS))        # [0,1,...,9]

# 08 tiered min_samples_leaf, IDENTICAL across all configs (descending order).
MSL_TIERS = [(0.030, 70), (0.015, 50)]

BASE = "base"
CANDIDATES = [f"K{k}" for k in K_VALUES]   # ["K3","K5","K10"]

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
    """Reproduce the shipped-08 config for one dataset and derive base + every
    K under the cand_C gate. Returns
    (preds, l2, l2_fired, fired, msl_val, n_fits) where preds maps
    config_name -> {row_id -> pos-class prob}.
      base = seed-0 prediction (== shipped 08).
      On a FIRING dataset (ratio >= 0.015): fit seeds 0..9 once, and each Kk =
        mean predict_proba over the first k seeds.
      On a NON-firing dataset: only seed-0 is fit and every Kk reuses that exact
        seed-0 array (byte-identical to base)."""
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)

    features = [c for c in train.columns if c not in ("row_id", "target")]
    cat_mask = [train[c].dtype == object for c in features]

    n = len(train)
    n_feat = len(features)
    ratio = n_feat / n

    l2_fired = ratio >= L2_GATE_THRESHOLD
    l2 = GATED_L2 if l2_fired else 0.0
    fired = ratio >= SEEDAVG_GATE_THRESHOLD  # cand_C seed-avg gate (0.015)
    msl_val = msl_for_ratio(ratio)

    row_ids = test["row_id"].tolist()

    if fired:
        # Fit all 10 seeds once; derive each K from nested prefixes.
        seed_preds = {}
        for s in ALL_SEEDS:
            seed_preds[s] = fit_one_seed(train, test, features, cat_mask,
                                         l2, msl_val, s)
        n_fits = len(ALL_SEEDS)
        base_vec = seed_preds[BASE_SEED]
        preds = {BASE: dict(zip(row_ids, base_vec.tolist()))}
        for k in K_VALUES:
            avg_vec = np.mean(np.vstack([seed_preds[s] for s in range(k)]),
                              axis=0)
            preds[f"K{k}"] = dict(zip(row_ids, avg_vec.tolist()))
    else:
        # Only seed-0 needed; every K reuses the exact base array.
        base_vec = fit_one_seed(train, test, features, cat_mask,
                                l2, msl_val, BASE_SEED)
        n_fits = 1
        base_map = dict(zip(row_ids, base_vec.tolist()))
        preds = {BASE: base_map}
        for k in K_VALUES:
            preds[f"K{k}"] = base_map  # byte-identical to base

    return preds, l2, l2_fired, fired, msl_val, n_fits


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
            preds, l2, l2_fired, fired, msl_val, n_fits = run_one(
                train_csv, test_csv)
            total_fits += n_fits
            rec["l2_fired"] = l2_fired
            rec["fires"] = bool(fired)     # cand_C gate (ratio >= 0.015)
            for cfg in ALL_CONFIGS:
                pub, prv = score_split(preds[cfg], sol)
                rec[f"{cfg}_pub"] = pub
                rec[f"{cfg}_prv"] = prv
            rec["msl"] = msl_val
            print(f"[OK] {name} (l2={l2}, l2_fired={l2_fired}, fires={fired}, "
                  f"msl={msl_val}, fits={n_fits}): "
                  f"base pub={rec['base_pub']:.6f} prv={rec['base_prv']:.6f} | "
                  f"K3 pub={rec['K3_pub']:.6f} prv={rec['K3_prv']:.6f} | "
                  f"K5 pub={rec['K5_pub']:.6f} prv={rec['K5_prv']:.6f} | "
                  f"K10 pub={rec['K10_pub']:.6f} prv={rec['K10_prv']:.6f}")
        except Exception as e:  # capture but keep going — CLEAN-RUN diagnostic
            exceptions.append((name, repr(e)))
            rec["l2_fired"] = False
            rec["fires"] = False
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

    firing_list = [r["dataset"] for r in rows if r.get("fires")]

    # ---- write results CSV ----
    #   per dataset: name, fires, base_pub, base_prv, and pub/prv AUC + delta
    #   vs base for K3/K5/K10.
    csv_path = os.path.join(OUT_DIR, "results.csv")
    fieldnames = ["dataset", "fires", "l2_fired", "msl", "base_pub", "base_prv"]
    for cfg in CANDIDATES:
        fieldnames += [f"{cfg}_pub", f"{cfg}_d_pub", f"{cfg}_prv", f"{cfg}_d_prv"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            out = {
                "dataset": r["dataset"],
                "fires": r.get("fires", ""),
                "l2_fired": r.get("l2_fired", ""),
                "msl": r.get("msl", ""),
                "base_pub": r.get("base_pub", ""),
                "base_prv": r.get("base_prv", ""),
            }
            for cfg in CANDIDATES:
                out[f"{cfg}_pub"] = r.get(f"{cfg}_pub", "")
                out[f"{cfg}_prv"] = r.get(f"{cfg}_prv", "")
                out[f"{cfg}_d_pub"] = delta(r, cfg, "pub")
                out[f"{cfg}_d_prv"] = delta(r, cfg, "prv")
            w.writerow(out)

    # ---- INVARIANT check: every K on any NON-firing dataset must be
    #      byte-identical to base (delta exactly 0), because it reuses the exact
    #      seed-0 array there. Required base-reproduction check. ----
    invariant_violations = []
    for cfg in CANDIDATES:
        for r in rows:
            if not r.get("fires"):  # cfg reused base seed-0 here
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
        header = (f"{'dataset':<10} {'fires':>5} {'msl':>4} {BASE:>9}")
        for cfg in CANDIDATES:
            header += f" {cfg:>9} {'d'+cfg:>11}"
        summary_lines.append(header)
        for r in rows:
            line = (f"{r['dataset']:<10} {str(bool(r.get('fires'))):>5} "
                    f"{str(r.get('msl')):>4} "
                    f"{r[f'{BASE}_{split}']:>9.4f}")
            for cfg in CANDIDATES:
                line += (f" {r[f'{cfg}_{split}']:>9.4f} "
                         f"{delta(r, cfg, split):>+11.5f}")
            summary_lines.append(line)

    # ---- gate firings / seed-averaging scope ----
    summary_lines.append("")
    summary_lines.append("=== SEED-AVERAGING SCOPE (cand_C gate, ratio>=0.015) ===")
    summary_lines.append(
        f"seed-averaging fires on ({len(firing_list)}): "
        f"{', '.join(firing_list) if firing_list else '(none)'}; "
        f"all K in {K_VALUES} use this SAME gate. Non-firing datasets = single "
        f"seed-0 (byte-identical to base) for every K.")

    # ---- INVARIANT report ----
    summary_lines.append("")
    summary_lines.append(
        "=== INVARIANT (every K on non-firing datasets identical to base, "
        "delta 0) ===")
    if invariant_violations:
        summary_lines.append("VIOLATED! a K's non-firing dataset differs from "
                             "base:")
        for cfg, ds, dp, dv in invariant_violations:
            summary_lines.append(
                f"  {cfg}/{ds}: pub d={dp:+.6g} prv d={dv:+.6g}")
    else:
        n_nonfire = len([r for r in rows if not r.get("fires")])
        summary_lines.append(
            f"OK: for every K (K3/K5/K10), each of the {n_nonfire} non-firing "
            f"datasets is byte-identical to base (delta exactly 0). Required "
            f"base-reproduction check.")

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

    # ---- K-comparison spotlight (does K3 match K5? does K10 beat K5?) ----
    summary_lines.append("")
    summary_lines.append("=== K-COMPARISON (mean deltas vs 08, zero-regression) ===")
    for cfg in CANDIDATES:
        mp = mean_delta(cfg, "pub")
        mv = mean_delta(cfg, "prv")
        rp = regressions(cfg, "pub")
        rv = regressions(cfg, "prv")
        zr = "YES" if (not rp and not rv) else "NO"
        summary_lines.append(
            f"  {cfg:<4} mean dPublic={mp:+.5f} dPrivate={mv:+.5f} "
            f"zero-regression={zr}")
    # K5 reference (== round29 cand_C)
    mp5, mv5 = mean_delta("K5", "pub"), mean_delta("K5", "prv")
    for cfg in CANDIDATES:
        if cfg == "K5":
            continue
        mp, mv = mean_delta(cfg, "pub"), mean_delta(cfg, "prv")
        rel_pub = "matches/beats" if mp >= mp5 - 1e-9 else "below"
        rel_prv = "matches/beats" if mv >= mv5 - 1e-9 else "below"
        summary_lines.append(
            f"  {cfg} vs K5(cand_C): Public {rel_pub} (d {mp-mp5:+.5f}), "
            f"Private {rel_prv} (d {mv-mv5:+.5f})")

    summary_lines.append("")
    if any_clean:
        summary_lines.append(
            f"OVERALL: clean improvement(s) over 08: {', '.join(clean_names)}. "
            f"Fewer seeds (lower K) that stay clean and match K5 are the "
            f"simplicity win; a higher K only wins if it beats K5 while staying "
            f"clean.")
    else:
        summary_lines.append(
            "OVERALL: NO clean improvement over 08 at any swept K.")

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
