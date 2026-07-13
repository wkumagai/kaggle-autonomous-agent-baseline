#!/usr/bin/env python
"""
bench_03 round21 — TWO-GATE extension of the shipped 06 gated-l2 recipe.
OFFLINE ONLY. No subprocess, no LLM, no Kaggle. Calls sklearn in-process.

Adapted from experiments/bench_03/round20_gated_reg2/replay.py.

Base recipe reproduced (verified vs
  `git show HEAD:submissions/02_early_stopping/agent/prompts/system.md` and
  `git show HEAD:submissions/06_ngated_l2/agent/prompts/system.md`):
  - load train.csv, test.csv with pandas
  - features = [c for c in train.columns if c not in ("row_id","target")]
  - cat_mask  = [train[c].dtype == object for c in features]
  - n = len(train); n_feat = len(features)
  - L2 GATE: fires when (n_feat / n) >= 0.010  (shipped 06)
  - base06: l2 = 1.0 if l2-gate fires else 0.0; every other HGB param default.
  - HistGradientBoostingClassifier(categorical_features=cat_mask, random_state=0,
                                   max_iter=300, early_stopping=True,
                                   l2_regularization=l2)
  - fit(train[features], train["target"])
  - pred = clf.predict_proba(test[features])[:, pos_class_1]
Score: ROC AUC on Public rows and Private rows separately, joined by row_id to
solution.csv, over all 16 datasets.

The QUESTION (round21 angle): round20's single-gate msl50 (min_samples_leaf=50
applied on the SAME 0.010 l2-gate) lifted train_09/13/15 by Private +0.012..+0.020
but REGRESSED the boundary dataset train_16 (ratio 0.0116, only just above 0.010)
by Private -0.0005. Can we keep those three big gains while dropping train_16 to
"no change" by gating the SECOND knob on a TIGHTER threshold?

Configs:
  base06       : l2 = 1.0 if (n_feat/n) >= 0.010 else 0.0; all other HGB params
                 default.  (== shipped 06)
  msl50_2gate  : IDENTICAL l2 gate & l2=1.0 as base06 (fires 09/13/15/16), PLUS a
                 SECOND, tighter gate: min_samples_leaf = 50 when (n_feat/n) >=
                 0.015 else 20 (the HGB default).  The second gate fires on
                 train_09 (0.0162) / train_13 (0.0180) / train_15 (0.0600) but
                 NOT on the boundary train_16 (0.0116) -> train_16 stays
                 byte-identical to base06.

Expected firing (per dataset_stats.csv ratios n_feat/n):
  train_09: 18/1109  = 0.0162  -> l2-gate YES, msl-gate YES
  train_13:  9/500   = 0.0180  -> l2-gate YES, msl-gate YES
  train_15: 30/500   = 0.0600  -> l2-gate YES, msl-gate YES
  train_16: 21/1809  = 0.0116  -> l2-gate YES, msl-gate NO  (== base06)
  all others          < 0.010  -> neither gate; byte-identical to 02/06.

INVARIANT: for msl50_2gate, only datasets where the msl-gate (>=0.015) fires can
differ from base06. The l2-gate-only dataset train_16 and all 12 non-firing
datasets are byte-identical to base06 and contribute exactly 0 to every delta.

Adoption criterion: msl50_2gate is a CLEAN IMPROVEMENT over base06 iff its mean
delta is positive on BOTH splits AND there are ZERO regressions on BOTH splits.
A single regression on either split => REJECTED.
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
OUT_DIR = os.path.join(REPO, "experiments", "bench_03", "round21_2gate_msl50")

L2_GATE_THRESHOLD = 0.010   # shipped 06 feature-to-row-ratio gate for l2
GATED_L2 = 1.0              # l2 applied when the l2-gate fires (shipped 06)
MSL_GATE_THRESHOLD = 0.015  # round21 tighter second gate for min_samples_leaf
GATED_MSL = 50             # min_samples_leaf applied when the msl-gate fires
# (HGB default min_samples_leaf is 20; used when the msl-gate does NOT fire.)

# Each config: (name, use_msl_gate).
# base06 never applies the second knob. msl50_2gate applies min_samples_leaf=50
# ONLY on datasets whose ratio clears the tighter MSL_GATE_THRESHOLD.
CONFIGS = [
    ("base06", False),
    ("msl50_2gate", True),
]
BASE = "base06"
CANDIDATES = ["msl50_2gate"]

N_DATASETS = 16


def auc_or_nan(y_true, y_score):
    """ROC AUC, or NaN if the subset has a single class (undefined)."""
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def run_one(train_csv, test_csv, use_msl_gate):
    """Reproduce the 06 gated-l2 recipe for one dataset. When use_msl_gate is
    True, ALSO apply min_samples_leaf=50 iff the ratio clears the tighter
    MSL_GATE_THRESHOLD (0.015).
    Returns (pred_map, l2, l2_fired, msl_fired, msl_val) where pred_map maps
    test row_id -> pos-class prob."""
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)

    features = [c for c in train.columns if c not in ("row_id", "target")]
    cat_mask = [train[c].dtype == object for c in features]

    n = len(train)
    n_feat = len(features)
    ratio = n_feat / n

    l2_fired = ratio >= L2_GATE_THRESHOLD
    l2 = GATED_L2 if l2_fired else 0.0

    msl_fired = use_msl_gate and (ratio >= MSL_GATE_THRESHOLD)

    kwargs = dict(
        categorical_features=cat_mask,
        random_state=0,
        max_iter=300,
        early_stopping=True,
        l2_regularization=l2,
    )
    # Only override min_samples_leaf when the tighter second gate fires; otherwise
    # leave it at the HGB default (20) so the classifier is byte-identical to 06.
    msl_val = GATED_MSL if msl_fired else None
    if msl_fired:
        kwargs["min_samples_leaf"] = GATED_MSL

    clf = HistGradientBoostingClassifier(**kwargs)
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
        for cfg_name, use_msl_gate in CONFIGS:
            try:
                pred_map, l2, l2_fired, msl_fired, msl_val = run_one(
                    train_csv, test_csv, use_msl_gate)
                pub, prv = score_split(pred_map, sol)
                rec[f"{cfg_name}_pub"] = pub
                rec[f"{cfg_name}_prv"] = prv
                rec[f"{cfg_name}_l2"] = l2
                # l2 gate firing is config-independent (both configs use it).
                rec["l2_fired"] = l2_fired
                # msl gate firing is only meaningful for the candidate config.
                if use_msl_gate:
                    rec["msl_fired"] = msl_fired
                print(f"[OK] {name} {cfg_name} (l2={l2}, l2_fired={l2_fired}, "
                      f"msl_fired={msl_fired}, msl={msl_val}): "
                      f"pub={pub:.6f} prv={prv:.6f}")
            except Exception as e:  # capture but keep going — CLEAN-RUN diagnostic
                exceptions.append((name, cfg_name, repr(e)))
                rec[f"{cfg_name}_pub"] = float("nan")
                rec[f"{cfg_name}_prv"] = float("nan")
                rec[f"{cfg_name}_l2"] = float("nan")
                print(f"[ERROR] {name} {cfg_name}: {e!r}")
        rows.append(rec)

    # ---- write results CSV ----
    csv_path = os.path.join(OUT_DIR, "results.csv")
    fieldnames = ["dataset", "l2_fired", "msl_fired"]
    for cfg, _ in CONFIGS:
        fieldnames += [f"{cfg}_pub", f"{cfg}_prv", f"{cfg}_l2"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    # ---- deltas & helpers (all vs base06) ----
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
        """Datasets where cfg is strictly worse than base06 on this split."""
        out = []
        for r in rows:
            dd = delta(r, cfg, split)
            if not math.isnan(dd) and dd < -eps:
                out.append((r["dataset"], dd))
        return out

    def l2_firing_datasets():
        return [r["dataset"] for r in rows if r.get("l2_fired")]

    def msl_firing_datasets():
        return [r["dataset"] for r in rows if r.get("msl_fired")]

    l2_fired_list = l2_firing_datasets()
    msl_fired_list = msl_firing_datasets()

    summary_lines = []

    # ---- per-dataset tables (Public, Private) ----
    summary_lines.append("=== PER-DATASET (Public) ===")
    header = (f"{'dataset':<10} {'l2G':>4} {'mslG':>5} {'base06':>9}")
    for cfg in CANDIDATES:
        header += f" {cfg:>13} {'d'+cfg:>13}"
    summary_lines.append(header)
    for r in rows:
        line = (f"{r['dataset']:<10} {str(bool(r.get('l2_fired'))):>4} "
                f"{str(bool(r.get('msl_fired'))):>5} {r['base06_pub']:>9.4f}")
        for cfg in CANDIDATES:
            line += f" {r[f'{cfg}_pub']:>13.4f} {delta(r, cfg, 'pub'):>+13.4f}"
        summary_lines.append(line)

    summary_lines.append("")
    summary_lines.append("=== PER-DATASET (Private) ===")
    header = (f"{'dataset':<10} {'l2G':>4} {'mslG':>5} {'base06':>9}")
    for cfg in CANDIDATES:
        header += f" {cfg:>13} {'d'+cfg:>13}"
    summary_lines.append(header)
    for r in rows:
        line = (f"{r['dataset']:<10} {str(bool(r.get('l2_fired'))):>4} "
                f"{str(bool(r.get('msl_fired'))):>5} {r['base06_prv']:>9.4f}")
        for cfg in CANDIDATES:
            line += f" {r[f'{cfg}_prv']:>13.4f} {delta(r, cfg, 'prv'):>+13.4f}"
        summary_lines.append(line)

    # ---- gate firings ----
    summary_lines.append("")
    summary_lines.append("=== GATE FIRINGS ===")
    summary_lines.append(
        f"L2 GATE (ratio>=0.010, base06 & candidate, l2=1.0) fired on "
        f"({len(l2_fired_list)}): "
        f"{', '.join(l2_fired_list) if l2_fired_list else '(none)'}")
    summary_lines.append(
        f"MSL GATE (ratio>=0.015, candidate only, min_samples_leaf=50) fired on "
        f"({len(msl_fired_list)}): "
        f"{', '.join(msl_fired_list) if msl_fired_list else '(none)'}")
    summary_lines.append(
        "(datasets where the MSL gate does NOT fire are byte-identical to "
        "base06 -> delta 0, including the boundary l2-only dataset train_16)")

    # ---- per-candidate summary vs base06 ----
    summary_lines.append("")
    summary_lines.append("=== SUMMARY (each candidate vs base06) ===")
    for cfg in CANDIDATES:
        mp = mean_delta(cfg, "pub")
        mv = mean_delta(cfg, "prv")
        wp, lp, tp = wlt(cfg, "pub")
        wv, lv, tv = wlt(cfg, "prv")
        summary_lines.append(
            f"{cfg}: mean Public d={mp:+.5f}  mean Private d={mv:+.5f}  "
            f"Public W/L/T={wp}/{lp}/{tp}  Private W/L/T={wv}/{lv}/{tv}")

    # ---- per-candidate firing-dataset deltas + regressions ----
    summary_lines.append("")
    summary_lines.append("=== PER-CANDIDATE DETAIL (MSL-gate firing datasets only; "
                         "all other deltas are exactly 0) ===")
    for cfg in CANDIDATES:
        summary_lines.append(f"--- {cfg} ---")
        for r in rows:
            if not r.get("msl_fired"):
                continue
            dp = delta(r, cfg, "pub")
            dv = delta(r, cfg, "prv")
            summary_lines.append(
                f"  {r['dataset']:<10} "
                f"pub {r['base06_pub']:.4f}->{r[f'{cfg}_pub']:.4f} ({dp:+.5f})   "
                f"prv {r['base06_prv']:.4f}->{r[f'{cfg}_prv']:.4f} ({dv:+.5f})")
        rp = regressions(cfg, "pub")
        rv = regressions(cfg, "prv")
        pub_str = ", ".join(f"{n}({d:+.5f})" for n, d in rp) if rp else "(none)"
        prv_str = ", ".join(f"{n}({d:+.5f})" for n, d in rv) if rv else "(none)"
        summary_lines.append(f"  Public regressions:  {pub_str}")
        summary_lines.append(f"  Private regressions: {prv_str}")

    # ---- verdict ----
    summary_lines.append("")
    summary_lines.append("=== VERDICT (adoption vs base06) ===")
    summary_lines.append(
        "Criterion: CLEAN IMPROVEMENT over 06 iff mean delta positive AND "
        "zero regressions on BOTH splits.")
    any_clean = False
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
            verdict = (f"ADOPTED-CLEAN-IMPROVEMENT "
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
            verdict = "REJECTED (" + "; ".join(reasons) + ")"
        summary_lines.append(f"{cfg}: {verdict}")
    summary_lines.append("")
    summary_lines.append(
        f"OVERALL: {'msl50_2gate is a CLEAN IMPROVEMENT over 06 (adoption candidate)' if any_clean else 'NO clean improvement; base06 (l2=1.0 only) remains best'}.")

    # ---- clean-run line ----
    summary_lines.append("")
    summary_lines.append(f"CLEAN RUN: {'YES' if not exceptions else 'NO'} "
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
