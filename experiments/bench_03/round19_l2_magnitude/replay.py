#!/usr/bin/env python
"""
bench_03 round19 — l2_regularization MAGNITUDE sweep within the adopted
feature-ratio gate, on top of the shipped 02_early_stopping simple-path recipe.

OFFLINE ONLY. No subprocess, no LLM, no Kaggle. Calls sklearn in-process.

Adapted from experiments/bench_03/round18_ngated_l2/replay.py.

Recipe reproduced (verified vs
  `git show HEAD:submissions/02_early_stopping/agent/prompts/system.md` and
  `git show HEAD:submissions/06_ngated_l2/agent/prompts/system.md`):
  - load train.csv, test.csv with pandas
  - features = [c for c in train.columns if c not in ("row_id","target")]
  - cat_mask  = [train[c].dtype == object for c in features]
  - n = len(train); n_feat = len(features)
  - GATE: l2 = L if (n_feat / n) >= 0.010 else 0.0
  - HistGradientBoostingClassifier(categorical_features=cat_mask, random_state=0,
                                   max_iter=300, early_stopping=True,
                                   l2_regularization=l2)
  - fit(train[features], train["target"])
  - pred = clf.predict_proba(test[features])[:, pos_class_1]
Score: ROC AUC on Public rows and Private rows separately, joined by row_id to solution.csv.

The QUESTION: within the SAME adopted gate (threshold 0.010 fixed), does a
different l2 MAGNITUDE L beat the shipped L=1.0? The gate threshold never
changes; only the magnitude L applied when the gate fires varies.

Configs:
  base   : l2 = 0.0 for every dataset                       (== shipped 02, baseline)
  L0.5   : l2 = 0.5 if (n_feat / n) >= 0.010 else 0.0
  L1.0   : l2 = 1.0 if (n_feat / n) >= 0.010 else 0.0        (== shipped 06)
  L2.0   : l2 = 2.0 if (n_feat / n) >= 0.010 else 0.0
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
OUT_DIR = os.path.join(REPO, "experiments", "bench_03", "round19_l2_magnitude")

GATE_THRESHOLD = 0.010  # fixed feature-to-row-ratio gate threshold


# --- gate functions: given (n, n_feat) -> l2_regularization value for this dataset ---
def make_gate(magnitude):
    """Return a gate fn that applies `magnitude` when the ratio gate fires, else 0.0."""
    def gate_fn(n, n_feat):
        return magnitude if (n_feat / n) >= GATE_THRESHOLD else 0.0
    return gate_fn


def gate_base(n, n_feat):
    return 0.0


CONFIGS = [
    ("base", gate_base),
    ("L0.5", make_gate(0.5)),
    ("L1.0", make_gate(1.0)),
    ("L2.0", make_gate(2.0)),
]
SWEEP_CONFIGS = ["L0.5", "L1.0", "L2.0"]

N_DATASETS = 16


def auc_or_nan(y_true, y_score):
    """ROC AUC, or NaN if the subset has a single class (undefined)."""
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def run_one(train_csv, test_csv, gate_fn):
    """Reproduce the 02/06 recipe for one dataset. The gate decides l2 from (n, n_feat).
    Returns (pred_map, l2) where pred_map maps test row_id -> positive-class prob."""
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)

    features = [c for c in train.columns if c not in ("row_id", "target")]
    cat_mask = [train[c].dtype == object for c in features]

    n = len(train)
    n_feat = len(features)
    l2 = gate_fn(n, n_feat)

    clf = HistGradientBoostingClassifier(
        categorical_features=cat_mask,
        random_state=0,
        max_iter=300,
        early_stopping=True,
        l2_regularization=l2,
    )
    clf.fit(train[features], train["target"])
    proba = clf.predict_proba(test[features])

    # positive-class column: match recipe's [:, 1] but be robust to class order.
    classes = list(clf.classes_)
    pos_idx = classes.index(1) if 1 in classes else 1
    pred = proba[:, pos_idx]

    return dict(zip(test["row_id"].tolist(), pred.tolist())), l2


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
        for cfg_name, gate_fn in CONFIGS:
            try:
                pred_map, l2 = run_one(train_csv, test_csv, gate_fn)
                pub, prv = score_split(pred_map, sol)
                rec[f"{cfg_name}_pub"] = pub
                rec[f"{cfg_name}_prv"] = prv
                rec[f"{cfg_name}_l2"] = l2
                print(f"[OK] {name} {cfg_name} (l2={l2}): pub={pub:.6f} prv={prv:.6f}")
            except Exception as e:  # capture but keep going — CLEAN-RUN diagnostic
                exceptions.append((name, cfg_name, repr(e)))
                rec[f"{cfg_name}_pub"] = float("nan")
                rec[f"{cfg_name}_prv"] = float("nan")
                rec[f"{cfg_name}_l2"] = float("nan")
                print(f"[ERROR] {name} {cfg_name}: {e!r}")
        rows.append(rec)

    # ---- write results CSV ----
    csv_path = os.path.join(OUT_DIR, "results.csv")
    fieldnames = ["dataset",
                  "base_pub", "L0.5_pub", "L1.0_pub", "L2.0_pub",
                  "base_prv", "L0.5_prv", "L1.0_prv", "L2.0_prv",
                  "L0.5_l2", "L1.0_l2", "L2.0_l2"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    # ---- deltas & summary ----
    def delta(rec, cfg, split):
        b = rec.get(f"base_{split}")
        c = rec.get(f"{cfg}_{split}")
        if b is None or c is None or math.isnan(b) or math.isnan(c):
            return float("nan")
        return c - b

    summary_lines = []
    summary_lines.append("=== PER-DATASET (Public) ===")
    summary_lines.append(
        f"{'dataset':<10} {'base':>9} {'L0.5':>9} {'dL0.5':>9} "
        f"{'L1.0':>9} {'dL1.0':>9} {'L2.0':>9} {'dL2.0':>9}")
    for r in rows:
        d05 = delta(r, "L0.5", "pub")
        d10 = delta(r, "L1.0", "pub")
        d20 = delta(r, "L2.0", "pub")
        summary_lines.append(
            f"{r['dataset']:<10} {r['base_pub']:>9.4f} "
            f"{r['L0.5_pub']:>9.4f} {d05:>+9.4f} "
            f"{r['L1.0_pub']:>9.4f} {d10:>+9.4f} "
            f"{r['L2.0_pub']:>9.4f} {d20:>+9.4f}")

    summary_lines.append("")
    summary_lines.append("=== PER-DATASET (Private) ===")
    summary_lines.append(
        f"{'dataset':<10} {'base':>9} {'L0.5':>9} {'dL0.5':>9} "
        f"{'L1.0':>9} {'dL1.0':>9} {'L2.0':>9} {'dL2.0':>9}")
    for r in rows:
        d05 = delta(r, "L0.5", "prv")
        d10 = delta(r, "L1.0", "prv")
        d20 = delta(r, "L2.0", "prv")
        summary_lines.append(
            f"{r['dataset']:<10} {r['base_prv']:>9.4f} "
            f"{r['L0.5_prv']:>9.4f} {d05:>+9.4f} "
            f"{r['L1.0_prv']:>9.4f} {d10:>+9.4f} "
            f"{r['L2.0_prv']:>9.4f} {d20:>+9.4f}")

    def mean_delta(cfg, split):
        vals = [delta(r, cfg, split) for r in rows]
        vals = [v for v in vals if not math.isnan(v)]
        return sum(vals) / len(vals) if vals else float("nan")

    def wlt(cfg, split, eps=1e-6):
        w = l = t = 0
        for r in rows:
            d = delta(r, cfg, split)
            if math.isnan(d):
                continue
            if d > eps:
                w += 1
            elif d < -eps:
                l += 1
            else:
                t += 1
        return w, l, t

    def regressions(cfg, split, eps=1e-6):
        """Datasets where cfg is strictly worse than base on this split."""
        out = []
        for r in rows:
            d = delta(r, cfg, split)
            if not math.isnan(d) and d < -eps:
                out.append((r["dataset"], d))
        return out

    def fired(cfg, eps=1e-9):
        """Datasets where this gated config chose l2 > 0 (gate fired)."""
        out = []
        for r in rows:
            l2 = r.get(f"{cfg}_l2")
            if l2 is not None and not (isinstance(l2, float) and math.isnan(l2)) and l2 > eps:
                out.append(r["dataset"])
        return out

    summary_lines.append("")
    summary_lines.append("=== GATE FIRINGS (datasets where l2 > 0) ===")
    # all sweep configs share the same gate (same threshold), so firings match.
    for cfg in SWEEP_CONFIGS:
        f = fired(cfg)
        summary_lines.append(f"{cfg} FIRED on ({len(f)}): {', '.join(f) if f else '(none)'}")

    summary_lines.append("")
    summary_lines.append("=== SUMMARY (each L vs base) ===")
    for cfg in SWEEP_CONFIGS:
        mp = mean_delta(cfg, "pub")
        mv = mean_delta(cfg, "prv")
        wp, lp, tp = wlt(cfg, "pub")
        wv, lv, tv = wlt(cfg, "prv")
        summary_lines.append(
            f"{cfg}: mean Public d={mp:+.4f}  mean Private d={mv:+.4f}  "
            f"Public W/L/T={wp}/{lp}/{tp}  Private W/L/T={wv}/{lv}/{tv}")

    summary_lines.append("")
    summary_lines.append("=== REGRESSIONS vs base (strictly worse) ===")
    for cfg in SWEEP_CONFIGS:
        rp = regressions(cfg, "pub")
        rv = regressions(cfg, "prv")
        pub_str = ", ".join(f"{n}({d:+.4f})" for n, d in rp) if rp else "(none)"
        prv_str = ", ".join(f"{n}({d:+.4f})" for n, d in rv) if rv else "(none)"
        summary_lines.append(f"{cfg} Public regressions:  {pub_str}")
        summary_lines.append(f"{cfg} Private regressions: {prv_str}")

    # ---- verdict: does any magnitude cleanly beat shipped L=1.0? ----
    # "cleanly beat" = better mean (Public) than L1.0 AND no NEW regression vs base
    # that L1.0 did not already have.
    summary_lines.append("")
    summary_lines.append("=== VERDICT ===")
    l10_pub_mean = mean_delta("L1.0", "pub")
    l10_pub_regs = set(n for n, _ in regressions("L1.0", "pub"))
    l10_prv_regs = set(n for n, _ in regressions("L1.0", "prv"))
    winners = []
    for cfg in ["L0.5", "L2.0"]:
        mp = mean_delta(cfg, "pub")
        cfg_pub_regs = set(n for n, _ in regressions(cfg, "pub"))
        cfg_prv_regs = set(n for n, _ in regressions(cfg, "prv"))
        new_pub_regs = cfg_pub_regs - l10_pub_regs
        new_prv_regs = cfg_prv_regs - l10_prv_regs
        better = mp > l10_pub_mean + 1e-6
        if better and not new_pub_regs and not new_prv_regs:
            winners.append((cfg, mp))
    if winners:
        best = max(winners, key=lambda x: x[1])
        summary_lines.append(
            f"{best[0]} cleanly beats shipped L=1.0 "
            f"(mean Public d={best[1]:+.4f} > L1.0 {l10_pub_mean:+.4f}, no new regressions).")
    else:
        summary_lines.append(
            f"No magnitude cleanly beats shipped L=1.0 "
            f"(L1.0 mean Public d={l10_pub_mean:+.4f}); L=1.0 remains best.")

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
