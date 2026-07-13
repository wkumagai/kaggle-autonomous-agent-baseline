#!/usr/bin/env python
"""
bench_03 round17 — single-knob replay harness for l2_regularization on top of the
shipped 02_early_stopping simple-path recipe.

OFFLINE ONLY. No subprocess, no LLM, no Kaggle. Calls sklearn in-process.

Recipe reproduced (verified vs `git show HEAD:submissions/02_early_stopping/agent/prompts/system.md`):
  - load train.csv, test.csv with pandas
  - features = [c for c in train.columns if c not in ("row_id","target")]
  - cat_mask  = [train[c].dtype == object for c in features]
  - HistGradientBoostingClassifier(categorical_features=cat_mask, random_state=0,
                                   max_iter=300, early_stopping=True)
  - fit(train[features], train["target"])
  - pred = clf.predict_proba(test[features])[:, 1]
Score: ROC AUC on Public rows and Private rows separately, joined by row_id to solution.csv.

Only knob varied across configs: l2_regularization (BASE=0.0, CAND-a=0.1, CAND-b=1.0).
"""
import os
import sys
import csv
import math
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

REPO = "/Users/kumacmini/kaggle-autonomous-agent-baseline-auto"
DATA_DIR = os.path.join(REPO, "data")
OUT_DIR = os.path.join(REPO, "experiments", "bench_03", "round17_l2simple")

CONFIGS = [
    ("base",   0.0),
    ("cand_a", 0.1),
    ("cand_b", 1.0),
]

N_DATASETS = 16
N_JOBS = 2  # modest, avoid CPU contention


def auc_or_nan(y_true, y_score):
    """ROC AUC, or NaN if the subset has a single class (undefined)."""
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def run_one(train_csv, test_csv, l2):
    """Reproduce the 02 recipe for one dataset with the given l2_regularization.
    Returns a dict row_id -> predicted positive-class probability (test rows)."""
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)

    features = [c for c in train.columns if c not in ("row_id", "target")]
    cat_mask = [train[c].dtype == object for c in features]

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

    return dict(zip(test["row_id"].tolist(), pred.tolist()))


def score_split(pred_map, sol):
    """Given pred_map (row_id->prob) and solution df, return (public_auc, private_auc)."""
    sol = sol.copy()
    sol["pred"] = sol["row_id"].map(pred_map)
    # every solution row must have a prediction; if any missing, that's a real error
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
        for cfg_name, l2 in CONFIGS:
            try:
                pred_map = run_one(train_csv, test_csv, l2)
                pub, prv = score_split(pred_map, sol)
                rec[f"{cfg_name}_pub"] = pub
                rec[f"{cfg_name}_prv"] = prv
                print(f"[OK] {name} {cfg_name} (l2={l2}): pub={pub:.6f} prv={prv:.6f}")
            except Exception as e:  # capture but keep going — CLEAN-RUN diagnostic
                exceptions.append((name, cfg_name, repr(e)))
                rec[f"{cfg_name}_pub"] = float("nan")
                rec[f"{cfg_name}_prv"] = float("nan")
                print(f"[ERROR] {name} {cfg_name} (l2={l2}): {e!r}")
        rows.append(rec)

    # ---- write results CSV ----
    csv_path = os.path.join(OUT_DIR, "results.csv")
    fieldnames = ["dataset",
                  "base_pub", "cand_a_pub", "cand_b_pub",
                  "base_prv", "cand_a_prv", "cand_b_prv"]
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
    summary_lines.append(f"{'dataset':<10} {'base':>9} {'cand_a':>9} {'dA':>10} {'cand_b':>9} {'dB':>10}")
    for r in rows:
        dA = delta(r, "cand_a", "pub")
        dB = delta(r, "cand_b", "pub")
        summary_lines.append(
            f"{r['dataset']:<10} {r['base_pub']:>9.4f} {r['cand_a_pub']:>9.4f} {dA:>+10.4f} "
            f"{r['cand_b_pub']:>9.4f} {dB:>+10.4f}")

    summary_lines.append("")
    summary_lines.append("=== PER-DATASET (Private) ===")
    summary_lines.append(f"{'dataset':<10} {'base':>9} {'cand_a':>9} {'dA':>10} {'cand_b':>9} {'dB':>10}")
    for r in rows:
        dA = delta(r, "cand_a", "prv")
        dB = delta(r, "cand_b", "prv")
        summary_lines.append(
            f"{r['dataset']:<10} {r['base_prv']:>9.4f} {r['cand_a_prv']:>9.4f} {dA:>+10.4f} "
            f"{r['cand_b_prv']:>9.4f} {dB:>+10.4f}")

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

    summary_lines.append("")
    summary_lines.append("=== SUMMARY ===")
    for cfg in ("cand_a", "cand_b"):
        mp = mean_delta(cfg, "pub")
        mv = mean_delta(cfg, "prv")
        wp, lp, tp = wlt(cfg, "pub")
        summary_lines.append(
            f"{cfg}: mean Public d={mp:+.4f}  mean Private d={mv:+.4f}  "
            f"Public W/L/T={wp}/{lp}/{tp}")

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
