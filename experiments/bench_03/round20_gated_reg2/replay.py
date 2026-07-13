#!/usr/bin/env python
"""
bench_03 round20 — SECOND regularization knob on top of the shipped 06 gated-l2
recipe. OFFLINE ONLY. No subprocess, no LLM, no Kaggle. Calls sklearn in-process.

Adapted from experiments/bench_03/round19_l2_magnitude/replay.py.

Base recipe reproduced (verified vs
  `git show HEAD:submissions/02_early_stopping/agent/prompts/system.md` and
  `git show HEAD:submissions/06_ngated_l2/agent/prompts/system.md`):
  - load train.csv, test.csv with pandas
  - features = [c for c in train.columns if c not in ("row_id","target")]
  - cat_mask  = [train[c].dtype == object for c in features]
  - n = len(train); n_feat = len(features)
  - GATE: fires when (n_feat / n) >= 0.010
  - base06: l2 = 1.0 if gate fires else 0.0; every other HGB param default.
  - HistGradientBoostingClassifier(categorical_features=cat_mask, random_state=0,
                                   max_iter=300, early_stopping=True,
                                   l2_regularization=l2)
  - fit(train[features], train["target"])
  - pred = clf.predict_proba(test[features])[:, pos_class_1]
Score: ROC AUC on Public rows and Private rows separately, joined by row_id to
solution.csv, over all 16 datasets.

The QUESTION (round handoff angle "(b)"): the shipped 06 already applies
l2=1.0 on the SAME feature-ratio gate (fires on train_09/13/15/16). On top of
that same gate & same l2=1.0, does ALSO tightening a SECOND regularization knob
— only on the firing datasets — cleanly improve them further, with zero new
regressions?

Configs (base = shipped 06):
  base06 : l2 = 1.0 if gate fires else 0.0; all other HGB params default. (== 06)
  msl40  : same gate & l2=1.0, PLUS min_samples_leaf=40 when gate fires (dflt 20)
  msl50  : same, min_samples_leaf=50 when gate fires
  mln20  : same gate & l2=1.0, PLUS max_leaf_nodes=20 when gate fires (dflt 31)
  mln15  : same, max_leaf_nodes=15 when gate fires

INVARIANT: when the gate does NOT fire, the classifier is byte-identical to
02/06 for EVERY config (l2=0.0, no extra knob, all defaults). So the 12
non-firing datasets are identical across all configs and contribute exactly 0
to every delta. Only the 4 firing datasets can move.

Adoption criterion (stated in verdict): a candidate is a CLEAN IMPROVEMENT over
06 only if its mean delta is positive AND there are ZERO regressions on BOTH
splits vs base06.
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
OUT_DIR = os.path.join(REPO, "experiments", "bench_03", "round20_gated_reg2")

GATE_THRESHOLD = 0.010  # fixed feature-to-row-ratio gate threshold (shipped 06)
GATED_L2 = 1.0          # l2 applied when the gate fires (shipped 06)

# Each config: (name, extra_key, extra_val).
# extra_key is None for base06 (no second knob). For candidates, the extra knob
# is applied ONLY when the gate fires, on top of l2=1.0.
CONFIGS = [
    ("base06", None, None),
    ("msl40", "min_samples_leaf", 40),
    ("msl50", "min_samples_leaf", 50),
    ("mln20", "max_leaf_nodes", 20),
    ("mln15", "max_leaf_nodes", 15),
]
BASE = "base06"
CANDIDATES = ["msl40", "msl50", "mln20", "mln15"]

N_DATASETS = 16


def auc_or_nan(y_true, y_score):
    """ROC AUC, or NaN if the subset has a single class (undefined)."""
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def run_one(train_csv, test_csv, extra_key, extra_val):
    """Reproduce the 06 gated-l2 recipe for one dataset, optionally adding one
    extra HGB knob ONLY when the gate fires.
    Returns (pred_map, l2, fired) where pred_map maps test row_id -> pos-class prob."""
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)

    features = [c for c in train.columns if c not in ("row_id", "target")]
    cat_mask = [train[c].dtype == object for c in features]

    n = len(train)
    n_feat = len(features)
    fired = (n_feat / n) >= GATE_THRESHOLD
    l2 = GATED_L2 if fired else 0.0

    # Non-firing datasets are byte-identical to 02/06: l2=0.0 (== default) and
    # NO extra knob, all HGB defaults. Firing datasets get l2=1.0 (all configs)
    # plus the candidate's one extra knob.
    kwargs = dict(
        categorical_features=cat_mask,
        random_state=0,
        max_iter=300,
        early_stopping=True,
        l2_regularization=l2,
    )
    if fired and extra_key is not None:
        kwargs[extra_key] = extra_val

    clf = HistGradientBoostingClassifier(**kwargs)
    clf.fit(train[features], train["target"])
    proba = clf.predict_proba(test[features])

    # positive-class column: match recipe's [:, 1] but be robust to class order.
    classes = list(clf.classes_)
    pos_idx = classes.index(1) if 1 in classes else 1
    pred = proba[:, pos_idx]

    return dict(zip(test["row_id"].tolist(), pred.tolist())), l2, fired


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
        for cfg_name, extra_key, extra_val in CONFIGS:
            try:
                pred_map, l2, fired = run_one(train_csv, test_csv, extra_key, extra_val)
                pub, prv = score_split(pred_map, sol)
                rec[f"{cfg_name}_pub"] = pub
                rec[f"{cfg_name}_prv"] = prv
                rec[f"{cfg_name}_l2"] = l2
                rec["fired"] = fired  # gate firing is config-independent
                print(f"[OK] {name} {cfg_name} (l2={l2}, fired={fired}): "
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
    fieldnames = ["dataset", "fired"]
    for cfg, _, _ in CONFIGS:
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
        """Datasets where cfg is strictly worse than base06 on this split."""
        out = []
        for r in rows:
            d = delta(r, cfg, split)
            if not math.isnan(d) and d < -eps:
                out.append((r["dataset"], d))
        return out

    def firing_datasets():
        return [r["dataset"] for r in rows if r.get("fired")]

    fired_list = firing_datasets()

    summary_lines = []

    # ---- per-dataset tables (Public, Private) ----
    summary_lines.append("=== PER-DATASET (Public) ===")
    header = f"{'dataset':<10} {'fired':>5} {'base06':>9}"
    for cfg in CANDIDATES:
        header += f" {cfg:>9} {'d'+cfg:>9}"
    summary_lines.append(header)
    for r in rows:
        line = f"{r['dataset']:<10} {str(bool(r.get('fired'))):>5} {r['base06_pub']:>9.4f}"
        for cfg in CANDIDATES:
            line += f" {r[f'{cfg}_pub']:>9.4f} {delta(r, cfg, 'pub'):>+9.4f}"
        summary_lines.append(line)

    summary_lines.append("")
    summary_lines.append("=== PER-DATASET (Private) ===")
    header = f"{'dataset':<10} {'fired':>5} {'base06':>9}"
    for cfg in CANDIDATES:
        header += f" {cfg:>9} {'d'+cfg:>9}"
    summary_lines.append(header)
    for r in rows:
        line = f"{r['dataset']:<10} {str(bool(r.get('fired'))):>5} {r['base06_prv']:>9.4f}"
        for cfg in CANDIDATES:
            line += f" {r[f'{cfg}_prv']:>9.4f} {delta(r, cfg, 'prv'):>+9.4f}"
        summary_lines.append(line)

    # ---- gate firings ----
    summary_lines.append("")
    summary_lines.append("=== GATE FIRINGS (config-independent; base06 l2=1.0 here) ===")
    summary_lines.append(
        f"GATE FIRED on ({len(fired_list)}): "
        f"{', '.join(fired_list) if fired_list else '(none)'}")
    summary_lines.append(
        "(non-firing datasets are byte-identical across all configs -> delta 0)")

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
    summary_lines.append("=== PER-CANDIDATE DETAIL (firing datasets only; "
                         "non-firing deltas are exactly 0) ===")
    for cfg in CANDIDATES:
        summary_lines.append(f"--- {cfg} ---")
        for r in rows:
            if not r.get("fired"):
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
    # CLEAN IMPROVEMENT over 06 iff mean delta positive AND zero regressions on
    # BOTH splits. We require mean positive on BOTH splits (strict reading).
    summary_lines.append("")
    summary_lines.append("=== VERDICT (adoption vs base06) ===")
    summary_lines.append(
        "Criterion: clean improvement over 06 iff mean delta positive AND "
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
        f"OVERALL: {'a second knob cleanly improves 06' if any_clean else 'NO second knob cleanly improves 06; base06 (l2=1.0 only) remains best'}.")

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
