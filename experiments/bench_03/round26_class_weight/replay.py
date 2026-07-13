#!/usr/bin/env python
"""
bench_03 round26 — class_weight sweep on the shipped 08 two-gate
ratio-tiered-msl recipe. OFFLINE ONLY. No subprocess, no LLM, no Kaggle. Calls
sklearn in-process.

Adapted from experiments/bench_03/round25_learning_rate/replay.py (dataset
loading, split/scoring, verdict, and summary machinery reused verbatim; the
swept knob is class_weight instead of learning_rate).

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
        min_samples_leaf=msl, class_weight=cw)
    where base uses HGB's DEFAULT class_weight (None) everywhere.
  - fit(train[features], train["target"])
  - pred = clf.predict_proba(test[features])[:, pos_class_1]
Score: ROC AUC on Public rows and Private rows separately, joined by row_id to
solution.csv, over all 16 datasets.

The QUESTION (round26 angle): Rounds 17-25 all tested the
regularization/early-stopping/learning-rate family (l2, min_samples_leaf,
max_leaf_nodes, validation_fraction, learning_rate) and NONE beat 08. Round 26
tests a genuinely orthogonal, previously untouched knob:
`HistGradientBoostingClassifier(class_weight=...)` (added in sklearn 1.6,
available in 1.9.0). AUC is rank-based, but reweighting the classes changes the
fitted probability ranking, so it CAN move AUC — especially on imbalanced
datasets where 'balanced' upweights the minority class.

Configs (all keep the l2 + msl gates + learning_rate identical to 08; they
differ ONLY in class_weight):
  base      : class_weight = None       everywhere                == shipped 08
  cw_global : class_weight = 'balanced' on EVERY dataset
  cw_gated  : class_weight = 'balanced' ONLY when the training set is imbalanced
              (minority-class fraction < 0.35); else None (== 08).

Minority-class fraction is computed from each dataset's TRAINING labels as
min(class_count) / total. The cw gate threshold is 0.35 (fires when the smaller
class is under 35% of training rows).

INVARIANT: whenever a config's resolved class_weight is None, that dataset is
byte-identical to base (delta exactly 0). This applies to every cw_gated dataset
whose minority fraction >= 0.35 (the required near-identity check) and, trivially,
to base itself. cw_global fires on every dataset so it has no non-firing datasets.

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
OUT_DIR = os.path.join(REPO, "experiments", "bench_03", "round26_class_weight")

L2_GATE_THRESHOLD = 0.010   # shipped 08 feature-to-row-ratio gate for l2 (FIXED)
GATED_L2 = 1.0              # l2 applied when the l2-gate fires (FIXED)
DEFAULT_MSL = 20            # sklearn HGB default, used when no msl tier is cleared.
CW_GATE_THRESHOLD = 0.35   # cw_gated fires 'balanced' when minority frac < this.

# 08 tiered min_samples_leaf, IDENTICAL across all configs (descending order).
MSL_TIERS = [(0.030, 70), (0.015, 50)]

# Each config: (name, cw_spec). cw_spec resolves per-dataset to a class_weight:
#   'none'     -> None always            (== base 08)
#   'balanced' -> 'balanced' always      (cw_global)
#   'gated'    -> 'balanced' if minority_frac < CW_GATE_THRESHOLD else None
CONFIGS = [
    ("base", "none"),
    ("cw_global", "balanced"),
    ("cw_gated", "gated"),
]
BASE = "base"
CANDIDATES = ["cw_global", "cw_gated"]

N_DATASETS = 16


def msl_for_ratio(ratio, tiers=MSL_TIERS):
    """First tier (threshold, msl) whose threshold the ratio clears wins;
    tiers must be given in descending-threshold order. Else DEFAULT_MSL."""
    for thr, val in tiers:
        if ratio >= thr:
            return val
    return DEFAULT_MSL


def cw_for_spec(spec, minority_frac):
    """Resolve a config's cw_spec to an actual sklearn class_weight value for a
    dataset with the given minority-class fraction. Returns (cw_value, fired)
    where fired means 'balanced' was actually applied (=> may differ from base).
    A resolved class_weight of None is IDENTICAL to base (fired=False)."""
    if spec == "none":
        return None, False
    if spec == "balanced":
        return "balanced", True
    if spec == "gated":
        if minority_frac < CW_GATE_THRESHOLD:
            return "balanced", True
        return None, False
    raise ValueError(f"unknown cw_spec: {spec!r}")


def auc_or_nan(y_true, y_score):
    """ROC AUC, or NaN if the subset has a single class (undefined)."""
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def run_one(train_csv, test_csv, cw_spec):
    """Reproduce the shipped 08 two-gate ratio-tiered-msl recipe for one
    dataset, applying the class_weight resolved from cw_spec (base == 08 uses
    None everywhere). The l2, msl and learning_rate settings are identical for
    all configs. Returns
    (pred_map, l2, l2_fired, msl_val, minority_frac, cw_val, cw_fired) where
    pred_map maps test row_id -> pos-class prob, cw_fired means 'balanced' was
    actually applied (spec != none AND, for gated, the minority gate cleared)."""
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

    # minority-class fraction from TRAINING labels (config-independent).
    counts = train["target"].value_counts()
    minority_frac = float(counts.min()) / float(counts.sum())

    cw_val, cw_fired = cw_for_spec(cw_spec, minority_frac)

    clf = HistGradientBoostingClassifier(
        categorical_features=cat_mask,
        random_state=0,
        max_iter=300,
        early_stopping=True,
        l2_regularization=l2,
        min_samples_leaf=msl_val,
        class_weight=cw_val,
    )
    clf.fit(train[features], train["target"])
    proba = clf.predict_proba(test[features])

    # positive-class column: match recipe's [:, 1] but be robust to class order.
    classes = list(clf.classes_)
    pos_idx = classes.index(1) if 1 in classes else 1
    pred = proba[:, pos_idx]

    return (dict(zip(test["row_id"].tolist(), pred.tolist())),
            l2, l2_fired, msl_val, minority_frac, cw_val, cw_fired)


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


def cw_str(cw_val):
    """Stable printable form of a class_weight value ('balanced' or 'None')."""
    return "balanced" if cw_val == "balanced" else "None"


def main():
    warnings.filterwarnings("ignore")
    os.makedirs(OUT_DIR, exist_ok=True)

    rows = []            # per-dataset results
    exceptions = []      # (dataset, config, message)
    skipped = []
    total_fits = 0

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
        for cfg_name, cw_spec in CONFIGS:
            try:
                (pred_map, l2, l2_fired, msl_val, minority_frac,
                 cw_val, cw_fired) = run_one(train_csv, test_csv, cw_spec)
                total_fits += 1
                pub, prv = score_split(pred_map, sol)
                rec[f"{cfg_name}_pub"] = pub
                rec[f"{cfg_name}_prv"] = prv
                rec[f"{cfg_name}_msl"] = msl_val
                rec[f"{cfg_name}_cw"] = cw_str(cw_val)
                rec[f"{cfg_name}_cw_fired"] = cw_fired
                # l2-gate, msl and minority_frac are config-independent; record once.
                rec["l2_fired"] = l2_fired
                rec["minority_frac"] = minority_frac
                print(f"[OK] {name} {cfg_name} (l2={l2}, l2_fired={l2_fired}, "
                      f"msl={msl_val}, min_frac={minority_frac:.4f}, "
                      f"cw={cw_str(cw_val)}, cw_fired={cw_fired}): "
                      f"pub={pub:.6f} prv={prv:.6f}")
            except Exception as e:  # capture but keep going — CLEAN-RUN diagnostic
                exceptions.append((name, cfg_name, repr(e)))
                rec[f"{cfg_name}_pub"] = float("nan")
                rec[f"{cfg_name}_prv"] = float("nan")
                rec[f"{cfg_name}_msl"] = float("nan")
                rec[f"{cfg_name}_cw"] = "ERR"
                rec[f"{cfg_name}_cw_fired"] = False
                print(f"[ERROR] {name} {cfg_name}: {e!r}")
        rows.append(rec)

    # ---- write results CSV ----
    csv_path = os.path.join(OUT_DIR, "results.csv")
    fieldnames = ["dataset", "l2_fired", "minority_frac"]
    for cfg, _ in CONFIGS:
        fieldnames += [f"{cfg}_pub", f"{cfg}_prv", f"{cfg}_msl",
                       f"{cfg}_cw", f"{cfg}_cw_fired"]
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

    # ---- INVARIANT check: any dataset with resolved class_weight=None (cw not
    #      fired) must be byte-identical to base (delta exactly 0). This is the
    #      required near-identity check for cw_gated's minority>=0.35 datasets. ----
    invariant_violations = []
    for cfg in CANDIDATES:
        for r in rows:
            if not r.get(f"{cfg}_cw_fired"):
                dp = delta(r, cfg, "pub")
                dv = delta(r, cfg, "prv")
                dp = 0.0 if math.isnan(dp) else dp
                dv = 0.0 if math.isnan(dv) else dv
                if dp != 0.0 or dv != 0.0:
                    invariant_violations.append(
                        (cfg, r["dataset"], dp, dv))

    summary_lines = []

    # ---- per-dataset tables (Public, Private) ----
    for split, tag in (("pub", "Public"), ("prv", "Private")):
        if split == "prv":
            summary_lines.append("")
        summary_lines.append(f"=== PER-DATASET ({tag}) ===")
        header = (f"{'dataset':<10} {'l2G':>4} {'msl':>4} "
                  f"{'minFrac':>8} {BASE:>9}")
        for cfg in CANDIDATES:
            header += f" {'cw':>9} {cfg:>10} {'d'+cfg:>11}"
        summary_lines.append(header)
        for r in rows:
            line = (f"{r['dataset']:<10} {str(bool(r.get('l2_fired'))):>4} "
                    f"{str(r.get(f'{BASE}_msl')):>4} "
                    f"{r.get('minority_frac', float('nan')):>8.4f} "
                    f"{r[f'{BASE}_{split}']:>9.4f}")
            for cfg in CANDIDATES:
                line += (f" {str(r.get(f'{cfg}_cw')):>9} "
                         f"{r[f'{cfg}_{split}']:>10.4f} {delta(r, cfg, split):>+11.5f}")
            summary_lines.append(line)

    # ---- gate firings ----
    summary_lines.append("")
    summary_lines.append("=== GATE FIRINGS ===")
    summary_lines.append(
        f"L2 GATE (ratio>=0.010, l2=1.0) fired on "
        f"({len(l2_fired_list)}): "
        f"{', '.join(l2_fired_list) if l2_fired_list else '(none)'}")
    for cfg, cw_spec in CONFIGS:
        fired = [r["dataset"] for r in rows if r.get(f"{cfg}_cw_fired")]
        if cfg == BASE:
            summary_lines.append(
                f"CW 'balanced' in '{cfg}' (0): (none)  "
                f"[base uses default class_weight=None everywhere]")
        else:
            summary_lines.append(
                f"CW 'balanced' in '{cfg}' ({len(fired)}): "
                f"{', '.join(fired) if fired else '(none)'}")
    summary_lines.append(
        f"(cw_gated fires 'balanced' iff training minority-class fraction < "
        f"{CW_GATE_THRESHOLD}; otherwise class_weight=None == base. cw_global "
        f"fires 'balanced' on every dataset.)")

    # ---- INVARIANT report ----
    summary_lines.append("")
    summary_lines.append(
        "=== INVARIANT (resolved class_weight=None => identical to base, delta 0) ===")
    if invariant_violations:
        summary_lines.append("VIOLATED! non-fired datasets differ from base:")
        for cfg, ds, dp, dv in invariant_violations:
            summary_lines.append(
                f"  {cfg}/{ds}: pub d={dp:+.6g} prv d={dv:+.6g}")
    else:
        summary_lines.append(
            "OK: every dataset whose resolved class_weight is None is "
            "byte-identical to base (delta exactly 0). This covers all cw_gated "
            "datasets with minority fraction >= 0.35 (required near-identity check).")

    # ---- which datasets actually differed ----
    summary_lines.append("")
    summary_lines.append("=== DATASETS THAT ACTUALLY DIFFER (candidate vs base) ===")
    for cfg in CANDIDATES:
        diff = differing_datasets(cfg)
        summary_lines.append(
            f"{cfg}: {', '.join(diff) if diff else '(none)'}")

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
                f"(cw {r.get(f'{BASE}_cw')}->{r.get(f'{cfg}_cw')}, "
                f"minFrac={r.get('minority_frac', float('nan')):.4f})  "
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
    summary_lines.append("")
    if any_clean:
        summary_lines.append(
            f"OVERALL: clean improvement over 08 found: {', '.join(clean_names)} "
            f"(orchestrator decides adoption).")
    else:
        summary_lines.append(
            "OVERALL: NO clean improvement over 08; base (shipped 08) remains "
            "best. class_weight='balanced' (global or minority-gated) did not "
            "cleanly beat the default class_weight=None.")

    # ---- clean-run line ----
    summary_lines.append("")
    clean_run = (not exceptions) and (not invariant_violations)
    summary_lines.append(
        f"CLEAN RUN={'YES' if clean_run else 'NO'} "
        f"(total_fits={total_fits}, exceptions={len(exceptions)}, "
        f"skipped={len(skipped)}, invariant_violations={len(invariant_violations)})")
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
