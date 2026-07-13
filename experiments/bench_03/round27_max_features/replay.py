#!/usr/bin/env python
"""
bench_03 round27 — max_features sweep on the shipped 08 two-gate
ratio-tiered-msl recipe. OFFLINE ONLY. No subprocess, no LLM, no Kaggle. Calls
sklearn in-process.

Adapted from experiments/bench_03/round26_class_weight/replay.py (dataset
loading, split/scoring, verdict, and summary machinery reused verbatim; the
swept knob is `max_features` instead of `class_weight`, and it is gated on the
overfit-prone L2 gate instead of a minority-fraction gate).

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
    where base uses HGB's DEFAULT max_features (1.0) everywhere.
  - fit(train[features], train["target"])
  - pred = clf.predict_proba(test[features])[:, pos_class_1]
Score: ROC AUC on Public rows and Private rows separately, joined by row_id to
solution.csv, over all 16 datasets.

The QUESTION (round27 angle): add ONE new knob on top of the FULL 08 config
(keep l2 gate, msl tiers, early_stopping EXACTLY as 08):
`HistGradientBoostingClassifier(max_features=<mf>)` — column subsampling per
split — applied ONLY to the overfit-prone datasets, i.e. where the SAME L2 gate
fires (ratio = n_feat/n >= 0.010 -> train_09/13/15/16). Column subsampling is a
classic variance-reducer for high feature-to-row datasets, so if anything helps
it should help exactly these four. The other 12 datasets get max_features's
default (1.0) and are byte-identical to 08.

Configs (all keep the l2 + msl gates + early_stopping identical to 08; they
differ ONLY in max_features on the L2-firing datasets):
  base    : max_features = 1.0 everywhere                    == shipped 08
  mf_0.7  : max_features = 0.7 on L2-firing datasets, else 1.0 (== 08)
  mf_0.5  : max_features = 0.5 on L2-firing datasets, else 1.0 (== 08)

IMPLEMENTATION INVARIANT: `max_features` is only ever *passed* to the constructor
when it resolves to a value < 1.0 (i.e. a candidate config on an L2-firing
dataset). For base, and for every non-firing dataset, the constructor call is
byte-identical to the shipped 08 call (no max_features kwarg). Therefore every
non-firing dataset MUST have delta exactly 0 vs base on both splits — this is
checked explicitly.

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
OUT_DIR = os.path.join(REPO, "experiments", "bench_03", "round27_max_features")

L2_GATE_THRESHOLD = 0.010   # shipped 08 feature-to-row-ratio gate for l2 (FIXED)
GATED_L2 = 1.0              # l2 applied when the l2-gate fires (FIXED)
DEFAULT_MSL = 20            # sklearn HGB default, used when no msl tier is cleared.
DEFAULT_MF = 1.0           # sklearn HGB default max_features (== 08 everywhere).

# 08 tiered min_samples_leaf, IDENTICAL across all configs (descending order).
MSL_TIERS = [(0.030, 70), (0.015, 50)]

# Each config: (name, mf_value). mf_value is the max_features applied ONLY on
# datasets where the L2 gate fires (ratio >= 0.010). Non-firing datasets always
# use the default 1.0 (identical to 08). base uses 1.0 everywhere (never fires).
#   base   -> 1.0 always            (== shipped 08)
#   mf_0.7 -> 0.7 on L2-firing, else 1.0
#   mf_0.5 -> 0.5 on L2-firing, else 1.0
CONFIGS = [
    ("base", 1.0),
    ("mf_0.7", 0.7),
    ("mf_0.5", 0.5),
]
BASE = "base"
CANDIDATES = ["mf_0.7", "mf_0.5"]

N_DATASETS = 16


def msl_for_ratio(ratio, tiers=MSL_TIERS):
    """First tier (threshold, msl) whose threshold the ratio clears wins;
    tiers must be given in descending-threshold order. Else DEFAULT_MSL."""
    for thr, val in tiers:
        if ratio >= thr:
            return val
    return DEFAULT_MSL


def mf_for_config(mf_value, l2_fired):
    """Resolve a config's max_features for a dataset. max_features is applied
    ONLY when the L2 gate fired AND the config's mf_value is < 1.0. Returns
    (mf_resolved, mf_fired) where mf_fired means a non-default (<1.0)
    max_features was actually applied (=> may differ from base). When mf_fired
    is False the constructor is byte-identical to 08 (no max_features kwarg)."""
    if mf_value < DEFAULT_MF and l2_fired:
        return mf_value, True
    return DEFAULT_MF, False


def auc_or_nan(y_true, y_score):
    """ROC AUC, or NaN if the subset has a single class (undefined)."""
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def run_one(train_csv, test_csv, mf_value):
    """Reproduce the shipped 08 two-gate ratio-tiered-msl recipe for one
    dataset, applying the max_features resolved from mf_value (base == 08 uses
    the default 1.0 everywhere). The l2 and msl settings are identical for all
    configs. Returns
    (pred_map, l2, l2_fired, msl_val, mf_resolved, mf_fired) where pred_map maps
    test row_id -> pos-class prob, mf_fired means a <1.0 max_features was
    actually applied (config mf_value < 1.0 AND the L2 gate fired)."""
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

    mf_resolved, mf_fired = mf_for_config(mf_value, l2_fired)

    # Build the constructor kwargs IDENTICAL to shipped 08; only add
    # max_features when it actually fires (<1.0 on an L2-firing dataset). This
    # guarantees base and every non-firing dataset are byte-identical to 08.
    kwargs = dict(
        categorical_features=cat_mask,
        random_state=0,
        max_iter=300,
        early_stopping=True,
        l2_regularization=l2,
        min_samples_leaf=msl_val,
    )
    if mf_fired:
        kwargs["max_features"] = mf_resolved

    clf = HistGradientBoostingClassifier(**kwargs)
    clf.fit(train[features], train["target"])
    proba = clf.predict_proba(test[features])

    # positive-class column: match recipe's [:, 1] but be robust to class order.
    classes = list(clf.classes_)
    pos_idx = classes.index(1) if 1 in classes else 1
    pred = proba[:, pos_idx]

    return (dict(zip(test["row_id"].tolist(), pred.tolist())),
            l2, l2_fired, msl_val, mf_resolved, mf_fired)


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


def mf_str(mf_resolved):
    """Stable printable form of a max_features value."""
    return f"{mf_resolved:g}"


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
        for cfg_name, mf_value in CONFIGS:
            try:
                (pred_map, l2, l2_fired, msl_val,
                 mf_resolved, mf_fired) = run_one(train_csv, test_csv, mf_value)
                total_fits += 1
                pub, prv = score_split(pred_map, sol)
                rec[f"{cfg_name}_pub"] = pub
                rec[f"{cfg_name}_prv"] = prv
                rec[f"{cfg_name}_msl"] = msl_val
                rec[f"{cfg_name}_mf"] = mf_str(mf_resolved)
                rec[f"{cfg_name}_mf_fired"] = mf_fired
                # l2-gate and msl are config-independent; record once.
                rec["l2_fired"] = l2_fired
                print(f"[OK] {name} {cfg_name} (l2={l2}, l2_fired={l2_fired}, "
                      f"msl={msl_val}, mf={mf_str(mf_resolved)}, "
                      f"mf_fired={mf_fired}): pub={pub:.6f} prv={prv:.6f}")
            except Exception as e:  # capture but keep going — CLEAN-RUN diagnostic
                exceptions.append((name, cfg_name, repr(e)))
                rec[f"{cfg_name}_pub"] = float("nan")
                rec[f"{cfg_name}_prv"] = float("nan")
                rec[f"{cfg_name}_msl"] = float("nan")
                rec[f"{cfg_name}_mf"] = "ERR"
                rec[f"{cfg_name}_mf_fired"] = False
                print(f"[ERROR] {name} {cfg_name}: {e!r}")
        rows.append(rec)

    # ---- write results CSV ----
    csv_path = os.path.join(OUT_DIR, "results.csv")
    fieldnames = ["dataset", "l2_fired"]
    for cfg, _ in CONFIGS:
        fieldnames += [f"{cfg}_pub", f"{cfg}_prv", f"{cfg}_msl",
                       f"{cfg}_mf", f"{cfg}_mf_fired"]
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

    # ---- INVARIANT check: any dataset where max_features did NOT fire (base,
    #      or any non-L2-firing dataset) must be byte-identical to base
    #      (delta exactly 0). This is the required base-reproduction check:
    #      the 12 non-firing datasets have delta 0.0 for every config. ----
    invariant_violations = []
    for cfg in CANDIDATES:
        for r in rows:
            if not r.get(f"{cfg}_mf_fired"):
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
        header = (f"{'dataset':<10} {'l2G':>4} {'msl':>4} {BASE:>9}")
        for cfg in CANDIDATES:
            header += f" {'mf':>5} {cfg:>10} {'d'+cfg:>11}"
        summary_lines.append(header)
        for r in rows:
            line = (f"{r['dataset']:<10} {str(bool(r.get('l2_fired'))):>4} "
                    f"{str(r.get(f'{BASE}_msl')):>4} "
                    f"{r[f'{BASE}_{split}']:>9.4f}")
            for cfg in CANDIDATES:
                line += (f" {str(r.get(f'{cfg}_mf')):>5} "
                         f"{r[f'{cfg}_{split}']:>10.4f} {delta(r, cfg, split):>+11.5f}")
            summary_lines.append(line)

    # ---- gate firings ----
    summary_lines.append("")
    summary_lines.append("=== GATE FIRINGS ===")
    summary_lines.append(
        f"L2 GATE (ratio>=0.010, l2=1.0) fired on "
        f"({len(l2_fired_list)}): "
        f"{', '.join(l2_fired_list) if l2_fired_list else '(none)'}")
    summary_lines.append(
        "max_features fires ONLY on the SAME L2-firing datasets (candidate "
        "configs only); base uses default max_features=1.0 everywhere.")
    for cfg, mf_value in CONFIGS:
        fired = [r["dataset"] for r in rows if r.get(f"{cfg}_mf_fired")]
        if cfg == BASE:
            summary_lines.append(
                f"max_features<1.0 in '{cfg}' (0): (none)  "
                f"[base uses default max_features=1.0 everywhere]")
        else:
            summary_lines.append(
                f"max_features={mf_value:g} in '{cfg}' ({len(fired)}): "
                f"{', '.join(fired) if fired else '(none)'}")

    # ---- INVARIANT report ----
    summary_lines.append("")
    summary_lines.append(
        "=== INVARIANT (non-firing datasets identical to base, delta 0) ===")
    if invariant_violations:
        summary_lines.append("VIOLATED! non-fired datasets differ from base:")
        for cfg, ds, dp, dv in invariant_violations:
            summary_lines.append(
                f"  {cfg}/{ds}: pub d={dp:+.6g} prv d={dv:+.6g}")
    else:
        summary_lines.append(
            "OK: every dataset where max_features did NOT fire is "
            "byte-identical to base (delta exactly 0). This covers all 12 "
            "non-L2-firing datasets for every candidate config (required "
            "base-reproduction check).")

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
                f"(mf {r.get(f'{BASE}_mf')}->{r.get(f'{cfg}_mf')})  "
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
            "best. max_features column-subsampling (0.7 or 0.5) on the "
            "L2-firing datasets did not cleanly beat the default 1.0.")

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
