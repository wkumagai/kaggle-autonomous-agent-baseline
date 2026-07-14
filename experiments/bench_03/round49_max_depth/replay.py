#!/usr/bin/env python
"""
bench_03 round49 — TREE-DEPTH CAP (max_depth) single-knob sweep (ALL 16).
OFFLINE ONLY. No subprocess, no LLM, no Kaggle, no network. Calls sklearn
in-process only. Writes ONLY under this round49 directory; never touches
submissions/.

GOAL (improvement-log angle "max-depth-cap")
--------------------------------------------
Base-08 fits HistGradientBoostingClassifier(...) with max_depth UNSET
(sklearn default None => tree depth is bounded only by max_leaf_nodes, not by
an explicit depth cap). max_depth is ORTHOGONAL to max_leaf_nodes (already
tested in a prior round): max_leaf_nodes caps the number of leaves, while
max_depth caps how deep any single tree may grow. Hypothesis: an explicit
shallow depth cap is extra regularization that MIGHT help noisy datasets, but
on an already-tuned model it usually just under-fits. This round measures the
clean offline delta of each candidate max_depth on ALL 16 datasets (NOT gated
to a subgroup).

Design (single-seed, random_state=0, NON-ensemble):
  BASE arm      = base-08 HGB exactly (reference column), all 16 datasets.
                  max_depth UNSET -> sklearn default None (no depth cap).
  CAND arms     = identical base-08 + the SINGLE knob `max_depth=k` in the
                  constructor, one arm per k in {3, 5, 8}, applied to ALL 16
                  datasets. Everything else stays byte-identical
                  (random_state=0, max_iter=300, early_stopping=True, l2 gate,
                  tiered msl gate, object-dtype categorical mask,
                  validation_fraction UNSET -> sklearn default 0.10,
                  n_iter_no_change UNSET -> sklearn default 10).

BASE recipe reproduced (== shipped 08), identical to round41/round48:
  features = [c for c in train.columns if c not in ("row_id","target")]
  cat_mask = [train[c].dtype == object for c in features]
  n=len(train); ratio = len(features)/n
  l2  = 1.0 if ratio >= 0.010 else 0.0
  msl = 70 if ratio>=0.030 else 50 if ratio>=0.015 else 20
  HistGradientBoostingClassifier(categorical_features=cat_mask, random_state=0,
      max_iter=300, early_stopping=True, l2_regularization=l2,
      min_samples_leaf=msl)   # max_depth NOT set -> default None
  pred = predict_proba(test)[:, class==1]

REPRODUCTION: the BASE column on ALL 16 datasets must match round48's base
column (full precision, read from round48 results.csv) to < 5e-6. This is the
identical base-08 recipe, so it must reproduce exactly (deterministic w.r.t.
random_state).

ADOPTION: ADOPT iff SOME candidate arm cleanly improves — mean ΔPublic > 0 AND
mean ΔPrivate > 0 with ZERO regression on EITHER split (no dataset with
ΔAUC < 0 on Public or Private). Any negative ΔAUC on any dataset/split, or a
net-negative/negligible mean on either split => REJECT that arm.
"""
import os

# keep the run polite / modest on CPU; HGB is deterministic w.r.t. random_state
# regardless of thread count, so this does not affect reproduction.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import csv
import math
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

REPO = "/Users/kumacmini/kaggle-autonomous-agent-baseline-auto"
DATA_DIR = os.path.join(REPO, "data")
BENCH_DIR = os.path.join(REPO, "experiments", "bench_03")
OUT_DIR = os.path.join(BENCH_DIR, "round49_max_depth")
STATS_CSV = os.path.join(BENCH_DIR, "dataset_stats.csv")
ROUND48_RESULTS = os.path.join(BENCH_DIR, "round48_niter_patience", "results.csv")

L2_GATE_THRESHOLD = 0.010   # shipped-08 ratio gate for l2 (FIXED)
GATED_L2 = 1.0
DEFAULT_MSL = 20
MSL_TIERS = [(0.030, 70), (0.015, 50)]
BASE_SEED = 0
N_DATASETS = 16
REPRO_TOL = 5e-6

BASE = "base"
# swept knob: max_depth. One arm per value, layered on identical base-08.
MAX_DEPTHS = [3, 5, 8]
CAND_NAMES = {k: f"cand_md{k}" for k in MAX_DEPTHS}   # e.g. 3 -> "cand_md3"
CAND_LIST = [CAND_NAMES[k] for k in MAX_DEPTHS]
ALL_CONFIGS = [BASE] + CAND_LIST


def load_stats(path=STATS_CSV):
    stats = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            stats[row["name"]] = {
                "n_train": int(row["n_train"]),
                "n_object_cols": int(row["n_object_cols"]),
            }
    return stats


def round48_base_anchors(path=ROUND48_RESULTS):
    """Read round48's base_pub/base_prv for ALL 16 datasets to anchor
    reproduction at full precision. Returns dict name -> (pub, prv) or None."""
    if not os.path.exists(path):
        return None
    anchors = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            name = row.get("dataset")
            try:
                anchors[name] = (float(row["base_pub"]), float(row["base_prv"]))
            except (KeyError, ValueError):
                pass
    return anchors


def msl_for_ratio(ratio, tiers=MSL_TIERS):
    for thr, val in tiers:
        if ratio >= thr:
            return val
    return DEFAULT_MSL


def auc_or_nan(y_true, y_score):
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def fit_hgb(train, test, features, cat_mask, l2, msl_val, seed, overrides=None):
    """Fit ONE shipped-08 HGB. validation_fraction left UNSET (sklearn default
    0.1, byte-identical to shipped 08). `overrides` adds/replaces individual
    kwargs for the candidate knob; when None this is the exact base-08 model."""
    kwargs = dict(
        categorical_features=cat_mask,
        random_state=seed,
        max_iter=300,
        early_stopping=True,
        l2_regularization=l2,
        min_samples_leaf=msl_val,
    )
    if overrides:
        kwargs.update(overrides)
    clf = HistGradientBoostingClassifier(**kwargs)
    clf.fit(train[features], train["target"])
    proba = clf.predict_proba(test[features])
    classes = list(clf.classes_)
    pos_idx = classes.index(1) if 1 in classes else 1
    return proba[:, pos_idx]


def run_one(name, train_csv, test_csv, stats):
    """Returns (preds, meta). preds maps config_name -> {row_id -> prob}.

    base   = seed-0, max_depth UNSET (== shipped 08).
    cand_k = identical + max_depth=k, applied to ALL 16 datasets, k in {3,5,8}.
    """
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)

    features = [c for c in train.columns if c not in ("row_id", "target")]
    cat_mask = [train[c].dtype == object for c in features]

    n = len(train)
    ratio = len(features) / n
    l2 = GATED_L2 if ratio >= L2_GATE_THRESHOLD else 0.0
    msl_val = msl_for_ratio(ratio)

    st = stats[name]
    row_ids = test["row_id"].tolist()

    # base = seed-0, max_depth UNSET (byte-identical to shipped 08).
    base_vec = fit_hgb(train, test, features, cat_mask, l2, msl_val, BASE_SEED)

    preds = {BASE: dict(zip(row_ids, base_vec.tolist()))}
    # cand arms: same base-08 + the SINGLE knob max_depth=k (ALL 16, no gating).
    for k in MAX_DEPTHS:
        cand_vec = fit_hgb(train, test, features, cat_mask, l2, msl_val,
                           BASE_SEED, overrides={"max_depth": k})
        preds[CAND_NAMES[k]] = dict(zip(row_ids, cand_vec.tolist()))

    meta = {
        "n_train": st["n_train"],
        "n_object_cols": st["n_object_cols"],
        "l2": l2,
        "msl": msl_val,
        "n_fits": 1 + len(MAX_DEPTHS),
    }
    return preds, meta


def score_split(pred_map, sol):
    sol = sol.copy()
    sol["pred"] = sol["row_id"].map(pred_map)
    if sol["pred"].isna().any():
        raise ValueError(f"{int(sol['pred'].isna().sum())} row_ids unmatched")
    pub = sol[sol["Usage"] == "Public"]
    prv = sol[sol["Usage"] == "Private"]
    return (auc_or_nan(pub["target"], pub["pred"]),
            auc_or_nan(prv["target"], prv["pred"]))


def main():
    warnings.filterwarnings("ignore")
    os.makedirs(OUT_DIR, exist_ok=True)

    stats = load_stats()
    anchors48 = round48_base_anchors()
    rows = []
    exceptions = []
    skipped = []
    total_fits = 0

    for i in range(1, N_DATASETS + 1):
        name = f"train_{i:02d}"
        d = os.path.join(DATA_DIR, name)
        train_csv = os.path.join(d, "train.csv")
        test_csv = os.path.join(d, "test.csv")
        sol_csv = os.path.join(d, "solution.csv")
        if not (os.path.exists(train_csv) and os.path.exists(test_csv)
                and os.path.exists(sol_csv)):
            print(f"[SKIP] {name}: missing files")
            skipped.append(name)
            continue

        sol = pd.read_csv(sol_csv)
        rec = {"dataset": name}
        try:
            preds, meta = run_one(name, train_csv, test_csv, stats)
            total_fits += meta["n_fits"]
            rec.update({
                "n_train": meta["n_train"],
                "n_object_cols": meta["n_object_cols"],
                "l2": meta["l2"],
                "msl": meta["msl"],
            })
            for cfg in ALL_CONFIGS:
                pub, prv = score_split(preds[cfg], sol)
                rec[f"{cfg}_pub"] = pub
                rec[f"{cfg}_prv"] = prv
            cand_str = "  ".join(
                f"md{k} pub={rec[CAND_NAMES[k]+'_pub']:.6f} "
                f"prv={rec[CAND_NAMES[k]+'_prv']:.6f}" for k in MAX_DEPTHS)
            print(f"[OK] {name} n_tr={meta['n_train']} obj={meta['n_object_cols']} "
                  f"l2={meta['l2']} msl={meta['msl']} fits={meta['n_fits']} "
                  f"base pub={rec['base_pub']:.6f} prv={rec['base_prv']:.6f}  "
                  f"{cand_str}")
        except Exception as e:
            exceptions.append((name, repr(e)))
            rec.update({"n_train": stats.get(name, {}).get("n_train", ""),
                        "n_object_cols": stats.get(name, {}).get("n_object_cols", ""),
                        "l2": float("nan"), "msl": float("nan")})
            for cfg in ALL_CONFIGS:
                rec[f"{cfg}_pub"] = float("nan")
                rec[f"{cfg}_prv"] = float("nan")
            print(f"[ERROR] {name}: {e!r}")
        rows.append(rec)

    # ---- delta helpers (cand vs base == shipped 08) ----
    def delta(rec, cand, split):
        b = rec.get(f"{BASE}_{split}")
        c = rec.get(f"{cand}_{split}")
        if b is None or c is None or math.isnan(b) or math.isnan(c):
            return float("nan")
        return c - b

    def mean_delta(cand, split):
        vals = [delta(r, cand, split) for r in rows]
        vals = [v for v in vals if not math.isnan(v)]
        return sum(vals) / len(vals) if vals else float("nan")

    def wlt(cand, split, eps=1e-6):
        w = l = t = 0
        for r in rows:
            dd = delta(r, cand, split)
            if math.isnan(dd):
                continue
            if dd > eps:
                w += 1
            elif dd < -eps:
                l += 1
            else:
                t += 1
        return w, l, t

    def regressions(cand, split, eps=1e-6):
        return [(r["dataset"], delta(r, cand, split)) for r in rows
                if not math.isnan(delta(r, cand, split))
                and delta(r, cand, split) < -eps]

    # ---- results.csv ----
    csv_path = os.path.join(OUT_DIR, "results.csv")
    fieldnames = ["dataset", "n_train", "n_object_cols", "l2", "msl",
                  "base_pub", "base_prv"]
    for k in MAX_DEPTHS:
        cn = CAND_NAMES[k]
        fieldnames += [f"{cn}_pub", f"{cn}_prv", f"{cn}_d_pub", f"{cn}_d_prv"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            out = {k: r.get(k, "") for k in
                   ["dataset", "n_train", "n_object_cols", "l2", "msl",
                    "base_pub", "base_prv"]}
            for k in MAX_DEPTHS:
                cn = CAND_NAMES[k]
                out[f"{cn}_pub"] = r.get(f"{cn}_pub", "")
                out[f"{cn}_prv"] = r.get(f"{cn}_prv", "")
                out[f"{cn}_d_pub"] = delta(r, cn, "pub")
                out[f"{cn}_d_prv"] = delta(r, cn, "prv")
            w.writerow(out)

    # ---- REPRODUCTION: base on ALL 16 matches round48 (tol<5e-6) ----
    repro = {}
    repro_ok = True
    repro_available = anchors48 is not None
    by_name = {r["dataset"]: r for r in rows}
    max_abs_dev = 0.0
    for i in range(1, N_DATASETS + 1):
        nm = f"train_{i:02d}"
        r = by_name.get(nm)
        mine = (r.get("base_pub"), r.get("base_prv")) if r else (None, None)
        ref = anchors48.get(nm) if anchors48 else None
        if ref is None or mine[0] is None or mine[1] is None \
                or (isinstance(mine[0], float) and math.isnan(mine[0])):
            okp = okv = False
            devp = devv = float("nan")
        else:
            devp = abs(mine[0] - ref[0])
            devv = abs(mine[1] - ref[1])
            okp = devp < REPRO_TOL
            okv = devv < REPRO_TOL
            max_abs_dev = max(max_abs_dev, devp, devv)
        repro[nm] = {"mine": mine, "ref": ref, "okp": okp, "okv": okv,
                     "devp": devp, "devv": devv}
        if not (okp and okv):
            repro_ok = False

    # ---- partition sanity (all 16 present) ----
    present = {r["dataset"] for r in rows if not (
        isinstance(r.get("base_pub"), float) and math.isnan(r.get("base_pub")))}
    all16_ok = (len(present) == N_DATASETS and not skipped)

    # ================= SUMMARY =================
    L = []
    L.append("=" * 78)
    L.append("bench_03 round49 — TREE-DEPTH CAP max_depth in {3,5,8} single-knob "
             "sweep (ALL 16)  [OFFLINE]")
    L.append("=" * 78)
    L.append("")
    L.append("SETUP:")
    L.append("  base == shipped 08 (git HEAD:submissions/08...): HGB, early_stopping,")
    L.append("    max_depth UNSET -> sklearn default None (no explicit depth cap;")
    L.append("    tree size bounded only by max_leaf_nodes default 31).")
    L.append("    base column = seed-0 for all 16 datasets.")
    L.append("  cand_md{k} == base + the SINGLE kwarg max_depth=k in the HGB")
    L.append("    constructor, one arm per k in {3,5,8}, applied to ALL 16 datasets")
    L.append("    (no subgroup gating). Everything else byte-identical")
    L.append("    (random_state=0, max_iter=300, early_stopping=True, l2 gate,")
    L.append("    tiered msl gate, categorical mask).")

    # ---- SWEEP HEADLINE (one line per candidate) ----
    L.append("")
    L.append("=== HEADLINE (each cand max_depth=k vs base == shipped 08, all 16) ===")
    headline = {}
    for k in MAX_DEPTHS:
        cn = CAND_NAMES[k]
        mp = mean_delta(cn, "pub")
        mv = mean_delta(cn, "prv")
        wp, lp, tp = wlt(cn, "pub")
        wv, lv, tv = wlt(cn, "prv")
        headline[k] = (mp, mv, (wp, lp, tp), (wv, lv, tv))
        L.append(f"  max_depth={k}: mean ΔPublic={mp:+.6f} (W/L/T {wp}/{lp}/{tp})  "
                 f"mean ΔPrivate={mv:+.6f} (W/L/T {wv}/{lv}/{tv})")

    # ---- REPRODUCTION ----
    L.append("")
    L.append("=== REPRODUCTION CHECK (base on ALL 16 vs round48, tol<5e-6) ===")
    if not repro_available:
        L.append("  round48 results.csv NOT found -> reproduction NOT anchored (FAIL).")
    else:
        for i in range(1, N_DATASETS + 1):
            nm = f"train_{i:02d}"
            rr = repro[nm]
            mp_, mv_ = rr["mine"]
            rp_, rv_ = rr["ref"] if rr["ref"] else (float("nan"), float("nan"))
            L.append(
                f"  {nm}: Public {mp_:.6f} vs r48 {rp_:.6f} "
                f"(|d|={rr['devp']:.2e}, {'YES' if rr['okp'] else 'NO'}); "
                f"Private {mv_:.6f} vs r48 {rv_:.6f} "
                f"(|d|={rr['devv']:.2e}, {'YES' if rr['okv'] else 'NO'})")
        L.append(f"  max |dev| over all 16x2 = {max_abs_dev:.2e}")
        L.append(f"  REPRODUCTION: {'PASS' if repro_ok else 'FAIL'}")

    # ---- PER-DATASET DELTAS (per candidate) ----
    for k in MAX_DEPTHS:
        cn = CAND_NAMES[k]
        for split, tag in (("pub", "Public"), ("prv", "Private")):
            L.append("")
            L.append(f"=== PER-DATASET ΔAUC ({tag}) — base vs max_depth={k} ===")
            L.append(f"{'dataset':<10} {'base':>10} {'cand':>10} {'delta':>11}")
            for r in rows:
                b = r.get(f"{BASE}_{split}")
                c = r.get(f"{cn}_{split}")
                dd = delta(r, cn, split)
                bstr = f"{b:>10.6f}" if isinstance(b, float) and not math.isnan(b) else f"{'nan':>10}"
                cstr = f"{c:>10.6f}" if isinstance(c, float) and not math.isnan(c) else f"{'nan':>10}"
                dstr = f"{dd:>+11.6f}" if not math.isnan(dd) else f"{'nan':>11}"
                L.append(f"{r['dataset']:<10} {bstr} {cstr} {dstr}")

    # ---- REGRESSIONS (per candidate) ----
    L.append("")
    L.append("=== REGRESSIONS (ΔAUC < -1e-6) ===")
    any_reg_printed = False
    for k in MAX_DEPTHS:
        cn = CAND_NAMES[k]
        regs_pub = regressions(cn, "pub")
        regs_prv = regressions(cn, "prv")
        if not regs_pub and not regs_prv:
            L.append(f"  max_depth={k}: NONE on either split.")
        else:
            any_reg_printed = True
            for n_, d_ in regs_pub:
                L.append(f"  max_depth={k} Public  {n_}: {d_:+.6f}")
            for n_, d_ in regs_prv:
                L.append(f"  max_depth={k} Private {n_}: {d_:+.6f}")
    _ = any_reg_printed

    # ---- ADOPTION / VERDICT (per candidate) ----
    L.append("")
    L.append("=== ADOPTION ANALYSIS ===")
    L.append("ADOPT iff SOME arm has mean ΔPublic > 0 AND mean ΔPrivate > 0 with")
    L.append("  ZERO regression on EITHER split (no dataset ΔAUC < 0 on Public or")
    L.append("  Private). Any regression, or net-negative/negligible mean on either")
    L.append("  split => REJECT that arm.")
    ADOPT_EPS = 1e-5   # negligible-mean guard
    adopt_arms = []
    for k in MAX_DEPTHS:
        cn = CAND_NAMES[k]
        mp, mv, _, _ = headline[k]
        regs_pub = regressions(cn, "pub")
        regs_prv = regressions(cn, "prv")
        zero_regs = (not regs_pub) and (not regs_prv)
        clean_gain = (mp > ADOPT_EPS and mv > ADOPT_EPS)
        is_adopt = zero_regs and clean_gain
        if is_adopt:
            adopt_arms.append(k)
        L.append("")
        L.append(f"  [max_depth={k}] zero_regressions="
                 f"{'YES' if zero_regs else 'NO'} "
                 f"(Public regs={len(regs_pub)}, Private regs={len(regs_prv)})")
        L.append(f"  [max_depth={k}] mean ΔPublic  = {mp:+.6f}  (clean gain: "
                 f"{'YES' if mp > ADOPT_EPS else 'NO'})")
        L.append(f"  [max_depth={k}] mean ΔPrivate = {mv:+.6f}  (clean gain: "
                 f"{'YES' if mv > ADOPT_EPS else 'NO'})")
        L.append(f"  [max_depth={k}] -> "
                 f"{'ADOPT-CANDIDATE' if is_adopt else 'REJECT'}")

    L.append("")
    L.append("=== VERDICT ===")
    if adopt_arms:
        arms = ", ".join(f"max_depth={k}" for k in adopt_arms)
        L.append(f"  ADOPT-CANDIDATE: {arms} cleanly improve BOTH splits with zero "
                 f"regression.")
    else:
        L.append("  REJECT: no max_depth in {3,5,8} beats base-08 (max_depth=None) "
                 "on BOTH splits with zero regressions. Depth-capping is extra "
                 "regularization on an already-tuned model -> base-08 stays at "
                 "least as good overall.")

    ship = "ADOPT-CANDIDATE" if adopt_arms else "REJECT"
    L.append("")
    L.append(f"SHIP VERDICT: {ship}")

    # ---- CLEAN RUN marker ----
    clean_run = ((not exceptions) and all16_ok and repro_ok and repro_available
                 and (not skipped))
    L.append("")
    L.append(f"CLEAN RUN: {'YES' if clean_run else 'NO'} "
             f"({len(exceptions)} exceptions)  "
             f"[total_fits={total_fits}, datasets_scored={len(present)}/16, "
             f"skipped={len(skipped)}, reproduction={'YES' if repro_ok else 'NO'}]")
    for name, msg in exceptions:
        L.append(f"  EXC {name}: {msg}")

    summary = "\n".join(L)
    print("\n" + summary)
    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
        f.write(summary + "\n")
    print(f"\n[WROTE] {csv_path}")
    print(f"[WROTE] {os.path.join(OUT_DIR, 'summary.txt')}")
    print("HARNESS_DONE")


if __name__ == "__main__":
    main()
