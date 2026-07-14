#!/usr/bin/env python
"""
bench_03 round59 — TRAINING-ROW DEDUPLICATION (base-08 on full train vs base-08
on drop_duplicates train) — ALL 16.
OFFLINE ONLY. No subprocess, no LLM, no Kaggle, no network. Calls sklearn
in-process only. Writes ONLY under this round59 directory; NEVER touches
submissions/.

GOAL (improvement-log angle "dedup")
------------------------------------
Every prior round in bench_03 fits the base-08 model on the FULL training frame.
Exact-duplicate training rows (identical across ALL columns) have never been
removed before fitting. Duplicate rows re-weight the loss toward whatever region
of feature space they occupy; dropping them is a distinct, untried, orthogonal
mechanism (it changes the training SAMPLE, not the model, hyperparameters, seed,
or aggregation space). This round tests that single lever.

Design (single mechanism = training-row deduplication):
  BASE arm  = base-08 HGB exactly (reference column), seed-0, fit on the FULL
              train frame. Byte-identical to shipped 08 -> reproduces round58's
              base column (base_pub/base_prv) with max|dev| = 0. This is the
              reproduction anchor that proves the harness is faithful.
  DEDUP arm = the IDENTICAL base-08 model with the IDENTICAL l2/msl values
              (computed from the FULL train, n=len(train), NOT recomputed after
              dedup), seed-0, fit on train.drop_duplicates(keep="first") — exact
              duplicate rows removed across all columns. Same features, same
              categorical mask (rebuilt on the deduped frame), same test set,
              same predict_proba scoring. The ONLY difference vs BASE is which
              training rows are present.

BASE / DEDUP recipe (== shipped 08):
  features = [c for c in train.columns if c not in ("row_id","target")]
  cat_mask = [frame[c].dtype == object for c in features]   # rebuilt per frame
  n=len(train)  (FULL train — gates use this for BOTH arms); ratio = len(feats)/n
  l2  = 1.0 if ratio >= 0.010 else 0.0
  msl = 70 if ratio>=0.030 else 50 if ratio>=0.015 else 20
  HistGradientBoostingClassifier(categorical_features=cat_mask, random_state=0,
      max_iter=300, early_stopping=True, l2_regularization=l2,
      min_samples_leaf=msl)
  pred = predict_proba(test)[:, class==1]

Single seed (seed 0) per arm — dedup is orthogonal to seed-averaging, so a clean
base-vs-dedup comparison needs no K-ensemble.

REPRODUCTION (MANDATORY — proves the harness is faithful, must be BIT-IDENTICAL):
  BASE column on ALL 16 must match round58's base column (round58 results.csv,
  base_pub/base_prv) with max|dev| = 0. If dev is non-zero, CLEAN RUN = NO.

Reported per-dataset (all 16, un-gated): n_before, n_after, rows_dropped,
base AUC (pub/prv), dedup AUC (pub/prv), delta = dedup - base. Aggregate: mean
delta over the 16, win/lose/tie (eps=1e-6), regressions list.
"""
import os

# keep the run polite / modest on CPU; HGB is deterministic w.r.t. random_state
# regardless of thread count, so this does not affect reproduction.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

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
OUT_DIR = os.path.join(BENCH_DIR, "round59_dedup")
STATS_CSV = os.path.join(BENCH_DIR, "dataset_stats.csv")
# Anchor against round58 (the immediately-preceding harness), whose base column
# encodes the exact base-08 config fit on the full train.
ROUND58_RESULTS = os.path.join(BENCH_DIR, "round58_margin_agg", "results.csv")

L2_GATE_THRESHOLD = 0.010   # shipped-08 ratio gate for l2 (FIXED)
GATED_L2 = 1.0
DEFAULT_MSL = 20
MSL_TIERS = [(0.030, 70), (0.015, 50)]
BASE_SEED = 0
N_DATASETS = 16
REPRO_TOL = 0.0             # BIT-IDENTICAL: base arm must match round58 exactly
WLT_EPS = 1e-6             # win/lose/tie threshold (per task spec)

BASE = "base"
DEDUP = "dedup"
ALL_CONFIGS = [BASE, DEDUP]


def load_stats(path=STATS_CSV):
    stats = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            stats[row["name"]] = {
                "n_train": int(row["n_train"]),
                "n_object_cols": int(row["n_object_cols"]),
            }
    return stats


def round58_anchors(path=ROUND58_RESULTS):
    """Read round58's base_pub/base_prv for ALL 16 datasets to anchor
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
                anchors[name] = None
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


def fit_hgb(train_frame, test, features, cat_mask, l2, msl_val, seed):
    """Fit ONE shipped-08 HGB on `train_frame` and return P(class==1) on test.
    validation_fraction / max_depth / interaction_cst / tol / monotonic_cst
    left UNSET (sklearn defaults, byte-identical to shipped 08). l2/msl are the
    base-08 gate values computed from the FULL train and passed in identically
    for both arms."""
    clf = HistGradientBoostingClassifier(
        categorical_features=cat_mask,
        random_state=seed,
        max_iter=300,
        early_stopping=True,
        l2_regularization=l2,
        min_samples_leaf=msl_val,
    )
    clf.fit(train_frame[features], train_frame["target"])
    proba = clf.predict_proba(test[features])
    classes = list(clf.classes_)
    pos_idx = classes.index(1) if 1 in classes else 1
    return proba[:, pos_idx]


def run_one(name, train_csv, test_csv, stats):
    """Returns (preds, meta). preds maps config_name -> {row_id -> score}.

    base  = seed-0, base-08 fit on the FULL train (byte-identical to shipped 08).
    dedup = seed-0, SAME base-08 (same l2/msl from full train) fit on
            train.drop_duplicates(keep="first").
    """
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)

    features = [c for c in train.columns if c not in ("row_id", "target")]

    # ---- gates computed from the FULL train, UNCHANGED from base-08 for BOTH
    # arms (dedup does NOT recompute l2/msl). ----
    n = len(train)
    ratio = len(features) / n
    l2 = GATED_L2 if ratio >= L2_GATE_THRESHOLD else 0.0
    msl_val = msl_for_ratio(ratio)

    row_ids = test["row_id"].tolist()
    preds = {}
    n_fits = 0

    # ---- BASE: seed-0, FULL train, base-08 (byte-identical to shipped 08). ----
    cat_mask_full = [train[c].dtype == object for c in features]
    base_proba = fit_hgb(train, test, features, cat_mask_full, l2, msl_val,
                         BASE_SEED)
    preds[BASE] = dict(zip(row_ids, base_proba.tolist()))
    n_fits += 1

    # ---- DEDUP: seed-0, SAME model + SAME l2/msl, fit on deduped train. ----
    # Drop rows that are exact duplicates across ALL columns (row_id included is
    # irrelevant to the model but row_id is unique per row, so dedup across all
    # columns EXCLUDING row_id is the meaningful operation for training rows).
    dedup_train = train.drop(columns=["row_id"]).drop_duplicates(keep="first")
    # rebuild categorical mask on the deduped frame (dtypes unchanged, but rebuilt
    # explicitly per spec); features list is identical (row_id already excluded).
    cat_mask_dd = [dedup_train[c].dtype == object for c in features]
    dedup_proba = fit_hgb(dedup_train, test, features, cat_mask_dd, l2, msl_val,
                          BASE_SEED)
    preds[DEDUP] = dict(zip(row_ids, dedup_proba.tolist()))
    n_fits += 1

    n_before = n
    n_after = len(dedup_train)

    st = stats[name]
    meta = {
        "n_train": st["n_train"],
        "n_object_cols": st["n_object_cols"],
        "l2": l2,
        "msl": msl_val,
        "n_features": len(features),
        "n_cat": sum(cat_mask_full),
        "gate_c": st["n_object_cols"] > 0,
        "n_before": n_before,
        "n_after": n_after,
        "rows_dropped": n_before - n_after,
        "n_fits": n_fits,
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
    anchors58 = round58_anchors()
    rows = []
    exceptions = []
    skipped = []
    single_class_skips = []
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
                "gate_c": meta["gate_c"],
                "l2": meta["l2"],
                "msl": meta["msl"],
                "n_before": meta["n_before"],
                "n_after": meta["n_after"],
                "rows_dropped": meta["rows_dropped"],
            })
            for cfg in ALL_CONFIGS:
                pub, prv = score_split(preds[cfg], sol)
                rec[f"{cfg}_pub"] = pub
                rec[f"{cfg}_prv"] = prv
                if math.isnan(pub) or math.isnan(prv):
                    single_class_skips.append((name, cfg))
            print(f"[OK] {name} n_tr={meta['n_train']} obj={meta['n_object_cols']} "
                  f"gateC={meta['gate_c']} feats={meta['n_features']} "
                  f"cat={meta['n_cat']} l2={meta['l2']} msl={meta['msl']} "
                  f"n_before={meta['n_before']} n_after={meta['n_after']} "
                  f"dropped={meta['rows_dropped']} fits={meta['n_fits']} | "
                  f"base pub={rec['base_pub']:.6f} prv={rec['base_prv']:.6f}  "
                  f"dedup pub={rec['dedup_pub']:.6f} prv={rec['dedup_prv']:.6f}")
        except Exception as e:
            exceptions.append((name, repr(e)))
            rec.update({"n_train": stats.get(name, {}).get("n_train", ""),
                        "n_object_cols": stats.get(name, {}).get("n_object_cols", ""),
                        "gate_c": stats.get(name, {}).get("n_object_cols", 0) > 0,
                        "l2": float("nan"), "msl": float("nan"),
                        "n_before": "", "n_after": "", "rows_dropped": ""})
            for cfg in ALL_CONFIGS:
                rec[f"{cfg}_pub"] = float("nan")
                rec[f"{cfg}_prv"] = float("nan")
            print(f"[ERROR] {name}: {e!r}")
        rows.append(rec)

    # ---- delta helpers (dedup vs base == shipped 08) ----
    def delta(rec, split):
        b = rec.get(f"{BASE}_{split}")
        c = rec.get(f"{DEDUP}_{split}")
        if b is None or c is None or math.isnan(b) or math.isnan(c):
            return float("nan")
        return c - b

    def mean_delta(split):
        vals = [delta(r, split) for r in rows]
        vals = [v for v in vals if not math.isnan(v)]
        return sum(vals) / len(vals) if vals else float("nan")

    def wlt(split, eps=WLT_EPS):
        w = l = t = 0
        for r in rows:
            dd = delta(r, split)
            if math.isnan(dd):
                continue
            if dd > eps:
                w += 1
            elif dd < -eps:
                l += 1
            else:
                t += 1
        return w, l, t

    def regressions(split, eps=WLT_EPS):
        return [(r["dataset"], delta(r, split)) for r in rows
                if not math.isnan(delta(r, split)) and delta(r, split) < -eps]

    def improvements(split, eps=WLT_EPS):
        return [(r["dataset"], delta(r, split)) for r in rows
                if not math.isnan(delta(r, split)) and delta(r, split) > eps]

    # ---- results.csv ----
    csv_path = os.path.join(OUT_DIR, "results.csv")
    fieldnames = ["dataset", "n_train", "n_object_cols", "gate_c", "l2", "msl",
                  "n_before", "n_after", "rows_dropped",
                  "base_pub", "base_prv", "dedup_pub", "dedup_prv",
                  "dedup_d_pub", "dedup_d_prv"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            out = {k: r.get(k, "") for k in
                   ["dataset", "n_train", "n_object_cols", "gate_c", "l2", "msl",
                    "n_before", "n_after", "rows_dropped",
                    "base_pub", "base_prv", "dedup_pub", "dedup_prv"]}
            out["dedup_d_pub"] = delta(r, "pub")
            out["dedup_d_prv"] = delta(r, "prv")
            w.writerow(out)

    # ---- REPRODUCTION: base on ALL 16 must be BIT-IDENTICAL to round58 base ----
    repro_available = anchors58 is not None
    by_name = {r["dataset"]: r for r in rows}
    repro = {}
    repro_ok = True
    max_dev_base = 0.0
    for i in range(1, N_DATASETS + 1):
        nm = f"train_{i:02d}"
        r = by_name.get(nm)
        mine = (r.get("base_pub"), r.get("base_prv")) if r else (None, None)
        ref = anchors58.get(nm) if anchors58 else None
        if ref is None or mine[0] is None or mine[1] is None \
                or (isinstance(mine[0], float) and math.isnan(mine[0])):
            okp = okv = False
            devp = devv = float("nan")
        else:
            devp = abs(mine[0] - ref[0])
            devv = abs(mine[1] - ref[1])
            okp = devp <= REPRO_TOL
            okv = devv <= REPRO_TOL
            max_dev_base = max(max_dev_base, devp, devv)
        repro[nm] = {"mine": mine, "ref": ref, "okp": okp, "okv": okv,
                     "devp": devp, "devv": devv}
        if not (okp and okv):
            repro_ok = False

    # ---- partition sanity (all 16 present) ----
    present = {r["dataset"] for r in rows if not (
        isinstance(r.get("base_pub"), float) and math.isnan(r.get("base_pub")))}
    all16_ok = (len(present) == N_DATASETS and not skipped)
    n_gate_c = sum(1 for r in rows if r.get("gate_c"))

    # ================= SUMMARY =================
    L = []
    L.append("=" * 78)
    L.append("bench_03 round59 — TRAINING-ROW DEDUPLICATION: base-08 full-train vs "
             "base-08 deduped-train (ALL 16)  [OFFLINE]")
    L.append("=" * 78)
    L.append("")
    L.append("SETUP:")
    L.append("  base   == shipped 08: HGB, early_stopping, cat_mask=object cols,")
    L.append("            seed-0, fit on the FULL train frame.")
    L.append("  dedup  == IDENTICAL base-08 model + IDENTICAL l2/msl (computed from")
    L.append("            the FULL train, NOT recomputed after dedup), seed-0, fit on")
    L.append("            train.drop_duplicates(keep='first') (exact-duplicate rows")
    L.append("            removed across all columns except the unique row_id).")
    L.append("  Single new mechanism = training-row deduplication (changes the SAMPLE")
    L.append("  only; model, hyperparameters, seed, and test set are unchanged).")
    L.append("")

    # ---- HEADLINE (un-gated, all 16) ----
    L.append("=== HEADLINE (dedup vs base == shipped 08, ALL 16, un-gated) ===")
    mp = mean_delta("pub")
    mv = mean_delta("prv")
    wp, lp, tp = wlt("pub")
    wv, lv, tv = wlt("prv")
    L.append(f"  mean ΔPublic ={mp:+.6f}  (W/L/T {wp}/{lp}/{tp})")
    L.append(f"  mean ΔPrivate={mv:+.6f}  (W/L/T {wv}/{lv}/{tv})")

    # ---- DEDUP FOOTPRINT ----
    L.append("")
    L.append("=== DEDUP FOOTPRINT (rows removed by drop_duplicates) ===")
    L.append(f"{'dataset':<10} {'n_before':>10} {'n_after':>10} {'dropped':>9} "
             f"{'drop_%':>8}")
    total_before = total_after = 0
    n_with_dupes = 0
    for r in rows:
        nb = r.get("n_before")
        na = r.get("n_after")
        if isinstance(nb, int) and isinstance(na, int):
            dropped = nb - na
            pct = 100.0 * dropped / nb if nb else 0.0
            total_before += nb
            total_after += na
            if dropped > 0:
                n_with_dupes += 1
            L.append(f"{r['dataset']:<10} {nb:>10} {na:>10} {dropped:>9} "
                     f"{pct:>7.3f}%")
        else:
            L.append(f"{r['dataset']:<10} {'nan':>10} {'nan':>10} {'nan':>9} "
                     f"{'nan':>8}")
    tot_dropped = total_before - total_after
    tot_pct = 100.0 * tot_dropped / total_before if total_before else 0.0
    L.append(f"  TOTAL: before={total_before} after={total_after} "
             f"dropped={tot_dropped} ({tot_pct:.3f}%)  "
             f"datasets_with_duplicates={n_with_dupes}/{len(rows)}")

    # ---- PER-DATASET DELTAS ----
    for split, tag in (("pub", "Public"), ("prv", "Private")):
        L.append("")
        L.append(f"=== PER-DATASET ΔAUC ({tag}) — base vs dedup ===")
        L.append(f"{'dataset':<10} {'gateC':>6} {'dropped':>8} {'base':>10} "
                 f"{'dedup':>10} {'delta':>11}")
        for r in rows:
            b = r.get(f"{BASE}_{split}")
            c = r.get(f"{DEDUP}_{split}")
            dd = delta(r, split)
            drp = r.get("rows_dropped")
            drs = f"{drp:>8}" if isinstance(drp, int) else f"{'nan':>8}"
            bstr = f"{b:>10.6f}" if isinstance(b, float) and not math.isnan(b) else f"{'nan':>10}"
            cstr = f"{c:>10.6f}" if isinstance(c, float) and not math.isnan(c) else f"{'nan':>10}"
            dstr = f"{dd:>+11.6f}" if not math.isnan(dd) else f"{'nan':>11}"
            L.append(f"{r['dataset']:<10} {str(bool(r.get('gate_c'))):>6} "
                     f"{drs} {bstr} {cstr} {dstr}")

    # ---- REGRESSIONS / IMPROVEMENTS (un-gated) ----
    L.append("")
    L.append("=== REGRESSIONS (ΔAUC < -1e-6) ===")
    any_reg = False
    for split, tag in (("pub", "Public"), ("prv", "Private")):
        for n_, d_ in regressions(split):
            L.append(f"  {tag:<7} {n_}: {d_:+.6f}")
            any_reg = True
    if not any_reg:
        L.append("  NONE on either split.")

    L.append("")
    L.append("=== IMPROVEMENTS (ΔAUC > +1e-6) ===")
    any_imp = False
    for split, tag in (("pub", "Public"), ("prv", "Private")):
        for n_, d_ in improvements(split):
            L.append(f"  {tag:<7} {n_}: {d_:+.6f}")
            any_imp = True
    if not any_imp:
        L.append("  NONE on either split.")

    # ---- REPRODUCTION: base vs round58 base (BIT-IDENTICAL) ----
    L.append("")
    L.append("=== REPRODUCTION CHECK (base on ALL 16 vs round58 base, tol=0) ===")
    if not repro_available:
        L.append("  round58 results.csv NOT found -> reproduction NOT anchored (FAIL).")
    else:
        for i in range(1, N_DATASETS + 1):
            nm = f"train_{i:02d}"
            rr = repro[nm]
            mp_, mv_ = rr["mine"]
            rp_, rv_ = rr["ref"] if rr["ref"] else (float("nan"), float("nan"))
            mp_s = f"{mp_:.6f}" if isinstance(mp_, float) and not math.isnan(mp_) else "nan"
            mv_s = f"{mv_:.6f}" if isinstance(mv_, float) and not math.isnan(mv_) else "nan"
            L.append(
                f"  {nm}: Public {mp_s} vs r58 {rp_:.6f} "
                f"(|d|={rr['devp']:.2e}, {'YES' if rr['okp'] else 'NO'}); "
                f"Private {mv_s} vs r58 {rv_:.6f} "
                f"(|d|={rr['devv']:.2e}, {'YES' if rr['okv'] else 'NO'})")
        L.append(f"  max |dev| over all 16x2 = {max_dev_base:.2e}")
        L.append(f"  REPRODUCTION (base==round58 base): "
                 f"{'PASS' if repro_ok else 'FAIL'}")

    # ---- ADOPTION / VERDICT (un-gated clean-win test) ----
    L.append("")
    L.append("=== ADOPTION ANALYSIS ===")
    L.append("ADOPT iff mean ΔPublic > 0 AND mean ΔPrivate > 0 with ZERO regression")
    L.append("  on EITHER split (over all 16). Any regression, or net-negative /")
    L.append("  negligible mean on either split => REJECT.")
    ADOPT_EPS = 1e-5
    regs_pub = regressions("pub")
    regs_prv = regressions("prv")
    zero_regs = (not regs_pub) and (not regs_prv)
    clean_gain = (mp > ADOPT_EPS and mv > ADOPT_EPS)
    is_adopt = zero_regs and clean_gain
    L.append("")
    L.append(f"  zero_regressions={'YES' if zero_regs else 'NO'} "
             f"(Public regs={len(regs_pub)}, Private regs={len(regs_prv)})")
    L.append(f"  mean ΔPublic  = {mp:+.6f}  (clean gain: "
             f"{'YES' if mp > ADOPT_EPS else 'NO'})")
    L.append(f"  mean ΔPrivate = {mv:+.6f}  (clean gain: "
             f"{'YES' if mv > ADOPT_EPS else 'NO'})")
    L.append("")
    L.append("=== VERDICT ===")
    if is_adopt:
        L.append("  ADOPT: training-row deduplication cleanly improves BOTH mean "
                 "splits with zero regression.")
    else:
        L.append("  REJECT (clean-win test): dedup does not improve BOTH mean splits "
                 "with zero regressions.")
    ship = "ADOPT" if is_adopt else "REJECT"
    L.append("")
    L.append(f"SHIP VERDICT: {ship}")

    # ---- CLEAN RUN marker ----
    clean_run = ((not exceptions) and all16_ok and repro_ok and repro_available
                 and (not skipped) and (not single_class_skips))
    L.append("")
    L.append(f"CLEAN RUN: {'YES' if clean_run else 'NO'} "
             f"({len(exceptions)} exceptions)  "
             f"[total_fits={total_fits}, datasets_scored={len(present)}/16, "
             f"skipped={len(skipped)}, single_class_skips={len(single_class_skips)}, "
             f"gate_c_datasets={n_gate_c}, "
             f"reproduction={'YES' if repro_ok else 'NO'} "
             f"(base_maxdev={max_dev_base:.2e})]")
    for name, msg in exceptions:
        L.append(f"  EXC {name}: {msg}")
    for name, cfg in single_class_skips:
        L.append(f"  SINGLE-CLASS {name}/{cfg}")

    summary = "\n".join(L)
    print("\n" + summary)
    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
        f.write(summary + "\n")
    print(f"\n[WROTE] {csv_path}")
    print(f"[WROTE] {os.path.join(OUT_DIR, 'summary.txt')}")
    print(f"FINAL_MARKER CLEAN_RUN={'YES' if clean_run else 'NO'} "
          f"SCORED={len(present)}/16 EXC={len(exceptions)} "
          f"REPRO_MAXDEV={max_dev_base:.2e} MEAN_DPUB={mp:+.6f} MEAN_DPRV={mv:+.6f}")
    print("HARNESS_DONE")


if __name__ == "__main__":
    main()
