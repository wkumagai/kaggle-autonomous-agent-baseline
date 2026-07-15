#!/usr/bin/env python
"""Actual-fit benchmark of candidate A_orgate (seed-avg K10, OR-gate) vs base-08.

WHY THIS FILE EXISTS
--------------------
The candidate recipe lives as PROSE in ./system.md (a minimal edit of shipped-08:
StringDtype-robust text detection + the OR-gate + K=10 seed-averaging). The
top-level experiments/bench_03/benchmark.py is the 03_cv_ensemble multi-stage
harness (it extracts a single fenced python block and replays go.py safety/xgb/
cat/lgb/blend stages) and therefore CANNOT score an 08-style single-HGB prose
candidate. The canonical fit harness for the 08 single-HGB recipe is the
round*/replay.py family. This script reuses round70_AB_gateC/replay.py's exact
mechanism verbatim:

  * read_frame  : pandas>=3.0 StringDtype -> numpy object shim (so text columns
                  are detected under the pandas-3.0.3 gbm_venv Kaggle-hazard env).
  * fit_hgb     : one shipped-08 HGB; only random_state varies across seeds.
  * seed cache  : K=10 fits (seeds 0..9); base = seed 0, A = prob-mean of all K.
  * score_split : AUC on the local solution.csv Public / Private usage split.

The ONLY thing this harness adds over round70's A arm is the *gate*: round70's A
fired on gate-C (n_object_cols > 0); here the candidate fires on the OR-gate
(n_object_cols > 0 OR n_train >= 5000). On firing datasets final = seed-avg;
on non-firing datasets final = base seed-0 (bit-identical to base-08).

Run it under the pandas-3.x env to replicate the Kaggle StringDtype hazard:
  experiments/bench_03/gbm_venv/bin/python \
      experiments/bench_03/candidates/A_orgate/replay_orgate.py --workers 2

OFFLINE ONLY. Writes ONLY under this A_orgate directory; NEVER touches submissions/.
"""
import os

# Keep the run polite on CPU; HGB is deterministic w.r.t. random_state regardless
# of thread count, so throttling threads does not change any result.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import argparse
import csv
import json
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
DATA_DIR = os.path.join(REPO, "data")
BENCH_DIR = os.path.join(REPO, "experiments", "bench_03")
STATS_CSV = os.path.join(BENCH_DIR, "dataset_stats.csv")
OUT_CSV = os.path.join(HERE, "results.csv")
OUT_TXT = os.path.join(HERE, "summary.txt")

# shipped-08 recipe constants (FIXED, identical to round70).
L2_GATE_THRESHOLD = 0.010
GATED_L2 = 1.0
DEFAULT_MSL = 20
MSL_TIERS = [(0.030, 70), (0.015, 50)]
BASE_SEED = 0
K = 10                       # seed-avg ensemble size (Candidate A)
N_DATASETS = 16
# OR-gate: seed-avg fires when the dataset has any categorical column OR is large.
# THRESH-TRAIN16 variant: lowered from 5000 -> 1000 so train_16 (n=1809,
# n_object_cols=0) ALSO fires. All 15 previously-firing datasets keep firing
# unchanged (their gate was already True), so their deltas are identical; only
# train_16 flips from non-firing to firing. This is the SINGLE gate change.
GATE_N = 1000
# Anchor tolerance for base-08 seed-0 vs dataset_stats baseline_hgb_auc_private.
BASE_ANCHOR_TOL = 5e-4


def read_frame(path):
    """Load a CSV and normalise pandas>=3.0 StringDtype columns back to numpy
    `object` so the shipped-08 `dtype == object` categorical detection works.
    Verbatim from round70_AB_gateC/replay.py."""
    df = pd.read_csv(path)
    for c in df.columns:
        if isinstance(df[c].dtype, pd.StringDtype):
            df[c] = df[c].astype(object)
    return df


def load_stats(path=STATS_CSV):
    stats = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            stats[row["name"]] = {
                "n_train": int(row["n_train"]),
                "n_object_cols": int(row["n_object_cols"]),
                "baseline_hgb_auc_private": float(row["baseline_hgb_auc_private"]),
            }
    return stats


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
    """Fit ONE shipped-08 HGB and return P(class==1) on test. Only random_state
    varies across the seed ensemble. Verbatim from round70."""
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


def score_split(pred_vec, row_ids, sol):
    sol = sol.copy()
    pred_map = dict(zip(row_ids, pred_vec.tolist()))
    sol["pred"] = sol["row_id"].map(pred_map)
    if sol["pred"].isna().any():
        raise ValueError(f"{int(sol['pred'].isna().sum())} row_ids unmatched")
    pub = sol[sol["Usage"] == "Public"]
    prv = sol[sol["Usage"] == "Private"]
    return (auc_or_nan(pub["target"], pub["pred"]),
            auc_or_nan(prv["target"], prv["pred"]))


def run_one(name):
    """Fit base-08 (seed-0) and candidate A_orgate on one dataset; score both.

    base    = seed-0 shipped-08 HGB.
    cand    = seed-avg (prob-mean over seeds 0..9) IF the OR-gate fires, else base.
    OR-gate = (n_object_cols > 0) OR (n_train >= GATE_N).
    """
    d = os.path.join(DATA_DIR, name)
    train = read_frame(os.path.join(d, "train.csv"))
    test = read_frame(os.path.join(d, "test.csv"))
    sol = pd.read_csv(os.path.join(d, "solution.csv"))

    features = [c for c in train.columns if c not in ("row_id", "target")]
    # Robust text detection: read_frame already normalised StringDtype -> object,
    # so this reproduces the shipped-08 mask under pandas 3.x. n_object_cols is
    # counted from the SAME robust detection the candidate uses for its gate.
    cat_mask = [train[c].dtype == object for c in features]
    n_object_cols = int(sum(cat_mask))

    n = len(train)
    ratio = len(features) / n
    l2 = GATED_L2 if ratio >= L2_GATE_THRESHOLD else 0.0
    msl_val = msl_for_ratio(ratio)
    gate = (n_object_cols > 0) or (n >= GATE_N)

    row_ids = test["row_id"].tolist()

    # Always fit seed-0 (that IS base-08). If the gate fires, fit seeds 1..9 too
    # and prob-mean all K; else the candidate is exactly base-08.
    base_vec = fit_hgb(train, test, features, cat_mask, l2, msl_val, BASE_SEED)
    n_fits = 1
    if gate:
        seed_vecs = [base_vec]
        for k in range(1, K):
            seed_vecs.append(fit_hgb(train, test, features, cat_mask, l2, msl_val, k))
            n_fits += 1
        cand_vec = np.mean(seed_vecs, axis=0)
    else:
        cand_vec = base_vec

    base_pub, base_prv = score_split(base_vec, row_ids, sol)
    cand_pub, cand_prv = score_split(cand_vec, row_ids, sol)

    # Anchor: on a non-firing dataset the candidate must be bit-identical to base.
    identical_to_base = bool(np.array_equal(cand_vec, base_vec))

    return {
        "dataset": name,
        "n_train": n,
        "n_object_cols": n_object_cols,
        "n_features": len(features),
        "ratio": ratio,
        "l2": l2,
        "msl": msl_val,
        "gate_fired": gate,
        "n_fits": n_fits,
        "base_public": base_pub,
        "base_private": base_prv,
        "cand_public": cand_pub,
        "cand_private": cand_prv,
        "delta_private": cand_prv - base_prv,
        "delta_public": cand_pub - base_pub,
        "cand_equals_base": identical_to_base,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--datasets", default=None, help="comma list like 01,16")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    stats = load_stats()
    if args.datasets:
        names = [f"train_{d.strip()}" for d in args.datasets.split(",")]
    else:
        names = [f"train_{i:02d}" for i in range(1, N_DATASETS + 1)]

    print(f"pandas {pd.__version__} | datasets={len(names)} | workers={args.workers}")
    print(f"OR-gate = (n_object_cols > 0) OR (n_train >= {GATE_N})")

    rows = {}
    exceptions = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, nm): nm for nm in names}
        for fut in as_completed(futs):
            nm = futs[fut]
            try:
                rows[nm] = fut.result()
                r = rows[nm]
                print(f"[{time.time()-t0:5.0f}s] OK {nm} "
                      f"obj={r['n_object_cols']} n={r['n_train']} "
                      f"gate={r['gate_fired']} fits={r['n_fits']} "
                      f"base_prv={r['base_private']:.4f} cand_prv={r['cand_private']:.4f} "
                      f"dprv={r['delta_private']:+.4f}", flush=True)
            except Exception as e:  # noqa: BLE001
                exceptions.append((nm, repr(e)))
                print(f"[{time.time()-t0:5.0f}s] EXC {nm}: {e!r}", flush=True)

    ordered = [rows[nm] for nm in names if nm in rows]
    clean_run = (len(ordered) == len(names)) and (len(exceptions) == 0)

    # ---- base-08 anchor vs dataset_stats baseline_hgb_auc_private ----
    anchor_lines = []
    anchor_ok = True
    for r in ordered:
        exp = stats.get(r["dataset"], {}).get("baseline_hgb_auc_private")
        if exp is None:
            continue
        dev = abs(r["base_private"] - exp)
        ok = dev < BASE_ANCHOR_TOL
        anchor_ok = anchor_ok and ok
        anchor_lines.append(
            f"  {r['dataset']:10s} base_prv={r['base_private']:.4f} "
            f"stats={exp:.4f} |Δ|={dev:.5f} {'OK' if ok else 'DRIFT'}")

    # ---- write results.csv ----
    fields = ["dataset", "n_train", "n_object_cols", "n_features", "ratio",
              "l2", "msl", "gate_fired", "n_fits",
              "base_public", "base_private", "cand_public", "cand_private",
              "delta_public", "delta_private", "cand_equals_base"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in ordered:
            w.writerow({k: r[k] for k in fields})

    # ---- aggregates ----
    mean_base = float(np.mean([r["base_private"] for r in ordered]))
    mean_cand = float(np.mean([r["cand_private"] for r in ordered]))
    mean_delta = mean_cand - mean_base
    worst_reg = min(r["delta_private"] for r in ordered)
    worst_ds = min(ordered, key=lambda r: r["delta_private"])["dataset"]
    fired = [r for r in ordered if r["gate_fired"]]
    non_fired = [r for r in ordered if not r["gate_fired"]]
    # Non-firing datasets MUST be bit-identical to base (delta exactly 0).
    nonfire_ok = all(r["cand_equals_base"] and r["delta_private"] == 0.0
                     for r in non_fired)

    L = []
    L.append("candidate A_orgate -- seed-avg K10, OR-gate -- ACTUAL FIT bench")
    L.append("=" * 66)
    L.append(f"env: pandas {pd.__version__} (StringDtype hazard env if 3.x)")
    L.append(f"recipe: base-08 HGB; OR-gate=(n_object_cols>0 OR n_train>={GATE_N}); "
             f"seed-avg K={K} prob-mean on firing datasets, seed-0 otherwise")
    L.append(f"datasets scored: {len(ordered)}/{len(names)}  exceptions: {len(exceptions)}")
    L.append(f"CLEAN RUN = {'YES' if clean_run else 'NO'}")
    L.append("")
    L.append("Per-dataset (delta = cand_private - base_private):")
    L.append(f"  {'dataset':10s} {'n_tr':>6s} {'obj':>4s} {'gate':>5s} {'fits':>4s} "
             f"{'base_prv':>9s} {'cand_prv':>9s} {'d_prv':>8s} {'==base':>7s}")
    for r in ordered:
        L.append(f"  {r['dataset']:10s} {r['n_train']:6d} {r['n_object_cols']:4d} "
                 f"{str(r['gate_fired']):>5s} {r['n_fits']:4d} "
                 f"{r['base_private']:9.4f} {r['cand_private']:9.4f} "
                 f"{r['delta_private']:+8.4f} {str(r['cand_equals_base']):>7s}")
    L.append("")
    L.append("OR-gate FIRING datasets — per-dataset delta_private:")
    for r in fired:
        L.append(f"  {r['dataset']:10s} obj={r['n_object_cols']:2d} n={r['n_train']:6d} "
                 f"d_prv={r['delta_private']:+.4f} d_pub={r['delta_public']:+.4f}")
    L.append("")
    L.append("Non-firing datasets (must equal base, delta==0):")
    for r in non_fired:
        L.append(f"  {r['dataset']:10s} obj={r['n_object_cols']} n={r['n_train']} "
                 f"cand==base={r['cand_equals_base']} d_prv={r['delta_private']:+.6f}")
    L.append(f"  ANCHOR non-firing==base: {'PASS' if nonfire_ok else 'FAIL'}")
    L.append("")
    L.append("base-08 anchor vs dataset_stats.baseline_hgb_auc_private "
             f"(|Δ|<{BASE_ANCHOR_TOL}):")
    L.extend(anchor_lines)
    L.append(f"  BASE ANCHOR: {'PASS' if anchor_ok else 'DRIFT (see rows)'}")
    L.append("")
    L.append("Aggregate over %d datasets:" % len(ordered))
    L.append(f"  mean base_private  = {mean_base:.4f}")
    L.append(f"  mean cand_private  = {mean_cand:.4f}")
    L.append(f"  MEAN DELTA         = {mean_delta:+.4f}")
    L.append(f"  worst regression   = {worst_reg:+.4f}  ({worst_ds})")
    L.append(f"  firing datasets    = {len(fired)}/{len(ordered)}")
    if exceptions:
        L.append("")
        L.append("EXCEPTIONS:")
        for nm, e in exceptions:
            L.append(f"  {nm}: {e}")

    with open(OUT_TXT, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n" + "\n".join(L))

    blob = {
        "env_pandas": pd.__version__,
        "clean_run": clean_run,
        "n_scored": len(ordered),
        "n_exceptions": len(exceptions),
        "mean_base_private": mean_base,
        "mean_cand_private": mean_cand,
        "mean_delta": mean_delta,
        "worst_regression": worst_reg,
        "worst_dataset": worst_ds,
        "n_firing": len(fired),
        "nonfire_equals_base": nonfire_ok,
        "base_anchor_ok": anchor_ok,
    }
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(blob, f, indent=1)
    print("\nRUN COMPLETE marker: clean_run=%s mean_delta=%+.4f" % (clean_run, mean_delta))


if __name__ == "__main__":
    main()
