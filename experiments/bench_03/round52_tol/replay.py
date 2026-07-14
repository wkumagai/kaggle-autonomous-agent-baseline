#!/usr/bin/env python
"""
bench_03 round52 — EARLY-STOPPING TOLERANCE (`tol`) single-knob sweep (ALL 16).
OFFLINE ONLY. No subprocess, no LLM, no Kaggle, no network. Calls sklearn
in-process only. Writes ONLY under this round52 directory; NEVER touches
submissions/.

GOAL (improvement-log angle "tol")
----------------------------------
Base-08's HistGradientBoostingClassifier leaves `tol` at the sklearn default
(1e-7). `tol` is the absolute tolerance used by the early-stopping check: a
boosting iteration counts as "no improvement" unless the validation score
improves by MORE than `tol`. Raising `tol` therefore makes early_stopping
trigger on smaller improvements => the model stops sooner => fewer boosting
iterations => more regularization. This round measures the clean offline delta
of raising `tol` on ALL 16 datasets (NOT gated to a subgroup), for TWO values:
`tol=1e-3` and `tol=1e-4`.

Design (single-seed, random_state=0, NON-ensemble):
  BASE arm   = base-08 HGB exactly (reference column), all 16 datasets.
               tol UNSET (sklearn default 1e-7), byte-identical to 08.
  CAND arms  = identical base-08 pipeline, EXCEPT `tol` is raised. Two arms:
               "tol1e3" (tol=1e-3) and "tol1e4" (tol=1e-4), each applied to
               ALL 16 datasets (no subgroup gating). Everything else stays
               byte-identical (random_state=0, max_iter=300,
               early_stopping=True, l2 gate, tiered msl gate, cat_mask =
               object-dtype columns only, validation_fraction UNSET -> sklearn
               default 0.10, n_iter_no_change UNSET -> default 10, max_depth
               UNSET, interaction_cst UNSET). This is a TRUE single knob: only
               the `tol` argument differs.

BASE recipe reproduced (== shipped 08), identical to round51:
  features = [c for c in train.columns if c not in ("row_id","target")]
  cat_mask = [train[c].dtype == object for c in features]
  n=len(train); ratio = len(features)/n
  l2  = 1.0 if ratio >= 0.010 else 0.0
  msl = 70 if ratio>=0.030 else 50 if ratio>=0.015 else 20
  HistGradientBoostingClassifier(categorical_features=cat_mask, random_state=0,
      max_iter=300, early_stopping=True, l2_regularization=l2,
      min_samples_leaf=msl)   # max_depth / interaction_cst / tol NOT set
  pred = predict_proba(test)[:, class==1]

REPRODUCTION: the BASE column on ALL 16 datasets must match round51's base
column (full precision, read from round51 results.csv, columns
base_pub/base_prv) to < 5e-6. This is the identical base-08 recipe, so it must
reproduce exactly (deterministic w.r.t. random_state).

ADOPTION: per candidate arm, ADOPT iff it cleanly improves — mean ΔPublic > 0
AND mean ΔPrivate > 0 with ZERO regression on EITHER split (no dataset with
ΔAUC < 0 on Public or Private). Any negative ΔAUC on any dataset/split, or a
net-negative/negligible mean on either split => REJECT. SHIP VERDICT = ADOPT
iff at least one candidate arm qualifies, else REJECT.
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
OUT_DIR = os.path.join(BENCH_DIR, "round52_tol")
STATS_CSV = os.path.join(BENCH_DIR, "dataset_stats.csv")
ROUND51_RESULTS = os.path.join(BENCH_DIR, "round51_interaction_cst",
                               "results.csv")

L2_GATE_THRESHOLD = 0.010   # shipped-08 ratio gate for l2 (FIXED)
GATED_L2 = 1.0
DEFAULT_MSL = 20
MSL_TIERS = [(0.030, 70), (0.015, 50)]
BASE_SEED = 0
N_DATASETS = 16
REPRO_TOL = 5e-6

BASE = "base"
# two candidate arms: raise tol -> earlier stopping -> more regularization
CAND1 = "tol1e3"   # tol=1e-3
CAND2 = "tol1e4"   # tol=1e-4
CANDS = [CAND1, CAND2]
CAND_TOL = {CAND1: 1e-3, CAND2: 1e-4}
ALL_CONFIGS = [BASE, CAND1, CAND2]


def load_stats(path=STATS_CSV):
    stats = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            stats[row["name"]] = {
                "n_train": int(row["n_train"]),
                "n_object_cols": int(row["n_object_cols"]),
            }
    return stats


def round51_base_anchors(path=ROUND51_RESULTS):
    """Read round51's base_pub/base_prv for ALL 16 datasets to anchor
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


def fit_hgb(train, test, features, cat_mask, l2, msl_val, seed, tol=None):
    """Fit ONE shipped-08 HGB. validation_fraction left UNSET (sklearn default
    0.1, byte-identical to shipped 08). The ONLY thing a candidate arm changes
    vs base is `tol` (base: None -> sklearn default 1e-7; cand: 1e-3 or 1e-4);
    every other kwarg and the feature frame are identical to base-08. `tol` is
    only passed when non-None so the base fit is byte-identical to shipped 08
    (no explicit kwarg)."""
    kwargs = dict(
        categorical_features=cat_mask,
        random_state=seed,
        max_iter=300,
        early_stopping=True,
        l2_regularization=l2,
        min_samples_leaf=msl_val,
    )
    if tol is not None:
        kwargs["tol"] = tol
    clf = HistGradientBoostingClassifier(**kwargs)
    clf.fit(train[features], train["target"])
    proba = clf.predict_proba(test[features])
    classes = list(clf.classes_)
    pos_idx = classes.index(1) if 1 in classes else 1
    return proba[:, pos_idx]


def run_one(name, train_csv, test_csv, stats):
    """Returns (preds, meta). preds maps config_name -> {row_id -> prob}.

    base   = seed-0, base-08 (tol UNSET == shipped 08).
    tol1e3 = identical base-08 pipeline + tol=1e-3, ALL 16 datasets.
    tol1e4 = identical base-08 pipeline + tol=1e-4, ALL 16 datasets.
    """
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)

    features = [c for c in train.columns if c not in ("row_id", "target")]
    base_cat_mask = [train[c].dtype == object for c in features]

    n = len(train)
    ratio = len(features) / n
    l2 = GATED_L2 if ratio >= L2_GATE_THRESHOLD else 0.0
    msl_val = msl_for_ratio(ratio)

    st = stats[name]
    row_ids = test["row_id"].tolist()

    preds = {}
    # base = seed-0, base-08 (byte-identical to shipped 08).
    base_vec = fit_hgb(train, test, features, base_cat_mask, l2, msl_val,
                       BASE_SEED)
    preds[BASE] = dict(zip(row_ids, base_vec.tolist()))
    # candidate arms = same base-08 pipeline + raised tol (ALL 16).
    n_fits = 1
    for cand in CANDS:
        cand_vec = fit_hgb(train, test, features, base_cat_mask, l2, msl_val,
                           BASE_SEED, tol=CAND_TOL[cand])
        preds[cand] = dict(zip(row_ids, cand_vec.tolist()))
        n_fits += 1

    meta = {
        "n_train": st["n_train"],
        "n_object_cols": st["n_object_cols"],
        "l2": l2,
        "msl": msl_val,
        "n_features": len(features),
        "n_base_cat": sum(base_cat_mask),
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
    anchors51 = round51_base_anchors()
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
            print(f"[OK] {name} n_tr={meta['n_train']} obj={meta['n_object_cols']} "
                  f"feats={meta['n_features']} base_cat={meta['n_base_cat']} "
                  f"l2={meta['l2']} msl={meta['msl']} fits={meta['n_fits']} "
                  f"base pub={rec['base_pub']:.6f} prv={rec['base_prv']:.6f}  "
                  f"tol1e3 pub={rec['tol1e3_pub']:.6f} prv={rec['tol1e3_prv']:.6f}  "
                  f"tol1e4 pub={rec['tol1e4_pub']:.6f} prv={rec['tol1e4_prv']:.6f}")
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

    def improvements(cand, split, eps=1e-6):
        return [(r["dataset"], delta(r, cand, split)) for r in rows
                if not math.isnan(delta(r, cand, split))
                and delta(r, cand, split) > eps]

    # ---- results.csv ----
    csv_path = os.path.join(OUT_DIR, "results.csv")
    fieldnames = ["dataset", "n_train", "n_object_cols", "l2", "msl",
                  "base_pub", "base_prv",
                  "tol1e3_pub", "tol1e3_prv", "tol1e3_d_pub", "tol1e3_d_prv",
                  "tol1e4_pub", "tol1e4_prv", "tol1e4_d_pub", "tol1e4_d_prv"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            out = {k: r.get(k, "") for k in
                   ["dataset", "n_train", "n_object_cols", "l2", "msl",
                    "base_pub", "base_prv",
                    "tol1e3_pub", "tol1e3_prv", "tol1e4_pub", "tol1e4_prv"]}
            out["tol1e3_d_pub"] = delta(r, CAND1, "pub")
            out["tol1e3_d_prv"] = delta(r, CAND1, "prv")
            out["tol1e4_d_pub"] = delta(r, CAND2, "pub")
            out["tol1e4_d_prv"] = delta(r, CAND2, "prv")
            w.writerow(out)

    # ---- REPRODUCTION: base on ALL 16 matches round51 (tol<5e-6) ----
    repro = {}
    repro_ok = True
    repro_available = anchors51 is not None
    by_name = {r["dataset"]: r for r in rows}
    max_abs_dev = 0.0
    for i in range(1, N_DATASETS + 1):
        nm = f"train_{i:02d}"
        r = by_name.get(nm)
        mine = (r.get("base_pub"), r.get("base_prv")) if r else (None, None)
        ref = anchors51.get(nm) if anchors51 else None
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
    L.append("bench_03 round52 — EARLY-STOPPING TOLERANCE (`tol`) single-knob "
             "sweep (ALL 16)  [OFFLINE]")
    L.append("=" * 78)
    L.append("")
    L.append("SETUP:")
    L.append("  base == shipped 08 (git HEAD:submissions/08...): HGB, early_stopping,")
    L.append("    cat_mask = object-dtype columns ONLY, tol UNSET (sklearn default")
    L.append("    1e-7). base column = seed-0, all 16.")
    L.append("  tol1e3 == base pipeline with tol=1e-3 added.")
    L.append("  tol1e4 == base pipeline with tol=1e-4 added.")
    L.append("    Raising tol makes early_stopping trigger on smaller improvements")
    L.append("    -> stops sooner -> more regularization. Applied to ALL 16 datasets")
    L.append("    (no subgroup gating). Everything else byte-identical")
    L.append("    (random_state=0, max_iter=300, early_stopping=True, l2 gate,")
    L.append("    tiered msl gate, cat_mask, max_depth/interaction_cst UNSET;")
    L.append("    ONLY the tol argument differs).")

    # ---- SWEEP HEADLINE ----
    L.append("")
    L.append("=== HEADLINE (candidate vs base == shipped 08, all 16) ===")
    head = {}
    for cand in CANDS:
        mp = mean_delta(cand, "pub")
        mv = mean_delta(cand, "prv")
        wp, lp, tp = wlt(cand, "pub")
        wv, lv, tv = wlt(cand, "prv")
        head[cand] = (mp, mv, (wp, lp, tp), (wv, lv, tv))
        L.append(f"  {cand}: mean ΔPublic={mp:+.6f} (W/L/T {wp}/{lp}/{tp})  "
                 f"mean ΔPrivate={mv:+.6f} (W/L/T {wv}/{lv}/{tv})")

    # ---- REPRODUCTION ----
    L.append("")
    L.append("=== REPRODUCTION CHECK (base on ALL 16 vs round51, tol<5e-6) ===")
    if not repro_available:
        L.append("  round51 results.csv NOT found -> reproduction NOT anchored (FAIL).")
    else:
        for i in range(1, N_DATASETS + 1):
            nm = f"train_{i:02d}"
            rr = repro[nm]
            mp_, mv_ = rr["mine"]
            rp_, rv_ = rr["ref"] if rr["ref"] else (float("nan"), float("nan"))
            L.append(
                f"  {nm}: Public {mp_:.6f} vs r51 {rp_:.6f} "
                f"(|d|={rr['devp']:.2e}, {'YES' if rr['okp'] else 'NO'}); "
                f"Private {mv_:.6f} vs r51 {rv_:.6f} "
                f"(|d|={rr['devv']:.2e}, {'YES' if rr['okv'] else 'NO'})")
        L.append(f"  max |dev| over all 16x2 = {max_abs_dev:.2e}")
        L.append(f"  REPRODUCTION: {'PASS' if repro_ok else 'FAIL'}")

    # ---- PER-DATASET DELTAS ----
    for cand in CANDS:
        for split, tag in (("pub", "Public"), ("prv", "Private")):
            L.append("")
            L.append(f"=== PER-DATASET ΔAUC ({tag}) — base vs {cand} ===")
            L.append(f"{'dataset':<10} {'base':>10} {'cand':>10} {'delta':>11}")
            for r in rows:
                b = r.get(f"{BASE}_{split}")
                c = r.get(f"{cand}_{split}")
                dd = delta(r, cand, split)
                bstr = f"{b:>10.6f}" if isinstance(b, float) and not math.isnan(b) else f"{'nan':>10}"
                cstr = f"{c:>10.6f}" if isinstance(c, float) and not math.isnan(c) else f"{'nan':>10}"
                dstr = f"{dd:>+11.6f}" if not math.isnan(dd) else f"{'nan':>11}"
                L.append(f"{r['dataset']:<10} {bstr} {cstr} {dstr}")

    # ---- REGRESSIONS / IMPROVEMENTS ----
    L.append("")
    L.append("=== REGRESSIONS (ΔAUC < -1e-6) ===")
    any_reg = False
    for cand in CANDS:
        rp = regressions(cand, "pub")
        rv = regressions(cand, "prv")
        if not rp and not rv:
            L.append(f"  {cand}: NONE on either split.")
        else:
            any_reg = True
            for n_, d_ in rp:
                L.append(f"  {cand} Public  {n_}: {d_:+.6f}")
            for n_, d_ in rv:
                L.append(f"  {cand} Private {n_}: {d_:+.6f}")

    L.append("")
    L.append("=== IMPROVEMENTS (ΔAUC > +1e-6) ===")
    for cand in CANDS:
        ip = improvements(cand, "pub")
        iv = improvements(cand, "prv")
        if not ip and not iv:
            L.append(f"  {cand}: NONE on either split.")
        else:
            for n_, d_ in ip:
                L.append(f"  {cand} Public  {n_}: {d_:+.6f}")
            for n_, d_ in iv:
                L.append(f"  {cand} Private {n_}: {d_:+.6f}")

    # ---- largest single moves ----
    def extreme(cand, split, sign):
        vals = [(r["dataset"], delta(r, cand, split)) for r in rows
                if not math.isnan(delta(r, cand, split))]
        if not vals:
            return None
        return (min(vals, key=lambda x: x[1]) if sign < 0
                else max(vals, key=lambda x: x[1]))

    L.append("")
    L.append("=== LARGEST SINGLE MOVES ===")
    # overall (across both candidate arms and both splits)
    glob_worst = None
    glob_best = None
    for cand in CANDS:
        L.append(f"  -- {cand} --")
        for split, tag in (("pub", "Public"), ("prv", "Private")):
            worst = extreme(cand, split, -1)
            best = extreme(cand, split, +1)
            if worst:
                L.append(f"    {tag} max regression : {worst[0]} {worst[1]:+.6f}")
                if glob_worst is None or worst[1] < glob_worst[2]:
                    glob_worst = (cand, tag, worst[1], worst[0])
            if best:
                L.append(f"    {tag} max improvement: {best[0]} {best[1]:+.6f}")
                if glob_best is None or best[1] > glob_best[2]:
                    glob_best = (cand, tag, best[1], best[0])
    if glob_worst:
        L.append(f"  OVERALL largest regression : {glob_worst[0]} {glob_worst[1]} "
                 f"{glob_worst[3]} {glob_worst[2]:+.6f}")
    if glob_best:
        L.append(f"  OVERALL largest improvement: {glob_best[0]} {glob_best[1]} "
                 f"{glob_best[3]} {glob_best[2]:+.6f}")

    # ---- ADOPTION / VERDICT ----
    L.append("")
    L.append("=== ADOPTION ANALYSIS ===")
    L.append("Per candidate arm: ADOPT iff mean ΔPublic > 0 AND mean ΔPrivate > 0")
    L.append("  with ZERO regression on EITHER split (no dataset ΔAUC < 0 on Public")
    L.append("  or Private). Any regression, or net-negative/negligible mean on")
    L.append("  either split => REJECT.")
    ADOPT_EPS = 1e-5   # negligible-mean guard
    adopt_flags = {}
    L.append("")
    for cand in CANDS:
        regs_pub = regressions(cand, "pub")
        regs_prv = regressions(cand, "prv")
        mp, mv = head[cand][0], head[cand][1]
        zero_regs = (not regs_pub) and (not regs_prv)
        clean_gain = (mp > ADOPT_EPS and mv > ADOPT_EPS)
        is_adopt = zero_regs and clean_gain
        adopt_flags[cand] = is_adopt
        L.append(f"  [{cand}] zero_regressions={'YES' if zero_regs else 'NO'} "
                 f"(Public regs={len(regs_pub)}, Private regs={len(regs_prv)})")
        L.append(f"  [{cand}] mean ΔPublic  = {mp:+.6f}  (clean gain: "
                 f"{'YES' if mp > ADOPT_EPS else 'NO'})")
        L.append(f"  [{cand}] mean ΔPrivate = {mv:+.6f}  (clean gain: "
                 f"{'YES' if mv > ADOPT_EPS else 'NO'})")
        L.append(f"  [{cand}] -> {'ADOPT' if is_adopt else 'REJECT'}")
        L.append("")

    is_adopt_any = any(adopt_flags.values())
    adopted = [c for c in CANDS if adopt_flags[c]]

    L.append("=== VERDICT ===")
    if is_adopt_any:
        L.append(f"  ADOPT: {', '.join(adopted)} cleanly improve BOTH splits with "
                 f"zero regression.")
    else:
        L.append("  REJECT: no tol value beats base-08 on BOTH splits with zero "
                 "regressions. Raising tol makes early_stopping halt sooner "
                 "(fewer boosting iterations); on an already-tuned model this "
                 "under-fits some datasets -> base-08 (tol default 1e-7) stays "
                 "at least as good overall.")

    ship = "ADOPT" if is_adopt_any else "REJECT"
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
