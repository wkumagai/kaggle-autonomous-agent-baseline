#!/usr/bin/env python
"""
bench_03 round50 — CATEGORICAL-MASK WIDENING BY CARDINALITY single-knob sweep
(ALL 16). OFFLINE ONLY. No subprocess, no LLM, no Kaggle, no network. Calls
sklearn in-process only. Writes ONLY under this round50 directory; NEVER touches
submissions/.

GOAL (improvement-log angle "cardinality-catmask")
--------------------------------------------------
Base-08 marks a feature categorical ONLY when its pandas dtype is object:
  cat_mask = [train[c].dtype == object for c in features]
Low-cardinality NUMERIC columns (small integer codes, one-hot-ish flags,
bucketed values) are therefore fed to HGB as ORDERED numeric features even
though they are really unordered categories. Hypothesis: telling HGB to treat
low-cardinality numeric columns as categorical (native categorical splits) MIGHT
help by removing a false ordering assumption — but on an already-tuned model it
usually just adds noise / changes split geometry with no net gain. This round
measures the clean offline delta of widening the categorical mask by cardinality
on ALL 16 datasets (NOT gated to a subgroup).

Design (single-seed, random_state=0, NON-ensemble):
  BASE arm      = base-08 HGB exactly (reference column), all 16 datasets.
                  cat_mask = object-dtype columns only (byte-identical to 08).
  CAND arms     = identical base-08 pipeline, EXCEPT the categorical mask is
                  widened: in ADDITION to object-dtype columns, any NON-object
                  column whose nunique(dropna=True) <= T is ALSO marked
                  categorical. One arm per T in {5, 10}, applied to ALL 16
                  datasets (no subgroup gating). Everything else stays
                  byte-identical (random_state=0, max_iter=300,
                  early_stopping=True, l2 gate, tiered msl gate,
                  validation_fraction UNSET -> sklearn default 0.10,
                  n_iter_no_change UNSET -> default 10, max_depth UNSET).

HGB-safety of widened columns (so native categorical splits never error):
  HistGradientBoostingClassifier native categorical splits need the categorical
  column values to be finite and (in older sklearn) non-negative integers below
  max_bins. To stay safe and byte-faithful we only ADD a non-object column to
  the widened mask if it is:
      * integer-typed (dtype.kind in 'iu'), OR
      * float whose non-null values are ALL integral AND non-negative AND
        max value < 255.
  A column failing this predicate is left NUMERIC (not forced). The candidate
  fit only flips mask bits on the exact same feature frame base-08 uses (no
  dtype mutation), so every non-widened column is byte-identical to base; the
  fit is wrapped so an arm that errors on a dataset is RECORDED, not dropped.

BASE recipe reproduced (== shipped 08), identical to round49:
  features = [c for c in train.columns if c not in ("row_id","target")]
  cat_mask = [train[c].dtype == object for c in features]
  n=len(train); ratio = len(features)/n
  l2  = 1.0 if ratio >= 0.010 else 0.0
  msl = 70 if ratio>=0.030 else 50 if ratio>=0.015 else 20
  HistGradientBoostingClassifier(categorical_features=cat_mask, random_state=0,
      max_iter=300, early_stopping=True, l2_regularization=l2,
      min_samples_leaf=msl)   # max_depth NOT set -> default None
  pred = predict_proba(test)[:, class==1]

REPRODUCTION: the BASE column on ALL 16 datasets must match round49's base
column (full precision, read from round49 results.csv, columns base_pub/base_prv)
to < 5e-6. This is the identical base-08 recipe, so it must reproduce exactly
(deterministic w.r.t. random_state).

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
OUT_DIR = os.path.join(BENCH_DIR, "round50_cardinality_catmask")
STATS_CSV = os.path.join(BENCH_DIR, "dataset_stats.csv")
ROUND49_RESULTS = os.path.join(BENCH_DIR, "round49_max_depth", "results.csv")

L2_GATE_THRESHOLD = 0.010   # shipped-08 ratio gate for l2 (FIXED)
GATED_L2 = 1.0
DEFAULT_MSL = 20
MSL_TIERS = [(0.030, 70), (0.015, 50)]
BASE_SEED = 0
N_DATASETS = 16
REPRO_TOL = 5e-6
MAX_CAT_VAL = 255           # HGB max_bins default; float-safe upper bound

BASE = "base"
# swept knob: cardinality threshold T for widening the categorical mask.
# One arm per T, layered on identical base-08 pipeline.
THRESHOLDS = [5, 10]
CAND_NAMES = {t: f"cardT{t}" for t in THRESHOLDS}   # e.g. 5 -> "cardT5"
CAND_LIST = [CAND_NAMES[t] for t in THRESHOLDS]
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


def round49_base_anchors(path=ROUND49_RESULTS):
    """Read round49's base_pub/base_prv for ALL 16 datasets to anchor
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


def is_safe_categorical(col):
    """Return True iff a NON-object column can be safely marked categorical for
    HGB native categorical splits (finite, non-negative integer-representable,
    below MAX_CAT_VAL). Integer-typed columns qualify unconditionally; float
    columns qualify only if all non-null values are integral, non-negative, and
    the max value < MAX_CAT_VAL. Anything else stays numeric."""
    kind = col.dtype.kind
    if kind in ("i", "u"):
        return True
    if kind == "f":
        nonnull = col.dropna().to_numpy()
        if nonnull.size == 0:
            return False
        if not np.all(np.isfinite(nonnull)):
            return False
        if not np.all(np.mod(nonnull, 1) == 0):
            return False
        if np.any(nonnull < 0):
            return False
        if nonnull.max() >= MAX_CAT_VAL:
            return False
        return True
    return False


def widened_mask(train, features, base_mask, threshold):
    """Base object-dtype mask, PLUS any non-object column with cardinality
    <= threshold that is HGB-safe to treat as categorical. Returns
    (mask, added_cols)."""
    mask = list(base_mask)
    added = []
    for i, c in enumerate(features):
        if base_mask[i]:
            continue
        col = train[c]
        if col.dtype == object:
            continue
        if col.nunique(dropna=True) <= threshold and is_safe_categorical(col):
            mask[i] = True
            added.append(c)
    return mask, added


def fit_hgb(train, test, features, cat_mask, l2, msl_val, seed):
    """Fit ONE shipped-08 HGB. validation_fraction left UNSET (sklearn default
    0.1, byte-identical to shipped 08). The ONLY thing a candidate arm changes
    vs base is `cat_mask` (the widened categorical mask); every kwarg and the
    feature frame itself are identical to base-08."""
    kwargs = dict(
        categorical_features=cat_mask,
        random_state=seed,
        max_iter=300,
        early_stopping=True,
        l2_regularization=l2,
        min_samples_leaf=msl_val,
    )
    clf = HistGradientBoostingClassifier(**kwargs)
    clf.fit(train[features], train["target"])
    proba = clf.predict_proba(test[features])
    classes = list(clf.classes_)
    pos_idx = classes.index(1) if 1 in classes else 1
    return proba[:, pos_idx]


def run_one(name, train_csv, test_csv, stats):
    """Returns (preds, meta). preds maps config_name -> {row_id -> prob}.

    base    = seed-0, object-dtype cat_mask (== shipped 08).
    cardT{t} = identical + categorical mask widened by cardinality<=t, applied
               to ALL 16 datasets, t in {5,10}.
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

    # base = seed-0, object-dtype mask (byte-identical to shipped 08).
    base_vec = fit_hgb(train, test, features, base_cat_mask, l2, msl_val,
                       BASE_SEED)

    preds = {BASE: dict(zip(row_ids, base_vec.tolist()))}
    added_counts = {}
    # cand arms: same base-08 pipeline + widened categorical mask (ALL 16).
    for t in THRESHOLDS:
        mask, added = widened_mask(train, features, base_cat_mask, t)
        added_counts[t] = len(added)
        cand_vec = fit_hgb(train, test, features, mask, l2, msl_val, BASE_SEED)
        preds[CAND_NAMES[t]] = dict(zip(row_ids, cand_vec.tolist()))

    meta = {
        "n_train": st["n_train"],
        "n_object_cols": st["n_object_cols"],
        "l2": l2,
        "msl": msl_val,
        "n_features": len(features),
        "n_base_cat": sum(base_cat_mask),
        "added_counts": added_counts,
        "n_fits": 1 + len(THRESHOLDS),
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
    anchors49 = round49_base_anchors()
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
            for t in THRESHOLDS:
                rec[f"n_added_T{t}"] = meta["added_counts"][t]
            for cfg in ALL_CONFIGS:
                pub, prv = score_split(preds[cfg], sol)
                rec[f"{cfg}_pub"] = pub
                rec[f"{cfg}_prv"] = prv
            cand_str = "  ".join(
                f"T{t}(+{meta['added_counts'][t]}) "
                f"pub={rec[CAND_NAMES[t]+'_pub']:.6f} "
                f"prv={rec[CAND_NAMES[t]+'_prv']:.6f}" for t in THRESHOLDS)
            print(f"[OK] {name} n_tr={meta['n_train']} obj={meta['n_object_cols']} "
                  f"feats={meta['n_features']} base_cat={meta['n_base_cat']} "
                  f"l2={meta['l2']} msl={meta['msl']} fits={meta['n_fits']} "
                  f"base pub={rec['base_pub']:.6f} prv={rec['base_prv']:.6f}  "
                  f"{cand_str}")
        except Exception as e:
            exceptions.append((name, repr(e)))
            rec.update({"n_train": stats.get(name, {}).get("n_train", ""),
                        "n_object_cols": stats.get(name, {}).get("n_object_cols", ""),
                        "l2": float("nan"), "msl": float("nan")})
            for t in THRESHOLDS:
                rec[f"n_added_T{t}"] = ""
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
                  "base_pub", "base_prv"]
    for t in THRESHOLDS:
        cn = CAND_NAMES[t]
        fieldnames += [f"n_added_T{t}", f"{cn}_pub", f"{cn}_prv",
                       f"{cn}_d_pub", f"{cn}_d_prv"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            out = {k: r.get(k, "") for k in
                   ["dataset", "n_train", "n_object_cols", "l2", "msl",
                    "base_pub", "base_prv"]}
            for t in THRESHOLDS:
                cn = CAND_NAMES[t]
                out[f"n_added_T{t}"] = r.get(f"n_added_T{t}", "")
                out[f"{cn}_pub"] = r.get(f"{cn}_pub", "")
                out[f"{cn}_prv"] = r.get(f"{cn}_prv", "")
                out[f"{cn}_d_pub"] = delta(r, cn, "pub")
                out[f"{cn}_d_prv"] = delta(r, cn, "prv")
            w.writerow(out)

    # ---- REPRODUCTION: base on ALL 16 matches round49 (tol<5e-6) ----
    repro = {}
    repro_ok = True
    repro_available = anchors49 is not None
    by_name = {r["dataset"]: r for r in rows}
    max_abs_dev = 0.0
    for i in range(1, N_DATASETS + 1):
        nm = f"train_{i:02d}"
        r = by_name.get(nm)
        mine = (r.get("base_pub"), r.get("base_prv")) if r else (None, None)
        ref = anchors49.get(nm) if anchors49 else None
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
    L.append("bench_03 round50 — CATEGORICAL-MASK WIDENING BY CARDINALITY "
             "(T in {5,10}) single-knob sweep (ALL 16)  [OFFLINE]")
    L.append("=" * 78)
    L.append("")
    L.append("SETUP:")
    L.append("  base == shipped 08 (git HEAD:submissions/08...): HGB, early_stopping,")
    L.append("    cat_mask = object-dtype columns ONLY. base column = seed-0, all 16.")
    L.append("  cardT{t} == base pipeline with the categorical mask WIDENED: in")
    L.append("    addition to object-dtype columns, any non-object column with")
    L.append("    nunique(dropna=True) <= t that is HGB-safe (int-typed, or float")
    L.append("    with all non-null values integral/non-negative/<255) is ALSO")
    L.append("    marked categorical. One arm per t in {5,10}, ALL 16 datasets")
    L.append("    (no subgroup gating). Everything else byte-identical")
    L.append("    (random_state=0, max_iter=300, early_stopping=True, l2 gate,")
    L.append("    tiered msl gate, max_depth UNSET; only mask bits differ, the")
    L.append("    feature frame is untouched).")

    # ---- SWEEP HEADLINE (one line per candidate) ----
    L.append("")
    L.append("=== HEADLINE (each cand cardT{t} vs base == shipped 08, all 16) ===")
    headline = {}
    for t in THRESHOLDS:
        cn = CAND_NAMES[t]
        mp = mean_delta(cn, "pub")
        mv = mean_delta(cn, "prv")
        wp, lp, tp = wlt(cn, "pub")
        wv, lv, tv = wlt(cn, "prv")
        headline[t] = (mp, mv, (wp, lp, tp), (wv, lv, tv))
        L.append(f"  cardT{t}: mean ΔPublic={mp:+.6f} (W/L/T {wp}/{lp}/{tp})  "
                 f"mean ΔPrivate={mv:+.6f} (W/L/T {wv}/{lv}/{tv})")

    # ---- widening footprint ----
    L.append("")
    L.append("=== WIDENING FOOTPRINT (extra numeric cols marked categorical) ===")
    L.append(f"{'dataset':<10} {'obj_cols':>9}"
             + "".join(f" {'+T'+str(t):>6}" for t in THRESHOLDS))
    for r in rows:
        line = f"{r['dataset']:<10} {str(r.get('n_object_cols','')):>9}"
        for t in THRESHOLDS:
            line += f" {'+'+str(r.get('n_added_T'+str(t),'')):>6}"
        L.append(line)

    # ---- REPRODUCTION ----
    L.append("")
    L.append("=== REPRODUCTION CHECK (base on ALL 16 vs round49, tol<5e-6) ===")
    if not repro_available:
        L.append("  round49 results.csv NOT found -> reproduction NOT anchored (FAIL).")
    else:
        for i in range(1, N_DATASETS + 1):
            nm = f"train_{i:02d}"
            rr = repro[nm]
            mp_, mv_ = rr["mine"]
            rp_, rv_ = rr["ref"] if rr["ref"] else (float("nan"), float("nan"))
            L.append(
                f"  {nm}: Public {mp_:.6f} vs r49 {rp_:.6f} "
                f"(|d|={rr['devp']:.2e}, {'YES' if rr['okp'] else 'NO'}); "
                f"Private {mv_:.6f} vs r49 {rv_:.6f} "
                f"(|d|={rr['devv']:.2e}, {'YES' if rr['okv'] else 'NO'})")
        L.append(f"  max |dev| over all 16x2 = {max_abs_dev:.2e}")
        L.append(f"  REPRODUCTION: {'PASS' if repro_ok else 'FAIL'}")

    # ---- PER-DATASET DELTAS (per candidate) ----
    for t in THRESHOLDS:
        cn = CAND_NAMES[t]
        for split, tag in (("pub", "Public"), ("prv", "Private")):
            L.append("")
            L.append(f"=== PER-DATASET ΔAUC ({tag}) — base vs cardT{t} ===")
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
    for t in THRESHOLDS:
        cn = CAND_NAMES[t]
        regs_pub = regressions(cn, "pub")
        regs_prv = regressions(cn, "prv")
        if not regs_pub and not regs_prv:
            L.append(f"  cardT{t}: NONE on either split.")
        else:
            for n_, d_ in regs_pub:
                L.append(f"  cardT{t} Public  {n_}: {d_:+.6f}")
            for n_, d_ in regs_prv:
                L.append(f"  cardT{t} Private {n_}: {d_:+.6f}")

    # ---- ADOPTION / VERDICT (per candidate) ----
    L.append("")
    L.append("=== ADOPTION ANALYSIS ===")
    L.append("ADOPT iff SOME arm has mean ΔPublic > 0 AND mean ΔPrivate > 0 with")
    L.append("  ZERO regression on EITHER split (no dataset ΔAUC < 0 on Public or")
    L.append("  Private). Any regression, or net-negative/negligible mean on either")
    L.append("  split => REJECT that arm.")
    ADOPT_EPS = 1e-5   # negligible-mean guard
    adopt_arms = []
    for t in THRESHOLDS:
        cn = CAND_NAMES[t]
        mp, mv, _, _ = headline[t]
        regs_pub = regressions(cn, "pub")
        regs_prv = regressions(cn, "prv")
        zero_regs = (not regs_pub) and (not regs_prv)
        clean_gain = (mp > ADOPT_EPS and mv > ADOPT_EPS)
        is_adopt = zero_regs and clean_gain
        if is_adopt:
            adopt_arms.append(t)
        L.append("")
        L.append(f"  [cardT{t}] zero_regressions="
                 f"{'YES' if zero_regs else 'NO'} "
                 f"(Public regs={len(regs_pub)}, Private regs={len(regs_prv)})")
        L.append(f"  [cardT{t}] mean ΔPublic  = {mp:+.6f}  (clean gain: "
                 f"{'YES' if mp > ADOPT_EPS else 'NO'})")
        L.append(f"  [cardT{t}] mean ΔPrivate = {mv:+.6f}  (clean gain: "
                 f"{'YES' if mv > ADOPT_EPS else 'NO'})")
        L.append(f"  [cardT{t}] -> "
                 f"{'ADOPT-CANDIDATE' if is_adopt else 'REJECT'}")

    L.append("")
    L.append("=== VERDICT ===")
    if adopt_arms:
        arms = ", ".join(f"cardT{t}" for t in adopt_arms)
        L.append(f"  ADOPT-CANDIDATE: {arms} cleanly improve BOTH splits with zero "
                 f"regression.")
    else:
        L.append("  REJECT: no cardinality threshold in {5,10} beats base-08 "
                 "(object-only cat_mask) on BOTH splits with zero regressions. "
                 "Treating low-cardinality numeric columns as categorical changes "
                 "split geometry on an already-tuned model -> base-08 stays at "
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
