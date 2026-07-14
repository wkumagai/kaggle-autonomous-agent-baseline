#!/usr/bin/env python
"""
bench_03 round51 — INTERACTION-CONSTRAINT (no_interactions) single-knob sweep
(ALL 16). OFFLINE ONLY. No subprocess, no LLM, no Kaggle, no network. Calls
sklearn in-process only. Writes ONLY under this round51 directory; NEVER touches
submissions/.

GOAL (improvement-log angle "interaction_cst")
----------------------------------------------
Base-08's HistGradientBoostingClassifier leaves `interaction_cst` at the sklearn
default (None) => trees may use UNLIMITED feature interactions. The single knob
tested here is `interaction_cst='no_interactions'`, which constrains HGB to an
ADDITIVE model (each tree splits on a single feature; no feature-interaction
terms). Hypothesis: on small-n / noisy datasets, forbidding interactions is a
regularizer that MIGHT curb overfitting of spurious interaction terms — but on an
already-tuned model it usually just removes useful signal / changes tree geometry
with no net gain. This round measures the clean offline delta of adding
`interaction_cst='no_interactions'` on ALL 16 datasets (NOT gated to a subgroup).

Design (single-seed, random_state=0, NON-ensemble):
  BASE arm  = base-08 HGB exactly (reference column), all 16 datasets.
              interaction_cst UNSET (sklearn default None), byte-identical to 08.
  CAND arm  = identical base-08 pipeline, EXCEPT `interaction_cst='no_interactions'`
              is added. One arm named "noint", applied to ALL 16 datasets (no
              subgroup gating). Everything else stays byte-identical
              (random_state=0, max_iter=300, early_stopping=True, l2 gate,
              tiered msl gate, cat_mask = object-dtype columns only,
              validation_fraction UNSET -> sklearn default 0.10,
              n_iter_no_change UNSET -> default 10, max_depth UNSET). This is a
              TRUE single knob: only interaction_cst differs.

BASE recipe reproduced (== shipped 08), identical to round50:
  features = [c for c in train.columns if c not in ("row_id","target")]
  cat_mask = [train[c].dtype == object for c in features]
  n=len(train); ratio = len(features)/n
  l2  = 1.0 if ratio >= 0.010 else 0.0
  msl = 70 if ratio>=0.030 else 50 if ratio>=0.015 else 20
  HistGradientBoostingClassifier(categorical_features=cat_mask, random_state=0,
      max_iter=300, early_stopping=True, l2_regularization=l2,
      min_samples_leaf=msl)   # max_depth / interaction_cst NOT set
  pred = predict_proba(test)[:, class==1]

REPRODUCTION: the BASE column on ALL 16 datasets must match round50's base
column (full precision, read from round50 results.csv, columns base_pub/base_prv)
to < 5e-6. This is the identical base-08 recipe, so it must reproduce exactly
(deterministic w.r.t. random_state).

ADOPTION: ADOPT iff the candidate arm cleanly improves — mean ΔPublic > 0 AND
mean ΔPrivate > 0 with ZERO regression on EITHER split (no dataset with
ΔAUC < 0 on Public or Private). Any negative ΔAUC on any dataset/split, or a
net-negative/negligible mean on either split => REJECT.
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
OUT_DIR = os.path.join(BENCH_DIR, "round51_interaction_cst")
STATS_CSV = os.path.join(BENCH_DIR, "dataset_stats.csv")
ROUND50_RESULTS = os.path.join(BENCH_DIR, "round50_cardinality_catmask",
                               "results.csv")

L2_GATE_THRESHOLD = 0.010   # shipped-08 ratio gate for l2 (FIXED)
GATED_L2 = 1.0
DEFAULT_MSL = 20
MSL_TIERS = [(0.030, 70), (0.015, 50)]
BASE_SEED = 0
N_DATASETS = 16
REPRO_TOL = 5e-6

BASE = "base"
CAND = "noint"   # single candidate arm: interaction_cst='no_interactions'
ALL_CONFIGS = [BASE, CAND]


def load_stats(path=STATS_CSV):
    stats = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            stats[row["name"]] = {
                "n_train": int(row["n_train"]),
                "n_object_cols": int(row["n_object_cols"]),
            }
    return stats


def round50_base_anchors(path=ROUND50_RESULTS):
    """Read round50's base_pub/base_prv for ALL 16 datasets to anchor
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


def fit_hgb(train, test, features, cat_mask, l2, msl_val, seed,
            interaction_cst=None):
    """Fit ONE shipped-08 HGB. validation_fraction left UNSET (sklearn default
    0.1, byte-identical to shipped 08). The ONLY thing the candidate arm changes
    vs base is `interaction_cst` (base: None -> default; cand:
    'no_interactions'); every other kwarg and the feature frame are identical to
    base-08. interaction_cst is only passed when non-None so the base fit is
    byte-identical to shipped 08 (no explicit kwarg)."""
    kwargs = dict(
        categorical_features=cat_mask,
        random_state=seed,
        max_iter=300,
        early_stopping=True,
        l2_regularization=l2,
        min_samples_leaf=msl_val,
    )
    if interaction_cst is not None:
        kwargs["interaction_cst"] = interaction_cst
    clf = HistGradientBoostingClassifier(**kwargs)
    clf.fit(train[features], train["target"])
    proba = clf.predict_proba(test[features])
    classes = list(clf.classes_)
    pos_idx = classes.index(1) if 1 in classes else 1
    return proba[:, pos_idx]


def run_one(name, train_csv, test_csv, stats):
    """Returns (preds, meta). preds maps config_name -> {row_id -> prob}.

    base  = seed-0, base-08 (interaction_cst UNSET == shipped 08).
    noint = identical base-08 pipeline + interaction_cst='no_interactions',
            applied to ALL 16 datasets.
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

    # base = seed-0, base-08 (byte-identical to shipped 08).
    base_vec = fit_hgb(train, test, features, base_cat_mask, l2, msl_val,
                       BASE_SEED)
    # cand = same base-08 pipeline + interaction_cst='no_interactions' (ALL 16).
    cand_vec = fit_hgb(train, test, features, base_cat_mask, l2, msl_val,
                       BASE_SEED, interaction_cst="no_interactions")

    preds = {
        BASE: dict(zip(row_ids, base_vec.tolist())),
        CAND: dict(zip(row_ids, cand_vec.tolist())),
    }

    meta = {
        "n_train": st["n_train"],
        "n_object_cols": st["n_object_cols"],
        "l2": l2,
        "msl": msl_val,
        "n_features": len(features),
        "n_base_cat": sum(base_cat_mask),
        "n_fits": 2,
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
    anchors50 = round50_base_anchors()
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
                  f"noint pub={rec['noint_pub']:.6f} prv={rec['noint_prv']:.6f}")
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
                  "noint_pub", "noint_prv", "noint_d_pub", "noint_d_prv"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            out = {k: r.get(k, "") for k in
                   ["dataset", "n_train", "n_object_cols", "l2", "msl",
                    "base_pub", "base_prv", "noint_pub", "noint_prv"]}
            out["noint_d_pub"] = delta(r, CAND, "pub")
            out["noint_d_prv"] = delta(r, CAND, "prv")
            w.writerow(out)

    # ---- REPRODUCTION: base on ALL 16 matches round50 (tol<5e-6) ----
    repro = {}
    repro_ok = True
    repro_available = anchors50 is not None
    by_name = {r["dataset"]: r for r in rows}
    max_abs_dev = 0.0
    for i in range(1, N_DATASETS + 1):
        nm = f"train_{i:02d}"
        r = by_name.get(nm)
        mine = (r.get("base_pub"), r.get("base_prv")) if r else (None, None)
        ref = anchors50.get(nm) if anchors50 else None
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
    L.append("bench_03 round51 — INTERACTION-CONSTRAINT (no_interactions) "
             "single-knob sweep (ALL 16)  [OFFLINE]")
    L.append("=" * 78)
    L.append("")
    L.append("SETUP:")
    L.append("  base == shipped 08 (git HEAD:submissions/08...): HGB, early_stopping,")
    L.append("    cat_mask = object-dtype columns ONLY, interaction_cst UNSET")
    L.append("    (sklearn default None). base column = seed-0, all 16.")
    L.append("  noint == base pipeline with interaction_cst='no_interactions'")
    L.append("    added (HGB constrained to an ADDITIVE model: no feature")
    L.append("    interactions). Applied to ALL 16 datasets (no subgroup gating).")
    L.append("    Everything else byte-identical (random_state=0, max_iter=300,")
    L.append("    early_stopping=True, l2 gate, tiered msl gate, cat_mask,")
    L.append("    max_depth UNSET; ONLY interaction_cst differs).")

    # ---- SWEEP HEADLINE ----
    L.append("")
    L.append("=== HEADLINE (noint vs base == shipped 08, all 16) ===")
    mp = mean_delta(CAND, "pub")
    mv = mean_delta(CAND, "prv")
    wp, lp, tp = wlt(CAND, "pub")
    wv, lv, tv = wlt(CAND, "prv")
    L.append(f"  noint: mean ΔPublic={mp:+.6f} (W/L/T {wp}/{lp}/{tp})  "
             f"mean ΔPrivate={mv:+.6f} (W/L/T {wv}/{lv}/{tv})")

    # ---- REPRODUCTION ----
    L.append("")
    L.append("=== REPRODUCTION CHECK (base on ALL 16 vs round50, tol<5e-6) ===")
    if not repro_available:
        L.append("  round50 results.csv NOT found -> reproduction NOT anchored (FAIL).")
    else:
        for i in range(1, N_DATASETS + 1):
            nm = f"train_{i:02d}"
            rr = repro[nm]
            mp_, mv_ = rr["mine"]
            rp_, rv_ = rr["ref"] if rr["ref"] else (float("nan"), float("nan"))
            L.append(
                f"  {nm}: Public {mp_:.6f} vs r50 {rp_:.6f} "
                f"(|d|={rr['devp']:.2e}, {'YES' if rr['okp'] else 'NO'}); "
                f"Private {mv_:.6f} vs r50 {rv_:.6f} "
                f"(|d|={rr['devv']:.2e}, {'YES' if rr['okv'] else 'NO'})")
        L.append(f"  max |dev| over all 16x2 = {max_abs_dev:.2e}")
        L.append(f"  REPRODUCTION: {'PASS' if repro_ok else 'FAIL'}")

    # ---- PER-DATASET DELTAS ----
    for split, tag in (("pub", "Public"), ("prv", "Private")):
        L.append("")
        L.append(f"=== PER-DATASET ΔAUC ({tag}) — base vs noint ===")
        L.append(f"{'dataset':<10} {'base':>10} {'cand':>10} {'delta':>11}")
        for r in rows:
            b = r.get(f"{BASE}_{split}")
            c = r.get(f"{CAND}_{split}")
            dd = delta(r, CAND, split)
            bstr = f"{b:>10.6f}" if isinstance(b, float) and not math.isnan(b) else f"{'nan':>10}"
            cstr = f"{c:>10.6f}" if isinstance(c, float) and not math.isnan(c) else f"{'nan':>10}"
            dstr = f"{dd:>+11.6f}" if not math.isnan(dd) else f"{'nan':>11}"
            L.append(f"{r['dataset']:<10} {bstr} {cstr} {dstr}")

    # ---- REGRESSIONS / IMPROVEMENTS ----
    L.append("")
    L.append("=== REGRESSIONS (ΔAUC < -1e-6) ===")
    regs_pub = regressions(CAND, "pub")
    regs_prv = regressions(CAND, "prv")
    if not regs_pub and not regs_prv:
        L.append("  noint: NONE on either split.")
    else:
        for n_, d_ in regs_pub:
            L.append(f"  noint Public  {n_}: {d_:+.6f}")
        for n_, d_ in regs_prv:
            L.append(f"  noint Private {n_}: {d_:+.6f}")

    L.append("")
    L.append("=== IMPROVEMENTS (ΔAUC > +1e-6) ===")
    imp_pub = improvements(CAND, "pub")
    imp_prv = improvements(CAND, "prv")
    if not imp_pub and not imp_prv:
        L.append("  noint: NONE on either split.")
    else:
        for n_, d_ in imp_pub:
            L.append(f"  noint Public  {n_}: {d_:+.6f}")
        for n_, d_ in imp_prv:
            L.append(f"  noint Private {n_}: {d_:+.6f}")

    # ---- largest single moves ----
    def extreme(split, sign):
        vals = [(r["dataset"], delta(r, CAND, split)) for r in rows
                if not math.isnan(delta(r, CAND, split))]
        if not vals:
            return None
        return (min(vals, key=lambda x: x[1]) if sign < 0
                else max(vals, key=lambda x: x[1]))

    L.append("")
    L.append("=== LARGEST SINGLE MOVES ===")
    for split, tag in (("pub", "Public"), ("prv", "Private")):
        worst = extreme(split, -1)
        best = extreme(split, +1)
        if worst:
            L.append(f"  {tag} max regression : {worst[0]} {worst[1]:+.6f}")
        if best:
            L.append(f"  {tag} max improvement: {best[0]} {best[1]:+.6f}")

    # ---- ADOPTION / VERDICT ----
    L.append("")
    L.append("=== ADOPTION ANALYSIS ===")
    L.append("ADOPT iff noint has mean ΔPublic > 0 AND mean ΔPrivate > 0 with")
    L.append("  ZERO regression on EITHER split (no dataset ΔAUC < 0 on Public or")
    L.append("  Private). Any regression, or net-negative/negligible mean on either")
    L.append("  split => REJECT.")
    ADOPT_EPS = 1e-5   # negligible-mean guard
    zero_regs = (not regs_pub) and (not regs_prv)
    clean_gain = (mp > ADOPT_EPS and mv > ADOPT_EPS)
    is_adopt = zero_regs and clean_gain
    L.append("")
    L.append(f"  [noint] zero_regressions={'YES' if zero_regs else 'NO'} "
             f"(Public regs={len(regs_pub)}, Private regs={len(regs_prv)})")
    L.append(f"  [noint] mean ΔPublic  = {mp:+.6f}  (clean gain: "
             f"{'YES' if mp > ADOPT_EPS else 'NO'})")
    L.append(f"  [noint] mean ΔPrivate = {mv:+.6f}  (clean gain: "
             f"{'YES' if mv > ADOPT_EPS else 'NO'})")
    L.append(f"  [noint] -> {'ADOPT' if is_adopt else 'REJECT'}")

    L.append("")
    L.append("=== VERDICT ===")
    if is_adopt:
        L.append("  ADOPT: interaction_cst='no_interactions' cleanly improves BOTH "
                 "splits with zero regression.")
    else:
        L.append("  REJECT: interaction_cst='no_interactions' does not beat base-08 "
                 "on BOTH splits with zero regressions. Constraining HGB to an "
                 "additive model changes tree geometry on an already-tuned model "
                 "-> base-08 stays at least as good overall.")

    ship = "ADOPT" if is_adopt else "REJECT"
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
