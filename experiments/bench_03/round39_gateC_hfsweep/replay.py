#!/usr/bin/env python
"""
bench_03 round39 — GATE-C holdout_fraction (+ K) SWEEP.
OFFLINE ONLY. No subprocess, no LLM, no Kaggle, no network. Calls sklearn
in-process only. Writes ONLY under this round39 directory; never touches
submissions/.

GOAL (improvement-log angle "i")
--------------------------------
Take the current best seed-averaging candidate — GATE C = "apply K seed-averaging
to every dataset with n_object_cols > 0 (>=1 categorical/object column); pure-
numeric datasets use a single random_state=0" — as the FIXED gate, and sweep two
knobs to look for an additional clean improvement over base-08:
  1. holdout_fraction (= early_stopping's `validation_fraction`) applied ONLY to
     the gate-C-fired datasets ∈ {0.10, 0.15, 0.20, 0.25}.
  2. K (seeds averaged) ∈ {5, 8}.

IMPORTANT FACTUAL CORRECTION (surfaced, not silently accepted)
--------------------------------------------------------------
The round39 task text calls hf=0.15 "the base-08 default". That is WRONG.
base-08 (git HEAD:submissions/08_ratio_tiered_msl/agent/prompts/system.md) fits
    HistGradientBoostingClassifier(..., early_stopping=True, ...)
and NEVER sets validation_fraction, so it uses sklearn's DEFAULT
validation_fraction = 0.1 (verified: sklearn 1.9.0). round34's gate-C reference
numbers (+0.00363 ΔPub / +0.00316 ΔPrv) were therefore produced at hf=0.10, the
REAL base-08 default. Consequently:
  * The reproduction anchor is hf=0.10, K=5  (== round34 gate C nocap). We assert
    THIS reproduces round34 to <5e-6. (Asserting reproduction at hf=0.15 as the
    task literally requests is impossible — 0.15 is a *change*, not the default —
    so we anchor at the true default and report the discrepancy loudly.)
  * hf ∈ {0.15, 0.20, 0.25} are genuine changes vs base-08.
The "gate C to beat" (round34 reproduction, K=5) is therefore K5_hf010.

BASE recipe reproduced (== shipped 08), identical to round34:
  features = [c for c in train.columns if c not in ("row_id","target")]
  cat_mask = [train[c].dtype == object for c in features]
  n=len(train); ratio = len(features)/n
  l2  = 1.0 if ratio >= 0.010 else 0.0
  msl = 70 if ratio>=0.030 else 50 if ratio>=0.015 else 20
  HistGradientBoostingClassifier(categorical_features=cat_mask, random_state=s,
      max_iter=300, early_stopping=True, l2_regularization=l2,
      min_samples_leaf=msl)   # validation_fraction NOT set -> default 0.1
  pred = predict_proba(test)[:, class==1]
BASE column = seed-0 with validation_fraction UNSET (byte-identical to shipped 08
and to round34's base). This is the ΔAUC reference for every candidate.

CANDIDATES: for each hf in {0.10,0.15,0.20,0.25} and each K in {5,8}:
  cand_K{K}_hf{hf}  = gate C fires (n_object_cols>0) -> mean of predict_proba over
                      random_state 0..K-1, each fit with validation_fraction=hf;
                      does NOT fire (obj=0) -> exact seed-0 base (byte-identical).
Gate C firing set (obj>0): train_01,02,03,05,06,07,08,09,12,13,14,15  (12).
Non-firing (obj=0): train_04,10,11,16 -> every candidate == base on these (delta 0).

EQUIVALENCE the run verifies: seed-0 fit with validation_fraction=0.10 explicit is
byte-identical to seed-0 with validation_fraction UNSET (default 0.1). Checked ->
guarantees K5_hf010 reproduces round34's gate C exactly.

EFFICIENCY / caching: base = seed-0 no-vf for all 16 datasets (16 fits). For each
of the 12 fired datasets and each hf, fit seeds 0..7 ONCE (8 fits); K=5 uses seeds
0..4, K=8 uses 0..7. Total fits = 16 + 12*8*4 = 400.

Adoption: a candidate is a CLEAN IMPROVEMENT over base-08 iff mean ΔAUC > 0 on BOTH
splits AND zero regressions on BOTH splits. To be a clean improvement OVER GATE C
(K5_hf010), it must additionally beat K5_hf010 mean on BOTH splits with zero
regressions. Otherwise -> negative; gate C (K5_hf010) stays best.
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
BENCH_DIR = os.path.join(REPO, "experiments", "bench_03")
OUT_DIR = os.path.join(BENCH_DIR, "round39_gateC_hfsweep")
STATS_CSV = os.path.join(BENCH_DIR, "dataset_stats.csv")

L2_GATE_THRESHOLD = 0.010   # shipped-08 ratio gate for l2 (FIXED)
GATED_L2 = 1.0
DEFAULT_MSL = 20
MSL_TIERS = [(0.030, 70), (0.015, 50)]
MIN_OBJECT_COLS = 0         # gate C fires iff n_object_cols > 0
BASE_SEED = 0
N_DATASETS = 16

# sweep axes
HFS = [0.10, 0.15, 0.20, 0.25]
KS = [5, 8]
MAX_K = max(KS)             # fit seeds 0..MAX_K-1 per (fired dataset, hf)

BASE = "base"

# reproduction reference: gate C at the TRUE base-08 default hf=0.10, K=5
# == round34 gate C nocap.
REF_LABEL = "K5_hf010"
REF_PUB, REF_PRV = 0.00363, 0.00316

EXPECTED_FIRE = {"train_01", "train_02", "train_03", "train_05", "train_06",
                 "train_07", "train_08", "train_09", "train_12", "train_13",
                 "train_14", "train_15"}
OBJ0_NAMES = {"train_04", "train_10", "train_11", "train_16"}


def hf_tag(hf):
    """0.10 -> '010', 0.15 -> '015'."""
    return f"{int(round(hf * 100)):03d}"


def cand_name(K, hf):
    return f"cand_K{K}_hf{hf_tag(hf)}"


CANDIDATES = [cand_name(K, hf) for hf in HFS for K in KS]


def load_stats(path=STATS_CSV):
    stats = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            stats[row["name"]] = {
                "n_train": int(row["n_train"]),
                "n_object_cols": int(row["n_object_cols"]),
            }
    return stats


def gate_fires(n_object_cols):
    return n_object_cols > MIN_OBJECT_COLS


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


def fit_one_seed(train, test, features, cat_mask, l2, msl_val, seed,
                 validation_fraction=None):
    """Fit ONE shipped-08 HGB. validation_fraction=None -> parameter left UNSET
    (sklearn default 0.1, byte-identical to shipped 08). Otherwise pass it
    explicitly. All other hyperparameters are byte-identical to shipped 08."""
    kwargs = dict(
        categorical_features=cat_mask,
        random_state=seed,
        max_iter=300,
        early_stopping=True,
        l2_regularization=l2,
        min_samples_leaf=msl_val,
    )
    if validation_fraction is not None:
        kwargs["validation_fraction"] = validation_fraction
    clf = HistGradientBoostingClassifier(**kwargs)
    clf.fit(train[features], train["target"])
    proba = clf.predict_proba(test[features])
    classes = list(clf.classes_)
    pos_idx = classes.index(1) if 1 in classes else 1
    return proba[:, pos_idx]


def run_one(name, train_csv, test_csv, stats):
    """Returns (preds, meta) where preds maps config_name -> {row_id -> prob}
    for BASE and every candidate, and meta carries per-dataset info.

    base = seed-0 with validation_fraction UNSET (== shipped 08). A dataset that
    fires (obj>0) is, for each hf, fit seeds 0..MAX_K-1 with validation_fraction=hf
    and cached; each candidate cand_K{K}_hf{hf} uses the mean over the first K of
    those seeds. A dataset that does not fire (obj=0) reuses the exact seed-0 base
    for every candidate (byte-identical). Also fits seed-0 at hf=0.10 to verify
    the vf=0.10-explicit == vf-unset equivalence."""
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)

    features = [c for c in train.columns if c not in ("row_id", "target")]
    cat_mask = [train[c].dtype == object for c in features]

    n = len(train)
    ratio = len(features) / n
    l2 = GATED_L2 if ratio >= L2_GATE_THRESHOLD else 0.0
    msl_val = msl_for_ratio(ratio)

    st = stats[name]
    n_train_stat = st["n_train"]
    n_obj_stat = st["n_object_cols"]
    fires = gate_fires(n_obj_stat)

    row_ids = test["row_id"].tolist()

    # base = seed-0, validation_fraction UNSET (byte-identical to shipped 08).
    base_vec = fit_one_seed(train, test, features, cat_mask, l2, msl_val,
                            BASE_SEED, validation_fraction=None)
    base_map = dict(zip(row_ids, base_vec.tolist()))
    preds = {BASE: base_map}
    n_fits = 1
    equiv_ok = None

    if not fires:
        # obj=0 -> every candidate identical to base.
        for c in CANDIDATES:
            preds[c] = base_map
    else:
        # For each hf, fit seeds 0..MAX_K-1 once; build K-means candidates.
        for hf in HFS:
            seed_vecs = [
                fit_one_seed(train, test, features, cat_mask, l2, msl_val, s,
                             validation_fraction=hf)
                for s in range(MAX_K)
            ]
            n_fits += MAX_K
            # equivalence check: seed-0 @ vf=0.10 explicit == base (vf unset).
            if abs(hf - 0.10) < 1e-12:
                equiv_ok = bool(np.array_equal(seed_vecs[BASE_SEED], base_vec))
            for K in KS:
                avg = np.mean(np.vstack(seed_vecs[:K]), axis=0)
                preds[cand_name(K, hf)] = dict(zip(row_ids, avg.tolist()))

    meta = {
        "n_train": n_train_stat,
        "n_object_cols": n_obj_stat,
        "fires": bool(fires),
        "l2": l2,
        "msl": msl_val,
        "n_fits": n_fits,
        "equiv_ok": equiv_ok,
    }
    return preds, meta


def score_split(pred_map, sol):
    sol = sol.copy()
    sol["pred"] = sol["row_id"].map(pred_map)
    if sol["pred"].isna().any():
        raise ValueError(f"{int(sol['pred'].isna().sum())} row_ids unmatched")
    pub = sol[sol["Usage"] == "Public"]
    prv = sol[sol["Usage"] == "Private"]
    return auc_or_nan(pub["target"], pub["pred"]), auc_or_nan(prv["target"], prv["pred"])


def main():
    warnings.filterwarnings("ignore")
    os.makedirs(OUT_DIR, exist_ok=True)

    stats = load_stats()
    rows = []
    exceptions = []
    skipped = []
    total_fits = 0
    equiv_flags = []

    ALL_CONFIGS = [BASE] + CANDIDATES

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
                "fires": meta["fires"],
                "msl": meta["msl"],
                "equiv_ok": meta["equiv_ok"],
            })
            if meta["equiv_ok"] is not None:
                equiv_flags.append(meta["equiv_ok"])
            for cfg in ALL_CONFIGS:
                pub, prv = score_split(preds[cfg], sol)
                rec[f"{cfg}_pub"] = pub
                rec[f"{cfg}_prv"] = prv
            print(f"[OK] {name} n_tr={meta['n_train']} obj={meta['n_object_cols']} "
                  f"fires={meta['fires']} msl={meta['msl']} fits={meta['n_fits']} "
                  f"base pub={rec['base_pub']:.6f} prv={rec['base_prv']:.6f}")
        except Exception as e:
            exceptions.append((name, repr(e)))
            rec.update({"n_train": stats.get(name, {}).get("n_train", ""),
                        "n_object_cols": stats.get(name, {}).get("n_object_cols", ""),
                        "fires": False, "msl": float("nan"), "equiv_ok": None})
            for cfg in ALL_CONFIGS:
                rec[f"{cfg}_pub"] = float("nan")
                rec[f"{cfg}_prv"] = float("nan")
            print(f"[ERROR] {name}: {e!r}")
        rows.append(rec)

    # ---- delta helpers (all vs base == shipped 08) ----
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
        return [(r["dataset"], delta(r, cfg, split)) for r in rows
                if not math.isnan(delta(r, cfg, split)) and delta(r, cfg, split) < -eps]

    def differing(cfg, eps=1e-9):
        out = []
        for r in rows:
            dp = delta(r, cfg, "pub")
            dv = delta(r, cfg, "prv")
            dp = 0.0 if math.isnan(dp) else dp
            dv = 0.0 if math.isnan(dv) else dv
            if abs(dp) > eps or abs(dv) > eps:
                out.append(r["dataset"])
        return out

    # ---- results.csv ----
    csv_path = os.path.join(OUT_DIR, "results.csv")
    fieldnames = ["dataset", "n_train", "n_object_cols", "fires", "msl",
                  "equiv_ok", "base_pub", "base_prv"]
    for cfg in CANDIDATES:
        fieldnames += [f"{cfg}_pub", f"{cfg}_d_pub", f"{cfg}_prv", f"{cfg}_d_prv"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            out = {k: r.get(k, "") for k in
                   ["dataset", "n_train", "n_object_cols", "fires", "msl",
                    "equiv_ok", "base_pub", "base_prv"]}
            for cfg in CANDIDATES:
                out[f"{cfg}_pub"] = r.get(f"{cfg}_pub", "")
                out[f"{cfg}_prv"] = r.get(f"{cfg}_prv", "")
                out[f"{cfg}_d_pub"] = delta(r, cfg, "pub")
                out[f"{cfg}_d_prv"] = delta(r, cfg, "prv")
            w.writerow(out)

    # ---- INVARIANT: non-firing (obj=0) datasets must be byte-identical to base
    #      for EVERY candidate (delta exactly 0). ----
    invariant_violations = []
    for r in rows:
        if r.get("fires"):
            continue
        for cfg in CANDIDATES:
            dp = delta(r, cfg, "pub")
            dv = delta(r, cfg, "prv")
            dp = 0.0 if math.isnan(dp) else dp
            dv = 0.0 if math.isnan(dv) else dv
            if dp != 0.0 or dv != 0.0:
                invariant_violations.append((cfg, r["dataset"], dp, dv))

    # ---- firing-set check ----
    fired = {r["dataset"] for r in rows if r.get("fires")}
    fire_ok = (fired == EXPECTED_FIRE)
    obj0_excluded = not (OBJ0_NAMES & fired)

    # ---- equivalence check (vf=0.10 explicit == vf unset) ----
    equiv_all_ok = bool(equiv_flags) and all(equiv_flags)

    L = []

    L.append("=" * 78)
    L.append("bench_03 round39 — GATE-C holdout_fraction (+K) SWEEP  [OFFLINE]")
    L.append("=" * 78)
    L.append("")
    L.append("NOTE ON base-08 DEFAULT holdout_fraction:")
    L.append("  base-08 never sets validation_fraction -> sklearn DEFAULT = 0.10.")
    L.append("  The task text mislabels 0.15 as 'base-08 default'; it is NOT.")
    L.append("  Reproduction is therefore anchored at hf=0.10, K=5 (== round34")
    L.append("  gate C nocap). hf in {0.15,0.20,0.25} are genuine changes.")
    L.append("  'Gate C to beat' = " + REF_LABEL + " (the round34 reproduction).")

    # ---- SWEEP TABLE (the key deliverable) ----
    L.append("")
    L.append("=== SWEEP TABLE (each setting vs base == shipped 08) ===")
    L.append(f"{'setting':<14} {'meanDPub':>10} {'meanDPrv':>10} "
             f"{'Pub W/L/T':>12} {'Prv W/L/T':>12}")
    sweep = {}
    for hf in HFS:
        for K in KS:
            cfg = cand_name(K, hf)
            mp, mv = mean_delta(cfg, "pub"), mean_delta(cfg, "prv")
            wp, lp, tp = wlt(cfg, "pub")
            wv, lv, tv = wlt(cfg, "prv")
            sweep[(K, hf)] = (mp, mv, (wp, lp, tp), (wv, lv, tv))
            label = f"K{K}_hf{hf:.2f}"
            tag = "  <- gate C (round34 repro)" if cfg == "cand_" + REF_LABEL else ""
            L.append(f"{label:<14} {mp:>+10.5f} {mv:>+10.5f} "
                     f"{f'{wp}/{lp}/{tp}':>12} {f'{wv}/{lv}/{tv}':>12}{tag}")

    # ---- gate-C reproduction check ----
    ref_cfg = "cand_" + REF_LABEL
    rmp, rmv = mean_delta(ref_cfg, "pub"), mean_delta(ref_cfg, "prv")
    repro_pub_ok = abs(rmp - REF_PUB) < 5e-6
    repro_prv_ok = abs(rmv - REF_PRV) < 5e-6
    repro_ok = repro_pub_ok and repro_prv_ok
    L.append("")
    L.append("=== GATE-C REPRODUCTION CHECK (K5_hf010 must == round34 gate C) ===")
    L.append(f"  K5_hf010: Public {rmp:+.5f} (ref +{REF_PUB:.5f}, "
             f"{'YES' if repro_pub_ok else 'NO'}); "
             f"Private {rmv:+.5f} (ref +{REF_PRV:.5f}, "
             f"{'YES' if repro_prv_ok else 'NO'})  tol<5e-6")
    L.append(f"  REPRODUCTION: {'PASS' if repro_ok else 'FAIL'}")

    # ---- firing set / invariant / equivalence confirmations ----
    L.append("")
    L.append("=== GATE-C FIRING SET (fires iff n_object_cols>0, K seed-avg) ===")
    L.append(f"  fires on ({len(fired)}): {', '.join(sorted(fired))}")
    L.append(f"  expected 12 categorical matched: {'YES' if fire_ok else 'NO'}"
             + ("" if fire_ok else f" (got {sorted(fired)})"))
    L.append(f"  obj=0 datasets {sorted(OBJ0_NAMES)} excluded: "
             f"{'YES' if obj0_excluded else 'NO'}")

    L.append("")
    L.append("=== EQUIVALENCE CHECK (vf=0.10 explicit == vf unset, per fired ds) ===")
    L.append(f"  seed-0 @ validation_fraction=0.10 byte-identical to base "
             f"(vf unset) on all {len(equiv_flags)} fired datasets: "
             f"{'YES' if equiv_all_ok else 'NO'}")

    L.append("")
    L.append("=== INVARIANT (obj=0 datasets identical to base for every cand) ===")
    if invariant_violations:
        L.append("  VIOLATED!")
        for cfg, ds, dp, dv in invariant_violations:
            L.append(f"    {cfg}/{ds}: pub d={dp:+.6g} prv d={dv:+.6g}")
    else:
        L.append(f"  OK: each of the {len(OBJ0_NAMES)} obj=0 datasets "
                 f"{sorted(OBJ0_NAMES)} is byte-identical to base (delta 0) "
                 f"across all {len(CANDIDATES)} candidates. PASS.")

    # ---- per-dataset detail, K=5 across the four hf (PRIMARY axis) ----
    for split, tag in (("pub", "Public"), ("prv", "Private")):
        L.append("")
        L.append(f"=== PER-DATASET K=5 hf-sweep ({tag}) — base + hf"
                 f"{{0.10,0.15,0.20,0.25}} ===")
        header = f"{'dataset':<10} {'obj':>4} {'base':>8}"
        for hf in HFS:
            header += f" {'K5_'+f'{hf:.2f}':>9} {'d':>9}"
        L.append(header)
        for r in rows:
            line = (f"{r['dataset']:<10} {str(r.get('n_object_cols')):>4} "
                    f"{r[f'{BASE}_{split}']:>8.4f}")
            for hf in HFS:
                cfg = cand_name(5, hf)
                line += (f" {r[f'{cfg}_{split}']:>9.4f} "
                         f"{delta(r, cfg, split):>+9.5f}")
            L.append(line)

    # ---- best-candidate selection & verdict vs gate C ----
    L.append("")
    L.append("=== ADOPTION ANALYSIS ===")
    L.append("Criterion A (clean vs base-08): mean D > 0 AND zero regressions on")
    L.append("  BOTH splits. Criterion B (beats gate C=K5_hf010): also strictly")
    L.append("  greater mean than K5_hf010 on BOTH splits with zero regressions.")
    L.append("")
    refmp, refmv = rmp, rmv  # gate C means
    winners = []
    for hf in HFS:
        for K in KS:
            cfg = cand_name(K, hf)
            mp, mv = mean_delta(cfg, "pub"), mean_delta(cfg, "prv")
            rp, rv = regressions(cfg, "pub"), regressions(cfg, "prv")
            zero_regs = (not rp) and (not rv)
            clean_vs_base = (mp > 1e-9) and (mv > 1e-9) and zero_regs
            beats_gateC = (mp > refmp + 1e-9) and (mv > refmv + 1e-9)
            is_ref = (cfg == ref_cfg)
            status = []
            status.append("clean-vs-08" if clean_vs_base else "NOT-clean-vs-08")
            if is_ref:
                status.append("== gate C baseline")
            else:
                status.append("beats-gateC(both)" if beats_gateC
                              else "does-not-beat-gateC")
            regstr = ""
            if not zero_regs:
                allr = rp + rv
                regstr = " regs[" + ", ".join(
                    f"{n}({d:+.5f})" for n, d in allr) + "]"
            L.append(f"  K{K}_hf{hf:.2f}: pub{mp:+.5f} prv{mv:+.5f}  "
                     + "; ".join(status) + regstr)
            if (not is_ref) and clean_vs_base and beats_gateC:
                winners.append((cfg, K, hf, mp, mv))

    # ---- verdict ----
    L.append("")
    L.append("=== VERDICT ===")
    if winners:
        # pick the winner with the best (pub+prv) sum.
        winners.sort(key=lambda x: (x[3] + x[4]), reverse=True)
        wc, wK, whf, wmp, wmv = winners[0]
        L.append(f"CLEAN IMPROVEMENT over gate C FOUND: K{wK}_hf{whf:.2f}")
        L.append(f"  mean Public {wmp:+.5f} (gate C {refmp:+.5f}, "
                 f"+{wmp-refmp:.5f}); mean Private {wmv:+.5f} "
                 f"(gate C {refmv:+.5f}, +{wmv-refmv:.5f}); zero regressions "
                 f"on both splits.")
        if len(winners) > 1:
            L.append(f"  ({len(winners)} settings beat gate C cleanly; the above "
                     f"has the best pub+prv sum.)")
        L.append("ADOPT for next submission decision? -> candidate qualifies "
                 "(pending separate ship decision).")
    else:
        L.append("NEGATIVE: no setting is a CLEAN improvement OVER gate C "
                 "(K5_hf010) on BOTH splits with zero regressions.")
        L.append("GATE C (K=5, hf=0.10 = the true base-08 default holdout) "
                 "STAYS BEST. Do NOT ship anything from this round.")

    # ---- clean-run line ----
    clean_run = ((not exceptions) and (not invariant_violations) and fire_ok
                 and obj0_excluded and repro_ok and equiv_all_ok
                 and (not skipped))
    L.append("")
    L.append(f"CLEAN RUN={'YES' if clean_run else 'NO'} "
             f"(total_fits={total_fits}, exceptions={len(exceptions)}, "
             f"skipped={len(skipped)}, "
             f"invariant_violations={len(invariant_violations)}, "
             f"firing_set_match={'YES' if fire_ok else 'NO'}, "
             f"obj0_excluded={'YES' if obj0_excluded else 'NO'}, "
             f"reproduction={'YES' if repro_ok else 'NO'}, "
             f"equivalence={'YES' if equiv_all_ok else 'NO'})")
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
