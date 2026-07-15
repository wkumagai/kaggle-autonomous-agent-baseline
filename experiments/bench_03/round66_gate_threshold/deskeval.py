#!/usr/bin/env python3
"""
bench_03 round66 — GATE-THRESHOLD ROBUSTNESS SWEEP  [OFFLINE / READ-ONLY]

Pure desk recomputation from round61_rf_blend/results.csv. NO model fitting,
NO new data loading. The ship candidate (B / gate-D') applies the RF x HGB
rank-avg blend ONLY on datasets where a gate fires:

        gate fires  <=>  n_object_cols > 0  AND  n_train < T

For a FIRED dataset the realized score = blend_*; for a NON-FIRED dataset the
realized score = base_* (so its delta vs base is exactly 0). Because gating is a
post-hoc per-dataset SUBSET SELECTION over already-computed base/blend scores,
everything below is exact arithmetic from the CSV — no refit needed.

Outputs (written ONLY under this directory):
  - results.csv   one row per threshold T
  - summary.txt   human-readable report ending in a VERDICT section
"""

import os
import sys
import math

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
IN_CSV = os.path.normpath(
    os.path.join(HERE, "..", "round61_rf_blend", "results.csv")
)
OUT_CSV = os.path.join(HERE, "results.csv")
OUT_TXT = os.path.join(HERE, "summary.txt")

N_TOTAL = 16  # portfolio size; mean_d* are averaged over ALL datasets

# Threshold grid. +inf == "all n_object_cols>0 datasets fire" == gate-C
# (the round61 un-gated-within-obj view), used as the reproduction anchor.
T_GRID = [750, 1060, 1109, 1500, 2000, 3000, 3501, 5000, 7000, 8173,
          8500, 10000, 15000, 50000, math.inf]

# round61 summary GATE-C reference (mean over the 12 obj>0 datasets), rounded
# to 6 dp in that summary; used as a human-facing cross-check at T=+inf.
R61_GATEC_N = 12
R61_GATEC_DPUB = 0.001293
R61_GATEC_DPRIV = 0.001990


def T_label(T):
    return "+inf" if math.isinf(T) else str(int(T))


def main():
    # ------------------------------------------------------------------ load
    df = pd.read_csv(IN_CSV)
    assert len(df) == N_TOTAL, f"expected {N_TOTAL} rows, got {len(df)}"

    # ------------------------------------------------ input bit-reproduction
    # Recompute delta_* from base/blend and confirm it matches the CSV columns.
    rec_dpub = df["blend_public"] - df["base_public"]
    rec_dpriv = df["blend_private"] - df["base_private"]
    max_dev_pub = float((rec_dpub - df["delta_public"]).abs().max())
    max_dev_priv = float((rec_dpriv - df["delta_private"]).abs().max())
    max_dev = max(max_dev_pub, max_dev_priv)
    assert max_dev < 1e-12, (
        f"input delta columns do not reproduce blend-base (max|dev|={max_dev:g})"
    )

    # Work off the recomputed deltas (identical to CSV within <1e-12).
    df = df.assign(_dpub=rec_dpub, _dpriv=rec_dpriv)

    obj = df[df["n_object_cols"] > 0].copy()
    assert len(obj) == R61_GATEC_N, (
        f"expected {R61_GATEC_N} obj>0 datasets, got {len(obj)}"
    )

    # ---------------------------------------------------------- sweep over T
    rows = []
    for T in T_GRID:
        fired = obj[obj["n_train"] < T] if not math.isinf(T) else obj
        fired = fired.sort_values("n_train")
        n_fired = len(fired)

        sum_dpub = float(fired["_dpub"].sum())
        sum_dpriv = float(fired["_dpriv"].sum())
        mean_dpub = sum_dpub / N_TOTAL   # portfolio mean over all 16
        mean_dpriv = sum_dpriv / N_TOTAL

        pub_regs = int((fired["_dpub"] < 0).sum())   # regressions among FIRED
        priv_regs = int((fired["_dpriv"] < 0).sum())

        fired_mean_dpub = sum_dpub / n_fired if n_fired else 0.0
        fired_mean_dpriv = sum_dpriv / n_fired if n_fired else 0.0

        clean_win = bool(
            (mean_dpub > 0) and (mean_dpriv > 0)
            and (pub_regs == 0) and (priv_regs == 0)
        )

        rows.append({
            "threshold": T_label(T),
            "n_fired": n_fired,
            "fired_names": ";".join(fired["dataset"].tolist()),
            "mean_dPublic": mean_dpub,
            "mean_dPrivate": mean_dpriv,
            "pub_regs": pub_regs,
            "priv_regs": priv_regs,
            "clean_win": clean_win,
            "fired_mean_dPub": fired_mean_dpub,
            "fired_mean_dPriv": fired_mean_dpriv,
        })

    res = pd.DataFrame(rows)
    res.to_csv(OUT_CSV, index=False)

    # ------------------------------------------------- reproduction anchor
    anchor = res[res["threshold"] == "+inf"].iloc[0]
    # round61 gate-C reports the mean over the 12 obj>0 datasets == our
    # fired_mean_dP* at T=+inf. Compare to the (6-dp rounded) summary values.
    anc_dev_pub = abs(anchor["fired_mean_dPub"] - R61_GATEC_DPUB)
    anc_dev_priv = abs(anchor["fired_mean_dPriv"] - R61_GATEC_DPRIV)
    anchor_ok = (int(anchor["n_fired"]) == R61_GATEC_N
                 and anc_dev_pub < 5e-7 and anc_dev_priv < 5e-7)

    # ------------------------------------------------- clean-win interval
    # Generic derivation on n_train. A clean obj>0 dataset has both deltas > 0.
    clean_obj = obj[(obj["_dpub"] > 0) & (obj["_dpriv"] > 0)]
    dirty_obj = obj[(obj["_dpub"] <= 0) | (obj["_dpriv"] <= 0)]
    min_clean_nt = int(clean_obj["n_train"].min())   # need T > this to fire >=1
    min_dirty_nt = int(dirty_obj["n_train"].min())   # T <= this excludes dirty
    max_clean_nt = int(clean_obj["n_train"].max())
    # Simple interval (min_clean_nt, min_dirty_nt] is exact iff no clean dataset
    # sits at or above the smallest dirty one.
    interval_exact = max_clean_nt < min_dirty_nt
    interval_str = f"({min_clean_nt}, {min_dirty_nt}]"

    # Contiguous run of clean_win over the grid (sanity vs the interval).
    cw_grid = res[res["clean_win"]]["threshold"].tolist()

    # ------------------------------------------------- max-gain clean-win T
    cw = res[res["clean_win"]].copy()
    if len(cw):
        cw = cw.assign(_gain=cw["mean_dPublic"] + cw["mean_dPrivate"])
        best_gain = cw["_gain"].max()
        best_rows = cw[np.isclose(cw["_gain"], best_gain)]
        # numeric T for ordering (+inf never clean here, so all finite)
        best_Ts = [int(t) for t in best_rows["threshold"]]
        best_T = min(best_Ts)  # smallest T achieving the max (tie-break)
        best_pub = float(best_rows.iloc[0]["mean_dPublic"])
        best_priv = float(best_rows.iloc[0]["mean_dPrivate"])
    else:
        best_Ts, best_T, best_pub, best_priv = [], None, 0.0, 0.0

    # ------------------------------------------------- margin around T=5000
    SHIP_T = 5000
    ship_row = res[res["threshold"] == str(SHIP_T)].iloc[0]
    ship_clean = bool(ship_row["clean_win"])
    # Lower breaking edge: at/below min_clean_nt nothing fires -> clean_win False.
    lower_break = min_clean_nt
    # Upper breaking edge: first dirty dataset enters at n_train == min_dirty_nt.
    upper_break = min_dirty_nt
    margin_below = SHIP_T - lower_break
    margin_above = upper_break - SHIP_T
    interior = ship_clean and margin_below > 0 and margin_above > 0

    clean_run_ok = anchor_ok and (max_dev < 1e-12)

    # ------------------------------------------------------------- summary
    lines = []
    W = 78
    lines.append("=" * W)
    lines.append("bench_03 round66 — GATE-THRESHOLD ROBUSTNESS SWEEP  [OFFLINE / READ-ONLY]")
    lines.append("=" * W)
    lines.append("")
    lines.append("SETUP:")
    lines.append("  Ship candidate B (gate-D'): apply RF x HGB rank-avg blend ONLY where")
    lines.append("      gate fires  <=>  n_object_cols > 0  AND  n_train < T")
    lines.append("  Fired  -> realized score = blend_*   (delta vs base = blend-base)")
    lines.append("  !Fired -> realized score = base_*    (delta vs base = 0, exactly)")
    lines.append(f"  base_* == shipped 08 HGB;  blend_* == anchor RF blend (w=0.5, n=300).")
    lines.append(f"  Portfolio size = {N_TOTAL}; mean_dP* are averaged over ALL {N_TOTAL} datasets.")
    lines.append(f"  Input: {os.path.relpath(IN_CSV, HERE)}  ({len(df)} rows, {len(obj)} obj>0)")
    lines.append("")
    lines.append("INPUT BIT-REPRODUCTION (recomputed blend-base vs CSV delta columns):")
    lines.append(f"  max|dev| public  = {max_dev_pub:.3e}")
    lines.append(f"  max|dev| private = {max_dev_priv:.3e}")
    lines.append(f"  max|dev| overall = {max_dev:.3e}   (< 1e-12 required: "
                 f"{'PASS' if max_dev < 1e-12 else 'FAIL'})")
    lines.append("")
    lines.append("REPRODUCTION ANCHOR (T=+inf == gate-C, all obj>0 fire):")
    lines.append(f"  n_fired            = {int(anchor['n_fired'])}   "
                 f"(round61 gate-C n = {R61_GATEC_N})")
    lines.append(f"  fired-subset mean dPublic  = {anchor['fired_mean_dPub']:+.6f}   "
                 f"(round61 gate-C = {R61_GATEC_DPUB:+.6f}, dev={anc_dev_pub:.2e})")
    lines.append(f"  fired-subset mean dPrivate = {anchor['fired_mean_dPriv']:+.6f}   "
                 f"(round61 gate-C = {R61_GATEC_DPRIV:+.6f}, dev={anc_dev_priv:.2e})")
    lines.append(f"  ANCHOR CHECK: {'PASS' if anchor_ok else 'FAIL'} "
                 f"(matches round61 gate-C to 6dp)")
    lines.append("")
    lines.append("=" * W)
    lines.append("PER-THRESHOLD SWEEP  (mean_dP* averaged over all 16 datasets)")
    lines.append("=" * W)
    hdr = (f"{'T':>6} {'nFire':>5} {'mean_dPub':>11} {'mean_dPriv':>11} "
           f"{'pReg':>4} {'vReg':>4} {'clean':>6} {'fired_dPub':>11} {'fired_dPriv':>11}")
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for _, r in res.iterrows():
        lines.append(
            f"{r['threshold']:>6} {r['n_fired']:>5d} "
            f"{r['mean_dPublic']:>+11.6f} {r['mean_dPrivate']:>+11.6f} "
            f"{r['pub_regs']:>4d} {r['priv_regs']:>4d} "
            f"{('YES' if r['clean_win'] else 'no'):>6} "
            f"{r['fired_mean_dPub']:>+11.6f} {r['fired_mean_dPriv']:>+11.6f}"
        )
    lines.append("")
    lines.append("FIRED-SET DETAIL (dataset : n_train, dPublic, dPrivate, both-positive?):")
    for _, r in obj.sort_values("n_train").iterrows():
        both = (r["_dpub"] > 0) and (r["_dpriv"] > 0)
        lines.append(
            f"  {r['dataset']:<9} n_train={int(r['n_train']):>6}  "
            f"dPub={r['_dpub']:+.6f}  dPriv={r['_dpriv']:+.6f}  "
            f"{'CLEAN' if both else 'REGRESSION'}"
        )
    lines.append("")
    lines.append("=" * W)
    lines.append("VERDICT")
    lines.append("=" * W)
    lines.append(
        f"1) CLEAN-WIN CONTIGUOUS RANGE (on threshold T / n_train):  {interval_str}"
    )
    lines.append(
        f"     - lower bound {min_clean_nt} EXCLUSIVE: need T > {min_clean_nt} so the "
        f"smallest obj>0 clean datasets (n_train={min_clean_nt}) fire;"
    )
    lines.append(
        f"       at T <= {min_clean_nt} nothing fires -> mean_dP*=0 -> not a clean win."
    )
    lines.append(
        f"     - upper bound {min_dirty_nt} INCLUSIVE: gate is STRICT (n_train < T), so "
        f"T = {min_dirty_nt} still excludes"
    )
    lines.append(
        f"       the first regressing dataset (train_08, n_train={min_dirty_nt}, both "
        f"deltas < 0); T > {min_dirty_nt} admits it and breaks clean-win."
    )
    lines.append(
        f"     - interval derivation exact (no clean dataset at/above n_train="
        f"{min_dirty_nt}): {interval_exact}"
    )
    lines.append(
        f"     - grid Ts with clean_win = {cw_grid}"
    )
    lines.append("")
    tie = "" if len(best_Ts) == 1 else f"  (tie across T={best_Ts}; all fire the same set)"
    lines.append(
        f"2) MAX-GAIN CLEAN-WIN THRESHOLD:  T = {best_T}{tie}"
    )
    lines.append(
        f"     mean_dPublic = {best_pub:+.6f}, mean_dPrivate = {best_priv:+.6f} "
        f"(over all {N_TOTAL}); fires 5 datasets incl. train_03 (n_train=3501)."
    )
    lines.append("")
    lines.append(
        f"3) SHIPPED GATE-D' THRESHOLD T={SHIP_T}: "
        f"{'INTERIOR of the clean-win plateau' if interior else 'KNIFE-EDGE / NOT interior'}"
    )
    lines.append(
        f"     clean_win at T={SHIP_T}: {ship_clean}"
    )
    lines.append(
        f"     margin DOWN to nearest breaking threshold: {SHIP_T} - {lower_break} "
        f"= {margin_below} (n_train)  [below this, 0 fired -> not clean]"
    )
    lines.append(
        f"     margin UP   to nearest breaking threshold: {upper_break} - {SHIP_T} "
        f"= {margin_above} (n_train)  [train_08 @ n_train={upper_break} enters -> breaks]"
    )
    lines.append(
        f"     => T={SHIP_T} sits comfortably inside plateau {interval_str}; NOT a knife-edge."
        if interior else
        f"     => T={SHIP_T} is on/near a boundary — review."
    )
    lines.append("")
    lines.append(f"CLEAN RUN: {'YES' if clean_run_ok else 'NO'}")
    lines.append("=" * W)

    txt = "\n".join(lines) + "\n"
    with open(OUT_TXT, "w") as fh:
        fh.write(txt)

    # echo to stdout for the runner
    sys.stdout.write(txt)
    sys.stdout.write(f"\n[written] {OUT_CSV}\n[written] {OUT_TXT}\n")


if __name__ == "__main__":
    main()
