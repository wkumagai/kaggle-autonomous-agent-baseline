#!/usr/bin/env python3
"""round84 re-audit: judge past accept/reject rounds against the round83 noise-floor ruler.

Pure re-analysis of already-computed CSVs. No model fitting, no replay, no benchmark.

Ruler (from round83 seed-jitter, 16-dataset mean-delta):
  1 sigma_mean  = 0.00130 (Public) / 0.00126 (Private)
  2 sigma_mean  = 0.00260 (Public) / 0.00252 (Private)

Per-arm classification (aggregate, 16-dataset mean delta):
  REAL       : |mean_delta| >= 2 sigma_mean on BOTH splits, SAME direction.
  SUGGESTIVE : 1 sigma_mean <= |mean_delta| < 2 sigma_mean on at least one split (and not REAL).
  NOISE      : |mean_delta| < 1 sigma_mean on both splits.

Per-dataset "signal": |delta| > that dataset's OWN round83 std (pub_std / prv_std),
SAME direction on BOTH splits.
"""

import csv
import math
import os

# --- ruler constants (aggregate mean-delta sigma) ---
SIGMA_PUB = 0.00130
SIGMA_PRV = 0.00126
TWO_SIGMA_PUB = 2 * SIGMA_PUB  # 0.00260
TWO_SIGMA_PRV = 2 * SIGMA_PRV  # 0.00252

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)  # experiments/bench_03


def load_csv(rel):
    path = os.path.join(BENCH, rel)
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def load_round83_std():
    """Return {dataset: (pub_std, prv_std)} from the round83 seed-jitter ruler."""
    rows = load_csv("round83_seed_jitter/results.csv")
    out = {}
    for row in rows:
        out[row["dataset"]] = (float(row["pub_std"]), float(row["prv_std"]))
    return out


def same_direction(a, b):
    """True if a and b are both strictly positive or both strictly negative."""
    return (a > 0 and b > 0) or (a < 0 and b < 0)


def classify_arm(md_pub, md_prv):
    ap, av = abs(md_pub), abs(md_prv)
    real = (ap >= TWO_SIGMA_PUB and av >= TWO_SIGMA_PRV and same_direction(md_pub, md_prv))
    if real:
        return "REAL"
    suggestive = (ap >= SIGMA_PUB) or (av >= SIGMA_PRV)
    if suggestive:
        return "SUGGESTIVE"
    return "NOISE"


def analyze_arm(rows, arm, base_pub_col, base_prv_col, std_map):
    """Compute mean deltas, z-scores, per-dataset deltas, and signal count for one arm."""
    deltas = []  # (dataset, dpub, dprv, own_pub_std, own_prv_std, is_signal)
    sum_pub = 0.0
    sum_prv = 0.0
    n = 0
    signal = 0
    for row in rows:
        ds = row["dataset"]
        dpub = float(row[f"{arm}_pub"]) - float(row[base_pub_col])
        dprv = float(row[f"{arm}_prv"]) - float(row[base_prv_col])
        ps, vs = std_map.get(ds, (float("nan"), float("nan")))
        is_sig = (abs(dpub) > ps and abs(dprv) > vs and same_direction(dpub, dprv))
        if is_sig:
            signal += 1
        deltas.append((ds, dpub, dprv, ps, vs, is_sig))
        sum_pub += dpub
        sum_prv += dprv
        n += 1
    md_pub = sum_pub / n
    md_prv = sum_prv / n
    z_pub = md_pub / SIGMA_PUB
    z_prv = md_prv / SIGMA_PRV
    label = classify_arm(md_pub, md_prv)
    return {
        "arm": arm,
        "md_pub": md_pub,
        "md_prv": md_prv,
        "z_pub": z_pub,
        "z_prv": z_prv,
        "label": label,
        "signal": signal,
        "n": n,
        "deltas": deltas,
    }


def print_arm_detail(res):
    print(f"  arm {res['arm']:<8} "
          f"mean_delta_pub={res['md_pub']:+.6f} (z={res['z_pub']:+.2f})  "
          f"mean_delta_prv={res['md_prv']:+.6f} (z={res['z_prv']:+.2f})  "
          f"=> {res['label']:<10} signal_datasets={res['signal']}/{res['n']}")
    if res["signal"] > 0:
        for ds, dpub, dprv, ps, vs, sig in res["deltas"]:
            if sig:
                print(f"      * {ds}: dpub={dpub:+.5f} (>{ps:.5f})  "
                      f"dprv={dprv:+.5f} (>{vs:.5f})")


def rank_best(arm_results):
    """Best arm = largest combined |z| (|z_pub| + |z_prv|)."""
    return max(arm_results, key=lambda r: abs(r["z_pub"]) + abs(r["z_prv"]))


def main():
    std_map = load_round83_std()

    print("=" * 78)
    print("round84 re-audit  --  past rounds vs round83 noise-floor ruler")
    print("=" * 78)
    print(f"aggregate mean-delta ruler:  1 sigma = {SIGMA_PUB:.5f} pub / {SIGMA_PRV:.5f} prv")
    print(f"                             2 sigma = {TWO_SIGMA_PUB:.5f} pub / {TWO_SIGMA_PRV:.5f} prv")
    print(f"round83 per-dataset std loaded for {len(std_map)} datasets")
    print()

    rounds = [
        ("round80", "round80_tiny_no_es/results.csv", "base_pub", "base_prv",
         ["F50", "F100", "F200"], "REJECT"),
        ("round81", "round81_l2_gate_threshold/results.csv", "base_pub", "base_prv",
         ["T008", "T005", "T002"], "REJECT"),
        ("round82", "round82_no_cat_mask/results.csv", "base_pub", "base_prv",
         ["ORDCAT", "ORD"], "REJECT"),
    ]

    verdict_rows = []  # (round, best_arm, md_pub, z_pub, md_prv, z_prv, label, signal)

    for name, rel, bp, bv, arms, old_verdict in rounds:
        rows = load_csv(rel)
        print("-" * 78)
        print(f"{name}  (past verdict: {old_verdict})  [{len(rows)} datasets]")
        arm_results = [analyze_arm(rows, a, bp, bv, std_map) for a in arms]
        for res in arm_results:
            print_arm_detail(res)
        best = rank_best(arm_results)
        print(f"  --> best arm: {best['arm']} ({best['label']})")
        verdict_rows.append((name, best["arm"], best["md_pub"], best["z_pub"],
                             best["md_prv"], best["z_prv"], best["label"], best["signal"], best["n"]))
        print()

    # ---- round82 ORDCAT identity control self-check ----
    print("-" * 78)
    print("SELF-CHECK: round82 ORDCAT is the mathematical-identity control (expect ~0 / NOISE)")
    r82 = load_csv("round82_no_cat_mask/results.csv")
    ordcat = analyze_arm(r82, "ORDCAT", "base_pub", "base_prv", std_map)
    ok = abs(ordcat["md_pub"]) < 1e-9 and abs(ordcat["md_prv"]) < 1e-9 and ordcat["label"] == "NOISE"
    print(f"  ORDCAT mean_delta_pub={ordcat['md_pub']:+.9f}  mean_delta_prv={ordcat['md_prv']:+.9f}  "
          f"label={ordcat['label']}")
    print(f"  self-check {'PASS' if ok else 'FAIL'}: identity control is ~0 and NOISE")
    print()

    # ---- 08 accept re-audit (special hardcoded one-dataset case) ----
    print("=" * 78)
    print("08 ACCEPT RE-AUDIT (07 -> 08: msl=70 tier for ratio>=0.030, fires ONLY on train_15)")
    print("=" * 78)
    T15_DPUB = 0.0039
    T15_DPRV = 0.0040
    t15_ps, t15_vs = std_map["train_15"]
    print(f"  recorded accept delta on train_15: +{T15_DPUB:.4f} Public / +{T15_DPRV:.4f} Private")
    print(f"  train_15 OWN seed-noise sigma (round83): pub_std={t15_ps:.6f}  prv_std={t15_vs:.6f}")
    pub_above = T15_DPUB > t15_ps
    prv_above = T15_DPRV > t15_vs
    print(f"    Public : {T15_DPUB:.4f} {'>' if pub_above else '<='} {t15_ps:.6f}  "
          f"=> {'ABOVE' if pub_above else 'BELOW'} its own 1-sigma noise floor")
    print(f"    Private: {T15_DPRV:.4f} {'>' if prv_above else '<='} {t15_vs:.6f}  "
          f"=> {'ABOVE' if prv_above else 'BELOW'} its own 1-sigma noise floor")
    per_ds_signal = pub_above and prv_above and same_direction(T15_DPUB, T15_DPRV)
    print(f"  per-dataset signal (above own std on BOTH, same dir): {per_ds_signal}")
    # aggregate mean over 16 datasets: only train_15 nonzero
    md08_pub = T15_DPUB / 16
    md08_prv = T15_DPRV / 16
    z08_pub = md08_pub / SIGMA_PUB
    z08_prv = md08_prv / SIGMA_PRV
    label08 = classify_arm(md08_pub, md08_prv)
    print(f"  16-dataset MEAN delta: pub={md08_pub:+.6f} (z={z08_pub:+.2f})  "
          f"prv={md08_prv:+.6f} (z={z08_prv:+.2f})  => {label08}")
    print(f"  CONCLUSION: train_15's +{T15_DPUB:.4f}/+{T15_DPRV:.4f} gain is "
          f"{'ABOVE' if per_ds_signal else 'BELOW'} train_15's own seed-noise sigma "
          f"on {'both splits' if per_ds_signal else 'at least one split'}; "
          f"aggregate mean is {label08}.")
    verdict_rows.append(("08(accept)", "msl70/train_15", md08_pub, z08_pub,
                         md08_prv, z08_prv, label08, 1 if per_ds_signal else 0, 16))
    print()

    # ---- final verdict table ----
    print("=" * 78)
    print("FINAL VERDICT TABLE")
    print("=" * 78)
    hdr = f"{'round':<12} {'best arm':<15} {'mean_dpub (z)':<20} {'mean_dprv (z)':<20} {'label':<11} {'#signal':<8}"
    print(hdr)
    print("-" * len(hdr))
    for (name, arm, mdp, zp, mdv, zv, lab, sig, n) in verdict_rows:
        pub_cell = f"{mdp:+.5f} ({zp:+.2f})"
        prv_cell = f"{mdv:+.5f} ({zv:+.2f})"
        print(f"{name:<12} {arm:<15} {pub_cell:<20} {prv_cell:<20} {lab:<11} {str(sig)+'/'+str(n):<8}")
    print()
    print("NOTE:")
    print("  - round80 F200 is REAL but NEGATIVE (a real regression) -> REJECT was correct.")
    print("  - round81 (T002) is NOISE, round82 (ORD) is SUGGESTIVE-negative -> REJECTs consistent with ruler.")
    print("  - 08's accept (train_15) is BELOW that dataset's own seed-noise sigma and aggregate NOISE;")
    print("    i.e. the +0.0039/+0.0040 'gain' is not distinguishable from seed jitter.")


if __name__ == "__main__":
    main()
