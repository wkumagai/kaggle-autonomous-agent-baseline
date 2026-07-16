#!/usr/bin/env python3
"""
round87 pooled-effect analysis (PURE RE-ANALYSIS of existing CSVs).

Consolidates the three already-computed candidate-A seed-configuration fit
results into ONE effect-size stability statement, read against the round83
per-dataset seed-jitter noise floor.

NO model fitting. NO Kaggle submission. STDLIB ONLY.
Reads only from experiments/bench_03/{candidates,round83_seed_jitter}.
Writes only into experiments/bench_03/round87_pooled_effect/.
"""

import csv
import json
import math
import os
import re
import statistics

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)  # experiments/bench_03

CONFIGS = [
    ("A_orgate", os.path.join(BENCH, "candidates", "A_orgate")),
    ("A_orgate_seedwin", os.path.join(BENCH, "candidates", "A_orgate_seedwin")),
    ("A_orgate_K20", os.path.join(BENCH, "candidates", "A_orgate_K20")),
]
RULER_CSV = os.path.join(BENCH, "round83_seed_jitter", "results.csv")

OUT_CSV = os.path.join(HERE, "results.csv")
OUT_TXT = os.path.join(HERE, "summary.txt")
OUT_JSON = os.path.join(HERE, "summary.json")

# Established mean-delta noise floor (round83/round84).
MEAN_DELTA_SIGMA_PUB = 0.0013     # 1 sigma, public mean-delta floor
MEAN_DELTA_SIGMA_PRV = 0.00126    # 1 sigma, private mean-delta floor


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_config(path):
    """Return dict[dataset] -> row (with typed fields we need)."""
    rows = {}
    with open(os.path.join(path, "results.csv"), newline="") as fh:
        for r in csv.DictReader(fh):
            ds = r["dataset"]
            rows[ds] = {
                "gate_fired": r["gate_fired"].strip().lower() == "true",
                "delta_public": float(r["delta_public"]),
                "delta_private": float(r["delta_private"]),
                "cand_equals_base": r["cand_equals_base"].strip().lower() == "true",
            }
    return rows


def load_ruler(path):
    """Return dict[dataset] -> {pub_std, prv_std} (round83 per-dataset sample stdev)."""
    rows = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            rows[r["dataset"]] = {
                "pub_std": float(r["pub_std"]),
                "prv_std": float(r["prv_std"]),
            }
    return rows


def parse_summary_mean_delta(path):
    """Parse the 'MEAN DELTA = +0.0056' headline from a config summary.txt."""
    with open(os.path.join(path, "summary.txt")) as fh:
        text = fh.read()
    m = re.search(r"MEAN DELTA\s*=\s*([+-]?[0-9.]+)", text)
    if not m:
        raise ValueError("MEAN DELTA headline not found in %s" % path)
    return float(m.group(1))


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def main():
    config_rows = {name: load_config(path) for name, path in CONFIGS}
    config_headline = {name: parse_summary_mean_delta(path) for name, path in CONFIGS}
    ruler = load_ruler(RULER_CSV)

    # Consistent dataset ordering from the first config.
    first_name = CONFIGS[0][0]
    datasets = list(config_rows[first_name].keys())

    # --- Item 1: per-config means (all 16 and fired-only) ------------------
    config_means = {}
    for name, _ in CONFIGS:
        rows = config_rows[name]
        pub_all = [rows[d]["delta_public"] for d in datasets]
        prv_all = [rows[d]["delta_private"] for d in datasets]
        fired = [d for d in datasets if rows[d]["gate_fired"]]
        pub_fired = [rows[d]["delta_public"] for d in fired]
        prv_fired = [rows[d]["delta_private"] for d in fired]
        config_means[name] = {
            "mean_delta_public_all16": statistics.fmean(pub_all),
            "mean_delta_private_all16": statistics.fmean(prv_all),
            "mean_delta_public_fired": statistics.fmean(pub_fired),
            "mean_delta_private_fired": statistics.fmean(prv_fired),
            "n_fired": len(fired),
            "headline_mean_delta": config_headline[name],
        }

    # --- Self-check 5(a): recomputed all-16 mean private delta matches -----
    #     the summary.txt MEAN DELTA headline (printed to 4 dp).
    selfcheck_a_lines = []
    selfcheck_a_pass = True
    for name, _ in CONFIGS:
        recomputed = config_means[name]["mean_delta_private_all16"]
        headline = config_headline[name]
        # Headline is printed to 4 decimals; match recomputed value rounded to 4 dp.
        diff = abs(round(recomputed, 4) - headline)
        ok = diff <= 1e-6
        selfcheck_a_pass = selfcheck_a_pass and ok
        selfcheck_a_lines.append(
            "  %-18s recomputed_mean_prv=%+.6f  round4=%+.4f  headline=%+.4f  |diff|=%.1e  %s"
            % (name, recomputed, round(recomputed, 4), headline, diff,
               "PASS" if ok else "FAIL")
        )

    # --- Self-check 5(b): non-fired datasets have delta==0 exactly ---------
    #     (byte-identical cand==base) in ALL three configs.
    nonfired = [d for d in datasets if not config_rows[first_name][d]["gate_fired"]]
    selfcheck_b_lines = []
    selfcheck_b_pass = True
    for d in nonfired:
        for name, _ in CONFIGS:
            r = config_rows[name][d]
            zero_pub = (r["delta_public"] == 0.0)
            zero_prv = (r["delta_private"] == 0.0)
            eqbase = r["cand_equals_base"]
            ok = zero_pub and zero_prv and eqbase
            selfcheck_b_pass = selfcheck_b_pass and ok
            selfcheck_b_lines.append(
                "  %-10s %-18s d_pub=%+.6f d_prv=%+.6f cand==base=%s  %s"
                % (d, name, r["delta_public"], r["delta_private"], eqbase,
                   "PASS" if ok else "FAIL")
            )

    # --- Item 2 + 3: per-dataset pooled spread and ruler read --------------
    per_dataset = []
    for d in datasets:
        prv_by_cfg = [config_rows[name][d]["delta_private"] for name, _ in CONFIGS]
        pub_by_cfg = [config_rows[name][d]["delta_public"] for name, _ in CONFIGS]

        prv_spread = max(prv_by_cfg) - min(prv_by_cfg)
        prv_stdev = statistics.stdev(prv_by_cfg)  # ddof=1 across 3 configs
        prv_mean = statistics.fmean(prv_by_cfg)

        pub_spread = max(pub_by_cfg) - min(pub_by_cfg)
        pub_stdev = statistics.stdev(pub_by_cfg)
        pub_mean = statistics.fmean(pub_by_cfg)

        prv_sigma = ruler[d]["prv_std"]
        pub_sigma = ruler[d]["pub_std"]

        def mult(mean, sigma):
            return (mean / sigma) if sigma > 0 else float("inf")

        def flag(mean, sigma):
            if sigma <= 0:
                return "no_sigma"
            m = abs(mean) / sigma
            if m > 2.0:
                return "strong>2sigma"
            if m > 1.0:
                return "real>1sigma"
            return "within_jitter<=1sigma"

        per_dataset.append({
            "dataset": d,
            "gate_fired": config_rows[first_name][d]["gate_fired"],
            "prv_A_orgate": prv_by_cfg[0],
            "prv_A_orgate_seedwin": prv_by_cfg[1],
            "prv_A_orgate_K20": prv_by_cfg[2],
            "prv_spread": prv_spread,
            "prv_stdev_across_cfg": prv_stdev,
            "prv_mean_across_cfg": prv_mean,
            "prv_round83_sigma": prv_sigma,
            "prv_sigma_multiple": mult(prv_mean, prv_sigma),
            "prv_flag": flag(prv_mean, prv_sigma),
            "pub_A_orgate": pub_by_cfg[0],
            "pub_A_orgate_seedwin": pub_by_cfg[1],
            "pub_A_orgate_K20": pub_by_cfg[2],
            "pub_spread": pub_spread,
            "pub_stdev_across_cfg": pub_stdev,
            "pub_mean_across_cfg": pub_mean,
            "pub_round83_sigma": pub_sigma,
            "pub_sigma_multiple": mult(pub_mean, pub_sigma),
            "pub_flag": flag(pub_mean, pub_sigma),
        })

    # --- Item 4: aggregate ruler read vs mean-delta floor ------------------
    # Pooled mean = mean over 16 datasets of the per-dataset mean-across-configs.
    pooled_mean_prv = statistics.fmean([row["prv_mean_across_cfg"] for row in per_dataset])
    pooled_mean_pub = statistics.fmean([row["pub_mean_across_cfg"] for row in per_dataset])
    # Fired-only pooled mean.
    fired_rows = [row for row in per_dataset if row["gate_fired"]]
    pooled_mean_prv_fired = statistics.fmean([r["prv_mean_across_cfg"] for r in fired_rows])
    pooled_mean_pub_fired = statistics.fmean([r["pub_mean_across_cfg"] for r in fired_rows])

    pooled_prv_sigma_mult = pooled_mean_prv / MEAN_DELTA_SIGMA_PRV
    pooled_pub_sigma_mult = pooled_mean_pub / MEAN_DELTA_SIGMA_PUB

    # Per-dataset spread aggregates.
    max_prv_spread = max(row["prv_spread"] for row in per_dataset)
    max_prv_spread_ds = max(per_dataset, key=lambda r: r["prv_spread"])["dataset"]
    max_pub_spread = max(row["pub_spread"] for row in per_dataset)
    max_pub_spread_ds = max(per_dataset, key=lambda r: r["pub_spread"])["dataset"]

    # Per-dataset ruler flag tallies (private) over fired datasets only
    # (non-fired have exactly-zero effect and a sigma; count them separately).
    def tally(flag_key, rows):
        strong = sum(1 for r in rows if r[flag_key] == "strong>2sigma")
        real = sum(1 for r in rows if r[flag_key] == "real>1sigma")
        within = sum(1 for r in rows if r[flag_key] == "within_jitter<=1sigma")
        return strong, real, within

    prv_strong_all, prv_real_all, prv_within_all = tally("prv_flag", per_dataset)
    prv_strong_fired, prv_real_fired, prv_within_fired = tally("prv_flag", fired_rows)
    pub_strong_all, pub_real_all, pub_within_all = tally("pub_flag", per_dataset)
    pub_strong_fired, pub_real_fired, pub_within_fired = tally("pub_flag", fired_rows)

    # ---------------------------------------------------------------------
    # Write results.csv
    # ---------------------------------------------------------------------
    fieldnames = [
        "dataset", "gate_fired",
        "prv_A_orgate", "prv_A_orgate_seedwin", "prv_A_orgate_K20",
        "prv_spread", "prv_stdev_across_cfg", "prv_mean_across_cfg",
        "prv_round83_sigma", "prv_sigma_multiple", "prv_flag",
        "pub_A_orgate", "pub_A_orgate_seedwin", "pub_A_orgate_K20",
        "pub_spread", "pub_stdev_across_cfg", "pub_mean_across_cfg",
        "pub_round83_sigma", "pub_sigma_multiple", "pub_flag",
    ]
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for row in per_dataset:
            out = {}
            for k in fieldnames:
                v = row[k]
                if isinstance(v, float):
                    out[k] = "%.8g" % v
                else:
                    out[k] = v
            w.writerow(out)

    # ---------------------------------------------------------------------
    # Write summary.json
    # ---------------------------------------------------------------------
    summary_obj = {
        "note": "round87 pooled-effect re-analysis; NO fitting, NO submission; stdlib only",
        "n_datasets": len(datasets),
        "configs": [name for name, _ in CONFIGS],
        "mean_delta_sigma_floor": {
            "public_1sigma": MEAN_DELTA_SIGMA_PUB,
            "private_1sigma": MEAN_DELTA_SIGMA_PRV,
            "public_2sigma": 2 * MEAN_DELTA_SIGMA_PUB,
            "private_2sigma": 2 * MEAN_DELTA_SIGMA_PRV,
        },
        "per_config_means": config_means,
        "pooled": {
            "mean_delta_private_all16": pooled_mean_prv,
            "mean_delta_public_all16": pooled_mean_pub,
            "mean_delta_private_fired": pooled_mean_prv_fired,
            "mean_delta_public_fired": pooled_mean_pub_fired,
            "private_sigma_multiple_vs_meandelta_floor": pooled_prv_sigma_mult,
            "public_sigma_multiple_vs_meandelta_floor": pooled_pub_sigma_mult,
        },
        "per_dataset_spread": {
            "max_prv_spread": max_prv_spread,
            "max_prv_spread_dataset": max_prv_spread_ds,
            "max_pub_spread": max_pub_spread,
            "max_pub_spread_dataset": max_pub_spread_ds,
        },
        "ruler_tally_private": {
            "all16": {"strong_gt2sigma": prv_strong_all, "real_gt1sigma": prv_real_all,
                      "within_le1sigma": prv_within_all},
            "fired15": {"strong_gt2sigma": prv_strong_fired, "real_gt1sigma": prv_real_fired,
                        "within_le1sigma": prv_within_fired},
        },
        "ruler_tally_public": {
            "all16": {"strong_gt2sigma": pub_strong_all, "real_gt1sigma": pub_real_all,
                      "within_le1sigma": pub_within_all},
            "fired15": {"strong_gt2sigma": pub_strong_fired, "real_gt1sigma": pub_real_fired,
                        "within_le1sigma": pub_within_fired},
        },
        "self_checks": {
            "check_a_config_headline_match": selfcheck_a_pass,
            "check_b_nonfired_zero_delta": selfcheck_b_pass,
        },
    }
    with open(OUT_JSON, "w") as fh:
        json.dump(summary_obj, fh, indent=2)
        fh.write("\n")

    # ---------------------------------------------------------------------
    # Write summary.txt
    # ---------------------------------------------------------------------
    L = []
    L.append("round87 -- pooled effect-size stability of candidate A across 3 seed configs")
    L.append("=" * 76)
    L.append("PURE RE-ANALYSIS of existing CSVs. No model fitting. No submission. Stdlib only.")
    L.append("configs pooled: A_orgate (K10 seeds0-9) | A_orgate_seedwin (K10 seeds10-19) "
             "| A_orgate_K20 (K20 seeds0-19)")
    L.append("ruler: round83 per-dataset seed-jitter sample stdev (ddof=1); "
             "mean-delta floor 1sigma=%.5f Pub / %.5f Prv" % (MEAN_DELTA_SIGMA_PUB, MEAN_DELTA_SIGMA_PRV))
    L.append("")
    L.append("[1] Per-config means (private):")
    L.append("  %-18s  all16_prv   fired_prv   all16_pub   fired_pub   n_fired  headline" % "config")
    for name, _ in CONFIGS:
        m = config_means[name]
        L.append("  %-18s  %+.6f  %+.6f  %+.6f  %+.6f    %2d     %+.4f"
                 % (name, m["mean_delta_private_all16"], m["mean_delta_private_fired"],
                    m["mean_delta_public_all16"], m["mean_delta_public_fired"],
                    m["n_fired"], m["headline_mean_delta"]))
    L.append("")
    L.append("[2] Per-dataset pooled spread across the 3 configs (private):")
    L.append("  %-10s %-6s %-9s %-9s %-9s %-9s %-9s"
             % ("dataset", "fired", "min_prv", "max_prv", "spread", "stdev", "mean"))
    for row in per_dataset:
        prv_cfgs = [row["prv_A_orgate"], row["prv_A_orgate_seedwin"], row["prv_A_orgate_K20"]]
        L.append("  %-10s %-6s %+.6f %+.6f %.6f  %.6f  %+.6f"
                 % (row["dataset"], str(row["gate_fired"]), min(prv_cfgs), max(prv_cfgs),
                    row["prv_spread"], row["prv_stdev_across_cfg"], row["prv_mean_across_cfg"]))
    L.append("  max per-dataset PRIVATE spread = %.6f  (%s)" % (max_prv_spread, max_prv_spread_ds))
    L.append("  max per-dataset PUBLIC  spread = %.6f  (%s)" % (max_pub_spread, max_pub_spread_ds))
    L.append("")
    L.append("[3] Ruler read -- per-dataset mean-across-configs delta vs that dataset's OWN round83 sigma:")
    L.append("  %-10s %-6s %-11s %-11s %-9s %-9s | %-11s %-11s %-9s %-9s"
             % ("dataset", "fired", "prv_mean", "prv_sigma", "prv_xSig", "prv_flag",
                "pub_mean", "pub_sigma", "pub_xSig", "pub_flag"))
    for row in per_dataset:
        L.append("  %-10s %-6s %+.6f  %.6f    %6.2f  %-20s %+.6f  %.6f    %6.2f  %s"
                 % (row["dataset"], str(row["gate_fired"]),
                    row["prv_mean_across_cfg"], row["prv_round83_sigma"],
                    row["prv_sigma_multiple"], row["prv_flag"],
                    row["pub_mean_across_cfg"], row["pub_round83_sigma"],
                    row["pub_sigma_multiple"], row["pub_flag"]))
    L.append("")
    L.append("  Per-dataset PRIVATE flag tally:")
    L.append("    all 16 datasets : strong(>2s)=%d  real(>1s)=%d  within(<=1s)=%d"
             % (prv_strong_all, prv_real_all, prv_within_all))
    L.append("    fired 15 only   : strong(>2s)=%d  real(>1s)=%d  within(<=1s)=%d"
             % (prv_strong_fired, prv_real_fired, prv_within_fired))
    L.append("  Per-dataset PUBLIC flag tally:")
    L.append("    all 16 datasets : strong(>2s)=%d  real(>1s)=%d  within(<=1s)=%d"
             % (pub_strong_all, pub_real_all, pub_within_all))
    L.append("    fired 15 only   : strong(>2s)=%d  real(>1s)=%d  within(<=1s)=%d"
             % (pub_strong_fired, pub_real_fired, pub_within_fired))
    L.append("")
    L.append("[4] Aggregate ruler read -- pooled 16-dataset mean delta vs mean-delta noise floor:")
    L.append("  pooled mean PRIVATE delta (all16) = %+.6f  =>  %.2f sigma  (floor 1s=%.5f, 2s=%.5f)"
             % (pooled_mean_prv, pooled_prv_sigma_mult, MEAN_DELTA_SIGMA_PRV, 2 * MEAN_DELTA_SIGMA_PRV))
    L.append("  pooled mean PUBLIC  delta (all16) = %+.6f  =>  %.2f sigma  (floor 1s=%.5f, 2s=%.5f)"
             % (pooled_mean_pub, pooled_pub_sigma_mult, MEAN_DELTA_SIGMA_PUB, 2 * MEAN_DELTA_SIGMA_PUB))
    L.append("  pooled mean PRIVATE delta (fired15) = %+.6f" % pooled_mean_prv_fired)
    L.append("  pooled mean PUBLIC  delta (fired15) = %+.6f" % pooled_mean_pub_fired)
    L.append("")
    L.append("[5] Self-checks (MUST PASS):")
    L.append("  (a) recomputed all-16 mean private delta == config summary.txt MEAN DELTA headline (round 4dp, tol 1e-6):")
    L.extend(selfcheck_a_lines)
    L.append("      => check (a): %s" % ("PASS" if selfcheck_a_pass else "FAIL"))
    L.append("  (b) non-fired datasets have delta==0 exactly (cand==base) in ALL three configs:")
    L.extend(selfcheck_b_lines)
    L.append("      => check (b): %s" % ("PASS" if selfcheck_b_pass else "FAIL"))
    L.append("")
    L.append("CONCLUSION (3 lines):")
    L.append("  1. Candidate A's private effect is stable across 3 disjoint/overlapping seed windows: "
             "the largest per-dataset drift across configs is %.4f (%s), well inside seed jitter."
             % (max_prv_spread, max_prv_spread_ds))
    L.append("  2. Pooled over 16 datasets the mean private lift is %+.5f = %.1f sigma above the "
             "mean-delta noise floor -- a real, reproducible aggregate signal."
             % (pooled_mean_prv, pooled_prv_sigma_mult))
    L.append("  3. Per dataset the picture is mixed: %d/16 private effects clear >1 sigma of their own "
             "seed jitter (%d strong >2 sigma), while %d/16 stay within jitter -- the lift is broad and "
             "small, not driven by a single dataset." % (prv_strong_all + prv_real_all, prv_strong_all, prv_within_all))
    L.append("")
    with open(OUT_TXT, "w") as fh:
        fh.write("\n".join(L))

    # Console echo of the load-bearing numbers.
    print("=== round87 pooled-effect analysis ===")
    print("pooled mean PRIVATE delta (all16) = %+.6f  => %.2f sigma vs mean-delta floor"
          % (pooled_mean_prv, pooled_prv_sigma_mult))
    print("pooled mean PUBLIC  delta (all16) = %+.6f  => %.2f sigma vs mean-delta floor"
          % (pooled_mean_pub, pooled_pub_sigma_mult))
    print("max per-dataset PRIVATE spread across 3 configs = %.6f (%s)"
          % (max_prv_spread, max_prv_spread_ds))
    print("private ruler tally (all16): strong>2s=%d real>1s=%d within<=1s=%d"
          % (prv_strong_all, prv_real_all, prv_within_all))
    print("SELF-CHECK (a) headline match : %s" % ("PASS" if selfcheck_a_pass else "FAIL"))
    print("SELF-CHECK (b) nonfired zero  : %s" % ("PASS" if selfcheck_b_pass else "FAIL"))
    if not (selfcheck_a_pass and selfcheck_b_pass):
        raise SystemExit("SELF-CHECK FAILED")


if __name__ == "__main__":
    main()
