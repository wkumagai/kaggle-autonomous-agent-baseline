#!/usr/bin/env python3
"""round85 seed-averaging ruler: how far does K-seed averaging shrink AUC jitter,
and at what K does the residual jitter fall below the round83 noise-floor ruler?

Pure OFFLINE re-analysis of an already-committed CSV. No model fitting, no replay,
no benchmark. Stdlib only (csv, math, statistics, json, itertools).

Candidate "A" is a seed-averaging ensemble: it averages the base model's AUC over
K random_state seeds. round83 gave us, per dataset, the base model's AUC under 10
seeds (0..9). We reuse those 10 numbers as raw material.

For each dataset and each block size K in {1,2,3,4,5}:
  * enumerate ALL C(10,K) distinct K-subsets of the 10 seeds,
  * take the MEAN AUC of each subset (this simulates one run of candidate A at K seeds),
  * sigma_K = population stdev (statistics.pstdev) of those subset-means.
K=1 reproduces the raw per-seed sigma; larger K shrinks it.

Finite-population sampling theory (sampling K of N=10 seeds WITHOUT replacement):
    Var(mean_K) = (sigma_1^2 / K) * (N - K) / (N - 1)
  => sigma_K   = (sigma_1 / sqrt(K)) * sqrt((N - K) / (N - 1))
The naive 1/sqrt(K) law is (sigma_1 / sqrt(K)); the empirical value sits BELOW it by
the finite-population correction factor sqrt((N-K)/(N-1)). We report both.

K=10 is the actually-shipped K, but only 1 subset exists (all 10) so its subset-mean
sigma is exactly 0 and cannot be measured by subsampling. We therefore EXTRAPOLATE the
residual jitter at K=10 as the 1/sqrt(K) prediction sigma_1/sqrt(10) -- an extrapolation,
not a measurement.

Ruler (round83 seed-jitter, 16-dataset MEAN-delta):
  1 sigma_mean = 0.00130 (Public) / 0.00126 (Private)
  2 sigma_mean = 0.00260 (Public) / 0.00252 (Private)
Candidate A must push its residual jitter below this mean-delta ruler to be
distinguishable from seed noise.
"""

import csv
import json
import math
import os
import statistics
from itertools import combinations

# --- round83 aggregate mean-delta ruler (what candidate A must beat) ---
SIGMA_MEAN_PUB = 0.00130
SIGMA_MEAN_PRV = 0.00126
TWO_SIGMA_PUB = 2 * SIGMA_MEAN_PUB  # 0.00260
TWO_SIGMA_PRV = 2 * SIGMA_MEAN_PRV  # 0.00252

N_SEEDS = 10            # seeds 0..9 recorded in round83
K_MEASURED = [1, 2, 3, 4, 5]  # block sizes we enumerate exhaustively
K_SHIP = 10             # the K candidate A actually ships (extrapolated only)

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)  # experiments/bench_03


def load_csv(rel):
    path = os.path.join(BENCH, rel)
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def seed_auc(row, split):
    """Return the 10 per-seed AUCs for one split ('pub' or 'prv') of one dataset."""
    return [float(row[f"seed_{i:02d}_{split}"]) for i in range(N_SEEDS)]


def sigma_of_subset_means(values, k):
    """Population stdev of the means of ALL C(len(values), k) k-subsets.

    k=1 -> pstdev of the raw values (the per-seed sigma).
    """
    means = [sum(sub) / k for sub in combinations(values, k)]
    return statistics.pstdev(means)


def analyze_dataset(row, split):
    """Return dict of sigma_K (empirical) for K in K_MEASURED plus the K=10
    extrapolation and the theory/ratio bookkeeping, for one dataset+split."""
    aucs = seed_auc(row, split)
    sigma1 = sigma_of_subset_means(aucs, 1)  # == population stdev of the 10 seeds
    out = {"sigma1": sigma1, "emp": {}, "theo": {}, "ratio": {}}
    for k in K_MEASURED:
        emp = sigma_of_subset_means(aucs, k)
        theo = sigma1 / math.sqrt(k)                 # naive 1/sqrt(K) law
        out["emp"][k] = emp
        out["theo"][k] = theo
        out["ratio"][k] = (emp / theo) if theo > 0 else float("nan")
    # K=10: extrapolated residual jitter (1/sqrt(K) prediction; NOT measurable).
    out["k10_extrap"] = sigma1 / math.sqrt(K_SHIP)
    return out


def median(xs):
    return statistics.median(xs)


def first_k_below(median_by_k, threshold):
    """Smallest K (ascending) whose median sigma_K < threshold, else None."""
    for k in sorted(median_by_k):
        if median_by_k[k] < threshold:
            return k
    return None


def main():
    rows = load_csv("round83_seed_jitter/results.csv")
    datasets = [r["dataset"] for r in rows]

    # ---- per-dataset analysis, both splits ----
    per_ds = {"pub": {}, "prv": {}}
    for split in ("pub", "prv"):
        for row in rows:
            per_ds[split][row["dataset"]] = analyze_dataset(row, split)

    # =====================================================================
    # SANITY-1: read-integrity check -- our recomputed per-seed stdev must
    # equal the stored pub_std / prv_std column (proves we read the right 10
    # numbers per dataset).
    #
    # NOTE (data fact surfaced during round85): round83 stored the SAMPLE
    # stdev (ddof=1, statistics.stdev), NOT the population stdev. The task
    # brief calls that column "population stdev", but the committed CSV matches
    # sample stdev to <1e-9 on all 16 datasets and differs from pstdev by the
    # ddof factor sqrt(N/(N-1)) = sqrt(10/9) ~= 1.0541. We therefore run the
    # read-integrity check against the SAMPLE stdev (the definition round83
    # actually used) and ALSO report the pstdev-vs-stored gap for transparency.
    #
    # sigma_K itself stays population-based (statistics.pstdev of the subset
    # means) exactly as specified -- that is the correct spread of the finite
    # set of subset-means and it makes the finite-population correction ratio
    # come out to exactly sqrt((N-K)/(N-1)).
    # =====================================================================
    max_diff_samp = 0.0   # stored vs recomputed SAMPLE stdev (the integrity check)
    max_diff_pop = 0.0    # stored vs recomputed POPULATION stdev (== sigma1); for the record
    for split in ("pub", "prv"):
        col = f"{split}_std"
        for row in rows:
            stored = float(row[col])
            aucs = seed_auc(row, split)
            samp = statistics.stdev(aucs)                     # ddof=1
            pop = per_ds[split][row["dataset"]]["sigma1"]     # ddof=0 == pstdev
            max_diff_samp = max(max_diff_samp, abs(samp - stored))
            max_diff_pop = max(max_diff_pop, abs(pop - stored))
    sanity1_pass = max_diff_samp < 1e-9
    max_diff = max_diff_samp  # the reported integrity diff

    # =====================================================================
    # SANITY-2: sigma_K monotonically non-increasing in K, every dataset/split.
    # =====================================================================
    sanity2_pass = True
    sanity2_offender = None
    for split in ("pub", "prv"):
        for ds in datasets:
            seq = [per_ds[split][ds]["emp"][k] for k in K_MEASURED]
            for a, b in zip(seq, seq[1:]):
                if b > a + 1e-15:  # tiny fp tolerance
                    sanity2_pass = False
                    sanity2_offender = (split, ds)
                    break

    # ---- aggregate: median & max of sigma_K across the 16 datasets ----
    agg = {"pub": {}, "prv": {}}
    for split in ("pub", "prv"):
        for k in K_MEASURED:
            vals = [per_ds[split][ds]["emp"][k] for ds in datasets]
            agg[split][k] = {"median": median(vals), "max": max(vals)}
        # K=10 extrapolated
        vals10 = [per_ds[split][ds]["k10_extrap"] for ds in datasets]
        agg[split]["k10_extrap"] = {"median": median(vals10), "max": max(vals10)}

    # median 1/sqrt(K) theory series (per-dataset theo, then median) for crossing
    # search across ALL K=1..10 (empirical where measured, extrapolation beyond).
    median_series = {"pub": {}, "prv": {}}
    for split in ("pub", "prv"):
        for k in range(1, N_SEEDS + 1):
            if k in agg[split]:  # measured empirically
                median_series[split][k] = agg[split][k]["median"]
            else:                # theoretical 1/sqrt(K) per dataset, then median
                vals = [per_ds[split][ds]["sigma1"] / math.sqrt(k) for ds in datasets]
                median_series[split][k] = median(vals)

    # crossing K for the two rulers (use PUB ruler on pub series, PRV on prv series)
    cross = {}
    cross["pub_2sigma"] = first_k_below(median_series["pub"], TWO_SIGMA_PUB)
    cross["pub_1sigma"] = first_k_below(median_series["pub"], SIGMA_MEAN_PUB)
    cross["prv_2sigma"] = first_k_below(median_series["prv"], TWO_SIGMA_PRV)
    cross["prv_1sigma"] = first_k_below(median_series["prv"], SIGMA_MEAN_PRV)

    # =====================================================================
    # write results.csv (per-dataset rows)
    # =====================================================================
    fieldnames = ["dataset"]
    for split in ("pub", "prv"):
        fieldnames.append(f"{split}_sigma1")
        for k in K_MEASURED:
            fieldnames.append(f"{split}_sigma_K{k}")
        for k in K_MEASURED:
            fieldnames.append(f"{split}_ratio_emp_theo_K{k}")
        fieldnames.append(f"{split}_sigma_K10_extrap")
    out_csv = os.path.join(HERE, "results.csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for ds in datasets:
            rec = {"dataset": ds}
            for split in ("pub", "prv"):
                d = per_ds[split][ds]
                rec[f"{split}_sigma1"] = f"{d['sigma1']:.10f}"
                for k in K_MEASURED:
                    rec[f"{split}_sigma_K{k}"] = f"{d['emp'][k]:.10f}"
                for k in K_MEASURED:
                    rec[f"{split}_ratio_emp_theo_K{k}"] = f"{d['ratio'][k]:.6f}"
                rec[f"{split}_sigma_K10_extrap"] = f"{d['k10_extrap']:.10f}"
            w.writerow(rec)

    # =====================================================================
    # console report
    # =====================================================================
    print("=" * 78)
    print("round85 seed-averaging ruler  --  K-seed averaging vs round83 noise floor")
    print("=" * 78)
    print(f"seeds per dataset (N): {N_SEEDS}   datasets: {len(datasets)}")
    print(f"round83 mean-delta ruler:  1 sigma = {SIGMA_MEAN_PUB:.5f} pub / {SIGMA_MEAN_PRV:.5f} prv")
    print(f"                           2 sigma = {TWO_SIGMA_PUB:.5f} pub / {TWO_SIGMA_PRV:.5f} prv")
    print()

    print("-" * 78)
    print(f"SANITY-1 (read integrity: recomputed per-seed SAMPLE stdev == stored "
          f"pub_std/prv_std): {'PASS' if sanity1_pass else 'FAIL'}   "
          f"max_abs_diff = {max_diff:.3e}")
    print(f"         NOTE: round83 stored SAMPLE stdev (ddof=1); vs population "
          f"stdev the gap is {max_diff_pop:.3e} = ddof factor sqrt(10/9)~=1.0541.")
    print(f"         sigma_K below uses population pstdev of subset-means as specified.")
    off = "" if sanity2_pass else f"  offender={sanity2_offender}"
    print(f"SANITY-2 (sigma_K non-increasing in K, all ds/splits): "
          f"{'PASS' if sanity2_pass else 'FAIL'}{off}")
    print()

    for split, label in (("pub", "Public"), ("prv", "Private")):
        print("-" * 78)
        print(f"{label}: per-dataset sigma_K across {len(datasets)} datasets")
        print(f"  {'K':>3} {'median_sigma':>14} {'max_sigma':>12} "
              f"{'median_ratio_emp/theo':>22} {'sqrt((N-K)/(N-1))':>18}")
        for k in K_MEASURED:
            med = agg[split][k]["median"]
            mx = agg[split][k]["max"]
            med_ratio = median([per_ds[split][ds]["ratio"][k] for ds in datasets])
            fpc = math.sqrt((N_SEEDS - k) / (N_SEEDS - 1))  # theory prediction of the ratio
            print(f"  {k:>3} {med:>14.6f} {mx:>12.6f} {med_ratio:>22.4f} {fpc:>18.4f}")
        k10 = agg[split]["k10_extrap"]
        print(f"  {'10*':>3} {k10['median']:>14.6f} {k10['max']:>12.6f} "
              f"{'(extrapolated sigma_1/sqrt(10), not measured)':>40}")
        print()

    print("-" * 78)
    print("HEADLINE: K at which per-dataset MEDIAN sigma_K first drops below the ruler")
    print("  (K=1..5 empirical; K=6..9 1/sqrt(K) theory; K=10 shipped/extrapolated)")
    print(f"  Public : below 2 sigma ({TWO_SIGMA_PUB:.5f}) at K = {cross['pub_2sigma']};  "
          f"below 1 sigma ({SIGMA_MEAN_PUB:.5f}) at K = {cross['pub_1sigma']}")
    print(f"  Private: below 2 sigma ({TWO_SIGMA_PRV:.5f}) at K = {cross['prv_2sigma']};  "
          f"below 1 sigma ({SIGMA_MEAN_PRV:.5f}) at K = {cross['prv_1sigma']}")
    print(f"  K=10 extrapolated median sigma: pub={agg['pub']['k10_extrap']['median']:.6f}  "
          f"prv={agg['prv']['k10_extrap']['median']:.6f}")
    print()

    # =====================================================================
    # summary.txt (Japanese prose, feeds a Japanese log)
    # =====================================================================
    def fmt_k(k):
        return "なし(K<=10で未達)" if k is None else f"K={k}"

    lines = []
    lines.append("round85: シード平均化(候補A)の残留ジッター vs round83ノイズ床ルーラー")
    lines.append("=" * 60)
    lines.append("")
    lines.append("【目的】")
    lines.append("候補Aは基準モデルのAUCをK個のシードで平均するアンサンブルである。")
    lines.append("round83で得た各データセット10シード(0..9)のAUCを素材に、K個平均が")
    lines.append("run毎ジッター(σ)をどこまで縮めるか、そしてどのKで残留σがround83の")
    lines.append("ルーラーを下回るかを、モデル再学習なしのオフライン再解析で測定した。")
    lines.append("")
    lines.append("【手法】")
    lines.append("各データセット・各split(pub/prv)について、K∈{1,2,3,4,5}で全C(10,K)部分集合を")
    lines.append("列挙し、各部分集合のAUC平均(=K seedでのA実行1回に相当)を取り、その母標準偏差")
    lines.append("(statistics.pstdev)をsigma_Kとした。K=1は生のシードσを再現する。")
    lines.append("有限母集団(N=10からK個非復元抽出)理論では")
    lines.append("  sigma_K = (sigma_1/√K)・√((N-K)/(N-1))")
    lines.append("であり、素朴な1/√K則(sigma_1/√K)より √((N-K)/(N-1)) 倍だけ下振れする。")
    lines.append("K=10(実出荷K)は部分集合が1個のみ(=σ=0)で測定不能のため、残留ジッターは")
    lines.append("sigma_1/√10 として外挿値で報告する(測定値ではない)。")
    lines.append("")
    lines.append("【自己検査】")
    lines.append(f"SANITY-1  読み取り整合性(再計算した各シードのSAMPLE stdevが既存の")
    lines.append(f"          pub_std/prv_std列と一致): {'PASS' if sanity1_pass else 'FAIL'}"
                 f"(最大絶対差 {max_diff:.3e})")
    lines.append(f"          → 各データセットで正しい10個の数値を読めていることを確認。")
    lines.append(f"  ※データ事実: round83のpub_std/prv_std列はSAMPLE stdev(ddof=1, ")
    lines.append(f"    statistics.stdev)で保存されており、課題文の言う\"population stdev\"では")
    lines.append(f"    ない。母標準偏差(pstdev)との差は {max_diff_pop:.3e}(=ddof係数 ")
    lines.append(f"    sqrt(10/9)≒1.0541倍)。sigma_K本体は指定通りsubset-meanのpstdevを使用。")
    lines.append(f"SANITY-2  全データセット・全splitでsigma_KがKについて単調非増加: "
                 f"{'PASS' if sanity2_pass else 'FAIL'}")
    lines.append("")
    for split, label in (("pub", "Public"), ("prv", "Private")):
        lines.append(f"【{label}: 16データセットのsigma_K集計】")
        lines.append(f"  {'K':>3} | {'median_σ':>12} | {'max_σ':>12} | {'median比(emp/理論)':>16}")
        lines.append("  " + "-" * 54)
        for k in K_MEASURED:
            med = agg[split][k]["median"]
            mx = agg[split][k]["max"]
            med_ratio = median([per_ds[split][ds]["ratio"][k] for ds in datasets])
            lines.append(f"  {k:>3} | {med:>12.6f} | {mx:>12.6f} | {med_ratio:>16.4f}")
        k10 = agg[split]["k10_extrap"]
        lines.append(f"  10* | {k10['median']:>12.6f} | {k10['max']:>12.6f} | "
                     f"(外挿 sigma_1/√10・測定値ではない)")
        lines.append("")
    lines.append("【1/√K則の検証】")
    lines.append("emp/理論(=sigma_1/√K)比の中央値は理論の有限母集団補正 √((N-K)/(N-1)) に")
    lines.append("一致する(Kが大きいほど1より小さくなる下振れ)。すなわち経験的な縮小は")
    lines.append("1/√K則に有限母集団補正を掛けた形でよく説明できる。")
    lines.append("")
    lines.append("【結論(ヘッドライン)】")
    lines.append("per-datasetのMEDIAN sigma_Kが各ルーラーを初めて下回るK")
    lines.append("(K=1..5は経験値、K=6..9は1/√K理論、K=10は出荷K/外挿):")
    lines.append(f"  Public : 2σ({TWO_SIGMA_PUB:.5f})未満 → {fmt_k(cross['pub_2sigma'])};  "
                 f"1σ({SIGMA_MEAN_PUB:.5f})未満 → {fmt_k(cross['pub_1sigma'])}")
    lines.append(f"  Private: 2σ({TWO_SIGMA_PRV:.5f})未満 → {fmt_k(cross['prv_2sigma'])};  "
                 f"1σ({SIGMA_MEAN_PRV:.5f})未満 → {fmt_k(cross['prv_1sigma'])}")
    lines.append(f"  K=10 外挿 median σ: pub={agg['pub']['k10_extrap']['median']:.6f}  "
                 f"prv={agg['prv']['k10_extrap']['median']:.6f}")
    lines.append("")
    lines.append("【注意】")
    lines.append("ここでのσはper-datasetのシード平均ジッターであり、比較対象のルーラーは")
    lines.append("16データセットのmean-delta(0.0013/0.0026)である点に留意。K=10の値は")
    lines.append("測定ではなく外挿。σを縮めても真の効果量が増えるわけではなく、A自身の")
    lines.append("run-to-run再現性がどこまで安定するかの上限を示すものである。")
    lines.append("")

    with open(os.path.join(HERE, "summary.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    # also dump a machine-readable json for downstream logs
    with open(os.path.join(HERE, "summary.json"), "w") as fh:
        json.dump({
            "sanity1_pass": sanity1_pass,
            "sanity1_max_abs_diff_vs_sample_stdev": max_diff_samp,
            "sanity1_max_abs_diff_vs_population_stdev": max_diff_pop,
            "sanity1_note": "round83 stored SAMPLE stdev (ddof=1); read-integrity checked vs sample stdev",
            "sanity2_pass": sanity2_pass,
            "ruler": {
                "sigma_mean_pub": SIGMA_MEAN_PUB, "sigma_mean_prv": SIGMA_MEAN_PRV,
                "two_sigma_pub": TWO_SIGMA_PUB, "two_sigma_prv": TWO_SIGMA_PRV,
            },
            "median_sigma_by_k": {
                s: {str(k): agg[s][k]["median"] for k in K_MEASURED} for s in ("pub", "prv")
            },
            "max_sigma_by_k": {
                s: {str(k): agg[s][k]["max"] for k in K_MEASURED} for s in ("pub", "prv")
            },
            "k10_extrap_median": {
                "pub": agg["pub"]["k10_extrap"]["median"],
                "prv": agg["prv"]["k10_extrap"]["median"],
            },
            "crossings": cross,
        }, fh, indent=2)

    print("wrote:", out_csv)
    print("wrote:", os.path.join(HERE, "summary.txt"))
    print("wrote:", os.path.join(HERE, "summary.json"))


if __name__ == "__main__":
    main()
