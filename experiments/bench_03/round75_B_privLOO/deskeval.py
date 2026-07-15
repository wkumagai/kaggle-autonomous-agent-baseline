#!/usr/bin/env python3
"""
round75_B_privLOO -- OFFLINE desk calculation (pure arithmetic, no fitting).

Question: Is Candidate B (gate-D' RF-blend)'s Private-AUC advantage over
Candidate A (pure seed-avg HGB) robust across the gate-D' firing set
{train_03, train_05, train_09, train_13, train_15}, or does it concentrate in
a few datasets? Specifically, does B's mean Private-AUC advantage survive
leaving out train_05 and/or train_09?

Input (read-only, values used verbatim -- NOT recomputed by fitting):
    experiments/bench_03/round73_AvsB_gateDprime/results.csv

For BOTH public and private we compute, over subsets of the 5 datasets:
    mean(dA), mean(dB), B_minus_A = mean(dB) - mean(dA), winner
(winner = A if mean(dA) > mean(dB) else B).

Subsets: full {03,05,09,13,15}; each leave-one-out; leave-both-out {05,09}.
"""

import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
INPUT_CSV = os.path.join(
    REPO, "experiments", "bench_03", "round73_AvsB_gateDprime", "results.csv"
)
SUMMARY = os.path.join(HERE, "summary.txt")

# Sanity anchors from round73 (tolerance 1e-6).
ANCHORS = {
    "dA_private_mean": 0.008965,
    "dB_private_mean": 0.009522,
    "dA_public_mean": 0.009234,
    "dB_public_mean": 0.007136,
}
TOL = 1e-6


def load_rows(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    # Parse the four value columns we need, keyed by dataset.
    data = {}
    for r in rows:
        ds = r["dataset"]
        data[ds] = {
            "dA_public": float(r["dA_public"]),
            "dB_public": float(r["dB_public"]),
            "dA_private": float(r["dA_private"]),
            "dB_private": float(r["dB_private"]),
        }
    return data


def mean(xs):
    return sum(xs) / len(xs)


def summarize(data, datasets, kind):
    """kind is 'public' or 'private'. Returns dict of means/diff/winner."""
    dA = mean([data[d][f"dA_{kind}"] for d in datasets])
    dB = mean([data[d][f"dB_{kind}"] for d in datasets])
    b_minus_a = dB - dA
    winner = "A" if dA > dB else "B"
    return {"meanA": dA, "meanB": dB, "B_minus_A": b_minus_a, "winner": winner}


def fmt_row(label, s):
    return (
        f"  {label:<28} meanA={s['meanA']:+.6f}  meanB={s['meanB']:+.6f}  "
        f"B_minus_A={s['B_minus_A']:+.6f}  winner={s['winner']}"
    )


def main():
    data = load_rows(INPUT_CSV)
    all_ds = ["train_03", "train_05", "train_09", "train_13", "train_15"]
    for d in all_ds:
        assert d in data, f"missing dataset {d} in input"

    lines = []
    lines.append("round75_B_privLOO -- Candidate B Private-AUC advantage robustness (LOO)")
    lines.append("Input: experiments/bench_03/round73_AvsB_gateDprime/results.csv")
    lines.append("winner = A if mean(dA) > mean(dB) else B ; B_minus_A = mean(dB) - mean(dA)")
    lines.append("")

    # ---- Full set + LOO + leave-both-out tables (public & private) ----
    subsets = [("FULL {03,05,09,13,15}", all_ds)]
    for d in all_ds:
        remaining = [x for x in all_ds if x != d]
        subsets.append((f"LOO drop {d}", remaining))
    both = [x for x in all_ds if x not in ("train_05", "train_09")]
    subsets.append(("LEAVE-BOTH drop {05,09}", both))

    for kind in ("public", "private"):
        lines.append(f"=== {kind.upper()} ===")
        for label, ds in subsets:
            lines.append(fmt_row(label, summarize(data, ds, kind)))
        lines.append("")

    # ---- Direct answers (a)-(d), all on PRIVATE ----
    full_priv = summarize(data, all_ds, "private")
    drop05 = summarize(data, [x for x in all_ds if x != "train_05"], "private")
    drop09 = summarize(data, [x for x in all_ds if x != "train_09"], "private")
    dropboth = summarize(data, both, "private")

    lines.append("=== DIRECT ANSWERS (Private-AUC) ===")
    lines.append(
        f"Full-set Private winner = {full_priv['winner']} "
        f"(meanA={full_priv['meanA']:+.6f}, meanB={full_priv['meanB']:+.6f}, "
        f"B_minus_A={full_priv['B_minus_A']:+.6f})"
    )

    # (a) drop train_05 alone -> flip B to A?
    a_flip = (full_priv["winner"] == "B") and (drop05["winner"] == "A")
    lines.append(
        f"(a) Drop train_05 alone: winner={drop05['winner']} "
        f"(B_minus_A={drop05['B_minus_A']:+.6f}). "
        f"Flip Private winner B->A? {'YES' if a_flip else 'NO'}"
    )

    # (b) drop train_09 alone -> flip B to A?
    b_flip = (full_priv["winner"] == "B") and (drop09["winner"] == "A")
    lines.append(
        f"(b) Drop train_09 alone: winner={drop09['winner']} "
        f"(B_minus_A={drop09['B_minus_A']:+.6f}). "
        f"Flip Private winner B->A? {'YES' if b_flip else 'NO'}"
    )

    # (c) drop both -> who wins and by how much?
    lines.append(
        f"(c) Drop both {{05,09}}: winner={dropboth['winner']}, "
        f"B_minus_A={dropboth['B_minus_A']:+.6f} "
        f"(A wins by {abs(dropboth['B_minus_A']):.6f} if winner=A; "
        f"B wins by {abs(dropboth['B_minus_A']):.6f} if winner=B)"
    )

    # (d) is B's full-set Private advantage concentrated in train_05?
    #     Compare train_05's per-dataset (dB_private - dA_private) against
    #     full-set mean B_minus_A_private * 5 (i.e. the total sum of advantages).
    t05_adv = data["train_05"]["dB_private"] - data["train_05"]["dA_private"]
    full_sum = full_priv["B_minus_A"] * 5.0  # sum of per-dataset (dB-dA)
    concentrated = t05_adv > full_sum
    lines.append(
        f"(d) train_05 per-dataset (dB_private - dA_private) = {t05_adv:+.6f}; "
        f"full-set mean B_minus_A_private * 5 (= sum of advantages) = {full_sum:+.6f}. "
        f"train_05 advantage > 5*mean? {'YES' if concentrated else 'NO'} "
        f"-> B's Private advantage is "
        f"{'CONCENTRATED in train_05 (others net-negative)' if concentrated else 'NOT solely concentrated in train_05'}"
    )
    lines.append("")

    # ---- Sanity anchor check ----
    computed = {
        "dA_private_mean": full_priv["meanA"],
        "dB_private_mean": full_priv["meanB"],
        "dA_public_mean": summarize(data, all_ds, "public")["meanA"],
        "dB_public_mean": summarize(data, all_ds, "public")["meanB"],
    }
    lines.append("=== SANITY ANCHOR (tol=1e-6) ===")
    all_ok = True
    for k, expected in ANCHORS.items():
        got = computed[k]
        ok = abs(got - expected) <= TOL
        all_ok = all_ok and ok
        lines.append(
            f"  {k:<18} expected={expected:+.6f} got={got:+.9f} "
            f"diff={got - expected:+.2e} {'PASS' if ok else 'FAIL'}"
        )
    lines.append(f"SANITY ANCHOR: {'PASS' if all_ok else 'FAIL'}")
    assert all_ok, "Sanity anchor FAILED -- computed means do not match round73."

    out = "\n".join(lines) + "\n"
    print(out)
    with open(SUMMARY, "w") as f:
        f.write(out)


if __name__ == "__main__":
    main()
