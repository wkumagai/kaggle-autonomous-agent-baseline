#!/usr/bin/env python3
"""round73 AvsB gate-D' desk evaluation (READ-ONLY, no refit).

Directly compares the two approved ship candidates on the gate-D' firing set:
  A = pure seed-averaged HGB (K10)  -> columns dA_public / dA_private
  B = gate-D' RF blend              -> columns dB_public / dB_private

Question answered: if the user ships exactly ONE single lever, on the identical
gate-D' firing set {train_03, train_05, train_09, train_13, train_15}, which lever
wins per-dataset, Public vs Private?

Input : ../round68_AB_interaction/results.csv  (committed, already contains the
        base-relative deltas dA_*/dB_* and the gate_dprime flag; NO refit needed)
Output: ./results.csv    (per-dataset A-vs-B duel table)
        ./summary.txt     (aggregate + verdict + CLEAN RUN + anchor match)
"""
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
IN_CSV = os.path.normpath(os.path.join(HERE, "..", "round68_AB_interaction", "results.csv"))
OUT_CSV = os.path.join(HERE, "results.csv")
OUT_TXT = os.path.join(HERE, "summary.txt")

# round68 summary anchors (dA / dB means over the gate-D' firing set).
ANCHOR = {
    "dA_public": 0.009234,
    "dA_private": 0.008965,
    "dB_public": 0.007136,
    "dB_private": 0.009522,
}
ANCHOR_TOL = 1e-6

# gate-D' firing set is expected to be exactly these 5 datasets.
EXPECTED_FIRING = {"train_03", "train_05", "train_09", "train_13", "train_15"}

# tie threshold on the (dA - dB) gap; below this we call it a tie.
TIE_EPS = 1e-9


def to_bool(s):
    return str(s).strip().lower() in ("true", "1", "yes")


def winner(dA, dB):
    gap = dA - dB
    if abs(gap) < TIE_EPS:
        return "tie"
    return "A" if gap > 0 else "B"


def main():
    log = []

    def emit(msg):
        print(msg)
        log.append(msg)

    emit(f"[round73] reading input: {IN_CSV}")
    with open(IN_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    emit(f"[round73] total rows in round68 results.csv: {len(rows)}")

    fired = [r for r in rows if to_bool(r["gate_dprime"])]
    fired_names = [r["dataset"] for r in fired]
    emit(f"[round73] gate_dprime==True rows ({len(fired)}): {fired_names}")

    # Assert we have exactly the expected 5-dataset firing set.
    assert len(fired) == 5, f"expected 5 gate-D' rows, got {len(fired)}"
    assert set(fired_names) == EXPECTED_FIRING, (
        f"firing set mismatch: {set(fired_names)} != {EXPECTED_FIRING}"
    )
    emit("[round73] ASSERT OK: firing set == {train_03,05,09,13,15}, n=5")

    # Per-dataset duel.
    duel_rows = []
    for r in fired:
        dA_pub = float(r["dA_public"])
        dA_prv = float(r["dA_private"])
        dB_pub = float(r["dB_public"])
        dB_prv = float(r["dB_private"])
        duel_rows.append({
            "dataset": r["dataset"],
            "dA_public": dA_pub,
            "dB_public": dB_pub,
            "diff_public_AminusB": dA_pub - dB_pub,
            "winner_public": winner(dA_pub, dB_pub),
            "dA_private": dA_prv,
            "dB_private": dB_prv,
            "diff_private_AminusB": dA_prv - dB_prv,
            "winner_private": winner(dA_prv, dB_prv),
        })

    # Write per-dataset duel table.
    fieldnames = [
        "dataset",
        "dA_public", "dB_public", "diff_public_AminusB", "winner_public",
        "dA_private", "dB_private", "diff_private_AminusB", "winner_private",
    ]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for d in duel_rows:
            w.writerow(d)
    emit(f"[round73] wrote per-dataset duel table: {OUT_CSV}")

    # Aggregates.
    n = len(duel_rows)
    mean_dA_pub = sum(d["dA_public"] for d in duel_rows) / n
    mean_dB_pub = sum(d["dB_public"] for d in duel_rows) / n
    mean_dA_prv = sum(d["dA_private"] for d in duel_rows) / n
    mean_dB_prv = sum(d["dB_private"] for d in duel_rows) / n

    winA_pub = sum(1 for d in duel_rows if d["winner_public"] == "A")
    winB_pub = sum(1 for d in duel_rows if d["winner_public"] == "B")
    tie_pub = sum(1 for d in duel_rows if d["winner_public"] == "tie")
    winA_prv = sum(1 for d in duel_rows if d["winner_private"] == "A")
    winB_prv = sum(1 for d in duel_rows if d["winner_private"] == "B")
    tie_prv = sum(1 for d in duel_rows if d["winner_private"] == "tie")

    all_A_pub_pos = all(d["dA_public"] > 0 for d in duel_rows)
    all_A_prv_pos = all(d["dA_private"] > 0 for d in duel_rows)
    all_B_pub_pos = all(d["dB_public"] > 0 for d in duel_rows)
    all_B_prv_pos = all(d["dB_private"] > 0 for d in duel_rows)

    # Anchor check (sanity that we read the right rows / no contamination).
    anchor_results = {
        "dA_public": (mean_dA_pub, ANCHOR["dA_public"]),
        "dA_private": (mean_dA_prv, ANCHOR["dA_private"]),
        "dB_public": (mean_dB_pub, ANCHOR["dB_public"]),
        "dB_private": (mean_dB_prv, ANCHOR["dB_private"]),
    }
    anchor_lines = []
    anchor_all_pass = True
    for key, (got, exp) in anchor_results.items():
        delta = abs(got - exp)
        ok = delta < ANCHOR_TOL
        anchor_all_pass = anchor_all_pass and ok
        anchor_lines.append(
            f"  {key:12s} computed={got:+.6f} anchor={exp:+.6f} |Δ|={delta:.2e} "
            f"-> {'PASS' if ok else 'FAIL'}"
        )
    clean_run = "YES" if anchor_all_pass else "NO"

    emit("")
    emit("[round73] === anchor check vs round68 summary ===")
    for ln in anchor_lines:
        emit(ln)
    emit(f"[round73] CLEAN RUN = {clean_run}")

    # Verdict.
    lb_winner = "A" if mean_dA_pub > mean_dB_pub else ("B" if mean_dB_pub > mean_dA_pub else "tie")
    pv_winner = "A" if mean_dA_prv > mean_dB_prv else ("B" if mean_dB_prv > mean_dA_prv else "tie")

    # Build summary.txt.
    lines = []
    lines.append("round73 -- A vs B single-lever duel on gate-D' firing set")
    lines.append("=" * 64)
    lines.append("A = pure seed-avg HGB (K10)   [dA_* columns]")
    lines.append("B = gate-D' RF blend          [dB_* columns]")
    lines.append(f"firing set (n={n}): {', '.join(sorted(fired_names))}")
    lines.append("READ-ONLY desk calc; deltas taken verbatim from round68 results.csv (NO refit).")
    lines.append("")
    lines.append("Per-dataset duel (delta vs base; winner = larger delta):")
    lines.append(
        f"  {'dataset':10s} {'dA_pub':>10s} {'dB_pub':>10s} {'win_pub':>8s}"
        f" {'dA_prv':>10s} {'dB_prv':>10s} {'win_prv':>8s}"
    )
    for d in duel_rows:
        lines.append(
            f"  {d['dataset']:10s} {d['dA_public']:+.6f} {d['dB_public']:+.6f}"
            f" {d['winner_public']:>8s} {d['dA_private']:+.6f} {d['dB_private']:+.6f}"
            f" {d['winner_private']:>8s}"
        )
    lines.append("")
    lines.append("Aggregate over firing set:")
    lines.append(f"  Public : mean dA={mean_dA_pub:+.6f}  mean dB={mean_dB_pub:+.6f}"
                 f"  (A-B={mean_dA_pub - mean_dB_pub:+.6f})")
    lines.append(f"  Private: mean dA={mean_dA_prv:+.6f}  mean dB={mean_dB_prv:+.6f}"
                 f"  (A-B={mean_dA_prv - mean_dB_prv:+.6f})")
    lines.append(f"  Public  win counts : A={winA_pub}  B={winB_pub}  tie={tie_pub}")
    lines.append(f"  Private win counts : A={winA_prv}  B={winB_prv}  tie={tie_prv}")
    lines.append("")
    lines.append("Regression check (any dataset with negative delta?):")
    lines.append(f"  A all-positive Public ={all_A_pub_pos}  Private={all_A_prv_pos}")
    lines.append(f"  B all-positive Public ={all_B_pub_pos}  Private={all_B_prv_pos}")
    lines.append("")
    lines.append("Anchor check vs round68 summary (|Δ|<1e-6 == PASS):")
    lines.extend(anchor_lines)
    lines.append(f"  ANCHOR MATCH: {'PASS' if anchor_all_pass else 'FAIL'}")
    lines.append(f"  CLEAN RUN = {clean_run}")
    lines.append("")
    lines.append("Conclusion:")
    lines.append(f"  LB / Public single-lever winner : {lb_winner}"
                 f"  (mean dA {mean_dA_pub:+.6f} vs dB {mean_dB_pub:+.6f})")
    lines.append(f"  Private single-lever winner     : {pv_winner}"
                 f"  (mean dA {mean_dA_prv:+.6f} vs dB {mean_dB_prv:+.6f})")

    with open(OUT_TXT, "w") as f:
        f.write("\n".join(lines) + "\n")
    emit("")
    emit(f"[round73] wrote summary: {OUT_TXT}")
    emit("")
    emit("\n".join(lines))


if __name__ == "__main__":
    main()
