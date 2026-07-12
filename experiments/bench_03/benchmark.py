#!/usr/bin/env python
"""Offline benchmark for submissions/03_cv_ensemble: replays the EXACT shipped
recipe (single source of truth = the fenced python block in prompts/system.md)
on all 16 local datasets, scores every candidate on the local solution.csv
Public/Private splits, simulates the prompt's final-selection rule, and checks
the pass gates required before the outer submit.

Run (from repo root, with the repo .venv):
  .venv/bin/python experiments/bench_03/benchmark.py                # full 16-dataset run
  .venv/bin/python experiments/bench_03/benchmark.py --datasets 01  # smoke test

go.py subprocesses run under a pandas-3.x venv (experiments/bench_03/gbm_venv)
to replicate the Kaggle sandbox 'str' dtype hazard; override with --go-python.
"""
import argparse
import ast
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BENCH_DIR = os.path.join(REPO, "experiments", "bench_03")
SYSTEM_MD = os.path.join(REPO, "submissions", "03_cv_ensemble", "agent", "prompts", "system.md")
DATA_DIR = os.path.join(REPO, "data")
STATS_CSV = os.path.join(BENCH_DIR, "dataset_stats.csv")
RESULTS_JSON = os.path.join(BENCH_DIR, "results.json")
RESULTS_CSV = os.path.join(BENCH_DIR, "benchmark_results.csv")

# Session replay order (the workflow's run_command sequence).
STAGES = ["safety", "xgb", "cat", "lgb", "blend"]
# Submission order (the workflow's submit_predictions sequence).
CANDIDATES = [
    ("safety", "sub_safety.csv"),
    ("xgb", "sub_xgb.csv"),
    ("cat", "sub_cat.csv"),
    ("lgb", "sub_lgb.csv"),
    ("blend_all", "sub_blend_all.csv"),
    ("blend_top2", "sub_blend_top2.csv"),
]
STAGE_TIMEOUT = 1800  # generous local runaway guard; gate (c) is checked separately

PLACEHOLDER_KEYS = [
    "problem_description", "metric_name", "metric_direction", "max_tool_calls",
    "max_submissions", "max_selections", "max_exec_seconds", "max_stdout_chars",
    "max_budget_usd", "max_llm_calls", "max_time_minutes", "task_prompt",
]

CSV_FIELDS = [
    "dataset", "n_train", "status",
    "secs_safety", "secs_xgb", "secs_cat", "secs_lgb", "secs_blend", "secs_total",
    "baseline_auc", "picked_candidate_A", "picked_candidate_B", "final_private",
    "delta_vs_baseline", "oracle_private", "never_selected_private",
    "blend_all_private", "max_stage_secs", "error",
]


def extract_script(md_path):
    """SINGLE SOURCE OF TRUTH: the fenced python block in system.md."""
    text = open(md_path).read()
    blocks = re.findall(r"```python\n(.*?)```", text, re.S)
    if len(blocks) != 1:
        raise SystemExit("expected exactly 1 fenced python block in %s, found %d" % (md_path, len(blocks)))
    return text, blocks[0]


def static_gates(md_path):
    """Gate (d): replay ADK inject_session_state over system.md (reference:
    scratchpad verify_03.py) — 0 illegal brace runs, script byte-identical
    after injection, 0 braces in script, ast.parse OK."""
    text, code = extract_script(md_path)
    bad = []
    used = set()

    def repl(m):
        var = m.group().lstrip("{").rstrip("}").strip()
        core = var[:-1] if var.endswith("?") else var
        if not core.isidentifier():
            return m.group()  # left as-is by ADK, safe
        if core in PLACEHOLDER_KEYS:
            used.add(core)
            return "<VAL_%s>" % core
        bad.append(m.group())
        return m.group()

    injected = re.sub(r"{+[^{}]*}+", repl, text)
    blocks2 = re.findall(r"```python\n(.*?)```", injected, re.S)
    ast_ok = True
    try:
        ast.parse(code)
    except SyntaxError:
        ast_ok = False
    gates = {
        "illegal_brace_runs": bad,
        "placeholders_used": sorted(used),
        "braces_in_script": code.count("{") + code.count("}"),
        "script_unchanged_after_injection": bool(blocks2 and blocks2[0] == code),
        "ast_parse_ok": ast_ok,
        "instruction_chars": len(text),
    }
    gates["pass"] = (not bad and gates["braces_in_script"] == 0
                     and gates["script_unchanged_after_injection"] and ast_ok)
    return gates


def run_validate_submission():
    """Gate (e): validate_submission.py on the shipped agent dir."""
    cmd = [os.path.join(REPO, ".venv", "bin", "python"), "validate_submission.py",
           "--agent-dir", "submissions/03_cv_ensemble/agent"]
    try:
        p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=300)
        return {"pass": p.returncode == 0,
                "output": (p.stdout + p.stderr)[-2000:]}
    except Exception as e:  # noqa: BLE001
        return {"pass": False, "output": "exception: %r" % (e,)}


def parse_result_line(stdout):
    """First 'RESULT <stage> key=val ...' line -> dict of floats."""
    for line in stdout.splitlines():
        if line.startswith("RESULT "):
            parts = line.split()
            out = {"stage": parts[1]}
            for tok in parts[2:]:
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    try:
                        out[k] = float(v)
                    except ValueError:
                        out[k] = v
            return out
    return None


def parse_cand_lines(stdout):
    """'CAND <filename> oof_auc=<v>' lines -> {filename: oof}."""
    out = {}
    for line in stdout.splitlines():
        if line.startswith("CAND "):
            parts = line.split()
            fname = parts[1]
            for tok in parts[2:]:
                if tok.startswith("oof_auc="):
                    out[fname] = float(tok.split("=", 1)[1])
    return out


def score_candidate(sub_path, solution):
    """Join on row_id; AUC per Usage split (exact offline public/private)."""
    sub = pd.read_csv(sub_path)
    id_col, pred_col = sub.columns[0], sub.columns[1]
    # The prediction column is frequently literally named "target" (same as
    # solution's ground-truth column), which would silently collide into
    # target_x/target_y on merge. Rename it first so downstream lookups by
    # name are safe regardless of the sample_submission's column naming.
    pred = sub[[id_col, pred_col]].rename(columns={id_col: "row_id", pred_col: "_pred"})
    m = solution.merge(pred, on="row_id", how="inner")
    if len(m) != len(solution):
        raise ValueError("row_id join mismatch for %s: %d vs %d" % (sub_path, len(m), len(solution)))
    scores = {}
    for split in ("Public", "Private"):
        part = m[m["Usage"] == split]
        scores[split.lower()] = float(roc_auc_score(part["target"].values, part["_pred"].values))
    return scores


def simulate_selection(cands):
    """Apply the prompt's Step-10 rule to candidates in submission order.

    cands: list of dicts (submission order) with id, public, private, oof
           (oof None for safety). Public scores rounded to 6dp as reported.
    Returns selection dict incl. final_private / oracle / never_selected /
    blend_all_always.
    """
    for c in cands:
        c["public_r"] = round(c["public"], 6)
    # A = highest public score (first in submission order on ties)
    a = max(cands, key=lambda c: c["public_r"])
    # B = highest OOF (safety has no OOF)
    with_oof = [c for c in cands if c.get("oof") is not None]
    b = max(with_oof, key=lambda c: c["oof"]) if with_oof else None
    if b is None or b["id"] == a["id"]:
        # collision (or no OOF info) -> B = 2nd-best public
        rest = sorted([c for c in cands if c["id"] != a["id"]],
                      key=lambda c: -c["public_r"])
        b = rest[0] if rest else a
    final_private = max(a["private"], b["private"])
    oracle = max(c["private"] for c in cands)
    top2_pub = sorted(cands, key=lambda c: -c["public_r"])[:2]
    never_selected = max(c["private"] for c in top2_pub)
    blend_all = next((c for c in cands if c["id"] == "blend_all"), None)
    return {
        "selected_A": a["id"], "selected_B": b["id"],
        "final_private": final_private,
        "oracle": oracle,
        "never_selected": never_selected,
        "blend_all_always": blend_all["private"] if blend_all else None,
    }


def run_dataset(ds, go_code, go_python):
    """PER-DATASET SESSION REPLAY in an isolated temp workdir."""
    ds_dir = os.path.join(DATA_DIR, ds)
    row = {"dataset": ds, "stages": {}, "candidates": {}, "status": "ok", "error": ""}
    work = tempfile.mkdtemp(prefix="bench03_%s_" % ds)
    try:
        for f in ("train.csv", "test.csv", "sample_submission.csv"):
            shutil.copy(os.path.join(ds_dir, f), os.path.join(work, f))
        go_path = os.path.join(work, "go.py")
        with open(go_path, "w") as fh:
            fh.write(go_code)
        row["n_train"] = int(sum(1 for _ in open(os.path.join(ds_dir, "train.csv"))) - 1)

        # Exact session order; the script itself sets OMP/OPENBLAS/MKL=4 via setdefault.
        for stage in STAGES:
            t0 = time.time()
            info = {"secs": None, "ok": False}
            try:
                p = subprocess.run([go_python, "go.py", stage], cwd=work,
                                   capture_output=True, text=True, timeout=STAGE_TIMEOUT)
                info["secs"] = round(time.time() - t0, 1)
                info["returncode"] = p.returncode
                info["stdout_head"] = p.stdout[:2000]
                if p.returncode != 0:
                    info["stderr_tail"] = p.stderr[-2000:]
                else:
                    info["ok"] = True
                    res = parse_result_line(p.stdout)
                    if res:
                        info["result"] = res
                    if stage == "blend":
                        info["cands"] = parse_cand_lines(p.stdout)
            except subprocess.TimeoutExpired:
                info["secs"] = round(time.time() - t0, 1)
                info["error"] = "timeout>%ds" % STAGE_TIMEOUT
            row["stages"][stage] = info

        # Score every produced sub_*.csv against solution.csv.
        solution = pd.read_csv(os.path.join(ds_dir, "solution.csv"))
        cand_oof = row["stages"].get("blend", {}).get("cands", {})
        sel_input = []
        for cid, fname in CANDIDATES:
            fpath = os.path.join(work, fname)
            if not os.path.exists(fpath):
                continue
            sc = score_candidate(fpath, solution)
            oof = cand_oof.get(fname)
            if oof is None and cid in ("xgb", "cat", "lgb"):
                res = row["stages"].get(cid, {}).get("result") or {}
                oof = res.get("oof_auc")
            row["candidates"][cid] = {"file": fname, "oof": oof,
                                      "public": sc["public"], "private": sc["private"]}
            sel_input.append({"id": cid, "public": sc["public"],
                              "private": sc["private"], "oof": oof})
        if not sel_input:
            row["status"] = "no_candidates"
            return row
        row["selection"] = simulate_selection(sel_input)
    except Exception as e:  # noqa: BLE001
        row["status"] = "error"
        row["error"] = repr(e)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return row


def flat_row(row, baseline):
    stages = row.get("stages", {})
    secs = {s: stages.get(s, {}).get("secs") for s in STAGES}
    model_secs = [v for s, v in secs.items() if v is not None]
    sel = row.get("selection", {})
    fp = sel.get("final_private")
    out = {
        "dataset": row["dataset"],
        "n_train": row.get("n_train"),
        "status": row.get("status"),
        "secs_safety": secs["safety"], "secs_xgb": secs["xgb"], "secs_cat": secs["cat"],
        "secs_lgb": secs["lgb"], "secs_blend": secs["blend"],
        "secs_total": round(sum(model_secs), 1) if model_secs else None,
        "baseline_auc": baseline,
        "picked_candidate_A": sel.get("selected_A"),
        "picked_candidate_B": sel.get("selected_B"),
        "final_private": fp,
        "delta_vs_baseline": round(fp - baseline, 4) if fp is not None and baseline is not None else None,
        "oracle_private": sel.get("oracle"),
        "never_selected_private": sel.get("never_selected"),
        "blend_all_private": sel.get("blend_all_always"),
        "max_stage_secs": max(model_secs) if model_secs else None,
        "error": row.get("error", ""),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=None,
                    help="comma list like 01,02 (default: all 16)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--go-python", default=os.path.join(BENCH_DIR, "gbm_venv", "bin", "python"),
                    help="interpreter for go.py subprocesses (pandas 3.x venv)")
    ap.add_argument("--skip-validate", action="store_true")
    ap.add_argument("--system-md", default=SYSTEM_MD,
                    help="path to system.md to score (default: shipped submissions/03_cv_ensemble path)")
    args = ap.parse_args()
    system_md = args.system_md

    if not os.path.exists(args.go_python):
        print("WARNING: %s not found, falling back to %s" % (args.go_python, sys.executable))
        args.go_python = sys.executable
    pdv = subprocess.run([args.go_python, "-c", "import pandas;print(pandas.__version__)"],
                         capture_output=True, text=True).stdout.strip()
    print("go.py interpreter: %s (pandas %s)" % (args.go_python, pdv))
    if not pdv.startswith("3."):
        print("WARNING: go.py env pandas is not 3.x — the pandas-3 'str' dtype hazard is NOT replicated")

    # Gate (d): static gates on the shipped prompt; extraction is the single source of truth.
    gates_d = static_gates(system_md)
    print("static gates:", json.dumps(gates_d))
    if not gates_d["pass"]:
        raise SystemExit("STATIC GATES FAILED — fix system.md before benchmarking")
    _, go_code = extract_script(system_md)

    gates_e = None
    if not args.skip_validate:
        gates_e = run_validate_submission()
        print("validate_submission: %s" % ("PASS" if gates_e["pass"] else "FAIL"))
        if not gates_e["pass"]:
            print(gates_e["output"])

    stats = pd.read_csv(STATS_CSV).set_index("name")
    if args.datasets:
        names = ["train_%s" % d.strip() for d in args.datasets.split(",")]
    else:
        names = sorted(d for d in os.listdir(DATA_DIR) if re.fullmatch(r"train_\d+", d))
    print("datasets: %s | workers=%d" % (",".join(names), args.workers))

    # Incremental CSV so partial progress survives.
    csv_exists = os.path.exists(RESULTS_CSV)
    csv_fh = open(RESULTS_CSV, "a", newline="")
    writer = csv.DictWriter(csv_fh, fieldnames=CSV_FIELDS)
    if not csv_exists:
        writer.writeheader()
        csv_fh.flush()

    all_rows = {}
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_dataset, ds, go_code, args.go_python): ds for ds in names}
        for fut in as_completed(futs):
            ds = futs[fut]
            try:
                row = fut.result()
            except Exception as e:  # noqa: BLE001
                row = {"dataset": ds, "status": "error", "error": repr(e), "stages": {}, "candidates": {}}
            baseline = float(stats.loc[ds, "baseline_hgb_auc_private"]) if ds in stats.index else None
            row["baseline_hgb_auc_private"] = baseline
            all_rows[ds] = row
            fr = flat_row(row, baseline)
            writer.writerow(fr)
            csv_fh.flush()
            print("[%5.0fs] done %s: final_private=%s delta=%s max_stage=%ss" % (
                time.time() - t_start, ds, fr["final_private"], fr["delta_vs_baseline"],
                fr["max_stage_secs"]), flush=True)
            # Incremental JSON too.
            with open(RESULTS_JSON, "w") as jf:
                json.dump({"static_gates": gates_d, "validate_submission": gates_e,
                           "go_python": args.go_python, "go_env_pandas": pdv,
                           "datasets": all_rows}, jf, indent=1)
    csv_fh.close()

    # ---- Summary table ----
    print("\n%-9s %7s %9s %9s %-11s %-11s %9s %7s %8s %8s %8s" % (
        "dataset", "n_train", "baseline", "final_pv", "pick_A", "pick_B",
        "delta", "maxstg", "oracle", "nevsel", "blendall"))
    flats = [flat_row(all_rows[ds], all_rows[ds].get("baseline_hgb_auc_private")) for ds in names if ds in all_rows]
    ok = [f for f in flats if f["final_private"] is not None and f["baseline_auc"] is not None]
    for f in flats:
        r = all_rows[f["dataset"]]
        sel = r.get("selection", {})
        print("%-9s %7s %9s %9s %-11s %-11s %9s %7s %8s %8s %8s" % (
            f["dataset"], f["n_train"],
            "%.4f" % f["baseline_auc"] if f["baseline_auc"] is not None else "-",
            "%.4f" % f["final_private"] if f["final_private"] is not None else "-",
            f["picked_candidate_A"] or "-", f["picked_candidate_B"] or "-",
            "%+.4f" % f["delta_vs_baseline"] if f["delta_vs_baseline"] is not None else "-",
            f["max_stage_secs"] if f["max_stage_secs"] is not None else "-",
            "%.4f" % sel["oracle"] if sel.get("oracle") is not None else "-",
            "%.4f" % sel["never_selected"] if sel.get("never_selected") is not None else "-",
            "%.4f" % sel["blend_all_always"] if sel.get("blend_all_always") is not None else "-"))

    if ok:
        m_final = float(np.mean([f["final_private"] for f in ok]))
        m_base = float(np.mean([f["baseline_auc"] for f in ok]))
        m_oracle = float(np.mean([f["oracle_private"] for f in ok]))
        m_never = float(np.mean([f["never_selected_private"] for f in ok]))
        m_blend = float(np.mean([f["blend_all_private"] for f in ok if f["blend_all_private"] is not None]))
        worst_reg = min(f["delta_vs_baseline"] for f in ok)
        max_stage = max(f["max_stage_secs"] for f in ok if f["max_stage_secs"] is not None)
        print("\nmeans over %d datasets:" % len(ok))
        print("  baseline_hgb      %.4f" % m_base)
        print("  final_private     %.4f  (hedge selection)" % m_final)
        print("  oracle            %.4f" % m_oracle)
        print("  never_selected    %.4f  (harness top-2-public fallback)" % m_never)
        print("  blend_all_always  %.4f" % m_blend)
        print("  mean delta        %+.4f | worst delta %+.4f | max stage %ss" % (
            m_final - m_base, worst_reg, max_stage))

        # ---- Pass gates ----
        # (c) is measured on the throttled (OMP=4) local wall time; at an assumed
        # 3x Kaggle-CPU slowdown a 100s local stage ~= the 300s per-command cap.
        gA = (m_final - m_base) >= 0.010
        gB = worst_reg >= -0.005
        gC = max_stage <= 100
        gD = gates_d["pass"]
        gE = gates_e["pass"] if gates_e else None
        # Wall-time budget flag: total recipe compute must fit inside 60 min at 3x slowdown.
        risky = [f["dataset"] for f in ok if f["secs_total"] and f["secs_total"] * 3 > 2400]
        print("\nPASS GATES:")
        print("  (a) mean delta >= +0.010 : %s (%+.4f)" % ("PASS" if gA else "FAIL", m_final - m_base))
        print("  (b) worst delta >= -0.005: %s (%+.4f)" % ("PASS" if gB else "FAIL", worst_reg))
        print("  (c) max stage <= 100s    : %s (%ss)" % ("PASS" if gC else "FAIL", max_stage))
        print("  (d) static gates         : %s" % ("PASS" if gD else "FAIL"))
        print("  (e) validate_submission  : %s" % ("PASS" if gE else ("SKIPPED" if gE is None else "FAIL")))
        if risky:
            print("  TIMEOUT-RISK datasets (total*3 > 40min): %s" % ",".join(risky))
        else:
            print("  no dataset at 60-min wall-time risk (total*3 <= 40min for all)")
        verdict = gA and gB and gC and gD and (gE is not False)
        print("\nOVERALL: %s" % ("ALL GATES PASS — ok to submit" if verdict else "GATES FAILED — do NOT submit"))
        with open(RESULTS_JSON) as jf:
            blob = json.load(jf)
        blob["summary"] = {
            "n_datasets": len(ok), "mean_baseline": m_base, "mean_final_private": m_final,
            "mean_oracle": m_oracle, "mean_never_selected": m_never,
            "mean_blend_all": m_blend, "mean_delta": m_final - m_base,
            "worst_delta": worst_reg, "max_stage_secs": max_stage,
            "gates": {"a_mean_delta": gA, "b_worst_delta": gB, "c_stage_time": gC,
                      "d_static": gD, "e_validate": gE},
            "timeout_risk_datasets": risky, "overall_pass": verdict,
        }
        with open(RESULTS_JSON, "w") as jf:
            json.dump(blob, jf, indent=1)

    bad = [f["dataset"] for f in flats if f["status"] != "ok"]
    if bad:
        print("\nFAILED datasets: %s" % ",".join(bad))
        sys.exit(1)


if __name__ == "__main__":
    main()
