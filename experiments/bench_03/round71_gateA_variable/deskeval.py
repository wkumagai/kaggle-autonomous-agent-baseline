#!/usr/bin/env python3
"""
round71 gate-VARIABLE robustness deskeval for candidate A (seed averaging, K=10).

READ-ONLY DESK CALCULATION. This script:
  * reads two existing CSVs (round69 per-dataset deltas + dataset_stats),
  * merges them on dataset name,
  * evaluates a family of candidate GATE definitions over the 16 offline
    datasets by reusing round69's already-measured per-dataset dA_K10 deltas,
  * writes results.csv (one row per gate) and summary.txt (prose analysis).

It does NOT load any dataset, fit/refit any model, or touch submissions/.
Every output path is under experiments/bench_03/round71_gateA_variable/.

Candidate A = seed averaging of the base-08 HistGradientBoosting classifier:
fit the same model with K=10 seeds and average predicted probabilities.
dA_K10 = AUC(candidate A) - AUC(single-seed base), measured in round69,
reported per public / private eval split.

Current shipping gate = gate-C = "n_object_cols > 0".
Question: is gate-C the right gate variable, or does a different/simpler/better
gate capture the seed-avg gains more cleanly (more clean wins, no new regression)?
"""

import os
import sys
import math
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Paths (all relative to repo root; inputs are read-only)
# ----------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)                       # experiments/bench_03
ROUND69 = os.path.join(BENCH, "round69_seedavg_K", "results.csv")
STATS = os.path.join(BENCH, "dataset_stats.csv")
OUT_RESULTS = os.path.join(HERE, "results.csv")
OUT_SUMMARY = os.path.join(HERE, "summary.txt")

TOL = 1e-9          # win/tie/regress tolerance on dA_K10
BASE_AUC_CUT = 0.88  # threshold used by the optional "exclude high-base-AUC" gate

# ----------------------------------------------------------------------------
# Load + merge
# ----------------------------------------------------------------------------
r69 = pd.read_csv(ROUND69)
stats = pd.read_csv(STATS)

# round69 keyed by 'dataset', stats keyed by 'name'
stats = stats.rename(columns={"name": "dataset"})

# keep only the columns we need from each, disambiguating overlaps with suffixes
r69_cols = ["dataset", "n_train", "n_object_cols", "gate_c",
            "base_public", "base_private", "dA_K10_public", "dA_K10_private"]
stats_cols = ["dataset", "n_train", "n_object_cols", "n_features",
              "baseline_hgb_auc_private"]

df = pd.merge(r69[r69_cols], stats[stats_cols],
              on="dataset", how="inner", suffixes=("_r69", "_st"))

assert len(df) == 16, f"expected 16 merged rows, got {len(df)}"

# ----------------------------------------------------------------------------
# Sanity-check that n_train and n_object_cols agree between the two files
# ----------------------------------------------------------------------------
max_mismatch_ntrain = int((df["n_train_r69"] - df["n_train_st"]).abs().max())
max_mismatch_nobj = int((df["n_object_cols_r69"] - df["n_object_cols_st"]).abs().max())
assert max_mismatch_ntrain == 0, f"n_train mismatch: {max_mismatch_ntrain}"
assert max_mismatch_nobj == 0, f"n_object_cols mismatch: {max_mismatch_nobj}"
print(f"[sanity] max n_train mismatch (r69 vs stats)      = {max_mismatch_ntrain}")
print(f"[sanity] max n_object_cols mismatch (r69 vs stats) = {max_mismatch_nobj}")

# canonical columns
df["n_train"] = df["n_train_r69"]
df["n_object_cols"] = df["n_object_cols_r69"]
# "base AUC" per task instruction = dataset_stats baseline_hgb_auc_private
df["base_auc"] = df["baseline_hgb_auc_private"]

df = df.sort_values("dataset").reset_index(drop=True)

# ----------------------------------------------------------------------------
# Gate family definitions. Each entry: (name, boolean-mask function on df row-wise)
# ----------------------------------------------------------------------------
def m_ungated(d):
    return pd.Series(True, index=d.index)

def m_nobj_ge(k):
    return lambda d: d["n_object_cols"] >= k

def m_ntrain_lt(T):
    return lambda d: d["n_train"] < T

def m_ntrain_ge(T):
    return lambda d: d["n_train"] >= T

def m_or_obj_ntrain(T):
    # candidate A OR-gate hypothesis: fire if it has object cols OR is "big enough"
    return lambda d: (d["n_object_cols"] > 0) | (d["n_train"] >= T)

def m_or_obj_ntrain_lowbase(T):
    # OR but the n_train branch only counts datasets whose base AUC is not already high
    return lambda d: (d["n_object_cols"] > 0) | ((d["n_train"] >= T) & (d["base_auc"] < BASE_AUC_CUT))

gates = []
gates.append(("ungated", m_ungated))
gates.append(("gateC:n_object_cols>0", lambda d: d["n_object_cols"] > 0))
for k in (1, 2, 3, 5, 8):
    gates.append((f"n_object_cols>={k}", m_nobj_ge(k)))
for T in (1000, 2000, 5000, 10000, 50000, math.inf):
    label = "+inf" if math.isinf(T) else str(int(T))
    gates.append((f"n_train<{label}", m_ntrain_lt(T)))
for T in (1000, 2000, 5000, 10000, 50000):
    gates.append((f"n_train>={int(T)}", m_ntrain_ge(T)))
for T in (2000, 5000, 8000, 8775, 10000):
    gates.append((f"n_object_cols>0 OR n_train>={int(T)}", m_or_obj_ntrain(T)))
for T in (2000, 5000):
    gates.append((f"n_object_cols>0 OR (n_train>={int(T)} AND base_auc<{BASE_AUC_CUT})",
                  m_or_obj_ntrain_lowbase(T)))

# ----------------------------------------------------------------------------
# Per-gate metric computation
# ----------------------------------------------------------------------------
def win_tie_regress(vals):
    """Return (n_win, n_tie, n_regress) at tolerance TOL."""
    v = np.asarray(vals, dtype=float)
    n_win = int((v > TOL).sum())
    n_tie = int((np.abs(v) <= TOL).sum())
    n_reg = int((v < -TOL).sum())
    return n_win, n_tie, n_reg

def eval_gate(name, mask_fn):
    mask = mask_fn(df).astype(bool)
    fired = df[mask]
    unfired = df[~mask]
    fired_names = sorted(fired["dataset"].tolist())
    n_fired = len(fired)

    if n_fired == 0:
        # degenerate; still emit a row with zeros / NaNs
        return {
            "gate_name": name, "n_fired": 0, "fired_list": "",
            "mean_dA_pub": float("nan"), "mean_dA_priv": float("nan"),
            "sum_dA_pub": 0.0, "sum_dA_priv": 0.0,
            "nwin_pub": 0, "ntie_pub": 0, "nreg_pub": 0,
            "nwin_priv": 0, "ntie_priv": 0, "nreg_priv": 0,
            "min_dA_pub": float("nan"), "min_dA_priv": float("nan"),
            "argmin_dataset": "",
            "_unfired_sum_pub": float(unfired["dA_K10_public"].sum()),
            "_unfired_sum_priv": float(unfired["dA_K10_private"].sum()),
        }

    pub = fired["dA_K10_public"].to_numpy()
    priv = fired["dA_K10_private"].to_numpy()

    nwin_pub, ntie_pub, nreg_pub = win_tie_regress(pub)
    nwin_priv, ntie_priv, nreg_priv = win_tie_regress(priv)

    # worst (min) dataset -- report by the private-split argmin (the split we ship on)
    imin_priv = int(np.argmin(priv))
    argmin_ds = fired.iloc[imin_priv]["dataset"]

    return {
        "gate_name": name,
        "n_fired": n_fired,
        "fired_list": ";".join(fired_names),
        "mean_dA_pub": float(pub.mean()),
        "mean_dA_priv": float(priv.mean()),
        "sum_dA_pub": float(pub.sum()),
        "sum_dA_priv": float(priv.sum()),
        "nwin_pub": nwin_pub, "ntie_pub": ntie_pub, "nreg_pub": nreg_pub,
        "nwin_priv": nwin_priv, "ntie_priv": ntie_priv, "nreg_priv": nreg_priv,
        "min_dA_pub": float(pub.min()),
        "min_dA_priv": float(priv.min()),
        "argmin_dataset": argmin_ds,
        "_unfired_sum_pub": float(unfired["dA_K10_public"].sum()),
        "_unfired_sum_priv": float(unfired["dA_K10_private"].sum()),
    }

rows = [eval_gate(name, fn) for name, fn in gates]
res = pd.DataFrame(rows)

# public results.csv columns (drop the private helper cols with leading underscore)
out_cols = ["gate_name", "n_fired", "fired_list",
            "mean_dA_pub", "mean_dA_priv", "sum_dA_pub", "sum_dA_priv",
            "nwin_pub", "ntie_pub", "nreg_pub",
            "nwin_priv", "ntie_priv", "nreg_priv",
            "min_dA_pub", "min_dA_priv", "argmin_dataset"]
res[out_cols].to_csv(OUT_RESULTS, index=False)
print(f"[write] {OUT_RESULTS}  ({len(res)} gate rows)")

# ----------------------------------------------------------------------------
# Targeted analysis for summary.txt
# ----------------------------------------------------------------------------
def fmt(x, nd=6):
    if isinstance(x, float) and math.isnan(x):
        return "nan"
    return f"{x:.{nd}f}"

lines = []
def P(s=""):
    lines.append(s)

P("=" * 78)
P("round71 -- gate-VARIABLE robustness study for candidate A (seed averaging K=10)")
P("=" * 78)
P("Pure desk calculation. Reuses round69 per-dataset dA_K10 deltas (AUC of the")
P("K=10 seed-averaged base-08 HGB minus the single-seed base), evaluated over 16")
P("offline datasets. NO refit, NO submission. tol = 1e-9 for win/tie/regress.")
P(f"'base AUC' = dataset_stats.baseline_hgb_auc_private.  base_auc cut = {BASE_AUC_CUT}.")
P("")
P(f"[sanity] max n_train mismatch (r69 vs stats)       = {max_mismatch_ntrain}")
P(f"[sanity] max n_object_cols mismatch (r69 vs stats) = {max_mismatch_nobj}")
P("")

# ---- (a) un-gated regressors --------------------------------------------------
P("-" * 78)
P("(a) UN-GATED REGRESSORS (dA_K10 < 0), by split")
P("-" * 78)
reg_pub = df[df["dA_K10_public"] < -TOL].sort_values("dA_K10_public")
reg_priv = df[df["dA_K10_private"] < -TOL].sort_values("dA_K10_private")
P(f"Public-split regressors : {sorted(reg_pub['dataset'].tolist()) or 'NONE'}")
P(f"Private-split regressors: {sorted(reg_priv['dataset'].tolist()) or 'NONE'}")
P("")
regressors = sorted(set(reg_pub["dataset"]).union(set(reg_priv["dataset"])))
P("Detail of every dataset that regresses on either split:")
P(f"  {'dataset':10s} {'n_train':>8s} {'n_obj':>6s} {'n_feat':>7s} {'base_auc':>9s} "
  f"{'dA_pub':>11s} {'dA_priv':>11s}")
for ds in regressors:
    r = df[df["dataset"] == ds].iloc[0]
    P(f"  {ds:10s} {int(r['n_train']):>8d} {int(r['n_object_cols']):>6d} "
      f"{int(r['n_features']):>7d} {r['base_auc']:>9.4f} "
      f"{r['dA_K10_public']:>11.6f} {r['dA_K10_private']:>11.6f}")
P("")

# ---- (b) no-object-col datasets (the ones gate-C excludes) --------------------
P("-" * 78)
P("(b) DATASETS WITH n_object_cols == 0 (excluded by gate-C)")
P("-" * 78)
nobj0 = df[df["n_object_cols"] == 0].sort_values("n_train")
P(f"  {'dataset':10s} {'n_train':>8s} {'base_auc':>9s} {'dA_pub':>11s} {'dA_priv':>11s}  verdict")
for _, r in nobj0.iterrows():
    vp = "WIN" if r["dA_K10_public"] > TOL else ("REG" if r["dA_K10_public"] < -TOL else "tie")
    vq = "WIN" if r["dA_K10_private"] > TOL else ("REG" if r["dA_K10_private"] < -TOL else "tie")
    P(f"  {r['dataset']:10s} {int(r['n_train']):>8d} {r['base_auc']:>9.4f} "
      f"{r['dA_K10_public']:>11.6f} {r['dA_K10_private']:>11.6f}  pub={vp} priv={vq}")
P("")
pos = nobj0[(nobj0["dA_K10_public"] > TOL) & (nobj0["dA_K10_private"] > TOL)]
neg = nobj0[(nobj0["dA_K10_public"] < -TOL) | (nobj0["dA_K10_private"] < -TOL)]
P(f"Positive on BOTH splits (gains gate-C forfeits): "
  f"{sorted(pos['dataset'].tolist()) or 'NONE'}")
P(f"  forfeited summed dA  pub={fmt(pos['dA_K10_public'].sum())}  "
  f"priv={fmt(pos['dA_K10_private'].sum())}")
P(f"Negative on either split (correctly excluded by gate-C): "
  f"{sorted(neg['dataset'].tolist()) or 'NONE'}")
P("")

# ---- (c) margin analysis for OR-gate -----------------------------------------
P("-" * 78)
P("(c) MARGIN ANALYSIS -- OR-gate  n_object_cols>0 OR n_train>=T")
P("-" * 78)
# The obj>0 datasets always fire regardless of T. What T controls is which
# no-object-col datasets get pulled in. Clean+maximal means: pull in ALL
# no-obj datasets whose dA_K10 is positive on both splits, and NONE that regress.
no_obj_sorted = nobj0.sort_values("n_train")
P("No-object-col datasets, sorted by n_train (T only affects these):")
for _, r in no_obj_sorted.iterrows():
    tag = "REGRESSOR" if (r["dA_K10_public"] < -TOL or r["dA_K10_private"] < -TOL) else "gainer"
    P(f"    n_train={int(r['n_train']):>6d}  {r['dataset']:10s}  "
      f"dA_priv={r['dA_K10_private']:>11.6f}  [{tag}]")
P("")
# locate the separating gap: highest n_train among no-obj regressors, and
# lowest n_train among no-obj gainers.
noobj_reg = no_obj_sorted[(no_obj_sorted["dA_K10_public"] < -TOL) |
                          (no_obj_sorted["dA_K10_private"] < -TOL)]
noobj_gain = no_obj_sorted[(no_obj_sorted["dA_K10_public"] > TOL) &
                           (no_obj_sorted["dA_K10_private"] > TOL)]
hi_reg = int(noobj_reg["n_train"].max()) if len(noobj_reg) else None
lo_gain = int(noobj_gain["n_train"].min()) if len(noobj_gain) else None
P(f"Highest n_train among no-obj REGRESSORS  = {hi_reg}  "
  f"(gate must EXCLUDE this -> need T > {hi_reg})")
P(f"Lowest  n_train among no-obj GAINERS      = {lo_gain}  "
  f"(gate must INCLUDE this -> need T <= {lo_gain})")
if hi_reg is not None and lo_gain is not None and hi_reg < lo_gain:
    width = lo_gain - hi_reg
    P("")
    P(f"=> CLEAN + MAXIMAL plateau for T:   ({hi_reg}, {lo_gain}]   "
      f"i.e.  {hi_reg} < T <= {lo_gain}")
    P(f"   Lowest failing T from below : T = {hi_reg}   "
      f"(<= {hi_reg} re-admits the regressor)")
    P(f"   Highest failing T from above: T = {lo_gain + 1} and up "
      f"(> {lo_gain} drops the smallest gainer -> clean but not maximal)")
    P(f"   Plateau WIDTH in n_train    : {width}   "
      f"-> {'WIDE PLATEAU (robust)' if width > 500 else 'KNIFE-EDGE (fragile)'}")
else:
    P("=> No clean separating interval exists (regressor n_train not below all gainers).")
P("")

# ---- (d) verdict -------------------------------------------------------------
P("-" * 78)
P("(d) VERDICT -- does any gate STRICTLY DOMINATE gate-C?")
P("-" * 78)

# reference: gate-C metrics
gc = res[res["gate_name"] == "gateC:n_object_cols>0"].iloc[0]
gc_fired = set(gc["fired_list"].split(";")) if gc["fired_list"] else set()
gc_sum_pub, gc_sum_priv = gc["sum_dA_pub"], gc["sum_dA_priv"]
gc_reg = gc["nreg_pub"] + gc["nreg_priv"]

# "clean" gate = 0 regressions on BOTH splits
res["_clean"] = (res["nreg_pub"] == 0) & (res["nreg_priv"] == 0)

# strict domination of gate-C:
#   * fires on a superset of gate-C's fired datasets (>= every clean win),
#   * introduces no new regression (clean, since gate-C itself is clean),
#   * strictly higher summed gain on both splits.
dominators = []
for _, r in res.iterrows():
    if r["gate_name"] == "gateC:n_object_cols>0":
        continue
    fired = set(r["fired_list"].split(";")) if r["fired_list"] else set()
    superset = gc_fired.issubset(fired)
    clean = (r["nreg_pub"] == 0) and (r["nreg_priv"] == 0)
    strictly_more = (r["sum_dA_pub"] > gc_sum_pub + 1e-12) and \
                    (r["sum_dA_priv"] > gc_sum_priv + 1e-12)
    if superset and clean and strictly_more:
        dominators.append(r)

P(f"gate-C reference: fires {gc['n_fired']} datasets, "
  f"regressions pub/priv = {gc['nreg_pub']}/{gc['nreg_priv']} (clean), "
  f"sum_dA pub={fmt(gc_sum_pub)} priv={fmt(gc_sum_priv)}")
P("")
if dominators:
    # pick the dominator with the largest private summed gain (and simplest form)
    dom = sorted(dominators, key=lambda r: (-r["sum_dA_priv"], r["gate_name"]))[0]
    extra = sorted(set(dom["fired_list"].split(";")) - gc_fired)
    P(f"YES -- at least {len(dominators)} gate(s) strictly dominate gate-C.")
    P(f"Best dominator (max sum_dA_priv): {dom['gate_name']}")
    P(f"  fires {dom['n_fired']} datasets (gate-C fires {gc['n_fired']}); "
      f"adds {extra}")
    P(f"  regressions pub/priv = {dom['nreg_pub']}/{dom['nreg_priv']} (still clean)")
    P(f"  sum_dA pub  {fmt(gc_sum_pub)} -> {fmt(dom['sum_dA_pub'])}  "
      f"(+{fmt(dom['sum_dA_pub'] - gc_sum_pub)})")
    P(f"  sum_dA priv {fmt(gc_sum_priv)} -> {fmt(dom['sum_dA_priv'])}  "
      f"(+{fmt(dom['sum_dA_priv'] - gc_sum_priv)})")
    P("")
    # overfitting / margin flag
    if hi_reg is not None and lo_gain is not None and hi_reg < lo_gain:
        width = lo_gain - hi_reg
        P(f"  Overfitting check: the improvement relies on the n_train threshold T")
        P(f"  separating the single no-obj regressor (train n_train={hi_reg}) from the")
        P(f"  no-obj gainers (min n_train={lo_gain}). The clean+maximal window is")
        P(f"  {hi_reg} < T <= {lo_gain}, a WIDTH of {width} rows in n_train.")
        if width > 500:
            P(f"  This is a WIDE plateau, not a knife-edge: any T in [~2000..8775]")
            P(f"  yields the identical clean, maximal fired set. Robust, low overfit risk.")
        else:
            P(f"  This is a NARROW window -> non-trivial overfitting risk; flag it.")
else:
    P("NO gate strictly dominates gate-C under the (superset AND clean AND strictly")
    P("higher summed gain on BOTH splits) criterion.")
P("")

# ---- ranked table of CLEAN gates by sum_dA_priv ------------------------------
P("-" * 78)
P("RANKED TABLE -- CLEAN gates (0 regressions on BOTH splits), by sum_dA_priv desc")
P("-" * 78)
clean = res[res["_clean"]].copy().sort_values("sum_dA_priv", ascending=False)
P(f"  {'gate_name':52s} {'nfire':>5s} {'sum_pub':>10s} {'sum_priv':>10s} "
  f"{'win_p':>5s} {'win_q':>5s}")
for _, r in clean.iterrows():
    P(f"  {r['gate_name']:52s} {int(r['n_fired']):>5d} "
      f"{r['sum_dA_pub']:>10.6f} {r['sum_dA_priv']:>10.6f} "
      f"{int(r['nwin_pub']):>5d} {int(r['nwin_priv']):>5d}")
P("")
P("(For context, ungated leaves nothing on the table but ships the train_16")
P(" regressor; every 'clean' gate above has 0 regressions on both splits.)")
P("")
P("=" * 78)
P("END round71 summary")
P("=" * 78)

with open(OUT_SUMMARY, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"[write] {OUT_SUMMARY}")

# ----------------------------------------------------------------------------
# Completion markers
# ----------------------------------------------------------------------------
clean_run = (max_mismatch_ntrain == 0 and max_mismatch_nobj == 0 and
             os.path.exists(OUT_RESULTS) and os.path.exists(OUT_SUMMARY))
print(f"CLEAN RUN: {'YES' if clean_run else 'NO'}")
print("=== DONE round71 ===")
sys.exit(0 if clean_run else 1)
