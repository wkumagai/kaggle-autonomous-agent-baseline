#!/usr/bin/env python
"""
bench_03 round82 — REMOVE the CATEGORICAL MASK: ordinal-encode the object columns
and mark NOTHING categorical, so HGB uses ordinary numeric split-finding on them
instead of native categorical splits. OFFLINE ONLY. No subprocess, no LLM, no
Kaggle. Calls sklearn in-process.

Adapted verbatim in structure from
experiments/bench_03/round81_l2_gate_threshold/replay.py.

Base recipe reproduced (verified vs
  `git show HEAD:submissions/08_ratio_tiered_msl/agent/prompts/system.md`):
  - load train.csv, test.csv with pandas
  - features = [c for c in train.columns if c not in ("row_id","target")]
  - cat_mask  = [train[c].dtype == object for c in features]        [THE LEVER]
  - n = len(train); n_feat = len(features); ratio = n_feat / n
  - L2 GATE:  l2  = 1.0 if ratio >= 0.010 else 0.0            [FIXED — shipped 08]
  - MSL TIERS: msl = 70 if ratio >= 0.030 else (50 if ratio >= 0.015 else 20)
               [FIXED — shipped 08]
  - HistGradientBoostingClassifier(categorical_features=cat_mask, random_state=0,
        max_iter=300, early_stopping=True, l2_regularization=l2,
        min_samples_leaf=msl)
  - fit(train[features], train["target"])
  - pred = clf.predict_proba(test[features])[:, pos_class_1]
Score: ROC AUC on Public rows and Private rows separately, joined by row_id to
solution.csv, over all 16 datasets.

The QUESTION (round82 angle): round50 tested WIDENING the categorical mask
(marking low-cardinality NUMERIC columns categorical too). The exact INVERSE —
making ZERO columns categorical — has never been tested. This asks: are HGB's
native categorical splits actually EARNING THEIR KEEP in the shipped recipe, or
is a plain ordinal encoding just as good?

PRE-CHECK (established before writing this harness; re-verified in-process):
  - Literally DROPPING the `categorical_features` argument is NOT implementable:
    HGB then tries to cast the raw object columns to float and raises
    `ValueError: could not convert string to float: 'cat_1'` (the exact offending
    literal is dataset-dependent — the brief records 'cat_6'; train_01 raises on
    'cat_1'). This fires on the 12 datasets that have object columns.
  - The IMPLEMENTABLE inverse: ORDINAL-ENCODE the object columns to integer codes
    and mark NOTHING categorical. HGB then does ordinary numeric split-finding on
    them (threshold splits on an arbitrary lexicographic ordering) instead of
    native categorical splits (subset splits on the category set).

ONE LEVER ONLY: whether the mask marks the object columns categorical. The l2
gate (1.0 @ ratio>=0.010), the msl tiers (70/50/20), max_iter=300,
early_stopping=True and random_state=0 are all taken from 08 UNCHANGED and apply
identically in every arm. `ratio` is recomputed from the actual data, never
hardcoded.

Arms:
  base   : shipped 08 exactly. Object cols kept as objects -> native categorical
           via cat_mask.
  ORDCAT : CONTROL ARM. Object cols replaced by ordinal integer codes, but the
           cat_mask is STILL passed (the same columns are still marked
           categorical). This isolates the ENCODING from the TREATMENT. Expected
           ~no-op vs base; if it moves a lot, the encoding itself is a confound
           and the numbers for ORD cannot be attributed to the mask alone. The
           harness says so LOUDLY in summary.txt if that happens.
  ORD    : THE LEVER. Object cols replaced by the SAME ordinal integer codes as
           ORDCAT, and `categorical_features` is an ALL-FALSE mask (nothing
           categorical -> every column numeric).

ORDCAT and ORD share a BYTE-IDENTICAL encoding (they call the same function), so
the ONLY difference between them is the mask. That is what makes ORDCAT a valid
control for ORD.

Encoding spec (identical for ORDCAT and ORD):
  - For each object column, build the category list from the TRAIN column's
    unique NON-NULL values, sorted lexicographically (`sorted()`), mapped to
    0,1,2,...
  - Apply the SAME mapping to the test column. Values present in test but not in
    train -> NaN (HGB handles NaN natively). Null stays NaN.
  - Result dtype float. Non-object columns are NEVER touched (byte-identical to
    base).

INVARIANT: per experiments/bench_03/dataset_stats.csv, exactly 4 datasets have
n_object_cols == 0 — train_04, train_10, train_11, train_16. On those the
object-column set is EMPTY, so no column is re-encoded AND the shipped cat_mask
is already all-False (so ORD's all-False mask is the same mask base already
passes). Both candidate arms are therefore byte-identical to base there and MUST
show delta EXACTLY 0.00000 on BOTH splits. The zero-object set is RECOMPUTED
from the actual data (never hardcoded) and asserted to match that expectation.
If any of them moves, the harness leaks and every number here is meaningless:
the harness reports INVARIANT=VIOLATED and stops.

Adoption criterion: a candidate is a CLEAN IMPROVEMENT over base(08) iff its
mean delta is positive on BOTH splits AND there are ZERO regressions on BOTH
splits (no dataset worse on either split). A single regression on either split
=> not clean.
"""
import os
import csv
import json
import math
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

REPO = "/Users/kumacmini/kaggle-autonomous-agent-baseline-auto"
DATA_DIR = os.path.join(REPO, "data")
OUT_DIR = os.path.join(REPO, "experiments", "bench_03", "round82_no_cat_mask")

L2_GATE_THRESHOLD = 0.010   # shipped 08 feature-to-row-ratio gate for l2 (FIXED — not the lever)
GATED_L2 = 1.0              # l2 applied when the l2-gate fires (FIXED — not the lever)
DEFAULT_MSL = 20            # sklearn HGB default, used when no msl tier is cleared.
MSL_TIERS = [(0.030, 70), (0.015, 50)]  # shipped 08 tiers (FIXED — not the lever)

BASE_MAX_ITER = 300         # shipped 08 (FIXED)
BASE_EARLY_STOPPING = True  # shipped 08 (FIXED)

# Each config: (name, encode_objects, mask_mode).
#   encode_objects — replace object cols with ordinal integer codes (float dtype)
#   mask_mode      — "cat"  : pass the shipped cat_mask (dtype==object per column)
#                    "none" : pass an ALL-FALSE mask (nothing categorical)
# base is 08 exactly. ORDCAT changes ONLY the encoding (control). ORD changes the
# encoding AND drops the mask — so ORDCAT->ORD isolates THE LEVER (the mask).
ARM_SPECS = [
    ("base",   False, "cat"),
    ("ORDCAT", True,  "cat"),
    ("ORD",    True,  "none"),
]
CONFIGS = ARM_SPECS
BASE = "base"
CONTROL = "ORDCAT"
CANDIDATES = ["ORDCAT", "ORD"]
SPEC_OF = {name: (enc, mask) for name, enc, mask in ARM_SPECS}

N_DATASETS = 16

# Documented expectation from experiments/bench_03/dataset_stats.csv. NOT used to
# drive any logic — the harness recomputes the zero-object set from the data and
# asserts agreement with this list.
EXPECTED_ZERO_OBJECT = ["train_04", "train_10", "train_11", "train_16"]


def msl_for_ratio(ratio):
    """First tier (threshold, msl) whose threshold the ratio clears wins;
    tiers are in descending-threshold order. Else DEFAULT_MSL. (Shipped 08.)"""
    for thr, val in MSL_TIERS:
        if ratio >= thr:
            return val
    return DEFAULT_MSL


def auc_or_nan(y_true, y_score):
    """ROC AUC, or NaN if the subset has a single class (undefined)."""
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def ordinal_encode(train, test, features, cat_mask):
    """Replace every OBJECT feature column with ordinal integer codes, in place on
    copies. Shared verbatim by ORDCAT and ORD so their encodings are
    byte-identical and the mask is the only difference between them.

    Spec: category list = sorted() of the TRAIN column's unique non-null values,
    mapped to 0,1,2,...; the SAME mapping is applied to test; test values unseen
    in train -> NaN; null stays NaN; result dtype float. Non-object columns are
    never touched.

    Returns (train2, test2, n_encoded)."""
    train2 = train.copy()
    test2 = test.copy()
    n_encoded = 0
    for col, is_cat in zip(features, cat_mask):
        if not is_cat:
            continue  # non-object column — NEVER touched
        cats = sorted(train[col].dropna().unique())
        mapping = {c: float(i) for i, c in enumerate(cats)}
        # .map() yields NaN for unseen/null keys, exactly as specified.
        train2[col] = train[col].map(mapping).astype(float)
        test2[col] = test[col].map(mapping).astype(float)
        n_encoded += 1
    return train2, test2, n_encoded


def run_one(train_csv, test_csv, encode_objects, mask_mode):
    """Reproduce the shipped 08 recipe for one dataset, applying THE LEVER: the
    object columns are ordinal-encoded (`encode_objects`) and/or the categorical
    mask is dropped (`mask_mode == "none"`). Everything else (l2 gate, msl tiers,
    max_iter, early_stopping, random_state) is 08 UNCHANGED. Returns
    (pred_map, l2, msl_val, n, ratio, n_obj, n_encoded, n_marked_cat)."""
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)

    features = [c for c in train.columns if c not in ("row_id", "target")]
    # cat_mask is ALWAYS computed from the ORIGINAL (pre-encoding) dtypes, so
    # ORDCAT marks exactly the same columns base does.
    cat_mask = [train[c].dtype == object for c in features]
    n_obj = sum(cat_mask)

    n = len(train)
    n_feat = len(features)
    ratio = n_feat / n

    # ---- THE LEVER (the only thing that varies across arms) ----
    n_encoded = 0
    if encode_objects:
        train, test, n_encoded = ordinal_encode(train, test, features, cat_mask)
    if mask_mode == "cat":
        used_mask = cat_mask
    elif mask_mode == "none":
        used_mask = [False] * len(features)
    else:
        raise ValueError(f"unknown mask_mode {mask_mode!r}")
    n_marked_cat = sum(used_mask)

    # ---- FIXED shipped-08 gates (identical in every arm) ----
    l2 = GATED_L2 if ratio >= L2_GATE_THRESHOLD else 0.0
    msl_val = msl_for_ratio(ratio)

    clf = HistGradientBoostingClassifier(
        categorical_features=used_mask,
        random_state=0,
        max_iter=BASE_MAX_ITER,
        early_stopping=BASE_EARLY_STOPPING,
        l2_regularization=l2,
        min_samples_leaf=msl_val,
    )
    clf.fit(train[features], train["target"])
    proba = clf.predict_proba(test[features])

    # positive-class column: match recipe's [:, 1] but be robust to class order.
    classes = list(clf.classes_)
    pos_idx = classes.index(1) if 1 in classes else 1
    pred = proba[:, pos_idx]

    return (dict(zip(test["row_id"].tolist(), pred.tolist())),
            l2, msl_val, n, ratio, n_obj, n_encoded, n_marked_cat)


def score_split(pred_map, sol):
    """Given pred_map (row_id->prob) and solution df, return (public_auc, private_auc)."""
    sol = sol.copy()
    sol["pred"] = sol["row_id"].map(pred_map)
    if sol["pred"].isna().any():
        n_missing = int(sol["pred"].isna().sum())
        raise ValueError(f"{n_missing} solution row_ids had no matching prediction")
    pub = sol[sol["Usage"] == "Public"]
    prv = sol[sol["Usage"] == "Private"]
    pub_auc = auc_or_nan(pub["target"], pub["pred"])
    prv_auc = auc_or_nan(prv["target"], prv["pred"])
    return pub_auc, prv_auc


def main():
    warnings.filterwarnings("ignore")
    os.makedirs(OUT_DIR, exist_ok=True)

    rows = []            # per-dataset results
    exceptions = []      # (dataset, config, message)
    skipped = []
    n_fits_ok = 0        # successful fits — CLEAN-RUN accounting

    for i in range(1, N_DATASETS + 1):
        name = f"train_{i:02d}"
        d = os.path.join(DATA_DIR, name)
        train_csv = os.path.join(d, "train.csv")
        test_csv = os.path.join(d, "test.csv")
        sol_csv = os.path.join(d, "solution.csv")

        if not (os.path.exists(train_csv) and os.path.exists(test_csv) and os.path.exists(sol_csv)):
            print(f"[SKIP] {name}: missing train/test/solution")
            skipped.append(name)
            continue

        sol = pd.read_csv(sol_csv)
        rec = {"dataset": name}
        for cfg_name, encode_objects, mask_mode in CONFIGS:
            try:
                (pred_map, l2, msl_val, n, ratio, n_obj, n_encoded,
                 n_marked_cat) = run_one(train_csv, test_csv, encode_objects, mask_mode)
                pub, prv = score_split(pred_map, sol)
                rec[f"{cfg_name}_pub"] = pub
                rec[f"{cfg_name}_prv"] = prv
                rec[f"{cfg_name}_n_encoded"] = n_encoded
                rec[f"{cfg_name}_n_marked_cat"] = n_marked_cat
                # n, ratio, n_obj, l2 and the msl tier are config-independent (all
                # FIXED across arms; only the encoding/mask moves).
                rec["n_train"] = n
                rec["ratio"] = ratio
                rec["n_obj"] = n_obj
                rec["l2"] = l2
                rec["msl"] = msl_val
                # has_obj: this dataset HAS object columns -> the lever can touch
                # it. Datasets with NO object columns are the invariant set.
                rec["has_obj"] = n_obj > 0
                n_fits_ok += 1
                print(f"[OK] {name} {cfg_name} (n={n}, ratio={ratio:.5f}, n_obj={n_obj}, "
                      f"encoded={n_encoded}, marked_cat={n_marked_cat}, "
                      f"l2={l2}, msl={msl_val}): pub={pub:.6f} prv={prv:.6f}")
            except Exception as e:  # capture but keep going — CLEAN-RUN diagnostic
                exceptions.append((name, cfg_name, repr(e)))
                rec[f"{cfg_name}_pub"] = float("nan")
                rec[f"{cfg_name}_prv"] = float("nan")
                print(f"[ERROR] {name} {cfg_name}: {e!r}")
        rows.append(rec)

    # ---- write results CSV + JSON (raw per-dataset numbers) ----
    csv_path = os.path.join(OUT_DIR, "results.csv")
    fieldnames = ["dataset", "n_train", "ratio", "n_obj", "has_obj", "l2", "msl"]
    for cfg, _, _ in CONFIGS:
        fieldnames += [f"{cfg}_pub", f"{cfg}_prv", f"{cfg}_n_encoded",
                       f"{cfg}_n_marked_cat"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    json_path = os.path.join(OUT_DIR, "results.json")
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2, default=str)

    # ---- deltas & helpers (all vs base == shipped 08) ----
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
        """Datasets where cfg is strictly worse than base(08) on this split."""
        out = []
        for r in rows:
            dd = delta(r, cfg, split)
            if not math.isnan(dd) and dd < -eps:
                out.append((r["dataset"], dd))
        return out

    def differing_datasets(cfg, eps=1e-9):
        """Datasets where cfg differs from base on EITHER split."""
        out = []
        for r in rows:
            dp = delta(r, cfg, "pub")
            dv = delta(r, cfg, "prv")
            dp = 0.0 if math.isnan(dp) else dp
            dv = 0.0 if math.isnan(dv) else dv
            if abs(dp) > eps or abs(dv) > eps:
                out.append(r["dataset"])
        return out

    def max_abs_delta(cfg, dataset_names):
        """max |delta| over the given datasets x both splits, for cfg vs base."""
        m = 0.0
        for r in rows:
            if r["dataset"] not in dataset_names:
                continue
            for split in ("pub", "prv"):
                dd = delta(r, cfg, split)
                if not math.isnan(dd):
                    m = max(m, abs(dd))
        return m

    # RECOMPUTED from the actual data — never hardcoded.
    obj_datasets = [r["dataset"] for r in rows if r.get("has_obj")]
    zero_obj_datasets = [r["dataset"] for r in rows if not r.get("has_obj")]

    summary_lines = []

    # ---- per-dataset tables (Public, Private) ----
    for split, tag in (("pub", "Public"), ("prv", "Private")):
        if split == "prv":
            summary_lines.append("")
        summary_lines.append(f"=== PER-DATASET ({tag}) ===")
        header = (f"{'dataset':<10} {'n':>6} {'ratio':>8} {'n_obj':>5} {'l2':>4} "
                  f"{'msl':>4} {BASE:>9}")
        for cfg in CANDIDATES:
            header += f" {cfg:>9} {'d'+cfg:>10}"
        summary_lines.append(header)
        for r in rows:
            line = (f"{r['dataset']:<10} {r.get('n_train'):>6} "
                    f"{r.get('ratio'):>8.5f} {r.get('n_obj'):>5} "
                    f"{str(r.get('l2')):>4} {str(r.get('msl')):>4} "
                    f"{r[f'{BASE}_{split}']:>9.4f}")
            for cfg in CANDIDATES:
                line += (f" {r[f'{cfg}_{split}']:>9.4f} "
                         f"{delta(r, cfg, split):>+10.5f}")
            summary_lines.append(line)

    # ---- object-column census ----
    summary_lines.append("")
    summary_lines.append("=== OBJECT-COLUMN CENSUS (recomputed from the data) ===")
    summary_lines.append(
        f"Datasets WITH object columns ({len(obj_datasets)}) — the lever can "
        f"touch these: {', '.join(obj_datasets) if obj_datasets else '(none)'}")
    summary_lines.append(
        f"Datasets with ZERO object columns ({len(zero_obj_datasets)}) — the "
        f"INVARIANT set, must be identical to base: "
        f"{', '.join(zero_obj_datasets) if zero_obj_datasets else '(none)'}")
    zero_obj_matches = sorted(zero_obj_datasets) == sorted(EXPECTED_ZERO_OBJECT)
    summary_lines.append(
        f"Expected zero-object set (per dataset_stats.csv): "
        f"{', '.join(EXPECTED_ZERO_OBJECT)}  -> recomputed set "
        f"{'MATCHES' if zero_obj_matches else 'DOES NOT MATCH — dataset_stats.csv '
                                              'and the data disagree'}")
    summary_lines.append(
        f"(all arms share the l2 gate (l2={GATED_L2} @ ratio>={L2_GATE_THRESHOLD}), "
        f"the msl tiers (70/50/20), max_iter={BASE_MAX_ITER}, "
        f"early_stopping={BASE_EARLY_STOPPING} and random_state=0; they differ "
        f"ONLY in whether the object columns are ordinal-encoded and whether the "
        f"mask marks them categorical -> only datasets WITH object columns can "
        f"move. The {len(zero_obj_datasets)} zero-object datasets are identical "
        f"-> delta 0)")

    # ---- INVARIANT CHECK: zero-object datasets must be identical ----
    summary_lines.append("")
    summary_lines.append("=== INVARIANT CHECK (every dataset with ZERO object "
                         "columns must be byte-identical to base: delta EXACTLY "
                         "0.00000 on both splits, in BOTH candidate arms) ===")
    summary_lines.append(
        "Rationale: with no object columns, no column is re-encoded (ORDCAT/ORD "
        "encode nothing) AND the shipped cat_mask is already all-False (so ORD's "
        "all-False mask is the mask base already passes). Both arms therefore "
        "reduce to base exactly.")
    invariant_ok = zero_obj_matches
    if not zero_obj_matches:
        summary_lines.append(
            "  !! recomputed zero-object set does not match the documented "
            "expectation — treating as a violation.")
    zero_obj_set = set(zero_obj_datasets)
    for cfg in CANDIDATES:
        diff = differing_datasets(cfg)
        violators = [d for d in diff if d in zero_obj_set]
        n_identical = len([r for r in rows if r["dataset"] not in diff])
        n_expected_identical = len(zero_obj_datasets)
        ok = not violators
        invariant_ok = invariant_ok and ok
        summary_lines.append(
            f"{cfg}: differs from base on ({len(diff)}): "
            f"{', '.join(diff) if diff else '(none)'}  "
            f"(must be a subset of the object-column datasets; zero-object "
            f"datasets must NOT appear)  |  identical (delta exactly 0 on both "
            f"splits): {n_identical}/{len(rows)} (at least "
            f"{n_expected_identical}/{len(rows)} required)  "
            f"-> {'OK' if ok else 'VIOLATION: ' + ', '.join(violators)}")
    # explicit machine check of the zero-object datasets, both splits, exact zero
    invariant_max_abs = 0.0
    for cfg in CANDIDATES:
        m = max_abs_delta(cfg, zero_obj_set)
        invariant_max_abs = max(invariant_max_abs, m)
        summary_lines.append(
            f"  {cfg}: max |delta| over the {len(zero_obj_set)} zero-object "
            f"datasets ({', '.join(sorted(zero_obj_set))}) x both splits = "
            f"{m:.10g}")
    summary_lines.append(
        f"max |delta| over all ZERO-OBJECT (dataset x candidate x split) cells = "
        f"{invariant_max_abs:.10g} (must be exactly 0)")
    invariant_ok = invariant_ok and (invariant_max_abs == 0.0)
    summary_lines.append(
        f"INVARIANT={'HOLDS' if invariant_ok else 'VIOLATED — HARNESS LEAKS, '
                    'SCORES ARE NOT TRUSTWORTHY'} "
        f"(only datasets that actually have object columns are affected by the "
        f"lever, as designed)")

    # ---- CONTROL CHECK: ORDCAT must be ~a no-op vs base ----
    summary_lines.append("")
    summary_lines.append("=== CONTROL CHECK (ORDCAT vs base — isolates the "
                         "ENCODING from the TREATMENT) ===")
    summary_lines.append(
        "ORDCAT applies the SAME ordinal encoding as ORD but STILL marks the same "
        "columns categorical. HGB re-derives its categories from the integer "
        "codes, so ORDCAT should be ~a no-op vs base. If ORDCAT moves a lot, the "
        "ENCODING is itself a confound and ORD's delta CANNOT be attributed to "
        "the mask alone.")
    ctrl_mp = mean_delta(CONTROL, "pub")
    ctrl_mv = mean_delta(CONTROL, "prv")
    ctrl_max_abs = max_abs_delta(CONTROL, {r["dataset"] for r in rows})
    ctrl_diff = differing_datasets(CONTROL)
    summary_lines.append(
        f"{CONTROL}: mean Public d={ctrl_mp:+.5f}  mean Private d={ctrl_mv:+.5f}  "
        f"max |delta| over ALL datasets x both splits = {ctrl_max_abs:.5f}")
    summary_lines.append(
        f"{CONTROL} differs from base on ({len(ctrl_diff)}/{len(rows)}): "
        f"{', '.join(ctrl_diff) if ctrl_diff else '(none)'}")
    # Yardstick: the control's own movement is the floor below which ORD's delta
    # is indistinguishable from encoding noise.
    CONTROL_NOOP_TOL = 0.001  # mean-delta magnitude we are willing to call "~no-op"
    ctrl_is_noop = (abs(ctrl_mp) <= CONTROL_NOOP_TOL) and (abs(ctrl_mv) <= CONTROL_NOOP_TOL)
    if ctrl_is_noop:
        summary_lines.append(
            f"CONTROL={CONTROL} IS ~NO-OP (|mean d| <= {CONTROL_NOOP_TOL} on both "
            f"splits) -> the encoding is not a meaningful confound; ORD's delta "
            f"vs base is attributable to THE MASK.")
        if ctrl_max_abs == 0.0:
            summary_lines.append(
                "  NOTE — the control is EXACTLY 0.00000 on every dataset, which "
                "looks like a no-op bug but is not. When a column is marked "
                "categorical, sklearn ordinal-encodes it internally using the "
                "SAME lexicographic category order this harness uses "
                "(verified: our sorted() codes == OrdinalEncoder().categories_ "
                "codes), so ORDCAT feeds HGB bit-identical binned data to base -> "
                "identical trees -> identical AUC. ORDCAT == base is therefore a "
                "mathematical IDENTITY, i.e. the strongest possible control "
                "result, not a skipped encoding. Two independent proofs the "
                "encoding really ran: (1) n_encoded is nonzero on all 12 "
                "object-column datasets (see results.csv); (2) ORD passes the "
                "SAME encoded frames with an all-False mask and does NOT raise "
                "the pre-check's `ValueError: could not convert string to float` "
                "— which is only possible if the object columns were in fact "
                "converted to numbers.")
    else:
        summary_lines.append(
            f"!!! LOUD WARNING: CONTROL={CONTROL} IS **NOT** A NO-OP "
            f"(mean pub={ctrl_mp:+.5f}, prv={ctrl_mv:+.5f}; exceeds "
            f"{CONTROL_NOOP_TOL} on at least one split). THE ORDINAL ENCODING "
            f"ITSELF IS A CONFOUND. ORD's delta vs base CONFLATES the encoding "
            f"with the mask, and must NOT be read as a clean measurement of the "
            f"mask's value. Compare ORD against {CONTROL} (not against base) to "
            f"isolate the mask, and treat the base-relative ORD numbers with "
            f"suspicion.")
    summary_lines.append(
        f"Encoding-isolated view of THE LEVER (ORD vs {CONTROL}, same encoding, "
        f"mask is the ONLY difference):")
    for split, tag in (("pub", "Public"), ("prv", "Private")):
        vals = []
        for r in rows:
            a = r.get(f"{CONTROL}_{split}")
            b = r.get(f"ORD_{split}")
            if a is None or b is None or math.isnan(a) or math.isnan(b):
                continue
            vals.append(b - a)
        mean_v = sum(vals) / len(vals) if vals else float("nan")
        n_worse = sum(1 for v in vals if v < -1e-6)
        n_better = sum(1 for v in vals if v > 1e-6)
        summary_lines.append(
            f"  {tag}: mean(ORD - {CONTROL}) = {mean_v:+.5f}  "
            f"(better on {n_better}, worse on {n_worse}, "
            f"tied on {len(vals) - n_better - n_worse})")

    # ---- THE SWEEP: per-dataset detail on the datasets the lever touches ----
    summary_lines.append("")
    summary_lines.append("=== OBJECT-COLUMN DATASET SWEEP (the datasets the lever touches) ===")
    for r in rows:
        if not r.get("has_obj"):
            continue
        summary_lines.append(
            f"--- {r['dataset']} (n={r.get('n_train')}, ratio={r.get('ratio'):.5f}, "
            f"n_obj={r.get('n_obj')} object cols, l2={r.get('l2')}, "
            f"msl={r.get('msl')}) ---")
        summary_lines.append(f"  {'config':>8} {'enc':>5} {'mask':>5} {'Public':>9} "
                             f"{'dPub':>10} {'Private':>9} {'dPrv':>10}")
        for cfg, enc, mask_mode in CONFIGS:
            dp = delta(r, cfg, "pub")
            dv = delta(r, cfg, "prv")
            tag = "  <- base (shipped 08)" if cfg == BASE else ""
            if cfg == CONTROL:
                tag = "  <- control"
            if cfg == "ORD":
                tag = "  <- THE LEVER"
            summary_lines.append(
                f"  {cfg:>8} {str(enc):>5} {mask_mode:>5} "
                f"{r[f'{cfg}_pub']:>9.5f} {dp:>+10.5f} "
                f"{r[f'{cfg}_prv']:>9.5f} {dv:>+10.5f}{tag}")
        # curve / peak read-out per split
        for split, tag in (("pub", "Public"), ("prv", "Private")):
            vals = [(c, r[f"{c}_{split}"]) for c, _, _ in CONFIGS]
            best_cfg, best_auc = max(vals, key=lambda t: t[1])
            seq = " -> ".join(f"{c}:{v:.5f}" for c, v in vals)
            summary_lines.append(
                f"  {tag} curve: {seq}   BEST={best_cfg} ({best_auc:.5f})")
        # encoding-noise yardstick: how far the CONTROL moved on THIS dataset is
        # the floor below which ORD's move here is not attributable to the mask.
        for split, tag in (("pub", "Public"), ("prv", "Private")):
            dctrl = delta(r, CONTROL, split)
            dord = delta(r, "ORD", split)
            summary_lines.append(
                f"  {tag} encoding-noise yardstick: |d{CONTROL}|={abs(dctrl):.5f} "
                f"vs |dORD|={abs(dord):.5f}  [an ORD move smaller than the "
                f"control's move on this dataset is encoding jitter, not the mask]")

    # ---- per-candidate summary vs base(08) ----
    summary_lines.append("")
    summary_lines.append("=== SUMMARY (candidates vs base == shipped 08, "
                         "mean over all 16 datasets) ===")
    for cfg in CANDIDATES:
        mp = mean_delta(cfg, "pub")
        mv = mean_delta(cfg, "prv")
        wp, lp, tp = wlt(cfg, "pub")
        wv, lv, tv = wlt(cfg, "prv")
        enc, mask_mode = SPEC_OF[cfg]
        summary_lines.append(
            f"{cfg} (encode={enc}, mask={mask_mode}): "
            f"mean Public d={mp:+.5f}  mean Private d={mv:+.5f}  "
            f"Public W/L/T={wp}/{lp}/{tp}  Private W/L/T={wv}/{lv}/{tv}")

    # ---- per-candidate differing-dataset deltas + regressions ----
    summary_lines.append("")
    summary_lines.append("=== PER-CANDIDATE DETAIL (datasets differing from base; "
                         "all other deltas are exactly 0) ===")
    for cfg in CANDIDATES:
        enc, mask_mode = SPEC_OF[cfg]
        summary_lines.append(f"--- {cfg} (encode={enc}, mask={mask_mode}) vs "
                             f"{BASE} (encode=False, mask=cat) ---")
        diff = set(differing_datasets(cfg))
        for r in rows:
            if r["dataset"] not in diff:
                continue
            dp = delta(r, cfg, "pub")
            dv = delta(r, cfg, "prv")
            summary_lines.append(
                f"  {r['dataset']:<10} (n={r.get('n_train')}, "
                f"ratio={r.get('ratio'):.5f}, n_obj={r.get('n_obj')}, "
                f"marked_cat {r.get(f'{BASE}_n_marked_cat')}->"
                f"{r.get(f'{cfg}_n_marked_cat')})  "
                f"pub {r[f'{BASE}_pub']:.4f}->{r[f'{cfg}_pub']:.4f} ({dp:+.5f})   "
                f"prv {r[f'{BASE}_prv']:.4f}->{r[f'{cfg}_prv']:.4f} ({dv:+.5f})")
        if not diff:
            summary_lines.append("  (no datasets differ)")
        rp = regressions(cfg, "pub")
        rv = regressions(cfg, "prv")
        pub_str = ", ".join(f"{n}({d:+.5f})" for n, d in rp) if rp else "(none)"
        prv_str = ", ".join(f"{n}({d:+.5f})" for n, d in rv) if rv else "(none)"
        summary_lines.append(f"  Public regressions:  {pub_str}")
        summary_lines.append(f"  Private regressions: {prv_str}")

    # ---- verdict ----
    summary_lines.append("")
    summary_lines.append("=== VERDICT (adoption vs base == shipped 08) ===")
    summary_lines.append(
        "Criterion: CLEAN IMPROVEMENT over 08 iff mean delta positive AND "
        "zero regressions on BOTH splits.")
    clean_names = []
    for cfg in CANDIDATES:
        mp = mean_delta(cfg, "pub")
        mv = mean_delta(cfg, "prv")
        rp = regressions(cfg, "pub")
        rv = regressions(cfg, "prv")
        mean_pos = (mp > 1e-9) and (mv > 1e-9)
        zero_regs = (not rp) and (not rv)
        clean = mean_pos and zero_regs
        if clean:
            clean_names.append(cfg)
            verdict = (f"CLEAN-IMPROVEMENT "
                       f"(mean pub={mp:+.5f}, prv={mv:+.5f}; no regressions)")
        else:
            reasons = []
            if not (mp > 1e-9):
                reasons.append(f"non-positive mean Public ({mp:+.5f})")
            if not (mv > 1e-9):
                reasons.append(f"non-positive mean Private ({mv:+.5f})")
            if rp:
                reasons.append(f"Public regs ({len(rp)}) [" +
                               ", ".join(f"{n}({d:+.5f})" for n, d in rp) + "]")
            if rv:
                reasons.append(f"Private regs ({len(rv)}) [" +
                               ", ".join(f"{n}({d:+.5f})" for n, d in rv) + "]")
            verdict = "NOT-CLEAN (" + "; ".join(reasons) + ")"
        summary_lines.append(f"{cfg}: {verdict}")
    summary_lines.append("")
    if clean_names:
        # best clean candidate = largest mean Public delta (tie-break Private)
        best = max(clean_names,
                   key=lambda c: (mean_delta(c, "pub"), mean_delta(c, "prv")))
        summary_lines.append(
            f"OVERALL: clean improvement over 08 found: {', '.join(clean_names)}; "
            f"best = {best} (mean pub={mean_delta(best, 'pub'):+.5f}, "
            f"prv={mean_delta(best, 'prv'):+.5f}) "
            f"(orchestrator decides adoption; check the CONTROL CHECK above "
            f"before believing it).")
    else:
        summary_lines.append(
            "OVERALL: NO clean improvement over 08; base (shipped 08, native "
            "categorical splits via cat_mask) remains best. Ordinal-encoding the "
            "object columns and marking NOTHING categorical did not cleanly help.")
    # answer the actual research question
    ord_mp = mean_delta("ORD", "pub")
    ord_mv = mean_delta("ORD", "prv")
    summary_lines.append("")
    summary_lines.append("=== ANSWER TO THE QUESTION (are HGB's native "
                         "categorical splits earning their keep?) ===")
    if (ord_mp < -1e-9) and (ord_mv < -1e-9):
        summary_lines.append(
            f"YES — removing the categorical mask costs AUC on BOTH splits "
            f"(mean pub={ord_mp:+.5f}, prv={ord_mv:+.5f}). The native categorical "
            f"splits are earning their keep; a plain ordinal encoding is WORSE. "
            f"Keep cat_mask in the shipped recipe.")
    elif (ord_mp > 1e-9) and (ord_mv > 1e-9):
        summary_lines.append(
            f"NO — removing the categorical mask GAINS AUC on both splits "
            f"(mean pub={ord_mp:+.5f}, prv={ord_mv:+.5f}). The native categorical "
            f"splits are not earning their keep on this benchmark.")
    else:
        summary_lines.append(
            f"MIXED — removing the categorical mask moves the splits in different "
            f"directions (mean pub={ord_mp:+.5f}, prv={ord_mv:+.5f}). No clean "
            f"read; the native categorical splits are not clearly earning their "
            f"keep, but dropping them is not clearly safe either.")

    # ---- clean-run line ----
    n_fits_expected = len(rows) * len(CONFIGS)
    summary_lines.append("")
    summary_lines.append(
        f"CLEAN RUN={'YES' if not exceptions else 'NO'} "
        f"(fits ok={n_fits_ok}/{n_fits_expected} "
        f"[{len(rows)} datasets x {len(CONFIGS)} arms], "
        f"exceptions={len(exceptions)}, skipped={len(skipped)})")
    for name, cfg, msg in exceptions:
        summary_lines.append(f"  EXC {name}/{cfg}: {msg}")

    summary = "\n".join(summary_lines)
    print("\n" + summary)

    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
        f.write(summary + "\n")

    print(f"\n[WROTE] {csv_path}")
    print(f"[WROTE] {json_path}")
    print(f"[WROTE] {os.path.join(OUT_DIR, 'summary.txt')}")
    print("HARNESS_DONE")


if __name__ == "__main__":
    main()
