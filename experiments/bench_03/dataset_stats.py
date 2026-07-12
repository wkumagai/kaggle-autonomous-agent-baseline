"""Per-dataset stats + baseline HistGradientBoosting private AUC for train_01..16."""
import os
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = "/Users/kumacmini/kaggle-autonomous-agent-baseline/data"
OUT = "/Users/kumacmini/kaggle-autonomous-agent-baseline/experiments/bench_03/dataset_stats.csv"


def process(name):
    d = os.path.join(ROOT, name)
    train = pd.read_csv(os.path.join(d, "train.csv"))
    test = pd.read_csv(os.path.join(d, "test.csv"))
    sol = pd.read_csv(os.path.join(d, "solution.csv"))

    feat_cols = [c for c in train.columns if c not in ("row_id", "target")]
    X = train[feat_cols]
    y = train["target"]
    obj_mask = [X[c].dtype == object for c in feat_cols]
    obj_cols = [c for c, m in zip(feat_cols, obj_mask) if m]

    missing_pct = float(X.isna().to_numpy().mean() * 100)
    max_card = int(max((X[c].nunique(dropna=True) for c in obj_cols), default=0))

    clf = HistGradientBoostingClassifier(categorical_features=obj_mask, random_state=0)
    Xt = X.copy()
    Xte = test[feat_cols].copy()
    for c in obj_cols:
        Xt[c] = Xt[c].astype("category")
        Xte[c] = pd.Categorical(Xte[c], categories=Xt[c].cat.categories)
    clf.fit(Xt, y)
    proba = clf.predict_proba(Xte)[:, 1]

    pred = pd.DataFrame({"row_id": test["row_id"], "pred": proba})
    merged = sol.merge(pred, on="row_id", how="left")
    priv = merged[merged["Usage"] == "Private"]
    auc = float(roc_auc_score(priv["target"], priv["pred"]))

    return dict(
        name=name,
        n_train=len(train),
        n_test=len(test),
        n_features=len(feat_cols),
        n_object_cols=len(obj_cols),
        missing_pct=round(missing_pct, 4),
        target_rate=round(float(y.mean()), 4),
        max_cat_cardinality=max_card,
        public_rows=int((sol["Usage"] == "Public").sum()),
        private_rows=int((sol["Usage"] == "Private").sum()),
        baseline_hgb_auc_private=round(auc, 4),
    )


def main():
    names = [f"train_{i:02d}" for i in range(1, 17)]
    with ProcessPoolExecutor(max_workers=8) as ex:
        rows = list(ex.map(process, names))
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
