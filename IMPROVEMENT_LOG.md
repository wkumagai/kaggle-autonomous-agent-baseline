# 改善ログ — Autonomous Agent Prediction (Beta)

このコンペは「予測結果」ではなく「自律的に学習・提出まで行うAIエージェント」を提出するコンペです。
ここでは、提出したエージェントの構成と、そのスコア（公開リーダーボードのROC AUC）の推移を記録します。
なるべくシンプルな構成を保ったまま、少しずつ改善していく方針です。

> **⚠️ 提出上限は「1日1件」です（2026-07-12 に実測で判明）。** Kaggle 側のエラーメッセージ:
> `Submission not allowed: Your team has used its daily Submission allowance (1) today`。
> つまり 1 UTC日につき提出できるのは 1 件だけ。毎時実行しても、その日すでに 1 件提出済みなら追加提出はできません。

> **📌 次回実行への申し送り（重要）:** `submissions/02_early_stopping/` は **作成・検証（validate_submission.py 合格）まで完了済みだが未提出**（本日分の提出枠を 01_baseline が使い切ったため）。
> 次回の UTC 日になったら、**新しい 03 を作らずに、まずこの 02 をそのまま提出**すること。手順:
> `(cd submissions/02_early_stopping/agent && rm -f ../submission.zip && zip -r ../submission.zip . -x '.*')` →
> `kaggle competitions submit -c autonomous-agent-prediction-beta -f submissions/02_early_stopping/submission.zip -m "02_early_stopping: early_stopping=True, max_iter=300"` →
> 採点後に下表の 02 行のスコアを更新する。

## スコア推移

| 日付 | submission_id | 変更内容 | 公開スコア (ROC AUC) | 前回比 |
|---|---|---|---|---|
| 2026-07-12 | 54591282 | **01_baseline**: `HistGradientBoostingClassifier` を1回だけ学習・提出する最小構成。列のdtype（object=カテゴリ/順序、数値=そのまま）から自動でカテゴリ列を判定するため、データセットごとに列名や種類が変わっても手直し不要。ツールも write_file / run_command / submit_predictions / get_status の4つだけ。 | 0.787 | (ベースライン) |
| 2026-07-13 予定 | （未提出） | **02_early_stopping**: 分類器の生成だけを `early_stopping=True, max_iter=300` に変更（他は 01 と同一）。ブースティング反復を最大300まで許しつつ、内部検証で頭打ちになれば自動停止する固定値の1点変更。**作成・検証は完了済み、本日は提出枠切れのため次UTC日に提出予定。** | 採点前 | — |

## 各回の詳細メモ

### 2026-07-12: 01_baseline (0.787)
- ねらい: まず「確実に動く」シンプルな土台を作ること。特徴量エンジニアリングや調整は一切なし。
- 構成: [submissions/01_baseline/](submissions/01_baseline/)
- 次に試したいこと: 単純な交差検証（学習データ内で分割して精度を確認してから提出する)、ハイパーパラメータの軽い調整（木の数・深さなど）、欠損値フラグ列の追加、など — ただしどれも「1つずつ」「シンプルさを保ったまま」試す。

### 2026-07-12: 02_early_stopping（作成済み・未提出）
- ねらい: 「木の本数の軽い調整」を1点だけ。デフォルト `max_iter=100`／early_stopping無効 を、`max_iter=300, early_stopping=True` に変更。より多くの反復を許しつつ、内部検証スコアが頭打ちになれば自動で停止するので、過学習を避けながら弱い信号を拾える可能性がある。
- 変更点: `submissions/02_early_stopping/agent/prompts/system.md` の分類器生成の1文のみ（＋説明1文）。`agent.yaml` の `name` を `early_stopping_agent` に。それ以外（列の扱い・カテゴリマスク・提出形式・ツール4つ・単発提出）は 01 と完全に同一。
- 検証: `python validate_submission.py --agent-dir submissions/02_early_stopping/agent` は全チェック合格（ADKコンパイル成功、tools=4）。
- 状態: **本日 2026-07-12 は 01_baseline で1日1件の提出枠を使い切ったため未提出。** 次のUTC日（2026-07-13）に上の「申し送り」手順でそのまま提出する。
- 構成: [submissions/02_early_stopping/](submissions/02_early_stopping/)
