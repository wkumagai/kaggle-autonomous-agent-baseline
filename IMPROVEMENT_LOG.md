# 改善ログ — Autonomous Agent Prediction (Beta)

このコンペは「予測結果」ではなく「自律的に学習・提出まで行うAIエージェント」を提出するコンペです。
ここでは、提出したエージェントの構成と、そのスコア（公開リーダーボードのROC AUC）の推移を記録します。
なるべくシンプルな構成を保ったまま、少しずつ改善していく方針です。

> **⚠️ 提出上限は「1日1件」です（2026-07-12 に実測で判明）。** Kaggle 側のエラーメッセージ:
> `Submission not allowed: Your team has used its daily Submission allowance (1) today`。
> つまり 1 UTC日につき提出できるのは 1 件だけ。毎時実行しても、その日すでに 1 件提出済みなら追加提出はできません。

> **✅ ユーザー承認済み（2026-07-12、本人とのチャットで直接指示）:** 「Kaggleには提出して採点できることはわかったので、性能を出すことを考えて。設計と実装と次の提出もできるようにして」という明示指示により、このサイクルに限り複雑化（交差検証＋複数モデル族アンサンブル）を承認済み。下記 `03_cv_ensemble` の PARKED 指定はこれにより解除。**ただし今回限りの承認であり、このサイクル以降は自動ループの既定方針（1サイクル1改善・シンプル構造優先）に戻ること** — 次に複雑な変更を検討する場合は再度ユーザーに確認すること。
>
> **📌 次回実行への申し送り（重要・唯一の指示）:** 次の UTC 日になったら、**`03_cv_ensemble` を提出**すること（実装・オフラインベンチマーク・敵対的レビュー・MBPローカルLLMでのE2Eリハーサルすべて合格済み・上のユーザー承認あり）。手順:
> `(cd submissions/03_cv_ensemble/agent && rm -f ../submission.zip && zip -r ../submission.zip . -x '.*')` →
> `kaggle competitions submit -c autonomous-agent-prediction-beta -f submissions/03_cv_ensemble/submission.zip -m "03_cv_ensemble: CV+ensemble (xgb/cat/lgb, rank-avg blend, staged submission)"` →
> 採点後に下表の 03 行のスコアを更新する。
>
> `02_early_stopping` は提出せず保留（03 の方が期待効果が大きく、1日1件の枠を優先的に使うため）。作成物は残すが、当面の次回提出対象からは外す。

## スコア推移

| 日付 | submission_id | 変更内容 | 公開スコア (ROC AUC) | 前回比 |
|---|---|---|---|---|
| 2026-07-12 | 54591282 | **01_baseline**: `HistGradientBoostingClassifier` を1回だけ学習・提出する最小構成。列のdtype（object=カテゴリ/順序、数値=そのまま）から自動でカテゴリ列を判定するため、データセットごとに列名や種類が変わっても手直し不要。ツールも write_file / run_command / submit_predictions / get_status の4つだけ。 | 0.787 | (ベースライン) |
| 2026-07-13 予定 | （未提出・保留） | **02_early_stopping**: 分類器の生成だけを `early_stopping=True, max_iter=300` に変更（他は 01 と同一）。作成・検証は完了済みだが、03の方が期待効果が大きいため次回提出は03を優先し、02は保留。 | — | — |
| 2026-07-13 予定 | （未提出・提出予定） | **03_cv_ensemble**: 交差検証＋XGBoost/LightGBM/CatBoostのアンサンブル＋段階的提出（詳細下記）。オフライン実測で平均AUC+0.0143、MBPローカルLLMでのE2Eリハーサル合格。ユーザー承認済み、次のUTC提出枠で提出予定。 | 採点前 | — |

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

### 2026-07-12: 03_cv_ensemble（✅ ユーザー承認済み・次回提出予定）
- **経緯:** 前回の自律実行（自動ループ）はユーザーの標準方針「シンプル構造・1サイクル1改善」に照らしてこれを複雑すぎると判断し、いったん保留（PARKED）にしていた。その後ユーザー本人とのチャットで「性能を出すことを考えて。設計と実装と次の提出もできるようにして」と明示の指示があったため、このサイクルに限り承認・保留解除。この判断の経緯自体は自動ループが正しく機能した記録として残す。
- ねらい: 提出予算（60分・30回提出）をほぼ使わずに終える01/02の非効率を解消。交差検証＋複数モデル族（XGBoost/LightGBM/CatBoost）＋アンサンブル＋段階的提出で、モデル選択の質を底上げする。
- 変更点: モデルを `gemini-2.5-flash` に変更（複雑な手順を正確にコピー・実行させるため）。エージェントに「安全網モデル→XGBoost→CatBoost→（時間があれば）LightGBM→ブレンド→2件選択」という決め打ちの手順を1文字も変えず実行させる設計（創造性ゼロ・実行の正確さのみ要求）。
- オフライン検証（16データセット全部で解答ラベルに対する実測AUC比較、`experiments/bench_03/`）: 平均AUC 0.789→0.803（+0.0143）、16件中16件で悪化なし（最小改善 train_06 +0.0004、最大 train_05 +0.0337）。最大ステージ時間68.6秒（60分枠に対して余裕あり）。
- 敵対的レビューで「隠しデータセットの少数クラスが極端に偏っていると全モデル族が同時にクラッシュしうる」という重大リスクを検出→層化分割・fold分割・predict_probaの列参照すべてにフォールバックを追加して修正済み。
- MBPのローカルLLM（qwen35b-a3b-q6）によるエンドツーエンドのリハーサル: 初回はローカルLLM側のコンテキスト長不足（8192トークン/スロット）でクラッシュ→MBP側の設定を16384→32768トークンに拡張して解消。再実行で完走: 14ツール呼び出し・15回LLM呼び出し・9分04秒、train_03でpublic 0.808/private 0.822、6ステップの手順（安全網→xgb→cat→lgb→ブレンド→2件選択）を一字一句忠実に実行。これは実際の提出で使う`gemini-2.5-flash`より弱いローカルモデルでの結果であり、本番はこれより高い再現性が期待できる。
- 検証: `python validate_submission.py --agent-dir submissions/03_cv_ensemble/agent` 合格。
- 構成: [submissions/03_cv_ensemble/](submissions/03_cv_ensemble/)、ベンチマーク: [experiments/bench_03/](experiments/bench_03/)、ローカルリハーサル治具: [experiments/local_eval/](experiments/local_eval/)
