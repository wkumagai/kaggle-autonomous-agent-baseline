# 改善ログ — Autonomous Agent Prediction (Beta)

このコンペは「予測結果」ではなく「自律的に学習・提出まで行うAIエージェント」を提出するコンペです。
ここでは、提出したエージェントの構成と、そのスコア（公開リーダーボードのROC AUC）の推移を記録します。
なるべくシンプルな構成を保ったまま、少しずつ改善していく方針です。

> **⚠️ 提出上限は「1日1件」です（2026-07-12 に実測で判明）。** Kaggle 側のエラーメッセージ:
> `Submission not allowed: Your team has used its daily Submission allowance (1) today`。
> つまり 1 UTC日につき提出できるのは 1 件だけ。毎時実行しても、その日すでに 1 件提出済みなら追加提出はできません。
> **🔑 訂正（2026-07-13 15:04 UTC 実測）: `ERROR` になった提出は1日1件枠を消費しない。** 03(ERROR, 07-13 00:02 UTC)と同じUTC日に 02 を提出したところ **受理された**（54651128, 07-13 15:04 UTC）。過去 run が「03がその日の枠を消費した(要確認)」と記していた前提は誤りで、**枠を消費するのは採点まで到達した提出（COMPLETE/PENDING）だけ**と判断してよい。日次判定は「その日に COMPLETE/PENDING の提出が既に1件あるか」で行うこと（ERRORは数えない）。

> **✅ ユーザー承認済み（2026-07-12、本人とのチャットで直接指示）:** 「Kaggleには提出して採点できることはわかったので、性能を出すことを考えて。設計と実装と次の提出もできるようにして」という明示指示により、このサイクルに限り複雑化（交差検証＋複数モデル族アンサンブル）を承認済み。下記 `03_cv_ensemble` の PARKED 指定はこれにより解除。**ただし今回限りの承認であり、このサイクル以降は自動ループの既定方針（1サイクル1改善・シンプル構造優先）に戻ること** — 次に複雑な変更を検討する場合は再度ユーザーに確認すること。
>
> **🚨🚨 次回実行への申し送り（最新・最優先・2026-07-13 ~16:0x UTC 更新／これより下の旧ブロックは履歴として残すが、操作指示はこの最上部ブロックが最新）:**
>
> **📌 直近サイクル(2026-07-13 17:0x UTC)でやったこと:** 提出履歴確認＝Kaggle上は 01/02(ともにCOMPLETE 0.787)・03(ERROR)のみで変化なし。**本日 2026-07-13 UTC枠は 02(COMPLETE, 15:04 UTC)が消費済み＝新規提出せず。** オフライン探索は round31(`max_leaf_nodes` 単ノブ)を実施し **不採用（両候補 mln_20/mln_15 とも mean 負・train_16 が回帰）＝08(既定 max_leaf_nodes=31)が最良のまま**（詳細は下の round31 メモ）。実提出キューは不変。**次に実提出できるのは次UTC日(2026-07-14 00:00 UTC 以降) に1件。**
>
> **今サイクルで起きたこと（3点）:**
> 1. **`02_early_stopping` を実提出し、`COMPLETE` で `Public 0.787` を獲得（submission_id=54651128, 07-13 15:04 UTC）。** 03 の ERROR 以降途切れていた「採点される提出」を再確立。**01_baseline(0.787) と同スコア** ＝ 02 の 01比オフライン+0.0016 の微改善は公開LBの表示解像度(0.787)では動かなかった。**含意: 単一HGBの微最適化(06/07/08/cand_C の +0.001〜0.002級)は公開LBの見かけ上ほぼ動かない可能性が高い**（実際の伸びしろ判断はユーザーの (A)/(c) 方針決定に律速される、という従来の見立てを補強）。
> 2. **ERROR提出は日次枠を消費しないと実測（上の「⚠️提出上限」訂正参照）。** 今後の日次判定は COMPLETE/PENDING が既に1件あるかで行う（ERRORは数えない）。
> 3. **🎉 ラウンド29 で単ノブ枯渇(R17-28)を破る初のクリーン改善候補 `cand_C`（seed-averaging）を発見・検証。** 詳細下記。**ただし seed-averaging は「複数モデルを学習しブレンド」＝従来ユーザー確認必須と切り分けてきた (c)ブレンド多様化カテゴリの構造変更のため、独断で submissions/09 を作って提出していない。ユーザー判断待ち（PushNotificationで報告済み）。**
>
> **🎉 ラウンド29 詳細（seed-averaging・変数分散削減）— 単ノブ探索(R17-28)で唯一救えなかった train_13(カウンタームーバー)を初めて改善:** 08 の全設定(l2/msl 2ゲート・early_stopping・特徴量)を固定し、**最終確率の作り方だけ**を「同一08 HGB を random_state 0..4 で K=5 回学習し predict_proba を平均」に変更（sklearn-only・保証外パッケージ非依存＝03の脆弱性なし）。専用ハーネス `experiments/bench_03/round29_seed_avg/replay.py`（`.venv`=grader保証パッケージ一致、`submissions/`一切不触・git status確認済、80 fit 全CLEAN RUN）で16データPublic/Private別AUC採点。
> - **cand_A（全16データにseed-avg）: mean ΔPub +0.00458 / ΔPrv +0.00425、W/L/T 15/1/13→実際は15/1/0。** ほぼ全データが改善するが **train_16 のみ回帰(−0.00238/−0.00458)** ＝「悪化ゼロ」を満たさず不採用。
> - **cand_B（l2ゲート比≥0.010発火データ=train_09/13/15/16 にseed-avg）:** train_16 の回帰を内包し不採用。
> - **✅ cand_C（採用基準クリア・要ユーザー確認で保留）: seed-avg を比 `n_feat/n≥0.015` のデータ=train_09/13/15 のみに発火（08が既に持つmsl厳ゲートと同じ0.015閾値を再利用・境界の train_16 は非発火で08とバイト同一）。** mean ΔPub **+0.00169** / ΔPrv **+0.00151**、**両split W/L/T=3/0/13＝回帰ゼロ**。発火3データ全て両split改善（train_15 +0.0102/+0.0127, train_09 +0.0099/+0.0063, **train_13 +0.0070/+0.0051**）。非発火13データ(train_16含む)は08にバイト同一。**08をクリーンにパレート改善する初の候補（R18の08採用以降、12ラウンドぶり）。**
> - **🔑 機構的知見: 小n高比データの弱点は early_stopping の内部holdout分割の分散だった。** random_stateを変えて平均すると分散が減り、どの単ノブ正則化でも逆方向に動いた train_13 が初めて素直に改善する。0.015ゲートが唯一の障害 train_16 を綺麗に外す。
>
> **したがって次にやること（優先順・最新）:**
> 1. **日次枠の判定:** 今日 2026-07-13 UTC は 02(COMPLETE)を提出済み＝**本日はもう提出しない**。次UTC日(2026-07-14)以降に下記から1件。
> 2. **【ユーザー判断待ち】cand_C(seed-avg)を submissions/09 として作って提出するか。** seed-avg は構造変更(K=5モデル)なので独断でシップしない。**ユーザーがOKなら:** `coder`/`architect` に委譲して `submissions/09_seedavg_gated/` を新設（08をコピーし、system.md の予測ステップを「n_feat/n≥0.015 のとき random_state 0..4 の predict_proba を平均、他は単一 random_state=0」に変更＝1点変更・sklearn-only・03脆弱性なし）、validate_submission.py 合格・self-diff確認の上でキュー先頭に入れる。**ユーザーがNGなら** cand_C は据え置き、下の安全キューを継続。
> 3. **ユーザー判断と独立に進めてよい安全キュー（単ノブ・1提出1変更・grader保証済）:** 2026-07-14=`06_ngated_l2`、以降 `07_2gate_msl50`、`08_ratio_tiered_msl`。ただし**02が01と同じ0.787表示だったことから、これらも公開LB表示は 0.787 のまま動かない見込み**。「板を新鮮に保つ／各層がgraderで通ることの確認」以上の価値は薄い。急がば cand_C(承認後) か 08 直提出の方が上積み期待は大きい。
> 4. **次サイクルのオフライン角度:** cand_C の成功で「分散削減(ensembling)」方向が有望と判明。次は (a) seed数 K のスイープ(K=3/10 で cand_C を上回るか)【✅実施＝下記round30。結論: K=5が knee。K=3 は利得の約25%を失い「安いのに同等」ではない／K=10 は 2倍の学習で ΔPub+0.00018・ΔPrv+0.00032 の微増のみ・回帰ゼロだがコスパ薄。**cand_C の K=5 が最適で確定。**】、(b) 発火ゲート閾値の微調整、(c) seed-avg と (A)複雑路線 go.py 堅牢化の合流検討。いずれも round29/replay.py を雛形に。**(A)複雑路線 go.py 堅牢化 と 本格的な (c)モデル族多様化は依然ユーザー確認必須の大設計変更**。
>   - **単ノブ探索は round31(`max_leaf_nodes`)で不採用＝これで08土台の simple 単ノブ角度は l2/msl/valfrac/lr/class_weight/max_features/max_bins/max_leaf_nodes まで実質枯渇（すべて不採用、08既定が最良）。** simple 単ノブでの上積みは尽きたと判断してよい。実効的な伸びしろは既に判明済みの **cand_C(seed-avg K=5・要ユーザー確認)** か **(A)複雑路線 go.py 堅牢化(要ユーザー確認)** に律速される。次サイクルの新角度は上の (b)発火ゲート閾値の微調整（cand_C の ratio≥0.015 ゲートを 0.010/0.012 等に広げ train_16 を巻き込まず利得を増やせるか＝seed-avg 枠内なので構造は不変、単ノブ的に安全に試せる）を最優先候補とする。
>
> **🔬 ラウンド30（2026-07-14, seed数Kスイープ）— cand_C の K=5 が knee と確定・(a)を消化:** round29 の cand_C ゲート（ratio=n_feat/n≥0.015 → train_09/13/15 のみ seed-avg・他13データは seed-0 とバイト同一）を固定し、平均する seed 数 K∈{3,5,10} だけを振った。専用ハーネス `experiments/bench_03/round30_seed_k_sweep/replay.py`（`.venv`=grader保証sklearn-only, 発火3データのみseed0..9を各1回fitしKごとに先頭K個の予測を平均＝計43fit・CLEAN RUN=YES・`submissions/`不触をgit status確認済）で16データPublic/Private別AUC採点。
>   - **K=3: mean ΔPub +0.00129 / ΔPrv +0.00108、両split W/L/T 3/0/13・回帰ゼロ。** クリーンだが K=5 の利得の約75%(Pub)/72%(Prv)しか取れない（特に train_13 が ΔPub+0.00344 と K5 の約半分）。「安いのに同等」ではなく、5→3 に削ると測定可能に edge を失う。単純化の無償ランチではない。
>   - **K=5（=cand_C を本ハーネスで完全再現）: +0.00169 / +0.00151、3/0/13。** round29 と一致し再現性を確認。
>   - **K=10: +0.00187 / +0.00183、3/0/13・回帰ゼロ。** K=5 を両splitで僅かに上回る唯一の設定だが、モデル数2倍に対し利得増は ΔPub+0.00018・ΔPrv+0.00032 のみ＝収穫逓減が明確。
>   - **結論: K=5 が effort/gain の knee。cand_C の K=5 で確定してよい（K=3 は弱すぎ・K=10 はコスパ薄）。** これは今後 cand_C を submissions/09 として提案する際の seed 数根拠になる。**ship 判断は依然ユーザー確認待ち（seed-avg=K本モデルの構造変更のため）。**
>
> **🔬 ラウンド31（2026-07-13 17:0x UTC, `max_leaf_nodes` 単ノブ）— 不採用・08既定(31)が最良のまま。simple単ノブ角度の枯渇を確認:** round28(max_bins)と同じ枠組みで、08 の全設定(l2/msl 2ゲート・early_stopping)を固定し、**L2ゲート発火データ(ratio=n_feat/n≥0.010 → train_09/13/15/16)のみに `max_leaf_nodes` を下げる** 1点変更を検証。専用ハーネス `experiments/bench_03/round31_max_leaf_nodes/replay.py`（round28/replay.py をテンプレに knob だけ差し替え・`.venv`=grader保証sklearn-only・48fit CLEAN RUN=YES・非発火12データのバイト同一 invariant 検査PASS・`submissions/`不触をgit status確認済）で16データPublic/Private別AUC採点。max_leaf_nodes ∈ {31(=08既定), 20, 15} を比較。
>   - **mln_20: mean ΔPub −0.00049 / ΔPrv −0.00046、W/L/T 0/1/15(両split)。不採用（mean 負・train_16 が回帰 Pub−0.00777/Prv−0.00744）。**
>   - **mln_15: mean ΔPub −0.00103 / ΔPrv −0.00126、Pub 1/1/14・Prv 0/2/14。不採用（mean 負・train_16 両split回帰＋train_09 Prv−0.00470）。**
>   - **🔑 機構的知見:** train_09/13/15 は msl ゲート(50/70)により既に葉数<31 のため leaf cap が効かず delta=0（08にバイト同一）。唯一 msl=20 で葉容量が残る train_16 だけに cap が刺さり、そこを一方的に害する。**葉数制限は 08 の弱点(小n高比)に効かず、副作用のみ。** これで l2/msl/valfrac/lr/class_weight/max_features/max_bins/max_leaf_nodes の simple単ノブが全て不採用となり、08土台の単ノブ探索は枯渇。伸びしろは cand_C(seed-avg) か複雑路線堅牢化（ともに要ユーザー確認）に律速。
>
> **（以下は履歴。2026-07-13 各ラウンドの旧「次にやること」。操作指示は上の最新ブロックが優先。）**
>
> **🚨 次回実行への申し送り（最重要・2026-07-13 更新）— `03_cv_ensemble` は実グレーダーで ERROR。複雑路線は要ユーザー判断:**
> `03_cv_ensemble`（submission_id=54625716）を 2026-07-13 00:02 UTC に提出したが、Kaggle 実グレーダーで **`SubmissionStatus.ERROR`（スコアなし）** になった。一方 `01_baseline`（単発・単一HGBの簡素構成）は 0.787 で成功している。**オフラインベンチ（go.py を直接実行してAUC測定）とMBPローカルLLMリハーサルは両方通っていたのに本番でERRORした**＝失敗は go.py のロジックではなく、**本番のエージェント実行環境（`gemini-2.5-flash` 駆動で6ステップの手順を実行させる複雑構造）側**にある可能性が高い。**これはユーザーが今回限り承認した「複雑化（CV＋複数モデル族アンサンブル）」路線そのものが実グレーダーで動かなかったということ**であり、`05_te_ngated`/`04_missing_count` も同じ複雑 go.py・6ステップ構造を共有するので**同様にERRORするリスクが高い**。
>
> **したがって次にやること（優先順）:**
> 1. **ユーザーの判断を仰ぐ（PushNotificationで報告済み）。** 選択肢は大きく2つ: **(A) 複雑路線を続けるなら 03 のERROR原因の調査・修正が必要**（Kaggleの実行ログ/エラー詳細の取得、手順の簡素化＝ステップ削減やモデル依存の削減など、`architect`/`coder` に委譲する設計変更）。**(B) シンプル路線に戻す**なら、実グレーダーで確実に通る `01_baseline` を土台に、`02_early_stopping`（単発・単一HGD＋early_stopping、複雑手順なし）のような1点改善だけを積む。CLAUDE.md の既定方針（シンプル優先）と今回のERRORを踏まえると **(B) が既定寄り**だが、複雑路線の承認はユーザーが与えたものなので独断で放棄せず確認する。
> 2. **提出枠について:** 03 は ERROR でも本日 2026-07-13 UTC の1日1件枠を消費した可能性が高い（要確認）。少なくとも直近提出が確定するまで新規提出しない。次UTC日(2026-07-14)以降、上のユーザー判断に従って **02_early_stopping（安全策）** か **05（複雑路線継続かつERROR原因解消後）** のいずれかを提出する。
> 3. **05/04 を「複雑路線が実グレーダーで動く」と確認できるまで提出しないこと**（03と同構造でERROR再現の恐れ）。
>
> **🔬 ERROR原因の再現診断（2026-07-13 01:xx UTC・ラウンド15）— 複雑路線の脆弱性を実証。推奨は (B) シンプル路線寄りへさらに傾く:**
> `03` の go.py（fenced pythonブロック）を、**xgboost/lightgbm/catboost を持たない環境（この repo の `.venv` と一致。Kaggle グレーダーのサンドボックスも同様の可能性が高い＝01のsystem.md自体が保証しているのは pandas/numpy/scikit-learn だけ）** で train_01 に対して段階実行し、失敗カスケードを実測した:
> - `safety` → **OK**（lightgbm不在→sklearn HGB フォールバックが発火し `sub_safety.csv` を書く。exit 0, val_auc=0.70）
> - `xgb` → **クラッシュ**（`import xgboost` で ImportError→`fit_fam` に fallback が無く stage 全体が落ちる。出力ファイルなし）
> - `cat` → **クラッシュ**（同上、catboost不在）
> - `lgb` → **クラッシュ**（.venv は lightgbm も不在。※base python3 には lightgbm はある＝グレーダーが何を持つかは不確定）
> - `blend` → `RESULT blend no_models`（ブレンド対象ゼロ、何も書かない。残るのは `sub_safety.csv` のみ）
>
> **結論（証拠に基づく）:** 03の売り（xgb/cat/lgb の複数族アンサンブル）は**グレーダーに保証されていない外部パッケージに依存**している。パッケージが欠けると3族すべてが落ち、クリーンな6手順が「クラッシュ＋リカバリ分岐だらけの茨の道」に化ける。Step3で `sub_safety.csv` を先に提出する安全網はあるので**パッケージ欠如＝即・総ERROR**とまでは断定できない（総ERRORは agent実行層＝gemini-2.5-flash が茨の道で tool-call/時間予算を使い切る・最終選択を誤る等に起因する可能性が高く、これはオフラインでは検証不能）。だが**この脆弱性は gbm_venv（3族入り）で回すオフラインベンチでは完全に不可視**で、「オフライン＋MBPローカルは通ったのに本番ERROR」を素直に説明する。**一方 `01_baseline`(0.787成功) と `02_early_stopping` は sklearn の HGB 単独＝グレーダーが保証する範囲だけを使うので、この脆弱性を構造的に持たない。**
>
> **したがって推奨の更新（A/Bの最終決定は依然ユーザー・PushNotificationで報告）:**
> - **(B) シンプル路線が既定として強く支持される。** 次の実提出（次UTC日 2026-07-14 以降。本日 2026-07-13 UTC枠は 03 が消費済み）は `submissions/02_early_stopping/`（sklearn-only・グレーダーで確実に動く・01への1点改善）を最優先とする。提出手順は上の「02の提出手順」参照。
> - **(A) 複雑路線を続けるなら**、go.py を「各族が該当パッケージ不在時に sklearn HGB へフォールバックする／xgb・cat・lgb の存在を前提にしない」よう作り直す設計変更が必須（`architect`/`coder` に委譲）。これは「シンプル1変更」を超える設計変更なので独断で着手せず、ユーザー確認を待つ。
> - 05/04/freq 等の複雑路線候補は上記(A)の堅牢化が済むまで提出しない（03同構造でERROR再現の恐れ）。
>
> **✅ シンプル路線の実証検証（ラウンド16, 2026-07-13 ~02 UTC）— (B)を確定的に裏付け。次の実提出は `02_early_stopping` で確定・キュー済み:** これまでの14ラウンドは全て「複雑な03 go.py（fenced block）を gbm_venv（xgb/lgb/cat 入り）で回す」FE差分探索で、**グレーダーが実際に保証する sklearn-only 環境で simple 路線を回したことが一度も無かった**。今回その空白を埋めた。`01_baseline`（HGB既定 max_iter=100）と `02_early_stopping`（HGB max_iter=300 + early_stopping=True）が system.md 通りに書くはずの train.py を、**xgboost/lightgbm/catboost を持たない本体 `.venv`（グレーダー保証パッケージと一致）** で16データ全部に対し実行し、solution.csv の Public/Private で AUC 採点した（検証ハーネスは scratchpad のみ・`submissions/` は一切不触）。結果:
> - **16データ全部が sklearn-only 環境でクラッシュ無しに完走（CLEAN RUN=True）。** これが最重要の de-risk 成果 — **02（および01）は03を殺した「保証外パッケージ依存」の脆弱性を構造的に持たない**ことを実証。02は実グレーダーで通る蓋然性が高い。
> - **ただし 02 vs 01 は「クリーンなパレート改善」ではない（従来の申し送りの過大評価を訂正）。** mean d_pub=**+0.00157** / d_prv=**+0.00059**、Public で **8勝3敗5分**。勝ちは反復増で効くデータに集中（train_05 +0.0176, train_13 +0.0081, train_02 +0.0079）だが、**明確な悪化が3件**（train_15 −0.0088, train_16 −0.0071, train_04 −0.0047＝early_stopping が早く止まりすぎ/内部holdoutでの学習データ減が害）。それでも平均は正で、かつ安全。よって**01の再提出（既に0.787で採点済＝枠の無駄）より、02で「sklearn-only+early_stopping」の実LB値を取りに行く方が正**。
> - **同サイクルで新角度も1つ検証（不採用）: `02b`=early_stopping の忍耐値を上げる（`n_iter_no_change=10→20`, sklearn-only・1ノブ）。** 02の3件の悪化を「もっと粘れば」直せるかの仮説。**逆に悪化: mean d_pub=+0.00031、Public 7勝9敗。** 忍耐を増やすと反復が伸びて過学習が増え、02の悪化を直すどころか新たな悪化に置き換え平均利得も縮む。**既定 patience=10 がこのsuiteのスイートスポット。02 as-shipped が simple の最良候補。**
>
> **したがって次にやること（優先順・更新）:**
> 1. **本日 2026-07-13 UTC は 03(ERROR) が1日1件枠を消費済み（現在 02:0x UTC で確認）＝新規提出不可。**
> 2. **次UTC日 2026-07-14 以降、`submissions/02_early_stopping/` を最優先で実提出する（(B)確定・上の検証で安全＋平均正を実証）。** 手順は下の「02の提出手順」。これで03 ERROR以降途切れた「採点される提出」を再確立する。
> 3. **(A) 複雑路線（03/05のgo.py堅牢化＝各族のsklearn HGBフォールバック実装）は依然ユーザー確認待ちの大設計変更**。独断着手しない。ただし 02 が安全確定した今、(B)を止める理由は無いので 02 提出は(A)判断と独立に進める。
> 4. FEはこのsuiteで飽和（R1-14）、simple のハイパラ(early_stopping忍耐)もR16で局所最適確認。実効的な伸びしろは (b)モデル族/ブレンド多様化のみで、これは設計変更のためユーザー確認要。次サイクルのオフライン探索は「まだ試していない直交角度」を1つ選ぶ（例: HGBの`l2_regularization`や`max_leaf_nodes`など simple 単ノブ、または 01↔02 の悪化3件に効く条件付き選択）。検証ハーネス雛形は今回の手法（sklearn-only .venv で01/02系を直接replay採点）を再利用可。
>
> **🔎 探索済み・不採用（ラウンド17, 2026-07-13 ~03 UTC）— ただし「n-gated l2」という有望リードを発見。simple路線初のsklearn-only単ノブ探索:** ラウンド16の「次の角度」提案に従い、**shipped 02(sklearn-only HGB, early_stopping, max_iter=300)の上で `l2_regularization` 単ノブ**を初検証（これまでの`reg_l2_bump`候補は複雑03 go.py側だったのに対し、今回はグレーダー保証のsklearn-only simple路線での初のl2探索）。02のtrain.pyロジックを`git show HEAD:submissions/02_early_stopping/agent/prompts/system.md`基準にin-process再現し、`.venv`(sklearn 1.9.0, xgb/lgb/cat無し)で16データ全部をPublic/Private別AUC採点する専用replayハーネスを新設（`submissions/`は一切不触・git status確認済）。l2 ∈ {0.0(=02), 0.1, 1.0} を比較。**まず重要な de-risk 成果: 3設定×16データ=48 fit 全てクラッシュ無し完走（CLEAN RUN=True）。simple路線がグレーダー環境で確実に動くことを再確認。** 採点結果:**両方とも不採用。**
> - **l2=0.1:** mean Public Δ=**−0.0012**, Private Δ=−0.0012, Public 7勝9敗。平均が負で明確に劣化。即・不採用。
> - **l2=1.0:** mean Public Δ=**+0.0010**（ノイズ域）, Private +0.0012, Public 10勝6敗。平均は僅かに正だが**明確な回帰を伴う**（train_05 −0.0077, train_03 −0.0039＝どちらも小n弱点データを更に悪化）。「meanが明確に正 かつ 悪化なし」の採用基準を満たさない。**不採用。**
> - **🔑 発見したリード（次サイクルの最優先角度）:** l2=1.0 が効いたデータは **02が01に対して回帰した3件そのもの**だった — train_15 (Public +0.0089/Priv +0.0053), train_16 (+0.0054/+0.0083), train_04 (Priv +0.0042)。**つまり「02の3つの回帰は大n側で、l2正則化を足すと綺麗に反転する」一方、l2は小n(train_05/03/13)を明確に害する。** これは05(n-gated TE)と全く同じ n依存の構造で、**「大n(例 n≥閾値)のときだけ l2_regularization を足す n-gated l2」**が次の有望候補（小nの害を避けつつ02の3回帰を直せる可能性）。素朴な全体一律l2は不採用だが、n-gatedにする価値が数値で裏付けられた。
> - 生ハーネス=`experiments/bench_03/simple_replay.py`、結果=`experiments/bench_03/round17_l2simple/{results.csv,summary.txt,run.log}`。このsklearn-only replayハーネスは今後の全simple路線候補で再利用する（benchmark.pyは複雑03専用で simple路線を検証できないため、これが simple路線の正式な検証ハーネス）。
>
> **したがって次にやること（優先順・ラウンド17末で更新）:**
> 1. **実提出キューは不変:** 次UTC日 **2026-07-14** に `submissions/02_early_stopping/` を最優先で実提出（(B)シンプル路線確定・R16で安全実証済）。本日 2026-07-13 UTC枠は 03(ERROR) が消費済みで新規提出不可。
> 2. **次サイクルのオフライン最優先角度 = 「n-gated l2_regularization」**（上のリード）。02の上で `if len(train) >= <閾値>: clf の l2_regularization=1.0 else 0.0`（列は増やさない・小n非発火で安全・大nの02回帰3件を狙う）を simple_replay ハーネスで検証。閾値は train_05/03/13(小n) を非発火に、train_15/16/04(大n) を発火にする値を dataset_stats.csv の n から決める。クリーンに02をパレート改善（悪化ゼロで大n底上げ）できれば採用候補として `submissions/06_*/` を新設。
> 3. (A)複雑路線(03/05のgo.py堅牢化)は依然ユーザー確認待ちの大設計変更。独断着手しない。
>
> **✅ 採用候補（ラウンド18, 2026-07-13 ~04 UTC）— ラウンド17のリード「n-gated l2」を実装・検証し `submissions/06_ngated_l2/` を新設（validate合格）。ただし正しい軸は「生n」ではなく「特徴量/行数の比(n_feat/n)」だった:** ラウンド17末の最優先角度に従い gated l2 を実装。**まず軸を実証で正した** — 専用replayハーネス `experiments/bench_03/round18_ngated_l2/replay.py`（`.venv`=sklearn-only, grader保証パッケージと一致）で 02(l2=0.0)を基準に2ゲートを比較（48fit全てCLEAN RUN=YES）:
> - **gate_ratio（採用）: `l2 = 1.0 if (n_feat/n) >= 0.010 else 0.0`。** train_09/13/15/16 の4データで発火、**全4データがPublic/Private両方で改善（悪化ゼロ）・残り12データは02とバイト同一** ＝**02のクリーンなパレート改善**。mean Public Δ=**+0.0014** / Private Δ=**+0.0010**、Public W/L/T=4/0/12。発火4データのPublic Δ: train_15 +0.0089, train_16 +0.0054, train_09 +0.0044, train_13 +0.0033。
> - **gate_nsmall（対照・棄却）: `l2 = 1.0 if n <= 1200 else 0.0`（＝ラウンド17申し送りの「生nゲート」素直解釈）。** train_05/09/13/15 で発火し、**train_05 がPublic −0.0077 / Private −0.0020 と両面回帰** ＝パレート改善にならない。**これで「効く軸は生nではなく feature/row 比（過学習しやすさ）」が実証された** — ラウンド17の「n-gated」表現は軸を取り違えており、round18で訂正。（最大の勝ち train_15 は n=500＝最小データなので、生n≥閾値ゲートでは原理的に拾えない。）
> - **戦略的価値: 06は 02が01(0.787,実グレーダー成功)に対して回帰していた3件のうち2件(train_15, train_16)を、まさにこのl2ゲートで埋める。** 02は01比で mean+0.0016 だが train_15 −0.0088 / train_16 −0.0071 / train_04 −0.0047 の3回帰を持つ「クリーンでない改善」だった。06はその train_15/16 を反転させるため、02より01に対しクリーンな改善に近づく。
> - **構造的安全性: 06は 02と同じ sklearn-only 単一HGD・単発構成に「l2値をデータ駆動で決める1行」を足しただけ**（新規列なし・保証外パッケージ依存なし・小/低比データは l2=0.0 で02と完全同一）。03をERRORさせた「保証外パッケージ依存×6ステップ複雑手順」の脆弱性を構造的に持たない。単ノブ1変更でシンプル方針を維持。
>
> **したがって次にやること（優先順・ラウンド18末で更新）:**
> 1. **本日 2026-07-13 UTC は 03(ERROR)が1日1件枠を消費済み＝新規提出不可。** 実提出は行っていない。
> 2. **次UTC日 2026-07-14: `submissions/02_early_stopping/` を実提出（キュー不変・(B)確定）。** 目的は 03 ERROR以降途切れた「採点される提出」を、grader保証の sklearn-only 最小変更で再確立し、simple路線が実グレーダーで通ることを再確認すること。手順は下記。
> 3. **その次 2026-07-15: `submissions/06_ngated_l2/` を実提出（採用候補・ラウンド18で02をクリーンにパレート改善と実証）。** 提出手順は下記。※もし 2026-07-14 の 02 が実グレーダーで問題なくスコアしたら、06も同一の sklearn-only 構造なので安全に続けられる。（枠は残り約25日分と潤沢なので、02で1枠使って再確立→06で改善、の2段が安全。急ぐなら 07-14 に 06 を直接出す選択も可＝06は02を内包し02の弱点も直すが、まず最小変更で再確立する保守案を既定とする。）
> 4. (A)複雑路線(03/05のgo.py堅牢化)は依然ユーザー確認待ちの大設計変更。独断着手しない。
> 5. **次サイクルのオフライン探索:** 素朴FE・TE系ハイパラは飽和確認済（R1-17）。gated-l2 は採用したので、次は (a) l2値そのものの微調整を発火データ内で（例 l2=0.5/2.0 を gate_ratio 内で・悪化なく上積みできるか）、(b) 同じ「過学習しやすさ軸」で他の正則化ノブ（`max_leaf_nodes` 縮小や `min_samples_leaf` 増）を同じ比ゲートで、あるいは (c) モデル族多様化（設計変更＝ユーザー確認要）。simple路線の検証は `experiments/bench_03/simple_replay.py` / `round18_ngated_l2/replay.py` の sklearn-only replay を再利用（benchmark.py は複雑03専用でsimple路線を検証できない）。
>
> **🔎 探索済み・不採用（ラウンド19, 2026-07-13 ~05 UTC）— ラウンド18の角度(a)「gate_ratio内のl2値スイープ」を検証。L=1.0(=06)が最適で確定・キュー不変:** round18採用の06(gate_ratio: `l2 = L if n_feat/n>=0.010 else 0.0`)の上で、**ゲート閾値0.010は固定したまま発火時の l2 magnitude L のみ** を L∈{0.5, 1.0, 2.0} でスイープ（baseは L=0.0=02）。専用replayハーネス `experiments/bench_03/round19_l2_magnitude/replay.py`（round18/replay.py をコピー拡張・sklearn-only `.venv`=grader保証パッケージと一致）で16データをPublic/Private別AUC採点。**まず de-risk: 全設定×16データ=64 fit 全てクラッシュ無し完走（CLEAN RUN=YES）。`submissions/` は一切不触（`git status --porcelain` は `experiments/bench_03/round19_l2_magnitude/` のみ・確認済）。** 全Lで発火データは同一（train_09/13/15/16、他12データはbaseとバイト同一）。採点結果:
> - **L0.5:** mean Public Δ=**+0.0009** / Private +0.0007, Public W/L/T=3/1/12。train_13 が Public −0.0001 / Private −0.0017 と回帰。**弱いうえ回帰あり＝不採用。**
> - **L1.0（=shipped 06）:** mean Public Δ=**+0.0014** / Private +0.0010, Public W/L/T=**4/0/12**, Private **4/0/12**。**両splitで回帰ゼロ・唯一のクリーン正。**
> - **L2.0:** mean Public Δ=+0.0014（L1.0と同値）だが **train_15 が Private −0.0020 の新規回帰**。Private W/L/T=3/1/12。meanは並ぶが「悪化なし」を満たさず**不採用。**
> - **結論:** l2 magnitude は L=1.0 が明確なスイートスポット（正の平均かつ両split回帰ゼロは L1.0 のみ）。**06(L=1.0) は magnitude ノブで局所最適が確定。提出キューは不変（07-14=02 → 07-15=06）。** 生ログ=`experiments/bench_03/round19_l2_magnitude/{results.csv,summary.txt}`。
> - **次サイクルの最優先角度 = 上記(b)「別の正則化ノブを同じ gate_ratio(n_feat/n>=0.010) で」**（例: 発火データのみ `max_leaf_nodes` を既定31から縮小、または `min_samples_leaf` を増やす）。magnitudeノブ(a)は本ラウンドで飽和したので(b)へ移る。検証は round19/replay.py を雛形に再利用（発火判定は共通、変える1ノブだけ差し替え）。
>
> **🔎 探索済み・不採用（ラウンド20, 2026-07-13 ~06 UTC）— ラウンド19の次角度(b)「同じ gate_ratio 内で l2=1.0 に第2の正則化ノブを重ねる」を検証。第2ノブは全て train_16 を回帰させ不採用。06(l2=1.0のみ)が依然最良:** shipped 06 のゲート(`n_feat/n>=0.010`・発火=train_09/13/15/16)と l2=1.0 はそのまま固定し、**発火データにだけ第2の正則化ノブを追加**して 06 をさらにクリーンに底上げできるかを検証（非発火12データは 02/06 と完全同一を保つ設計）。専用replayハーネス `experiments/bench_03/round20_gated_reg2/replay.py`（round19/replay.py を拡張・sklearn-only `.venv`=grader保証パッケージと一致）で16データPublic/Private別AUC採点。**de-risk: 5設定×16データ=80 fit 全てクラッシュ無し完走（CLEAN RUN=YES）、`git status --porcelain` は `experiments/bench_03/round20_gated_reg2/` のみ＝`submissions/` 一切不触・確認済。** 全設定で発火は同一（train_09/13/15/16）、非発火12データは全設定でバイト同一（delta 0）。base=06(l2=1.0)に対する4候補:
> - **msl40（`min_samples_leaf=20→40` を発火時のみ）:** mean Public +0.00001 / Private +0.00058、W/L/T=3/1/12。train_09 Public −0.0030 と train_16 Private −0.0035 を回帰 → **不採用。**
> - **msl50（`min_samples_leaf=50`）: 4候補中最大の平均利得（Public +0.00089 / Private +0.00251）。** train_09/13/15 を両split明確に底上げ（train_13 Private +0.0196, train_09 Private +0.0118, train_15 Private +0.0092）。**だが train_16 が僅かに回帰（Public −0.00069 / Private −0.00048）→「悪化ゼロ」を満たさず不採用。**
> - **mln20（`max_leaf_nodes=31→20`）:** 平均が両split負（Public −0.00013）、train_16 を両split回帰（Public −0.0078 / Private −0.0074）→ **不採用。**
> - **mln15（`max_leaf_nodes=15`）:** 平均 Public 負（−0.00089）、train_15/16 を両split回帰 → **不採用。**
> - **結論:** 06 の上に第2正則化ノブを重ねても、発火4データのうち **train_16 が必ず回帰**するため誰もクリーンに 06 を上回らない。**06(l2=1.0のみ)がこのゲート下の局所最適。**
> - **🔑 発見したリード（次サイクルの最優先角度）:** msl50 は train_09/13/15 に大きく効き（Private で +0.009〜+0.020）、唯一の障害が train_16 の極僅かな回帰（Private −0.0005）だった。**train_16 はゲート比 21/1809=0.0116 で閾値0.010をギリギリ超えて発火する「境界データ」**（他の3発火データは比0.016〜0.060と明確に高い）。つまり **「第2ノブ(min_samples_leaf増)はより厳しい別ゲート（例 n_feat/n>=0.015）で発火させ、境界の train_16 を除外」**すれば、msl50 の train_09/13/15 の大きな利得を悪化ゼロで取れる可能性がある。これが次ラウンドの最優先候補（l2ゲートは0.010のまま、msl系の第2ノブだけ 0.015 の厳ゲートにする「2段ゲート」）。生ログ=`experiments/bench_03/round20_gated_reg2/{results.csv,summary.txt,run.log}`。
>
> **したがって次にやること（優先順・ラウンド20末で更新）:**
> 1. **本日 2026-07-13 UTC は 03(ERROR)が1日1件枠を消費済み＝新規提出せず（確認済: 提出履歴の当日UTC分は 03 のみ）。**
> 2. **実提出キューは不変:** 次UTC日 **2026-07-14** に `submissions/02_early_stopping/` を最優先で実提出（(B)シンプル路線・R16で安全実証済）、その次 **2026-07-15** に `submissions/06_ngated_l2/`（採用候補・R18で02をクリーンにパレート改善）。手順は下記。
> 3. **次サイクルのオフライン最優先角度 = 上記リード「2段ゲート msl50」**（l2は比0.010ゲートのまま、min_samples_leaf=50 は比0.015のより厳しいゲートで発火＝境界の train_16 を除外し train_09/13/15 の大利得を悪化ゼロで取りに行く）。検証は round20/replay.py を雛形に再利用。
> 4. (A)複雑路線(03/05のgo.py堅牢化)は依然ユーザー確認待ちの大設計変更。独断着手しない。
>
> **✅ 採用候補（ラウンド21, 2026-07-13 ~07 UTC）— ラウンド20の最優先リード「2段ゲート msl50」を実装・検証し `submissions/07_2gate_msl50/` を新設（validate合格）。06をクリーンにパレート改善:** ラウンド20で「単一ゲートの msl50 は train_09/13/15 を大きく底上げ(Private +0.012〜+0.020)するが境界データ train_16(比0.0116)だけ僅かに回帰して不採用」と判明していた。その回帰を、**第2ノブ(min_samples_leaf=50)だけを l2ゲートより厳しい比0.015で発火**させることで除外できるという仮説を、専用replayハーネス `experiments/bench_03/round21_2gate_msl50/replay.py`（`.venv`=sklearn-only, grader保証パッケージと一致・round20/replay.pyを拡張）で検証。**de-risk: 16データ全fitクラッシュ無し完走(CLEAN RUN=YES)、`git status --porcelain` は round21配下のみ＝`submissions/`一切不触・確認済。** base=06(l2=1.0のみ)に対する候補 msl50_2gate:
> - **ゲート挙動（設計通り）**: L2ゲート(比≥0.010・06から不変)は train_09/13/15/16 で発火(l2=1.0)。新規の第2 msl ゲート(比≥0.015)は train_09(0.0162)/13(0.0180)/15(0.0600) で発火(min_samples_leaf=50)、**境界 train_16(0.0116)は非発火→06と完全同一**。
> - **結果: mean Public Δ=+0.00093 / Private Δ=+0.00254、W/L/T=3/0/13（両split回帰ゼロ）。** 発火3データ全て両split改善（train_13 Private +0.01955, train_09 Private +0.01181, train_15 Private +0.00920 ＝ round20 単一ゲート msl50 の利得と数値完全一致）。**train_16 は Public/Private とも Δ=0.00000（06にバイト同一）。** 他12データも06と同一。
> - **採用基準（mean両split正 かつ 両split回帰ゼロ）を満たす CLEAN IMPROVEMENT。** → `submissions/07_2gate_msl50/` を新設（06をコピーし system.md に min_samples_leaf 第2ゲートを1つ足すだけ・validate_submission.py 合格・既存提出物に差分なし確認済）。構造は 06と同じ sklearn-only 単一HGB・単発に「msl値をデータ駆動で決める第2ゲート1本」を足しただけ（新規列なし・保証外パッケージ非依存＝03の脆弱性を持たない）。
> - **戦略的価値: 07は 06の弱点(境界 train_16)を悪化させずに、l2だけでは動かなかった train_09/13(小n)を Private で大きく底上げする。** 提出キューの改善段を 02→06→07 と一段深くする。生ログ=`experiments/bench_03/round21_2gate_msl50/{results.csv,summary.txt,run.log}`。
>
> **🔎 探索済み・不採用（ラウンド22, 2026-07-13 ~08 UTC）— ラウンド21の最優先角度「07の第2mslゲート内での min_samples_leaf magnitude スイープ」を検証。msl=50(=07)が依然スイートスポットで確定・キュー不変:** 07の2ゲート構造(l2比≥0.010 / msl比≥0.015)はそのまま固定し、**発火時の `min_samples_leaf` magnitude のみ**を {40, 60, 70} で 50(=shipped 07) と比較（ラウンド19のl2 magnitudeスイープと同型）。専用replayハーネス `experiments/bench_03/round22_msl_magnitude/replay.py`（round21/replay.pyを拡張・sklearn-only `.venv`=grader保証パッケージと一致）で16データPublic/Private別AUC採点。**de-risk: 4設定×16=64 fit 全てクラッシュ無し完走(CLEAN RUN=YES)、`git status --porcelain` は `round22_msl_magnitude/` 配下のみ＝`submissions/`一切不触・確認済。** 全設定で発火は同一（mslゲートは train_09/13/15 のみ発火、train_16 は非発火で全設定 msl=20＝07と同一、他12データもバイト同一）。base=07(msl=50)に対する候補:
> - **msl40:** mean Public Δ=**−0.00092** / Private Δ=**−0.00174**、W/L/T=1/2/13。train_09(−0.0089/−0.0116)・train_13(−0.0063/−0.0126) を明確に回帰。葉サイズを下げると害＝**不採用。**
> - **msl60:** mean Public Δ=+0.00003 / Private Δ=+0.00000（実質フラット）。train_09 を両split・train_15 を Private で僅かに回帰 → 「悪化ゼロ」を満たさず**不採用。**
> - **msl70:** mean Public Δ=+0.00023 / Private Δ=**−0.00016**（Private平均が負）。**train_15 は大きく改善(+0.0039/+0.0040)する一方 train_09 が両split回帰(−0.0022/−0.0041)・train_13 も Private回帰** → 不採用。
> - **結論:** min_samples_leaf magnitude は **50(=07)が明確なスイートスポット**（正の平均かつ両split回帰ゼロは 50 のみ。40は一律悪化、60はフラット、70はPrivate負）。**07は msl magnitude ノブで局所最適が確定。提出キューは不変（02→06→07）。**
> - **🔑 発見したリード（次サイクルの最優先角度）:** 発火3データは **同じ方向に動かない** — train_15 は msl を上げるほど単調に改善（msl70で +0.0039/+0.0040）する一方、train_09 は逆に msl を上げると回帰する「カウンタームーバー」、train_13 は中間。単一のグローバル msl 値では3データを同時に底上げできない。**つまり「msl magnitude を比(n_feat/n)でさらに階層化する ratio-tiered msl」**（例: 最高比の train_15(0.060) だけ msl=70、中比の train_09(0.016)/13(0.018) は msl=50 のまま）が train_15 の単調ゲインを悪化ゼロで拾える可能性。これが次ラウンドの最優先候補。生ログ=`experiments/bench_03/round22_msl_magnitude/{results.csv,summary.txt,run.log}`。
>
> **したがって次にやること（優先順・ラウンド22末で更新）:**
> 1. **本日 2026-07-13 UTC は 03(ERROR)が1日1件枠を消費済み＝新規提出せず（確認済: 提出履歴の当日UTC分は 03 のみ・現在 ~08 UTC）。**
> 2. **実提出キュー（3段・不変）:** 次UTC日 **2026-07-14** に `submissions/02_early_stopping/`（(B)シンプル路線・grader保証で確実に動く再確立）、**2026-07-15** に `submissions/06_ngated_l2/`（02をクリーンにパレート改善）、**2026-07-16** に `submissions/07_2gate_msl50/`（06をクリーンにパレート改善）。各提出手順は下記。※もし 07-14 の 02 が実グレーダーで無事スコアしたら 06/07 も同一 sklearn-only 構造なので安全に続けられる。
> 3. **次サイクルのオフライン最優先角度 = 上記リード「ratio-tiered msl」**（07の2ゲート構造はそのまま、msl発火データの中でさらに比で階層化。最高比 train_15(0.060) は msl=70、中比 train_09/13 は msl=50 のまま＝07と同一を保つ2段目のmagnitudeゲート）。狙いは train_15 の単調ゲイン(+0.004)を train_09 の回帰を起こさず拾い、07をクリーンにパレート改善すること。round22/replay.py を雛形に再利用（比の階層で msl 値を分岐する1本を足すだけ）。閾値は dataset_stats.csv の比で train_15 のみを高tierに、09/13 を低tierに分ける値（例 比≥0.03）に設定。
> 4. (A)複雑路線(03/05のgo.py堅牢化)は依然ユーザー確認待ちの大設計変更。独断着手しない。
>
> **✅ 採用候補（ラウンド23, 2026-07-13 ~09 UTC）— ラウンド22の最優先リード「ratio-tiered msl」を実装・検証し `submissions/08_ratio_tiered_msl/` を新設（validate合格）。07をクリーンにパレート改善:** ラウンド22で「msl magnitude は3発火データ(train_09/13/15)が同方向に動かず、train_15 は msl↑で単調改善(+0.004)する一方 train_09 は回帰するカウンタームーバー」と判明していた。この非対称性を、**msl発火データの中を同じ比(n_feat/n)でさらに階層化**し、最高比の train_15(0.060)だけ msl=70 に上げ、中比の train_09(0.0162)/13(0.0180)は 07 と同じ msl=50 のまま据え置くことで解けるという仮説を、専用replayハーネス `experiments/bench_03/round23_ratio_tiered_msl/replay.py`（`.venv`=sklearn-only, grader保証パッケージと一致・round22/replay.py を拡張）で検証。**de-risk: 2設定×16=32 fit 全てクラッシュ無し完走(CLEAN RUN=YES)、`git status --porcelain` は `round23_ratio_tiered_msl/` 配下のみ＝`submissions/`一切不触・確認済。** base=07(msl発火時=50固定)に対する候補 tiered(`msl = 70 if ratio>=0.030 else (50 if ratio>=0.015 else 20)`):
> - **ゲート挙動（設計通り）**: 新規の高tier(比≥0.030)は **train_15(0.060)のみ発火**し msl 50→70。中比 train_09(0.0162)/13(0.0180) は高tier非発火で msl=50 のまま(07と同一)、train_16(0.0116)は msl-gate非発火で msl=20(07と同一)、非発火12データもバイト同一。**つまり 07 と異なるのは train_15 ただ1データのみ。**
> - **結果: 変化したのは train_15 のみ。mean Public Δ=+0.00024 / Private Δ=+0.00025（＝train_15単独ゲインを16で平均）、W/L/T=1/0/15（両split回帰ゼロ）。** train_15 Public 0.8372→0.8411 (+0.00385)、Private 0.8396→0.8436 (+0.00403)＝ラウンド22の msl70 単独スイープで観測した train_15 のゲインと一致。他15データは 07 とバイト同一（delta 0.00000）。
> - **採用基準（mean両split正 かつ 両split回帰ゼロ）を満たす CLEAN IMPROVEMENT。** → `submissions/08_ratio_tiered_msl/` を新設（07をコピーし system.md の msl ゲートを1行だけ「70/50/20 の3tier」に拡張・validate_submission.py 合格・self-diff で変更は当該ゲート行1本のみと確認・既存提出物に差分なし確認済）。構造は 07と同じ sklearn-only 単一HGB・単発に「高比のみ msl を一段上げる比tier1本」を足しただけ（新規列なし・保証外パッケージ非依存＝03の脆弱性を持たない）。
> - **戦略的価値: 08は 07が拾えなかった train_15 の単調ゲインを、07の他データ(特にカウンタームーバー train_09)を一切悪化させずに取る。** 提出キューの改善段を 02→06→07→08 と一段深くする。生ログ=`experiments/bench_03/round23_ratio_tiered_msl/{results.csv,summary.txt}`。
>
> **🔎 探索済み・不採用（ラウンド24, 2026-07-13 ~10 UTC）— ラウンド23の候補(b)「validation_fraction を高比データで広げる」を実装・検証。不採用。simple単ノブ探索の飽和が一層濃厚に:** shipped 08 の l2/msl 2ゲートはそのまま固定し、**early_stopping の `validation_fraction` のみ**を、L2ゲートと同じ比(n_feat/n≥0.010)発火データ(train_09/13/15/16)で {0.15, 0.20} に広げて 0.1(=08既定) と比較（round23/replay.py を拡張した専用ハーネス `experiments/bench_03/round24_valfrac/replay.py`・sklearn-only `.venv`=grader保証パッケージと一致）。**de-risk: 3設定×16=48 fit 全CLEAN RUN=YES(exceptions=0)、`git status --porcelain` は `round24_valfrac/` 配下のみ＝`submissions/`一切不触・確認済。** 非発火12データは全設定バイト同一(delta 0)、差が出るのは発火4データのみ。base=08 に対し:
> - **vf015（0.1→0.15）: mean Public Δ=−0.00050 / Private Δ=−0.00065、Public W/L/T=1/3/12・Private 0/4/12。不採用。** train_13 だけ Public +0.0055 だが Private は −0.0005 と反転、train_09/15/16 は両split回帰（train_16 が最悪 −0.0067/−0.0073）。
> - **vf020（0.1→0.20）: mean Public Δ=−0.00070 / Private Δ=−0.00107、Public 1/3/12・Private 0/4/12。不採用（vf015より悪化幅拡大）。**
> - **知見:** 仮説（小n高比データは 0.1 holdout が~50–180行で早期停止シグナルが不安定→広げれば安定）は**反証された**。widening は holdout を増やす代わりに**極小データの学習行を削る副作用**が支配的で、train_13 を除く全発火データが回帰。**early_stopping の validation_fraction は既定 0.1 がこのsuiteのスイートスポット**。**train_16(比0.0116・msl-gate非発火の境界データ)が一貫して最も大きく回帰**＝境界データは正則化系のどのノブを触っても脆い（R20/R21のmsl境界回帰と同じ傾向）。生ログ=`experiments/bench_03/round24_valfrac/{results.csv,summary.txt}`。
> - **⚠️ 方針上の含意:** l2(R17-19)・msl(R20-23)・max_leaf_nodes(R20)・validation_fraction(R24) と、**shipped 単一HGB の正則化・早期停止系の単ノブはほぼ出尽くし、いずれも 08 を超えるクリーン改善を出せていない**（08=直近唯一の採用も train_15 単独 +0.004＝16平均で +0.00024 の極小ゲイン）。**オフライン単ノブ探索の実効的な伸びしろは枯渇に近い。** 最大の未回収リスク/価値は「02〜08 のどれも実グレーダーで未採点（01=0.787 が唯一の実LB値）」という点にあり、これは 1日1提出のボトルネックと A/B の大方針(ユーザー判断待ち)に律速される。次サイクルも探索は継続する(タスク指示)が、期待値は低いことを明記しておく。
>
> **したがって次にやること（優先順・ラウンド24末で更新）:**
> 1. **本日 2026-07-13 UTC は 03(ERROR)が1日1件枠を消費済み＝新規提出せず（確認済: 提出履歴の当日UTC分は 03 のみ・現在 ~10 UTC）。**
> 2. **実提出キュー（4段・不変）:** 次UTC日 **2026-07-14** に `submissions/02_early_stopping/`（(B)シンプル路線・grader保証で確実に動く再確立）、**2026-07-15** に `submissions/06_ngated_l2/`、**2026-07-16** に `submissions/07_2gate_msl50/`、**2026-07-17** に `submissions/08_ratio_tiered_msl/`。各提出手順は下記。※4段とも同一の sklearn-only 単一HGB・単発構造なので、07-14 の 02 が実グレーダーで無事スコアすれば以降も安全に続く。**キューはR24で不変（08が依然ベスト）。**
> 3. **次サイクルのオフライン角度（期待値低・上の含意参照）:** 正則化・早期停止系の単ノブは R17-24 でほぼ枯渇（l2/msl/mln/valfrac すべて 08 を超えず）。まだ触っていない直交ノブは実質 **`learning_rate`**（既定0.1を高比データで下げる＝遅学習で正則化）くらい。これを round24/replay.py 雛形で1本試す。ただし他の正則化ノブが全滅している以上ヒットの見込みは薄い。**ここまで単ノブが枯れたら、次の実効的な伸びしろは (A)複雑路線のgo.py堅牢化 か (c)モデル族/ブレンド多様化——どちらも大設計変更でユーザー確認必須**。独断で複雑化しない。
> 4. (A)複雑路線(03/05のgo.py堅牢化)は依然ユーザー確認待ちの大設計変更。独断着手しない。
>
> **🔎 探索済み・不採用（ラウンド25, 2026-07-13 ~11 UTC）— ラウンド24の最後の未検証直交ノブ「learning_rate を高比データで下げる」を実装・検証。不採用。単ノブ探索の枯渇が確定的に:** shipped 08 の l2/msl 2ゲートはそのまま固定し、**HGB の `learning_rate` のみ**を、L2ゲートと同じ比(n_feat/n≥0.010)発火データ(train_09/13/15/16)で {0.07, 0.05} に下げて既定0.1(=08)と比較（round24/replay.py を拡張した専用ハーネス `experiments/bench_03/round25_learning_rate/replay.py`・sklearn-only `.venv`=grader保証パッケージと一致）。**de-risk: 3設定×16=48 fit 全CLEAN RUN=YES(exceptions=0/skipped=0/invariant_violations=0)、`git status --porcelain` は `round25_learning_rate/` 配下のみ＝`submissions/`一切不触・確認済。** 非発火12データは全設定バイト同一(delta 0)、差が出るのは発火4データのみ。base=08 に対し:
> - **lr07（0.1→0.07）: mean Public Δ=−0.00035 / Private Δ=−0.00022、Public W/L/T=1/3/12・Private 2/2/12。不採用（両split平均負）。** train_09 だけ両split改善(Public +0.0031/Private +0.0017)だが、train_13(Public −0.0051/Private −0.0058)・train_16(Public −0.0027)が明確に回帰。
> - **lr05（0.1→0.05）: mean Public Δ=−0.00078 / Private Δ=−0.00097、Public 1/3/12・Private 1/3/12。不採用（lr07より悪化幅拡大）。** train_13 が最悪（Public −0.0100/Private −0.0115）、train_15 も両split回帰。
> - **🔑 知見（この角度は ratio 軸では救済不能＝dead end）:** 遅学習に対し train_09 と train_13 が**逆方向に動くカウンタームーバー**（train_09 は lr↓で改善、train_13 は大きく回帰）。ラウンド22の msl と同じ非対称だが、**今回は ratio-tier で分離できない**——train_09 の比は 18/1109=**0.0162**、train_13 の比は 9/500=**0.0180** で、**救いたい train_09 の方が比が低い**。「高比だけ lr を下げる」tier では原理的に必ず train_13(高比側)も巻き込むため、round23 のような tier 分岐で train_09 単独を拾うことができない。→ **learning_rate は不採用で確定・この軸の派生も閉じた。**
> - **⚠️ 方針上の含意（更新）:** l2(R17-19)・msl(R20-23)・max_leaf_nodes(R20)・validation_fraction(R24)・learning_rate(R25) と、**shipped 単一HGB の正則化・早期停止・学習率系の単ノブは完全に出尽くし、いずれも 08 を超えるクリーン改善を出せなかった**（08=直近唯一の採用も train_15 単独 +0.004＝16平均+0.00024の極小）。**オフライン単ノブ探索の伸びしろは枯渇。** 次の実効的な伸びしろは (A)複雑路線 go.py 堅牢化 か (c)モデル族/ブレンド多様化のみで、**どちらも「シンプル1変更」を超える大設計変更でユーザー確認必須**。次サイクル以降も探索は継続する(タスク指示)が、単ノブでは期待値が実質ゼロであることを明記する。生ログ=`experiments/bench_03/round25_learning_rate/{results.csv,summary.txt,run.log}`。
>
> **したがって次にやること（優先順・ラウンド25末で更新）:**
> 1. **本日 2026-07-13 UTC は 03(ERROR)が1日1件枠を消費済み＝新規提出せず（確認済: 提出履歴の当日UTC分は 03 のみ・現在 ~11 UTC）。**
> 2. **実提出キュー（4段・不変）:** 次UTC日 **2026-07-14** に `submissions/02_early_stopping/`、**2026-07-15** に `submissions/06_ngated_l2/`、**2026-07-16** に `submissions/07_2gate_msl50/`、**2026-07-17** に `submissions/08_ratio_tiered_msl/`。**キューはR25で不変（08が依然ベスト、単ノブ探索は枯渇）。** 各提出手順は下記。
> 3. **次サイクルのオフライン角度（期待値ほぼゼロ・上の含意参照）:** 正則化・早期停止・学習率の単ノブは R17-25 で完全枯渇。残る simple 単ノブは実質 `max_iter` くらいだが early_stopping 併用のため意味が薄い。**単ノブでヒットは見込めない。** 次に試すなら「まだ触っていない直交な simple 前処理」（例: 数値列の欠損補完戦略の変更、単調制約など）を1本だけ round25/replay.py 雛形で試す。それも枯れたら、実効的な伸びしろは (A)go.py 堅牢化 or (c)モデル族多様化＝**大設計変更でユーザー確認必須**（独断で複雑化しない）。
> 4. (A)複雑路線(03/05のgo.py堅牢化)は依然ユーザー確認待ちの大設計変更。独断着手しない。
>
> **🔎 探索済み・不採用（ラウンド26, 2026-07-13 ~12 UTC）— 正則化族が枯れたので初の非正則化・直交ノブ「class_weight」を検証。不採用。だが「このsuiteは事実上クラス均衡」という新事実を確定し、不均衡系アプローチの軸を丸ごと閉じた:** R17-25 は全て正則化/早期停止/学習率族の単ノブで、いずれも 08 を超えなかった。R26 は初めて**族の外**の直交ノブ `HistGradientBoostingClassifier(class_weight=...)`（sklearn 1.6+、.venv=1.9.0で利用可）を検証。AUCはランクベースだがクラス再重み付けは学習される確率のランキングを変えうるので動く可能性がある。08 の l2/msl 2ゲート・early_stopping・特徴量は全て固定し、class_weight ノブのみを差し替え。専用replayハーネス `experiments/bench_03/round26_class_weight/replay.py`（round25/replay.py を拡張・sklearn-only `.venv`=grader保証パッケージと一致）で16データPublic/Private別AUC採点。**de-risk: 3設定×16=48 fit 全CLEAN RUN=YES(exceptions=0/skipped=0/invariant_violations=0)、`git status --porcelain` は `round26_class_weight/` 配下のみ＝`submissions/`一切不触・確認済。base再現は round25 baseとバイト一致(max abs diff=0.0)。** base=08 に対する2候補:
> - **cw_gated（`class_weight='balanced'` を「学習データのマイノリティ比 < 0.35」の不均衡データだけに適用）: 発火ゼロ。mean Public/Private Δ=+0.00000、W/L/T=0/0/16。** **🔑 発火ゼロの理由＝新事実: この16データは全て事実上クラス均衡（マイノリティ比 0.489〜0.500）**で、どのデータも 0.35 ゲートを越えない → cw_gated は全16データで 08 とバイト同一。**不採用（＝08そのもの）。**
> - **cw_global（`class_weight='balanced'` を全データに適用）: mean Public Δ=−0.00166 / Private Δ=−0.00129、Public W/L/T=6/8/2・Private 6/8/2。不採用（両split平均負＋両split回帰8件、最悪 train_16 Public −0.00918/Private −0.00770）。** 均衡データを再重み付けするとノイズ注入になり左右対称に僅かに悪化。
> - **🔑 知見（不均衡系の軸を閉じる）: このsuiteは~50/50均衡なので、class_weight・少数クラスオーバーサンプリング・focal的な不均衡対策は構造的に無効**（発火しないか、均衡データを歪めて悪化）。正則化族(R17-25)に続き、**不均衡族もこのsuiteでは伸びしろ無しと確定**。
> - **⚠️ 方針上の含意（更新）: 単ノブ正則化(R17-25)＋不均衡対策(R26)がいずれも 08 を超えず。オフラインの simple 探索で残る直交角度は「数値欠損の明示補完」「単調制約」等の前処理系ごく僅かのみで、いずれも期待値は低い。** 実効的な伸びしろは依然 (A)複雑路線 go.py 堅牢化 or (c)モデル族/ブレンド多様化＝**大設計変更でユーザー確認必須**。次サイクルも探索は継続する(タスク指示)が単ノブの期待値は実質ゼロ。生ログ=`experiments/bench_03/round26_class_weight/{results.csv,summary.txt,run.log}`。
>
> **🔎 探索済み・不採用（ラウンド27, 2026-07-13 ~13 UTC）— 正則化・不均衡族が枯れたので初の「ランダム化系」直交ノブ `max_features`（列サブサンプリング, sklearn 1.4+）を検証。不採用。列サブサンプリングもこのsuiteでは伸びしろ無し:** R17-26 で葉サイズ/重み/学習率/クラス重みの単ノブが全滅したので、R27 は初めて**per-split の特徴量サブサンプリング** `HistGradientBoostingClassifier(max_features=<mf>)` を検証。葉サイズ/重み系(l2/msl)とは別系統のランダム化正則化なので効く可能性があった。08 の l2/msl 2ゲート・early_stopping・特徴量は全て固定し、`max_features` のみを L2ゲートと同じ比(n_feat/n≥0.010)発火データ(train_09/13/15/16)に適用、mf∈{0.7, 0.5} を 1.0(=08既定) と比較。専用replayハーネス `experiments/bench_03/round27_max_features/replay.py`（round26/replay.py を拡張・sklearn-only `.venv`=grader保証パッケージと一致）で16データPublic/Private別AUC採点。**de-risk: 3設定×16=48 fit 全CLEAN RUN=YES(exceptions=0/skipped=0/invariant_violations=0)、`git status --porcelain` は `round27_max_features/` 配下のみ＝`submissions/`一切不触・確認済。非発火12データは全設定で08とバイト同一(delta 0)・検証済。** base=08 に対する2候補:
> - **mf=0.7:** mean Public Δ=**−0.00048** / Private Δ=**−0.00076**、Public W/L/T=2/2/12・Private 2/2/12。**不採用（両split平均負）。** train_09(+0.0078/+0.0017)・train_15(+0.0034/+0.0042)は改善するが、**train_13(−0.0153/−0.0135)・train_16(−0.0036/−0.0046)が両split明確に回帰**して相殺以上に悪化。
> - **mf=0.5:** mean Public Δ=**−0.00055** / Private Δ=**−0.00070**、Public 0/4/12・Private 1/3/12。**不採用（0.7より悪化幅拡大・発火4データほぼ全滅）。**
> - **🔑 知見（列サブサンプリング軸を閉じる）: train_13 が再びカウンタームーバー**（R22 msl↑・R25 lr↓でも回帰した高比・極小n=500データ）。列を間引くと train_09/15 の利得を train_13/16 の回帰が必ず上回る。**葉/重み(l2/msl)系に続き、ランダム化(列サブサンプリング)系もこのsuiteでは 08 を超えられないと確定。**
> - **⚠️ 方針上の含意（更新・R27）: 正則化(R17-25)＋不均衡(R26)＋ランダム化列サブサンプリング(R27) の単ノブ族がすべて 08 を超えず。simple 単ノブ探索は実質完全枯渇。** 唯一 08 が拾えていない一貫パターンは「train_13(高比・n=500の極小・カウンタームーバー)を悪化させずに底上げする手が無い」こと。残る simple 直交角度は前処理系（欠損明示補完＝HGBネイティブNaN処理で効果薄・単調制約＝generic設定困難）ごく僅かで期待値ほぼゼロ。**実効的な伸びしろは (A)複雑路線 go.py 堅牢化 or (c)モデル族/ブレンド多様化＝大設計変更でユーザー確認必須**。生ログ=`experiments/bench_03/round27_max_features/{results.csv,summary.txt,run.log}`。
>
> **したがって次にやること（優先順・ラウンド27末で更新）:**
> 1. **本日 2026-07-13 UTC は 03(ERROR)が1日1件枠を消費済み＝新規提出せず（確認済: 提出履歴の当日UTC分は 03 のみ・現在 ~13 UTC）。**
> 2. **実提出キュー（4段・不変）:** 次UTC日 **2026-07-14** に `submissions/02_early_stopping/`、**2026-07-15** に `submissions/06_ngated_l2/`、**2026-07-16** に `submissions/07_2gate_msl50/`、**2026-07-17** に `submissions/08_ratio_tiered_msl/`。**キューはR27で不変（08が依然ベスト）。** 各提出手順は下記。
> 3. **次サイクルのオフライン角度（期待値ほぼゼロ）:** 正則化族(R17-25)・不均衡族(R26)・ランダム化列サブサンプリング(R27)ともに 08 を超えず単ノブ探索は枯渇。残る simple 直交前処理は「数値列の欠損明示補完(HGBはNaNネイティブ処理なので効果薄の見込み)」「単調制約(generic設定が難しい)」程度。1本だけ round27/replay.py 雛形で試す。それも枯れたら実効的な伸びしろは (A)go.py 堅牢化 or (c)モデル族多様化＝**大設計変更でユーザー確認必須**（独断で複雑化しない）。
> 4. (A)複雑路線(03/05のgo.py堅牢化)は依然ユーザー確認待ちの大設計変更。独断着手しない。
>
> **🔎 探索済み・不採用（ラウンド28, 2026-07-13 ~14 UTC）— 正則化「族の外」だがまだ触っていなかった直交ノブ `max_bins`（ヒストグラム分割粒度, HGB既定255）を検証。不採用。単ノブ探索の枯渇がさらに強固に:** R17-27 で葉/重み(l2/msl/mln)・学習率・クラス重み・列サブサンプリングが全滅したので、R28 は**別機構**の正則化 `HistGradientBoostingClassifier(max_bins=<mb>)` を検証。max_bins を下げると特徴量の離散化が粗くなり、極小データの過学習を抑える分散削減効果があるので、l2/msl とは異なる経路で効く可能性があった。08 の l2/msl 2ゲート・early_stopping・特徴量は全て固定し、`max_bins` のみを L2ゲートと同じ比(n_feat/n≥0.010)発火データ(train_09/13/15/16)に適用、mb∈{127, 63} を既定255(=08)と比較。専用replayハーネス `experiments/bench_03/round28_max_bins/replay.py`（round27/replay.py を拡張・sklearn-only `.venv`=grader保証パッケージと一致）で16データPublic/Private別AUC採点。**de-risk: 3設定×16=48 fit 全CLEAN RUN=YES(exceptions=0/skipped=0/invariant_violations=0)、`git status --porcelain` は `round28_max_bins/` 配下のみ＝`submissions/`一切不触・確認済。非発火12データは全設定で08とバイト同一(delta 0)・検証済。** base=08 に対する2候補:
> - **mb=127:** mean Public Δ=**−0.00173** / Private Δ=**−0.00184**、Public W/L/T=0/4/12・Private 0/4/12。**不採用（両split平均負・発火4データ全てが両split回帰）。** 最悪 train_13(−0.0134/−0.0161)・train_16(−0.0094/−0.0106)。分割粒度を半減すると極小データがむしろ情報を失い一律悪化。
> - **mb=63:** mean Public Δ=**−0.00000** / Private Δ=**−0.00019**、Public 2/2/12・Private 1/3/12。**不採用（両split平均が非正）。** train_09 は両split改善(+0.0015/+0.0021)するが、train_15(Public −0.0030)・train_16・train_13(Private −0.0019) が回帰して相殺。
> - **🔑 知見（ヒストグラム分割粒度の軸も閉じる）: train_13 が再びカウンタームーバー**（R22 msl↑・R25 lr↓・R27 列間引きでも回帰した高比・極小n=500データ）。max_bins を粗くしても train_09 の改善を train_13/15/16 の回帰が上回る。**葉/重み・学習率・クラス重み・列サブサンプリングに続き、ヒストグラム分割粒度もこのsuiteでは 08 を超えられないと確定。**
> - **⚠️ 方針上の含意（更新・R28）: 正則化(R17-25)＋不均衡(R26)＋列サブサンプリング(R27)＋分割粒度(R28) の単ノブ族が**すべて 08 を超えず。simple 単ノブ探索は実質完全に枯渇**。唯一 08 が拾えない一貫パターンは「train_13(高比・n=500極小・カウンタームーバー)を悪化させずに底上げする単ノブが存在しない」こと——どのノブでも train_13 は他の発火データと逆方向に動く。残る simple 直交前処理（欠損明示補完＝HGBネイティブNaN処理で効果薄、単調制約＝generic設定困難）は期待値ほぼゼロ。**実効的な伸びしろは (A)複雑路線 go.py 堅牢化 or (c)モデル族/ブレンド多様化＝大設計変更でユーザー確認必須**。生ログ=`experiments/bench_03/round28_max_bins/{results.csv,summary.txt,run.log}`。
>
> **したがって次にやること（優先順・ラウンド28末で更新）:**
> 1. **本日 2026-07-13 UTC は 03(ERROR)が1日1件枠を消費済み＝新規提出せず（確認済: 提出履歴の当日UTC分は 03 のみ・現在 ~14 UTC）。**
> 2. **実提出キュー（4段・不変）:** 次UTC日 **2026-07-14** に `submissions/02_early_stopping/`、**2026-07-15** に `submissions/06_ngated_l2/`、**2026-07-16** に `submissions/07_2gate_msl50/`、**2026-07-17** に `submissions/08_ratio_tiered_msl/`。**キューはR28で不変（08が依然ベスト）。** 各提出手順は下記。
> 3. **次サイクルのオフライン角度（期待値ほぼゼロ）:** 正則化族(R17-25)・不均衡(R26)・列サブサンプリング(R27)・分割粒度(R28) すべて 08 を超えず単ノブ探索は完全枯渇。残る simple 直交前処理は「欠損明示補完(HGBはNaNネイティブなので効果薄)」「単調制約(generic設定困難)」程度で、1本試す価値はあるが期待値ほぼゼロ。**枯れたら実効的な伸びしろは (A)go.py 堅牢化 or (c)モデル族多様化＝大設計変更でユーザー確認必須**（独断で複雑化しない）。単ノブが完全に枯れた事実は、次にユーザーへ (A)/(c) の方針判断を仰ぐ材料として明記する。
> 4. (A)複雑路線(03/05のgo.py堅牢化)は依然ユーザー確認待ちの大設計変更。独断着手しない。
>
> `08_ratio_tiered_msl`（採用候補・07をクリーンにパレート改善）の提出手順:
> `(cd submissions/08_ratio_tiered_msl/agent && rm -f ../submission.zip && zip -r ../submission.zip . -x '.*')` →
> `kaggle competitions submit -c autonomous-agent-prediction-beta -f submissions/08_ratio_tiered_msl/submission.zip -m "08_ratio_tiered_msl: on top of 07, min_samples_leaf=70 on high tier n_feat/n>=0.030 (train_15)"`
>
> `07_2gate_msl50`（採用候補・06をクリーンにパレート改善）の提出手順:
> `(cd submissions/07_2gate_msl50/agent && rm -f ../submission.zip && zip -r ../submission.zip . -x '.*')` →
> `kaggle competitions submit -c autonomous-agent-prediction-beta -f submissions/07_2gate_msl50/submission.zip -m "07_2gate_msl50: on top of 06, min_samples_leaf=50 on stricter gate n_feat/n>=0.015"`
>
> `06_ngated_l2`（採用候補・02をクリーンにパレート改善）の提出手順:
> `(cd submissions/06_ngated_l2/agent && rm -f ../submission.zip && zip -r ../submission.zip . -x '.*')` →
> `kaggle competitions submit -c autonomous-agent-prediction-beta -f submissions/06_ngated_l2/submission.zip -m "06_ngated_l2: gated l2_regularization=1.0 when n_features/n_rows>=0.010 (on top of 02)"`
>
> `02_early_stopping`（安全なフォールバック）の提出手順:
> `(cd submissions/02_early_stopping/agent && rm -f ../submission.zip && zip -r ../submission.zip . -x '.*')` →
> `kaggle competitions submit -c autonomous-agent-prediction-beta -f submissions/02_early_stopping/submission.zip -m "02_early_stopping: single HGB + early_stopping (simple fallback after 03 ERROR)"`
>
> （参考・複雑路線を続ける場合の 05 提出手順）
> `(cd submissions/05_te_ngated/agent && rm -f ../submission.zip && zip -r ../submission.zip . -x '.*')` →
> `kaggle competitions submit -c autonomous-agent-prediction-beta -f submissions/05_te_ngated/submission.zip -m "05_te_ngated: n-gated OOF target encoding (n<1500 cats only)"`
>
> **📋 05の次のキュー（ラウンド7で更新）:** `03_cv_ensemble` の実LBスコアが確定したら、**次の実提出候補は `submissions/05_te_ngated/`（n-gated target encoding = 03 ＋「n<1500 の小規模データセットだけに OOF ターゲットエンコーディング列を足す」1点変更。作成済み・`validate_submission.py`合格済み・下記ラウンド7参照）を最優先**とする。理由: オフライン実測で 03 を**パレート優越**する（n≥1500 の12データは 03 とバイト同一＝弱点 train_06 含めゼロリスク、小n の train_13 +0.0095 / train_05 +0.0053 / train_15 +0.0018 を悪化なしで底上げ、平均 +0.0143→+0.0154）。従来キューの `submissions/04_missing_count/`（平均+0.0144）と `freq-encode-cats` はその次の候補に後退。**注意: これらはまだ実LBで較正していないオフライン代理指標なので、03のLBスコアが出てから最終判断すること。**
>
> **🔎 探索済み・保留（ラウンド5, 2026-07-12）:** 新しい角度「頻度エンコーディング(freq-encode-cats)」を検証済み。03ベースに対しては弱点train_06を+0.0004→+0.0013へ底上げし全16データで悪化なしのクリーンな改善だが、キュー済みの04_missing_count(train_06 +0.0012)と実質同点かつより複雑なため**不採用**。ただしfreq(値の頻度)とmissing-count(行の欠損数)は直交する情報源なので、**04採用後の土台に対する次の1変更**として再検討する価値あり（＝「新しい角度」として再びゼロから探すのではなく、この案を優先的に検討してよい）。候補=`experiments/bench_03/candidates/freq-encode-cats/`。
>
> **🔎 探索済み・保留（ラウンド6, 2026-07-12 16:xx UTC）:** 新しい角度「OOFターゲットエンコーディング(target-encode-oof)」を検証済み。**不採用。** 平均delta +0.0154（03=+0.0143）で数値上は上回るが、その利得はすべて小n(500〜1060)カテゴリ3件(train_13/15/05)に集中する一方、**狙った弱点train_06はむしろ+0.0013→+0.0006と悪化**し、train_01/03/08でも小回帰。採用基準（平均を明確に上回る／弱点を悪化なく改善）をクリーンに満たさない。**知見: TEはn依存で、小n(≲1000)カテゴリで明確に効き(train_13 +0.0095)、中〜大の全カテゴリ(train_06)では僅かに害。** 次の優先角度案＝「n閾値未満のときだけ`__te`列を足すn-gated target encoding」（小nの利得を取り train_06の回帰を避ける）。候補=`experiments/bench_03/candidates/target-encode-oof/`、詳細は下の「ラウンド6」メモ。
>
> **✅ 採用候補（ラウンド7, 2026-07-12 ~17-18 UTC）:** ラウンド6で「次の優先角度」に挙げた **n-gated target encoding** を検証し、**採用候補として `submissions/05_te_ngated/` を作成・validate合格**。TEブロック全体を `if len(X) < 1500 and len(cats) > 0:` で囲い、小規模データセットだけに `__te` 列を足す1点変更。クリーンな同条件（workers=2）比較で **03 をパレート優越**: n≥1500 の12データは 03 とバイト同一（弱点 train_06 は +0.0004 のまま＝ラウンド6の非gated版が起こした train_06 回帰を完全に回避）、小n では train_13 +0.0309→+0.0404 / train_05 +0.0337→+0.0390 / train_15 +0.0213→+0.0231 を悪化なしで底上げ、train_09 は不変。平均 +0.0143→+0.0154、悪化データセットゼロ。これがこれまでで最もクリーンな候補（利得は大きく再現性のある小nゲイン由来・下振れ数学的にゼロ）。**本日(2026-07-12 UTC)は 01_baseline で提出枠を使い切っているため未提出。次UTC日はまず 03 を提出（較正）→その後 05_te_ngated を最優先で提出。**
>
> **🔎 探索済み・不採用（ラウンド9, 2026-07-12 ~19 UTC）:** 新しい直交角度「**レアカテゴリ集約(rare-cat-collapse)**」を検証済み。**不採用（no-op）。** `load()`のカテゴリ処理で、train内出現回数<5 のレベルだけを単一トークン`__rare__`に畳み込む1点変更（列は増やさない純粋なデノイズ・target encodingとも直交・大nでは非発火で安全）。クリーンな同条件(workers=2)実測で **16データ全部が 03 とバイト同一**（平均+0.0143・最悪train_06 +0.0004、全て03と同値、悪化ゼロ）。**知見: この16データのカテゴリ列には閾値5未満のロングテール・レベルが実質存在しないか、GBMのネイティブ・カテゴリ処理が既にレア水準を吸収しており、明示的なレア畳み込みは4桁AUC解像度で追加信号ゼロ。** freq/missing-count/missing-flags/rare-collapse と、カテゴリ列の「頻度/欠損/レア」系デノイズ角度は一通り no-op を確認できたので、今後の探索は「新情報を足す」系（target encoding=05で採用済／カテゴリ交互作用など）か「モデル族の多様化」系に絞るのが効率的。キュー不変（03→05→04/freq）。候補=`experiments/bench_03/candidates/rare-cat-collapse/`。
>
> **🔎 探索済み・不採用（ラウンド10, 2026-07-12 ~20 UTC）:** ラウンド9で挙げた「新情報を足す」系の未検証角度「**カテゴリ交互作用(cat-interactions)**」を **05_te_ngated の上に**検証。**不採用。** 低カーディナリティのカテゴリ列ペア（`card_a*card_b<=100`）だけを文字列連結して native categorical 列を最大3本足す1点変更（TEが小n専用なのに対し、これは大nの全カテゴリ弱点 train_06 を狙う直交角度）。クリーンな同条件(workers=2)実測: 平均delta **+0.0152**（05=+0.0154、−0.0002＝ノイズ域で上回らない）、狙いの train_06 は +0.0004→+0.0008 と微増するが **train_05 が 05比 −0.0047 明確に回帰**（train_13 は +0.0037 改善）。「05のmeanをクリーンに上回る／ゼロ回帰でtrain_06改善」の採用基準をどちらも満たさない。**知見: この16データのカテゴリ列は概ね高カーディナリティで `card*card<=100` にほぼ掛からず交互作用列がほとんど生成されない安全no-opに近く、効果はノイズ域・局所回帰だけが残る。** 「新情報を足す」系の残り有望角度は target encoding のバリアント（05で採用済）に絞られてきたので、次は **(b) モデル族の多様化系**（ブレンド重みの原理的改善など）へ軸足を移すのが効率的。ただしモデル族追加は「シンプル1変更」を超える設計変更になりうるので、大きくするならユーザー確認を挟む。キュー不変（03→05→04/freq）。候補=`experiments/bench_03/candidates/cat-interactions/`。
>
> **🔎 探索済み・不採用（ラウンド11, 2026-07-12 ~21 UTC）:** 初めて**数値側**の角度「**数値行集約(numeric-row-aggregates)**」を **05_te_ngated の上に**検証。**不採用。** `load()` の return 直前で、数値特徴量が2列以上のときだけ行方向の平均 `__num_row_mean` と標準偏差 `__num_row_std` の2列を足す1点変更（全カテゴリの train_06 は数値列<2で非発火＝設計上の安全no-op、大nの弱点 train_12 を狙う直交角度）。**候補と05ベースを同セッション・同条件(workers=2)で両方実測**（stale reference 回避）: 平均delta **候補+0.0147 vs 05+0.0154（−0.0007）**、16データ中**11データで05より悪化**。特に**狙った train_12 が +0.0031→+0.0019 と悪化**、小nTEも train_13/05/09 揃って回帰（+0.0404→+0.0378 / +0.0390→+0.0365 / +0.0289→+0.0262）、train_06 は +0.0004 で05とバイト一致（no-opガード正常）。「05のmeanをクリーンに上回る／弱点を悪化なく改善」の採用基準をどちらも満たさない。**知見: スケールの異なる生の数値列を素の行平均/分散で集約すると、大スケール列に支配されたノイズ2列が増えるだけで、木は低信号の分割候補が増えた分フィットが希釈され過半で小幅悪化。行内集約は既にアクセス可能な情報の劣化再表現に留まり新判別情報を足さない。** これで数値側の素朴FEも no-op〜微悪と確認でき、**この16データ suite は素朴な特徴量エンジニアリングに対し概ね飽和**という累積結論が補強された。実効的な伸びしろは (b)モデル族/ブレンドの多様化に絞られる（設計変更のためユーザー確認要）。キュー不変（03→05→04/freq）。候補=`experiments/bench_03/candidates/numeric-row-aggregates/`。
>
> **🔎 探索済み・不採用（ラウンド12, 2026-07-12 ~22 UTC）:** 新情報を足す系が飽和したため、初めて**採用済みの勝ち技(TE)自体のハイパラ**を突く角度「**TE平滑化強化(te-smoothing-20)**」を **05_te_ngated の上に**検証。**不採用。** 05のOOFターゲットエンコーディング(`len(X)<1500`ゲート)のBayes平滑化定数を `smoothing=10.0→20.0` に上げるだけの1数値変更（レア水準の符号化を全体平均へより強く寄せる正則化強化・列数不変・n≥1500の12データはゲート非発火でバイト同一）。**候補と05ベースを同セッション・同条件(workers=2)で両方実測。** 変化したのは小n3データのみ: **train_05 が +0.0390→+0.0337 と明確に回帰（oracleも 0.6851→0.6833 と低下＝選択由来でない真の特徴量劣化）**、train_13 +0.0404→+0.0421・train_15 +0.0231→+0.0234 は小幅改善、train_09 は不変、train_06(大n非発火)含む残り13データはバイト同一。平均delta **05=+0.01538 vs 候補=+0.01517（−0.00021）** で05を上回らず、しかも最小の悪化許容(train_05 −0.0053)を出す。「05のmeanをクリーンに上回る／悪化なし」の採用基準を両方満たさない。**知見: 平滑化を強めると最小n(≲1000)で符号化がレア水準の弱信号ごと全体平均へ潰れ、小nでの効き方が非一貫(1悪化/2改善/1不変)になる＝過正則化の兆候。05の smoothing=10 が既にこのsuiteのスイートスポット近傍。** これで「勝ち技のハイパラ微調整」も伸びしろ無しと確認でき、**素朴FE飽和＋勝ち技も局所最適**という累積結論がさらに補強された。実効的な伸びしろは (b)モデル族/ブレンドの多様化に事実上絞られる（設計変更のためユーザー確認要）。キュー不変（03→05→04/freq）。候補=`experiments/bench_03/candidates/te-smoothing-20/`。
>
> **🔎 探索済み・不採用（ラウンド13, 2026-07-12 ~23 UTC）:** ラウンド5の着眼「頻度(count)エンコーディングはTEと直交」を、**05の小nゲート内でTE列に加えcount列(`c+"__cnt"`)も足す**形(`load()`に3行追加・目的不使用でリークなし・n≥1500非発火で train_06 バイト同一)で初検証。**不採用。** 05ベースと候補を同セッション・同条件(workers=2)で両方完走させ比較: 平均delta **05=+0.01538 vs 候補=+0.01483（−0.00055）**。ゲート発火する小n3データが**揃って悪化・改善ゼロ**（train_05 +0.0390→+0.0365 / train_13 +0.0404→+0.0354 / train_15 +0.0231→+0.0218、train_09不変、train_06非発火で不変）。「05のmeanを上回る／悪化なし」を両方満たさず。**知見: countはTEに対し直交な追加信号にならず、最も脆い最小nでTEの目的平均を低信号列で希釈し一様に小幅悪化。ラウンド12(TE平滑化)と同じ「小nは追加操作に弱い」パターンで、小nの利得は目的平均(TE)そのものに由来という結論を補強。特徴量側は素朴FE・勝ち技への追加符号化・勝ち技ハイパラすべて飽和と確定。** 残る伸びしろは (b)モデル族/ブレンド多様化のみで、これは設計変更のため実行前にユーザー確認要。キュー不変（03→05→04/freq）。候補=`experiments/bench_03/candidates/ngated-freq-encode/`。
> （注: この回、`coder` サブエージェントがベンチマーク完走前に「1/16完了」で早期リターンしバックグラウンド実行が中断していたため、オーケストレータが自己diff(意図した3行のみ)確認後に両ベンチを完了マーカー確認まで走らせ直して判定した。**サブエージェントの「実行中」報告を最終結果として扱わない**という既存ルールが実際に効いた事例。）
>
> **🔎 探索済み・不採用（ラウンド14, 2026-07-13 ~00 UTC / 03提出後の同サイクル）:** 採用済みの勝ち技(TE)の**もう一方のハイパラ＝nゲート閾値そのもの**を突く角度「**TEゲート拡張(te-gate-widen-4000)**」を **05_te_ngated の上に**検証。**不採用。** ゲート条件 `if len(X) < 1500` の閾値だけを `1500→4000` に上げる1数値変更（TE発火対象を「n<1500の小規模」から「n<4000の中規模まで」に広げる。この16データで新たにTE発火するのは中規模カテゴリ唯一の train_03(n=3501,7カテゴリ)のみ＝設計上、他15データはゲート挙動不変でバイト同一になるはず）。**候補と05ベースを同セッション・同条件(workers=2)で両方完走させ比較。** 結果は設計通りに外科的: **15データが05とバイト同一**（train_06含む）、変化は狙い通り train_03 のみで **+0.0176→+0.0172（−0.0004）と回帰**。平均delta **05=+0.01538 vs 候補=+0.01536（−0.00002）**。「05のmeanをクリーンに上回る／悪化なし」の採用基準を両方満たさない。**知見: ラウンド6で示した「TEはn依存で小n(≲1100)に効き中〜大では僅かに害」を、ゲート**上側**境界で定量確認。ラウンド12が平滑化定数(smoothing=10が最適)を確認したのに続き、**ゲート閾値も1500が上側で最適**（4000へ広げると中規模train_03を1件巻き込んで悪化）。TEハイパラ(平滑化・ゲート閾値)は両方向とも局所最適で、05はこのsuiteでのTEスイートスポット。** これで「勝ち技のハイパラ探索」はほぼ出尽くし、**素朴FE飽和＋TE両ハイパラ局所最適**の累積結論をさらに補強。実効的な伸びしろは (b)モデル族/ブレンド多様化に事実上限定（設計変更のためユーザー確認要）。キュー不変（05→04/freq）。候補=`experiments/bench_03/candidates/te-gate-widen-4000/`、生ログ=`experiments/bench_03/round14/`。
>
> **🔎 探索済み・不採用（ラウンド8, 2026-07-12 ~18 UTC）:** 新しい直交角度「**列ごとの欠損フラグ(missing-flags-per-col)**」を検証済み。**不採用（no-op）。** 数値列で欠損があるものだけに `<col>__isna`(0/1)列を足す1点変更（カテゴリ列は既に`.fillna("nan")`で欠損を明示レベル化済みなので対象外＝ラウンド1の「行ごとの欠損カウント」集約とは別物・直交）。クリーンな同条件(workers=2)実測で **16データ全部が 03 とdelta一致**（平均 +0.0143・最悪 train_06 +0.0004、いずれも03と同値）。GBM各族(xgb/lgb/cat/HGB)がNaNを学習的デフォルト方向で既に処理するため、明示フラグは4桁AUC解像度で追加信号ゼロ。**知見: この16データの欠損は概ね非情報的(MCAR/MAR)で、ネイティブNaN処理が既に信号を汲み尽くしている。** キュー不変（03→05→04/freq）。候補=`experiments/bench_03/candidates/missing-flags-per-col/`。
>
> **✅ 保留候補の決着（2026-07-12 ラウンド4）:** ラウンド3で「再検証待ち」だった②depth削減(round2_cat_frac_depth_reduction)・③順序列二重エンコード(04b_ordinal_dualencode)は、クリーンな基準で再検証し**両方とも不採用**に決着した（③=no-op、②=弱点train_06に効くが単純なキュー済み04_missing_countに劣る）。「再検証待ち」の候補はもう無い。次回のオフライン round は新しい角度を1つ選ぶこと。
>
> **🏃 方針の訂正（2026-07-12、ユーザー指示により）:** 「オフライン改善ラウンド2」で書いた『03の実LBスコアが出るまで候補チャーンを止める』は誤った判断だった。Final Submission Deadlineは2026-08-06、まだ約25日ある一方、実提出は1日1件しかできないボトルネックなので、**オフラインでの候補探索・検証は毎回のサイクルで継続すること。「弱点に効く一手が今回は見当たらない」という理由だけで探索自体を止めない。** 明確に効果のある候補が無ければ「今回は不採用」を記録して次のサイクルでも別の角度から探索を続ける。実提出はあくまで1日1件のボトルネックなので、そのボトルネックの手前(オフライン検証)を絞る理由はない。

## スコア推移

| 日付 | submission_id | 変更内容 | 公開スコア (ROC AUC) | 前回比 |
|---|---|---|---|---|
| 2026-07-12 | 54591282 | **01_baseline**: `HistGradientBoostingClassifier` を1回だけ学習・提出する最小構成。列のdtype（object=カテゴリ/順序、数値=そのまま）から自動でカテゴリ列を判定するため、データセットごとに列名や種類が変わっても手直し不要。ツールも write_file / run_command / submit_predictions / get_status の4つだけ。 | 0.787 | (ベースライン) |
| 2026-07-13 予定 | （未提出・保留） | **02_early_stopping**: 分類器の生成だけを `early_stopping=True, max_iter=300` に変更（他は 01 と同一）。作成・検証は完了済みだが、03の方が期待効果が大きいため次回提出は03を優先し、02は保留。 | — | — |
| 2026-07-13 | 54625716 | **03_cv_ensemble**: 交差検証＋XGBoost/LightGBM/CatBoostのアンサンブル＋段階的提出（詳細下記）。オフライン実測で平均AUC+0.0143、MBPローカルLLMでのE2Eリハーサル合格。ユーザー承認済み。**⚠️ 2026-07-13 00:02 UTC 提出したが Kaggle 実グレーダー上で `ERROR`（スコアなし）。** オフライン(go.py直接実行)とMBPローカルLLMリハーサルは通ったが、本番のエージェント実行環境(gemini-2.5-flash駆動の6ステップ手順)で失敗。詳細下記・要ユーザー判断。 | **ERROR** | — |
| 保留（要判断） | （未提出・採用候補・**リスク再評価要**） | **05_te_ngated**: 03 ＋「n<1500 の小規模データセットだけに OOFターゲットエンコーディング列を足す」1点変更。オフラインで03をパレート優越。**ただし 03 と同じ複雑な go.py / 6ステップ・エージェント構造を共有するため、03がERRORした今、05も同様にERRORする可能性が高い。03のERROR原因が判明・解消するまで 05 の提出は保留。** | オフライン+0.0011 | — |
| 2026-07-13 | 54651128 | **02_early_stopping**（実提出・COMPLETE）: 01_baseline に `early_stopping=True, max_iter=300` を足すだけの単発・単一HGB。03のERROR以降途切れた「採点される提出」を grader保証の sklearn-only 最小変更で再確立。**実グレーダーで問題なく採点され、simple路線がgraderで確実に通ることを実証**（06/07/08/cand_C も同一構造なので de-risk 完了）。**ただし公開LBは 01 と同じ 0.787** ＝ 01比オフライン+0.0016 の微改善は公開スコアの表示解像度では動かず。03(ERROR)と同UTC日に受理された＝ERROR提出は日次枠を消費しないことも実測。 | **0.787** | ±0.000（=01） |
| 2026-07-14以降 提出予定（安全キュー・**ラウンド18で検証済**） | （未提出・作成済み・validate合格） | **06_ngated_l2**: 02 に「n_feat/n≥0.010 の過学習しやすいデータだけ `l2_regularization=1.0`」のデータ駆動1行ゲートを足した単発・単一HGB。02をクリーンにパレート改善（mean Public +0.0014・悪化ゼロ）。※公開LBは 02 同様 0.787 のまま動かない見込み。 | オフライン: 02比+0.0014（悪化ゼロ） | — |
| 保留（要ユーザー確認・**ラウンド29で検証済・採用基準クリア**） | （未提出・**submissions/09未作成**・要ユーザーOK） | **cand_C (seedavg_gated)**: 08 の予測ステップだけを「n_feat/n≥0.015 のデータ(train_09/13/15)は random_state 0..4 の predict_proba を平均、他は単一seed」に変更（sklearn-only・K=5・03脆弱性なし）。**単ノブ枯渇(R17-28)を破り08をクリーンにパレート改善する初の候補**（mean Public +0.00169/Private +0.00151・両split回帰ゼロ・train_13を初めて改善）。**seed-avg は構造変更(複数モデル+ブレンド)のため独断シップせず、ユーザー判断待ち。** | オフライン: 08比+0.00169（悪化ゼロ） | — |
| 2026-07-15 提出予定（採用候補・**ラウンド18で検証済**） | （未提出・作成済み・validate合格） | **06_ngated_l2**: 02 に「特徴量数/行数の比が高い(n_feat/n≥0.010)過学習しやすいデータだけ `l2_regularization=1.0`、他は0.0」という**データ駆動の1行ゲート**を足した単発・単一HGB。sklearn-only・新規列なし・保証外パッケージ非依存で03の脆弱性を持たない。**02をクリーンにパレート改善**（4データ改善・悪化ゼロ・12データ02と同一、mean Public +0.0014/Private +0.0010）。02が01比で回帰していた train_15/16 の2件をこのl2で反転。 | オフライン: 02比+0.0014（悪化ゼロ） | — |
| 2026-07-16 提出予定（採用候補・**ラウンド21で検証済**） | （未提出・作成済み・validate合格） | **07_2gate_msl50**: 06 に「特徴量数/行数の比が高い(n_feat/n≥**0.015**)データだけ `min_samples_leaf=50`（既定20）」という**第2の、より厳しいデータ駆動ゲート**を1本足した単発・単一HGB（l2ゲートは06から不変の比≥0.010）。厳ゲートで境界データ train_16 を除外し、train_09/13/15(小n)を Private で大きく底上げ。sklearn-only・新規列なし・保証外パッケージ非依存で03の脆弱性を持たない。**06をクリーンにパレート改善**（発火3データ改善・悪化ゼロ・他13データ06と同一、mean Public +0.00093/Private +0.00254）。 | オフライン: 06比+0.00093（悪化ゼロ） | — |

## 各回の詳細メモ

### 2026-07-13（~16 UTC）: ラウンド30 — seed数Kスイープ（cand_C の K=5 が knee と確定・提出なし＝当日UTC枠は02が消費済み）
- **背景:** round29 で発見した cand_C（ratio=n_feat/n≥0.015 のデータ=train_09/13/15 のみ K=5 seed-avg・他13はseed-0とバイト同一）は 08 をクリーンにパレート改善する唯一の候補だが、K=5 が最適 seed 数かは未検証だった。ship 判断はユーザー確認待ち（構造変更）なので、この offline サイクルは「K を振って cand_C の設計根拠を固める」ことに充てた。
- **手法:** ハーネス `experiments/bench_03/round30_seed_k_sweep/replay.py`（round29 を雛形・sklearn-only `.venv`=grader保証一致）。cand_C ゲート固定、平均 seed 数 K∈{3,5,10} のみ変更。発火3データのみ seed 0..9 を各1回 fit し先頭K個を平均＝計43fit。CLEAN RUN=YES（例外0・invariant違反0・exit0）、`submissions/` 不触を git status で確認。
  - **K=3:** mean ΔPub **+0.00129**/ΔPrv +0.00108、両split W/L/T=3/0/13・回帰ゼロ。だが K=5 の利得の約75%(Pub)/72%(Prv)しか取れない（train_13 ΔPub+0.00344＝K5の約半分）。**「安いのに同等」ではない**＝単純化の無償ランチにはならない。
  - **K=5（=cand_C 完全再現）:** +0.00169/+0.00151、3/0/13。round29 と一致・再現性確認。
  - **K=10:** +0.00187/+0.00183、3/0/13・回帰ゼロ。K=5 を両splitで僅かに上回る唯一の設定だが、モデル数2倍に対し ΔPub+0.00018・ΔPrv+0.00032 のみ＝収穫逓減が明確。
- **結論:** **K=5 が effort/gain の knee。cand_C は K=5 で確定してよい**（K=3 は弱すぎ・K=10 はコスパ薄）。これは cand_C を submissions/09 として提案する際の seed 数根拠になる。ship 判断は依然ユーザー確認待ち。次サイクルの残る offline 角度は (b)発火ゲート閾値の微調整。
- **提出:** なし。本日 2026-07-13 UTC枠は 02(COMPLETE)が消費済みのため新規提出せず（ルール通り）。

### 2026-07-13（~15 UTC）: 02実提出(0.787再確立) ＋ ラウンド29 — seed-averaging（🎉単ノブ枯渇を破る初のクリーン改善候補 cand_C／train_13を初改善／要ユーザー確認で保留）
- **02実提出:** `02_early_stopping` を提出→ `COMPLETE / Public 0.787`（submission_id=54651128）。03のERROR以降の「採点される提出」を再確立し、simple路線がgraderで確実に通ることを実証。**ただし 01(0.787)と同スコア**＝単一HGB微最適化は公開LB表示解像度では動かないと判明。**ERROR提出は日次枠を消費しない**ことも同時に実測（03と同UTC日に02が受理）。
- **ラウンド29(seed-averaging):** R17-28で正則化/不均衡/ランダム化/分割粒度の単ノブが全て08を超えられず、唯一の一貫ブロッカーは train_13(n=500・高比・カウンタームーバー)だった。R29は初めて**ハイパラでなく最終確率の作り方**を変える角度＝「同一08 HGB を random_state 0..4 で K=5 学習し predict_proba を平均（early_stopping内部holdout分割の分散を削減）」を検証。ハーネス=`experiments/bench_03/round29_seed_avg/replay.py`（sklearn-only `.venv`=grader保証一致・80fit全CLEAN RUN・`submissions/`一切不触・git確認済）。
  - **cand_A(全16発火):** mean ΔPub +0.00458/ΔPrv +0.00425、Pub/Prv W/L/T=15/1/0。ほぼ全改善だが **train_16 のみ回帰(−0.00238/−0.00458)** →悪化ゼロ違反で不採用。
  - **cand_B(l2ゲート比≥0.010発火=09/13/15/16):** train_16回帰を内包→不採用。
  - **✅ cand_C(比≥0.015発火=train_09/13/15のみ・境界train_16を除外＝08のmsl厳ゲートと同一0.015閾値を再利用):** mean ΔPub **+0.00169**/ΔPrv **+0.00151**、**両split W/L/T=3/0/13＝回帰ゼロ**。発火3データ全改善（train_15 +0.0102/+0.0127, train_09 +0.0099/+0.0063, **train_13 +0.0070/+0.0051**）。非発火13データ(train_16含む)は08にバイト同一。**R18の08採用以来12ラウンドぶりに08をクリーンにパレート改善する初の候補。**
  - **機構的知見:** 小n高比データの弱点は early_stopping の内部holdout分割の分散。seedを変えて平均すると分散が減り、どの単ノブでも逆方向に動いた train_13 が初めて素直に改善。
  - **⚠️ 保留理由:** seed-avg は「複数モデル学習＋ブレンド」＝従来ユーザー確認必須と切り分けてきた (c)ブレンド多様化カテゴリの構造変更。独断で submissions/09 を作らず、ユーザー判断待ち（PushNotification報告済み）。承認されれば `submissions/09_seedavg_gated/` を coder/architect 委譲で新設し安全キュー先頭へ。生ログ=`experiments/bench_03/round29_seed_avg/{results.csv,summary.txt,run.log}`。

### 2026-07-13: ラウンド27 — max_features 列サブサンプリング（❌不採用・初のランダム化系直交ノブ／列サブサンプリング軸を閉じた）

R17-26 で葉サイズ/重み/学習率/クラス重みの単ノブが全て 08 を超えなかったため、R27 は初めて**per-split の特徴量サブサンプリング** `HistGradientBoostingClassifier(max_features=<mf>)`（sklearn 1.4+、`.venv`=1.9.0で利用可）を検証。葉/重み系(l2/msl)とは別系統のランダム化正則化。08 の l2/msl 2ゲート・early_stopping・特徴量は固定し、`max_features` のみを L2ゲートと同じ比(n_feat/n≥0.010)発火データ(train_09/13/15/16)に適用、mf∈{0.7, 0.5} を 1.0(=08既定)と比較。ハーネス `experiments/bench_03/round27_max_features/replay.py`（round26/replay.py 拡張・sklearn-only `.venv`=grader保証パッケージと一致）で16データPublic/Private別AUC採点。**3設定×16=48 fit 全CLEAN RUN=YES(exceptions=0/skipped=0/invariant_violations=0)、`submissions/`不触（git status で `round27_max_features/` のみ確認）、非発火12データは全設定で08とバイト同一(delta 0)検証済。** base=08 に対し:

- **mf=0.7:** mean Public Δ=−0.00048 / Private Δ=−0.00076、Public W/L/T=2/2/12・Private 2/2/12 → 不採用（両split平均負）。train_09(+0.0078/+0.0017)・train_15(+0.0034/+0.0042)は改善するが train_13(−0.0153/−0.0135)・train_16(−0.0036/−0.0046)が両split明確に回帰し相殺以上に悪化。
- **mf=0.5:** mean Public Δ=−0.00055 / Private Δ=−0.00070、Public 0/4/12・Private 1/3/12 → 不採用（0.7より悪化拡大・発火4データほぼ全滅）。

**知見:** train_13(高比・n=500極小)が R22 msl↑・R25 lr↓に続き再びカウンタームーバー。列を間引くと train_09/15 の利得を train_13/16 の回帰が必ず上回る。**葉/重み(l2/msl)系・不均衡(R26)系に続き、ランダム化(列サブサンプリング)系もこのsuiteでは 08 を超えられないと確定。** 正則化(R17-25)＋不均衡(R26)＋ランダム化(R27)の単ノブ族が全滅＝simple 単ノブ探索は実質完全枯渇。唯一 08 が拾えない一貫パターンは「train_13 を悪化させずに底上げする手が無い」こと。残る simple 直交角度は前処理系ごく僅か（欠損明示補完・単調制約）で期待値ほぼゼロ、実効的伸びしろは (A)go.py 堅牢化 or (c)モデル族多様化＝大設計変更（ユーザー確認必須）に限られる。生ログ=`experiments/bench_03/round27_max_features/{results.csv,summary.txt,run.log}`。

### 2026-07-13: ラウンド26 — class_weight（❌不採用・初の非正則化直交ノブ／「このsuiteは事実上クラス均衡」を確定し不均衡系の軸を閉じた）

R17-25 の正則化/早期停止/学習率族が全て 08 を超えなかったため、初めて族の外の直交ノブ `HistGradientBoostingClassifier(class_weight=...)`（sklearn 1.6+、`.venv`=1.9.0で利用可）を検証。08 の l2/msl 2ゲート・early_stopping・特徴量は固定し class_weight ノブのみ差し替え。ハーネス `experiments/bench_03/round26_class_weight/replay.py`（round25/replay.py 拡張・sklearn-only `.venv`=grader保証パッケージと一致）で16データPublic/Private別AUC採点。**3設定×16=48 fit 全CLEAN RUN=YES(exceptions=0/invariant_violations=0)、`submissions/`不触（git status で `round26_class_weight/` のみ確認）、base再現は round25 baseとバイト一致。** base=08 に対し:

- **cw_gated（`'balanced'` を学習データのマイノリティ比<0.35のときだけ）:** 発火ゼロ。mean Public/Private Δ=+0.00000、W/L/T=0/0/16。**発火ゼロの理由＝新事実: この16データは全て事実上クラス均衡（マイノリティ比 0.489〜0.500）**でゲートを越えない → 全16データで08とバイト同一。不採用（＝08）。
- **cw_global（`'balanced'` を全データ）:** mean Public Δ=−0.00166 / Private Δ=−0.00129、W/L/T=6/8/2（両split）。両split平均負＋両split回帰8件（最悪 train_16 Public −0.00918/Private −0.00770）。均衡データの再重み付けはノイズ注入で左右対称に僅か悪化。不採用。

**知見:** このsuiteは~50/50均衡なので class_weight・少数クラスオーバーサンプリング等の不均衡対策は構造的に無効。**正則化族(R17-25)に続き不均衡族(R26)もこのsuiteでは伸びしろ無しと確定。** 単ノブ探索の実効的な枯渇がさらに濃厚。残る simple 直交角度は前処理系ごく僅か（欠損明示補完・単調制約）で期待値は低く、実効的伸びしろは (A)複雑路線 go.py 堅牢化 or (c)モデル族多様化＝大設計変更（ユーザー確認必須）に限られる。生ログ=`experiments/bench_03/round26_class_weight/{results.csv,summary.txt,run.log}`。

### 2026-07-13: ラウンド25 — learning_rate を高比データで下げる（❌不採用・単ノブ探索の枯渇を確定）

shipped 08 の l2/msl 2ゲートを固定し、HGB の **`learning_rate` のみ**を L2ゲートと同じ比(n_feat/n≥0.010)発火データ(train_09/13/15/16)で {0.07, 0.05} に下げ、既定0.1(=08)と比較。ハーネス `experiments/bench_03/round25_learning_rate/replay.py`（round24/replay.py 拡張・sklearn-only `.venv`）で16データPublic/Private別AUC採点。**3設定×16=48 fit 全CLEAN RUN=YES、`submissions/`不触（git status で `round25_learning_rate/` のみ）。** base=08 に対し lr07（mean Public −0.00035/Private −0.00022）・lr05（mean Public −0.00078/Private −0.00097）とも不採用。**知見: 遅学習に対し train_09(lr↓で改善)と train_13(大きく回帰)がカウンタームーバーだが、救いたい train_09 の比(0.0162)が train_13(0.0180)より低いため ratio-tier で分離不能＝この軸は dead end。** l2/msl/mln/valfrac/lr の正則化系単ノブが完全枯渇。生ログ=`experiments/bench_03/round25_learning_rate/{results.csv,summary.txt,run.log}`。

### 2026-07-13: ラウンド24 — validation_fraction を高比データで広げる（❌不採用・既定0.1が最適／単ノブ探索の枯渇を確認）

ラウンド23末の候補(b)を検証。shipped 08 の l2/msl 2ゲートを固定し、`early_stopping` の **`validation_fraction` のみ**を L2ゲートと同じ比(n_feat/n≥0.010)発火データ(train_09/13/15/16)で {0.15, 0.20} に広げ、0.1(=08既定)と比較。ハーネス `experiments/bench_03/round24_valfrac/replay.py`（round23/replay.py 拡張・sklearn-only `.venv`=grader保証パッケージと一致）で16データPublic/Private別AUC採点。**3設定×16=48 fit 全CLEAN RUN=YES、`submissions/`不触（git status で `round24_valfrac/` のみ確認）。** 非発火12データは全設定バイト同一、差が出るのは発火4データのみ。

- **vf015（0.1→0.15）:** mean Public Δ=−0.00050 / Private Δ=−0.00065、Public W/L/T=1/3/12・Private 0/4/12 → 不採用。
- **vf020（0.1→0.20）:** mean Public Δ=−0.00070 / Private Δ=−0.00107、Public 1/3/12・Private 0/4/12 → 不採用（悪化拡大）。
- **判断: 不採用。** 唯一 train_13 が Public で改善(+0.0055/+0.0065)するが Private では回帰、他の発火データ(train_09/15/16)は両split回帰。仮説「小n高比データは0.1 holdout(~50–180行)が不安定→広げれば安定」は**反証**——widening は極小データの学習行を削る副作用が支配的。**validation_fraction は既定 0.1 がスイートスポット。** 境界データ train_16(比0.0116)が一貫して最大の回帰＝境界は正則化系のどのノブでも脆い(R20/R21と同傾向)。
- **知見/含意:** l2(R17-19)・msl(R20-23)・max_leaf_nodes(R20)・validation_fraction(R24) と **単一HGBの正則化・早期停止系の単ノブはほぼ出尽くし、いずれも 08 を超えるクリーン改善なし**。オフライン単ノブ探索は枯渇に近い。残る未検証の直交単ノブは `learning_rate` くらい。実効的な伸びしろは (A)複雑路線の堅牢化か (c)モデル族多様化＝大設計変更(ユーザー確認要)へ移りつつある。生ログ=`experiments/bench_03/round24_valfrac/{results.csv,summary.txt}`。

### 2026-07-13: ラウンド23 — ratio-tiered msl（✅採用候補・08_ratio_tiered_msl作成／07をクリーンにパレート改善）

ラウンド22末の最優先リードを実装・検証した回。**結論: msl発火データを同じ比(n_feat/n)でさらに階層化し、最高比の train_15 だけ msl=70 に上げる「ratio-tiered msl」は 07 をクリーンにパレート改善し、`submissions/08_ratio_tiered_msl/` を新設（validate合格）。**

- **背景（R22の非対称性）:** msl発火3データは msl↑に対し同方向に動かない — train_15 は単調改善(+0.004)、train_09 はカウンタームーバー(回帰)、train_13 は中間。単一グローバル msl では3データを同時に底上げできず 50 が均衡だった。ratio-tier で train_15 だけを高 msl に隔離すれば train_09 の回帰を避けて train_15 のゲインを取れるという仮説。
- **ハーネス:** `experiments/bench_03/round23_ratio_tiered_msl/replay.py`（round22/replay.py 拡張・sklearn-only `.venv`=grader保証パッケージと一致）で16データPublic/Private別AUC採点。2設定×16=32 fit 全CLEAN RUN=YES、`submissions/` 不触（`git status --porcelain` は `round23_ratio_tiered_msl/` のみ確認）。
- **設定:** base=07（msl発火時=50固定）／候補 tiered=`msl = 70 if ratio>=0.030 else (50 if ratio>=0.015 else 20)`。l2ゲート(1.0 @0.010)は両設定同一。
- **ゲート挙動（設計通り）:** 新規高tier(比≥0.030)は **train_15(0.060)のみ発火**（msl 50→70）。中比 train_09(0.0162)/13(0.0180) は高tier非発火で msl=50(07と同一)、train_16(0.0116)は msl-gate非発火で msl=20(07と同一)、非発火12データもバイト同一。**07と異なるのは train_15 ただ1データ。**
- **結果: mean Public Δ=+0.00024 / Private Δ=+0.00025、W/L/T=1/0/15（両split回帰ゼロ）。** train_15 Public 0.8372→0.8411 (+0.00385) / Private 0.8396→0.8436 (+0.00403)＝R22の msl70 単独スイープで見た train_15 ゲインと一致。他15データは 07 とバイト同一（delta 0）。
- **判断: 採用候補。** 「mean両split正 かつ 両split回帰ゼロ」を満たす CLEAN IMPROVEMENT。`submissions/08_ratio_tiered_msl/` を新設（07コピー＋system.md の msl ゲート行を70/50/20の3tierに拡張・self-diffで変更は当該1行のみ確認・validate合格・既存提出物に差分なし）。構造は07と同じ sklearn-only 単一HGB・単発に比tier1本を足しただけ（新規列・保証外パッケージ非依存＝03の脆弱性なし）。提出キューを 02→06→07→08 と一段延長。
- **知見/リード:** ゲート系(l2/msl)の magnitude・tier・閾値は R17-23 で一通り局所最適を確認（l2=1.0 @0.010、msl=50 @0.015、msl=70 @0.030）。simple単ノブ探索は飽和気味。次の未検証直交角度候補: (a) l2 にも同じ比tier（高比で l2 1.0→2.0、R19の全体2.0が起こした train_15 Private回帰を tier化で回避できるか）、(b) `validation_fraction` を高比データで広げる、(c) 飽和なら モデル族/ブレンド多様化（設計変更＝ユーザー確認要）。生ログ=`experiments/bench_03/round23_ratio_tiered_msl/{results.csv,summary.txt}`。

### 2026-07-13: ラウンド22 — 07の第2mslゲート内での min_samples_leaf magnitude スイープ（❌不採用・msl=50が最適／ratio-tiered mslのリード発見）

ラウンド21末の最優先角度を検証。shipped 07 の2ゲート構造(l2比≥0.010 / msl比≥0.015、msl発火=train_09/13/15)を固定し、**発火時の min_samples_leaf magnitude のみ**を {40, 60, 70} で 50(=07) と比較（ラウンド19のl2 magnitudeスイープと同型）。ハーネス `experiments/bench_03/round22_msl_magnitude/replay.py`（round21/replay.py 拡張・sklearn-only `.venv`=grader保証パッケージと一致）で16データPublic/Private別AUC採点。**4設定×16=64 fit 全CLEAN RUN=YES、`submissions/`不触（git status で `round22_msl_magnitude/` のみ確認）。** train_16 は msl比0.0116<0.015 で全設定非発火（msl=20）、非発火12データも全設定バイト同一＝07と差が出るのは train_09/13/15 のみ。base=07(msl=50) に対する3候補:

- **msl40:** mean Public Δ=−0.00092 / Private Δ=−0.00174、W/L/T=1/2/13。train_09(−0.0089/−0.0116)・train_13(−0.0063/−0.0126) 回帰。葉サイズ縮小は一律に害 → 不採用。
- **msl60:** mean Public Δ=+0.00003 / Private Δ=+0.00000（フラット）。train_09 両split・train_15 Private を僅かに回帰 → 「悪化ゼロ」不成立で不採用。
- **msl70:** mean Public Δ=+0.00023 / Private Δ=−0.00016（Private平均が負）。train_15 を大きく改善(+0.0039/+0.0040) するが train_09 が両split回帰(−0.0022/−0.0041)・train_13 も Private回帰 → 不採用。
- **判断: 不採用。** 「mean両split正 かつ 両split回帰ゼロ」を満たすのは 50(=07) のみ。min_samples_leaf magnitude は 50 がスイートスポットで、07はこのノブで局所最適確定。キュー不変（02→06→07）。
- **知見/リード:** 発火3データは同方向に動かない — train_15 は msl を上げるほど**単調に改善**(msl70で +0.004)、train_09 は逆に msl を上げると**回帰するカウンタームーバー**、train_13 は中間。単一グローバル msl では3データを同時に底上げできず 50 が均衡点。**→ 次角度: msl を比でさらに階層化する「ratio-tiered msl」**（最高比 train_15(0.060) だけ msl=70、中比 train_09/13 は msl=50 のまま＝07と同一を保つ）で train_15 の単調ゲインを train_09 の回帰なしに拾えるか。生ログ=`experiments/bench_03/round22_msl_magnitude/{results.csv,summary.txt,run.log}`。

### 2026-07-13: ラウンド20 — gate内でl2=1.0に第2正則化ノブを重ねる（❌不採用・train_16が必ず回帰／2段ゲートのリード発見）

ラウンド19の次角度(b)を検証。shipped 06 のゲート(`n_feat/n>=0.010`, 発火=train_09/13/15/16)と l2=1.0 を固定し、**発火データにだけ第2の正則化ノブ**を足して 06 をさらに底上げできるか。非発火12データは 02/06 と完全同一を保つ設計。ハーネス `experiments/bench_03/round20_gated_reg2/replay.py`（round19/replay.py 拡張・sklearn-only `.venv`）で16データPublic/Private別AUC採点。**5設定×16=80 fit 全CLEAN RUN=YES、`submissions/`不触（git status で `round20_gated_reg2/` のみ確認）。** 非発火12データは全設定でバイト同一（delta 0）。

- **msl40（min_samples_leaf 20→40）:** mean Public +0.00001 / Private +0.00058。train_09 Public −0.0030・train_16 Private −0.0035 を回帰 → 不採用。
- **msl50（min_samples_leaf=50）:** 4候補中**最大の平均利得**（Public +0.00089 / Private +0.00251）。train_13 Private +0.0196・train_09 Private +0.0118・train_15 Private +0.0092 と小n高比データを大きく底上げ。**唯一の障害が train_16 の極僅かな回帰（Public −0.00069 / Private −0.00048）** → 「悪化ゼロ」基準を満たさず不採用。
- **mln20（max_leaf_nodes 31→20）:** 平均が両split負、train_16 を両split回帰（Public −0.0078） → 不採用。
- **mln15（max_leaf_nodes=15）:** 平均 Public 負、train_15/16 を両split回帰 → 不採用。
- **結論:** 06 の上に第2正則化ノブを重ねても、発火4データのうち **train_16 が必ず回帰**するため誰も 06 をクリーンに上回らない。06(l2=1.0のみ)がこのゲート下の局所最適。
- **知見/リード:** train_16 はゲート比 21/1809=**0.0116** で閾値0.010をギリギリ超える「境界データ」（他3発火データは比0.016〜0.060と明確に高い＝より過学習しやすい）。第2ノブ(min_samples_leaf増)は train_09/13/15 の高比データには明確に効くが、境界の train_16 では逆効果。**→ 次角度: 第2ノブだけ比0.015のより厳しいゲートで発火させ train_16 を除外する「2段ゲート」**（l2は0.010のまま）。これで msl50 の大利得を悪化ゼロで取れる可能性。生ログ=`experiments/bench_03/round20_gated_reg2/{results.csv,summary.txt,run.log}`。

### 2026-07-13: ラウンド19 — gate_ratio内のl2 magnitudeスイープ（❌不採用・L=1.0=06が最適で確定）

ラウンド18の次角度(a)「発火時のl2値そのものの微調整」を検証。06のゲート閾値(n_feat/n≥0.010)は固定したまま、発火時の magnitude L のみ L∈{0.5,1.0,2.0} でスイープ（base=L=0.0=02）。ハーネス `experiments/bench_03/round19_l2_magnitude/replay.py`（round18/replay.py をコピー拡張・sklearn-only `.venv`）で16データPublic/Private別AUC採点。全設定×16=64 fit CLEAN RUN=YES、`submissions/` 不触（git status で確認）。全Lで発火は同一（train_09/13/15/16）。
- **L0.5:** mean Public +0.0009（弱い）＋ train_13 が両split回帰 → 不採用。
- **L1.0(=06):** mean Public +0.0014 / Private +0.0010、両split W/L/T=4/0/12（**回帰ゼロ・唯一のクリーン正**）。
- **L2.0:** mean Public +0.0014（L1.0と同値）だが train_15 Private −0.0020 の新規回帰 → 不採用。
- **結論:** magnitudeノブは L=1.0 がスイートスポットで飽和。06は magnitude で局所最適確定、キュー不変（07-14=02→07-15=06）。次角度は(b)別の正則化ノブ（`max_leaf_nodes`縮小/`min_samples_leaf`増）を同じ gate_ratio で。

### 2026-07-13: ラウンド18 — gated l2_regularization（✅採用候補・06_ngated_l2作成／軸は生nでなく feature/row比だと実証）

ラウンド17末の最優先角度「n-gated l2」を実装・検証した回。**結論: 特徴量数/行数の比でゲートする `l2` は 02 をクリーンにパレート改善し、`submissions/06_ngated_l2/` を新設（validate合格）。ただしラウンド17の「生nゲート」表現は軸を取り違えており、round18で feature/row 比へ訂正した。**

- **ハーネス:** `experiments/bench_03/round18_ngated_l2/replay.py`。ラウンド17の `simple_replay.py` を拡張し、02のレシピ（HGB, categorical_features=dtype-mask, random_state=0, max_iter=300, early_stopping=True）をin-process再現、`.venv`（sklearn 1.9.0, xgb/lgb/cat 無し＝grader保証パッケージと一致）で16データをPublic/Private別AUC採点。`submissions/` は一切不触（`git status --porcelain` は `experiments/bench_03/round18_ngated_l2/` と後述の `submissions/06_ngated_l2/` のみ）。
- **設定:** base=l2 0.0（＝shipped 02）／gate_ratio=`l2=1.0 if n_feat/n>=0.010 else 0.0`（仮説）／gate_nsmall=`l2=1.0 if n<=1200 else 0.0`（対照＝生nゲート）。48fit全てクラッシュ無し（CLEAN RUN=YES）。
- **gate_ratio（採用）:** train_09/13/15/16 で発火＝**4データ全てPublic/Private両面で改善・悪化ゼロ**、残り12データは02とバイト同一。mean Public Δ=+0.0014 / Private Δ=+0.0010、Public W/L/T=4/0/12。発火4データPublic Δ: train_15 +0.0089・train_16 +0.0054・train_09 +0.0044・train_13 +0.0033（Private も全て≥0）。**02のクリーンなパレート改善。**
- **gate_nsmall（棄却）:** train_05/09/13/15 で発火。**train_05 が Public −0.0077 / Private −0.0020 と両面回帰**（Public W/L/T=3/1/12）。→ パレート改善にならない。**これで「効く軸＝生nではなく feature/row 比（過学習しやすさ）」を実証。** 最大の勝ち train_15 は n=500＝最小データなので、生n≥閾値ゲートでは原理的に拾えない＝ラウンド17の「大nで発火」という記述は誤りだった。
- **なぜ比が効くか:** l2 が助けたのは高次元/低サンプル（train_15=30feat/500行, train_09=18/1109, train_13=9/500, train_16=21/1809＝比≥0.011）で、害したのは低比（train_05=9/1060=0.0085, train_03=18/3501=0.0051）。過学習しやすいデータほど l2 が効くという教科書的な挙動で、比ゲートはこれを直接捉える。
- **戦略的含意:** 06 は 02が01(0.787,実グレーダー成功)比で回帰していた3件(train_15 −0.0088 / train_16 −0.0071 / train_04 −0.0047)のうち **train_15・train_16 の2件をこのl2ゲートで反転**する。02は01比 mean+0.0016 だが3回帰を持つ「クリーンでない改善」であり、06はより01にクリーンな改善へ近づく。
- **成果物:** `submissions/06_ngated_l2/`（agent.yaml の name を `ngated_l2_agent` に、system.md の Step1 モデル生成バレットに「n=len(train), n_feat=len(features), l2=1.0 if n_feat/n>=0.010 else 0.0」を足し `l2_regularization=l2` を渡す**1点変更のみ**。02とのdiffはこのバレットだけ＝自己diff確認済）。`validate_submission.py`→`VALIDATION SUCCESSFUL`。実提出は本日枠(03 ERRORが消費)のため未提出、キューは 2026-07-14=02 →2026-07-15=06。生ログ=`experiments/bench_03/round18_ngated_l2/{results.csv,summary.txt}`。

### 2026-07-13: ラウンド17 — simple路線初のsklearn-only単ノブ探索: `l2_regularization`（不採用・ただしn-gated l2の有望リード発見）
- **背景:** 03複雑路線がERRORし、実提出は(B)sklearn-only simple路線(02)に確定。R16までのオフライン探索は全て複雑03 go.py上のFE差分だった。今回はR16提案に従い、**実際に出荷される02の上で simple単ノブ `l2_regularization` を初検証**。
- **手法:** benchmark.py は複雑03専用でsimple路線を検証できないため、**simple路線専用のsklearn-only replayハーネス `experiments/bench_03/simple_replay.py` を新設**。02のtrain.pyロジック（`git show HEAD:submissions/02_early_stopping/agent/prompts/system.md`基準・HGB categorical_features=dtype==object mask, random_state=0, max_iter=300, early_stopping=True）をin-processで16データに再現し、`.venv`(sklearn 1.9.0, xgb/lgb/cat 無し)でPublic/Private別にAUC採点。l2 ∈ {0.0, 0.1, 1.0} を比較。coderサブエージェントに委譲（絶対厳守ルール伝達済・`submissions/`不触をgit statusで確認）。
- **結果:** 48 fit 全クラッシュ無し完走（CLEAN RUN）。l2=0.1: mean Pub Δ=−0.0012(7勝9敗)→即不採用。l2=1.0: mean Pub Δ=+0.0010(ノイズ域,10勝6敗)だが train_05 −0.0077 / train_03 −0.0039 の明確な小n回帰を伴い採用基準未達→不採用。
- **🔑 発見:** l2=1.0が効いたのは**02が01に対し回帰した3件(train_15/16/04)そのもの**（train_15 Pub +0.0089, train_16 +0.0054, train_04 Priv +0.0042）。害は小n(train_05/03/13)に集中。05(n-gated TE)と同型のn依存構造 → **次サイクル最優先角度=「n-gated l2（大nのみl2=1.0を発火）」**。閾値は dataset_stats.csv のnで小n非発火・大n発火に設定。
- **提出:** 本日 2026-07-13 UTC枠は 03(ERROR)が消費済みで無提出。実提出キュー不変（2026-07-14に02_early_stopping）。

### 2026-07-13: ラウンド15 — 03 ERROR原因の再現診断（探索＝診断。新規候補は作らず）
- **背景:** 03(複雑路線)が実グレーダーで ERROR。オフラインベンチは gbm_venv(xgb/lgb/cat 全部入り)で go.py を回すため、パッケージ依存の脆弱性が原理的に不可視。この回は新FE候補を1つ足す代わりに、**より価値の高い「本番ERRORの再現診断」**を探索として実施した。
- **手法:** 03 の go.py を抽出し、**xgb/lgb/cat を持たない `.venv`**（Kaggleサンドボックスに近い最小構成）で train_01 に対し safety→xgb→cat→lgb→blend を段階実行。
- **結果:** safety のみ成功（sklearn HGB フォールバック、`sub_safety.csv` 生成）。xgb/cat/lgb は ImportError で全クラッシュ（`fit_fam` に族フォールバックが無い）。blend は `no_models`。詳細は冒頭「🔬 ERROR原因の再現診断」ブロック参照。
- **結論:** 03の複数族アンサンブルはグレーダー非保証の外部パッケージ(xgboost/catboost)依存で、欠けると3族が落ち6手順が失敗リカバリの茨道に化ける。Step3の安全網提出があるため総ERROR即断はできないが（総ERRORは agent実行層＝オフライン検証不能に起因の公算）、**この脆弱性がオフライン通過↔本番ERRORの乖離を素直に説明**。sklearn-only の 01/02 は構造的に無縁。→ 推奨は (B)シンプル路線寄りにさらに傾く。A/Bの最終判断はユーザー（報告済み）。
- **提出:** 本日 2026-07-13 UTC枠は 03(ERROR)が消費済みのため無提出。次UTC日は 02_early_stopping（sklearn-only・安全策）を最優先候補として据え置き。

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

### 2026-07-12: オフライン改善ラウンド1（不採用・仕組み自体は正常動作）
- **背景:** 実提出は1日1回だが、オフラインの実測ベンチマーク（`experiments/bench_03/benchmark.py`、16データセット全部で正解ラベルと突き合わせ、LLM不使用・無料）は無制限に回せる。この回では現行ベスト(03_cv_ensemble, 平均AUC 0.8033)に対する改善候補を4つ、それぞれ1点だけの変更で並行して試した: ①小規模データセット(n<1500)のシード数を3→5に増やす、②OOF-AUC加重のブレンド候補を追加、③大規模データセット(n≥20000)のシード数を1→2に増やす、④行ごとの欠損値カウントを特徴量に1列追加。
- **結果: 4件とも不採用。** ①③はノイズレベルの差(+0.0001)で実質変化なし。③はむしろ狙った大規模データセット(train_12)で僅かに悪化。②は一度も選ばれず実質no-op。④「欠損値カウント特徴量」が最も有望で、狙い通り弱点だったtrain_06(9列全カテゴリ)のdeltaを0.0004→0.0012に伸ばしたが、平均では+0.0001とまだノイズと区別できる水準ではなく、今回は不採用（次回以降の再検証候補として保留）。
- **副産物（重要）:** ベンチマーク治具を`--workers 6`で回すと、各候補の学習が内部で`n_jobs=4`スレッドを使うため最大24スレッドがCPU10コアを取り合い、train_02のステージ時間が実測68.6秒→約320秒に水増しされることが判明（4候補すべてに同じ倍率で起きたため、コード変更由来ではなく治具の並列度設定が原因と特定）。低並列(`--workers 2`)での再検証を実施中。
- 変更なし: `submissions/03_cv_ensemble` が引き続き次回提出の候補（上の申し送りブロック参照）。
- 構成: 候補一式 [experiments/bench_03/candidates/](experiments/bench_03/candidates/)

### 2026-07-12（13:01 UTC）: オフライン改善ラウンド2（新規採用なし・03を据え置き）
- **状況:** 実提出枠は本日分（2026-07-12 UTC）を 01_baseline（02:07 UTC 提出）で既に使い切っており、この時点では新規提出できない。03_cv_ensemble の提出は次のUTC日（2026-07-13）まで待ち。よってこの回はオフライン評価のみ。
- **弱点の特定:** 直近フル結果（`benchmark_results.csv`）の delta_vs_baseline を見ると、最弱は train_06(+0.0004, 9列すべてカテゴリ, n=10803) と train_12(+0.0031, n=49432)。この2つは `oracle_private ≈ final_private ≈ baseline_auc` で、**アンサンブル自体がベースラインをほぼ上回れていない=データ側の性能天井**であることが数値で確認できる（選択ミスではない）。さらにラウンド1の「大規模nのシード増（n≥20000で1→2）」は狙ったtrain_12でむしろ僅かに悪化しており、計算量を足しても引き出せる信号が無いことを裏づける。
- **判断（新規候補は今回試さず）:** 天井データセットに効く「1点だけ・シンプル・他データを悪化させない」変更は現状見当たらず、無理に候補を足すと(a) まだ実LBスコアで較正していないオフライン代理指標への過剰適合、(b) 厳格な「どのデータセットも悪化させない」ゲートに対する回帰、の二重リスクになる。したがって**03の実LBスコアが出るまで候補チャーンを止める**のが規律的に正しいと判断。03は検証済みで次回提出の唯一の候補として据え置き。
- **低並列(workers=2)再検証について:** ラウンド1で「実施中」とした低並列リランは、スレッド競合が影響するのは AUC ではなく**ステージ時間のみ**であり、`recheck_03_w2.log` で最大ステージ時間 69.6秒（最大級の train_02）＝60分枠に対し余裕、と確認済み。よってタイミング面の懸念は解消、AUC側の結論（ラウンド1の4件不採用）は不変。追加リランは不要と判断し打ち切り。
- 変更なし: `submissions/03_cv_ensemble` が引き続き次回（2026-07-13 UTC）提出の候補。次アクション＝03を提出→実LBスコアで較正してから次の1点改善を選ぶ。

### 2026-07-12（訂正・補足）: 低並列再検証は完了していた（missing-count-featureはクリーンに確認）
- 上のラウンド2記述時点では、`recheck_03_w2.log`（03本体）しか完了しておらず、同じバックグラウンド駆動スクリプトが直後に走らせていた `recheck_missingcount_w2.log`（missing-count-feature候補）はまだ実行中だった。「追加リランは不要と判断し打ち切り」は早計だったので、完了後の実測で補足する。
- **完了後の実測（workers=2、スレッド競合なしのクリーンな条件）:**
  - `03_cv_ensemble`（現行ベスト）: 平均delta +0.0143、最悪delta +0.0004（train_06）、最大ステージ69.6秒 — 全ゲートPASS。ラウンド1の数値と一致（水増しされていたのはタイミングのみだったという診断の裏付け）。
  - `missing-count-feature`: 平均delta +0.0144、**最悪delta +0.0012（train_06、03の3倍）**、最大ステージ38.0秒 — 全ゲートPASS。ラウンド1で見えていた「train_06の弱点改善」がノイズではなく本物だったとクリーンな条件下で確認できた。
- **判断:** 1行だけの単純な変更（行ごとの欠損値カウントを特徴量に1列追加）で、平均は横ばいながら弱点データセットの最悪ケースを明確に底上げする、規律に沿った「シンプルで良い変更」と判断。ただし**03の提出枠を横取りしない** — 03は既に敵対的レビュー・MBP実機リハーサルまで完了した唯一の次回提出候補として維持し、`04_missing_count`（`submissions/04_missing_count/`、`validate_submission.py`合格済み）は**03の実LBスコア確定後に検討する次の1点改善**としてキューに追加する。
- 次アクション（更新）: ① 03を提出（申し送りブロックの通り）→ ② 実LBスコアが出たらオフライン代理指標との整合を確認 → ③ 問題なければ `04_missing_count` を次の実提出候補として同じ手順で提出。

### 2026-07-12: オフライン改善ラウンド3（⚠️ 事故発生・復旧済み／新規採用なし）
- **背景:** ユーザーから「提出まで時間があるのに、なぜ改善をやめるのか」との指摘を受け、方針を訂正（探索を止めない）。新しい切り口4つ（①正則化を軽く追加、②カテゴリ比率が高いデータセットだけ木を浅くする、③順序列を数値＋カテゴリの二重エンコード、④無意味な列を除去）を並行して試した。
- **⚠️ 事故: 次回提出予定の `submissions/03_cv_ensemble/agent/prompts/system.md`（本番用・審査済みファイル）が、候補①(正則化)を検証する過程で誤って直接書き換えられていた。** 原因は、候補実装エージェントへの指示が「候補は `experiments/bench_03/candidates/<name>/` に書く」とは伝えていたが「本番ファイルには絶対に触れるな」と明示していなかったため。**発見後ただちに `git checkout` で本番ファイルを復元し、`validate_submission.py` で正常であることを再確認済み。** 03自体は無傷（実際にKaggleへは何も誤って提出されていない）。この事故により、①の書き換え中に本番ファイルを参照した②・④の一部エージェントが「本番ファイルに既に正則化が入っている」という誤った前提で結果を報告する連鎖的な混乱が発生した。
- **各候補の実際の結果（事故の影響を精査した上で）:**
  - **①正則化(reg_l2_bump)**: 唯一クリーンに検証できた候補。平均delta +0.0145（現行+0.0143とほぼ同値）、train_06 +0.0011（現行+0.0004より良いが、キュー済みの04より劣る）、**train_12 +0.0029（現行+0.0031よりわずかに悪化）**。全体として横ばい〜微妙、不採用。
  - **②カテゴリ比率で木を浅く(round2_cat_frac_depth_reduction)**: 実装エージェントが上記事故で汚染された本番ファイルを基準にしてしまい、意図した「木を浅くする」変更を実質適用し損ねた不良な検証だった。**技法自体は未検証のまま。** 次回、クリーンな基準から作り直して再検証が必要。
  - **③順序列の二重エンコード(04b_ordinal_dualencode)**: 実装は正しく分離されていたが、ベンチマークの完了を待たずにエージェントが結果を返してしまい、**数値が取れていない。** 次回再実行が必要。
  - **④無意味な列を除去(drop-degenerate-columns)**: 実装は正しく分離されており、平均delta +0.0143・train_06 +0.0004・train_12 +0.0031と**現行と完全に同一（no-op）** — 木ベースモデルは元々シグナルの無い列を分割に使わないため、明示的に列を落としても学習結果は変わらないという納得のいく結果。採用の意味なし。ステージ時間104.4秒でタイミングゲート不合格だったが、これは同時間帯に毎時自動タスクも別ディレクトリで同種のベンチマークを走らせていたことによるCPU競合の可能性が高い（要クリーン再検証）。
- **恒久対策（実施済み）:** 今後のオフライン探索エージェントへの指示に「`submissions/` 配下は絶対に書き換えない、比較対象は必ず `git show HEAD:submissions/03_cv_ensemble/agent/prompts/system.md` などコミット済みの内容から取得し、生きているファイルを直接読ませない」を明記するルールを追加した。
- 変更なし: `submissions/03_cv_ensemble` が引き続き次回提出の候補（無傷・検証済み）。`04_missing_count` がその次のキュー。②③は再検証待ちとして次回以降に持ち越し。

### 2026-07-12（14:00頃 UTC）: オフライン改善ラウンド4（ラウンド3で保留した②③をクリーンに再検証・両方とも不採用）
- **状況:** 実提出枠は本日分（2026-07-12 UTC）を 01_baseline（02:07 UTC）で使い切っており新規提出不可。03_cv_ensemble は次のUTC日（2026-07-13）まで提出待ち。よってこの回はオフライン評価のみ。方針どおり、前回「再検証待ち」とマークした2候補を最優先で処理した。作業ディレクトリは競合回避のため `~/kaggle-autonomous-agent-baseline-auto`。ベンチマーク前に `ps aux | grep benchmark.py` で他プロセス無しを確認、`--workers 2`（10コア／各学習n_jobs=4）で実行、いずれも完了を待って結果を取得した。
- **③ 順序列の二重エンコード（04b_ordinal_dualencode）— 不採用（no-op）:** 候補ファイルはコミット済みベースからのクリーンな差分（順序列に `_ordcat` カテゴリ版を1列追加するだけ）であることを `diff <(git show HEAD:...) 候補` で確認済み。実測: 平均delta **+0.0144**（現行03=+0.0143、ノイズ域）、**最悪delta +0.0004（train_06）＝現行03と完全に同一で弱点は一切改善せず**、最大ステージ38.2秒・全ゲートPASS。キュー済みの `04_missing_count`（train_06を+0.0012に底上げ）に明確に劣り、採用の意味なし。
- **② カテゴリ比率で木を浅く（round2_cat_frac_depth_reduction）— 不採用（クリーン再検証完了・弱点は改善するが単純なキュー済み候補に劣る）:** ラウンド3では事故で汚染されたベースを掴み、かつ正則化(REG_L2/reg_lambda/l2_leaf_reg)が混入した「2変更同時」の不良候補になっていた。今回 `coder` サブエージェントに「書き込みは候補ファイル1つのみ・`submissions/`は一切触るな・ベースは必ず `git show HEAD:...`・正則化は入れない」を明示して**depth削減のみのクリーンな1変更**に作り直し、`grep -nE 'REG_L2|reg_lambda|l2_leaf_reg'` で正則化混入ゼロを確認、自己diffで意図した差分（cat_frac算出＋各呼び出し側の受け取り、fit_famの`deep=cat_frac<0.7`条件で lgb num_leaves 31→15 / xgb max_depth 6→4 / cat depth 6→4）だけであることを確認した上でベンチマーク。実測: 平均delta **+0.0144**（現行とノイズ域）、最悪delta **+0.0008（train_06、現行+0.0004より改善）**、最大ステージ41.4秒・全ゲートPASS。**技法自体は弱点train_06に効く方向は正しいと確定**したが、(a)平均は横ばい、(b)3モデル族にまたがる分だけ `04_missing_count`（1行追加）より複雑なのに、その弱点改善幅（+0.0008）は04（+0.0012）より小さい。「最もシンプルで、悪化なく、弱点を最も底上げする1変更」という規律では、より単純で効果も大きい 04_missing_count が支配的。よって②は不採用（技法は記録に残すが採用しない）。
- **判断:** ラウンド3で持ち越した2候補はこれで両方クリーンに決着（ともに不採用）。弱点train_06に対する現時点の最良の1点改善は依然 `04_missing_count`（+0.0012）で不変。無理に新候補を足すと未較正のオフライン代理指標への過剰適合リスクがあるため、この回での新規採用はなし。
- **キュー（不変）:** 次アクション＝①次のUTC日（2026-07-13）に `03_cv_ensemble` を提出（申し送りブロックの手順どおり）→ ②実LBスコアが出たらオフライン代理指標との整合を確認 → ③問題なければ `04_missing_count` を次の実提出候補として提出。②③（depth削減・順序二重エンコード）は「再検証待ち」から外し、決着済み（不採用）とする。
- 構成: 候補一式 [experiments/bench_03/candidates/](experiments/bench_03/candidates/)

### 2026-07-12（15:xx UTC）: オフライン改善ラウンド5（新しい角度＝頻度エンコーディングを検証・不採用／03を据え置き）
- **状況:** 実提出枠は本日分（2026-07-12 UTC）を 01_baseline（02:07 UTC）で使い切っており新規提出不可。03_cv_ensemble は次のUTC日（2026-07-13）まで提出待ち。よってこの回はオフライン評価のみ。前回「再検証待ち」の候補はもう無いので、方針どおり**新しい角度を1つ**選んだ。作業ディレクトリは `~/kaggle-autonomous-agent-baseline-auto`、ベンチマーク前に `ps aux | grep benchmark.py` で他プロセス無しを確認、`--workers 2` で実行し完了を待って結果取得。
- **選んだ角度＝頻度（カウント）エンコーディング（freq-encode-cats）:** これまでの不採用候補（シード増・depth削減・順序二重エンコード・退化列除去）はいずれも「同じ情報の並べ替え」で、木ベースモデルには新情報を与えないため天井（train_06/train_12）を破れなかった。今回は初めて**新しい情報源**を1つ足す: 各カテゴリ列について「その値が train+test 全体で何回出現するか」を数値列 `<列名>_freq` として追加する（カテゴリ列自体はそのまま残す＝1情報源のみ追加のクリーンな1変更）。全カテゴリのtrain_06のような弱点に、値の周辺頻度という新シグナルが効く可能性を狙った。
- **絶対厳守ルールの遵守:** 候補は `coder` サブエージェントに作成させ、書き込みは `experiments/bench_03/candidates/freq-encode-cats/system.md` の1ファイルのみ、ベースは `git show HEAD:submissions/03_cv_ensemble/agent/prompts/system.md`、`submissions/` は一切読まない・触らない、を明示。`diff <(git show HEAD:...) 候補` で意図した4行挿入（`load()` の return 直前に頻度列生成ループ）だけが差分であることを確認済み。
- **実測（workers=2・スレッド競合なし、`experiments/bench_03/logs/freq_encode_w2.log`）:** 平均delta **+0.0144**（現行03=+0.0143、ノイズ域）、**最悪delta +0.0013（train_06）**、最大ステージ36.0秒・全16データセットで悪化なし・全ゲートPASS。弱点別: train_06 +0.0013（03=+0.0004、04_missing_count=+0.0012）、train_12 +0.0032（03=+0.0031）、train_01 +0.0065（03=+0.0055）。
- **判断: 不採用（キュー済みの04と実質同点で、より複雑）。** 03ベースに対しては弱点train_06を+0.0004→+0.0013へ明確に底上げし全データで悪化なしの「クリーンな良い変更」だが、既にキュー済みの `04_missing_count`（train_06 +0.0012）と**train_06で+0.0001差＝ノイズ域の同点**であり、freqは「カテゴリ列1つにつき1数値列追加」で04の「行あたり欠損数の1列追加」より複雑。「最もシンプルで、悪化なく、弱点を最も底上げする1変更」という規律では、効果が同等でより単純な04が支配的。よってfreq-encodeは記録に残すが採用せず、04を次々回提出候補として維持する。
- **補足（次の角度のヒント）:** freq-encode（値の頻度）と missing-count（行の欠損数）は直交する情報源であり、両者の併用は理屈上さらに効く可能性があるが、それは「1サイクル1変更」規律に反する2変更同時なので今回は試さない。03の実LBスコアが出て代理指標が較正できた後、04採用済みの土台に対する次の1変更としてfreqを再検討する余地はある。
- **キュー（不変）:** 次アクション＝①次のUTC日（2026-07-13）に `03_cv_ensemble` を提出 → ②実LBスコアが出たらオフライン代理指標との整合を確認 → ③問題なければ `04_missing_count` を提出。freq-encode は「04採用後の再検討候補」として保留（再検証待ちではなく、04を土台にした将来の1変更案）。
- 構成: 候補 [experiments/bench_03/candidates/freq-encode-cats/](experiments/bench_03/candidates/freq-encode-cats/)、ログ [experiments/bench_03/logs/freq_encode_w2.log](experiments/bench_03/logs/freq_encode_w2.log)

### 2026-07-12（16:xx UTC）: オフライン改善ラウンド6（新しい角度＝OOFターゲットエンコーディングを検証・不採用／03を据え置き）
- **状況:** 実提出枠は本日分（2026-07-12 UTC）を 01_baseline（02:07 UTC）で使い切っており新規提出不可（`kaggle competitions submissions` で確認、当日提出は 01 の1件のみ）。03_cv_ensemble は次のUTC日（2026-07-13、約8時間後）まで提出待ち。よってこの回はオフライン評価のみ。方針どおり**新しい角度を1つ**選んだ。作業ディレクトリ `~/kaggle-autonomous-agent-baseline-auto`、ベンチマーク前に `ps aux | grep benchmark.py` で他プロセス無しを確認、`--venv/bin/python ... --workers 2` で実行し `OVERALL:` 行が出るまで待って結果取得。参照用の `benchmark_results.csv` / `results.json` は実行前に退避し、実行後に復元（候補runがappend/上書きするため汚さない）。
- **選んだ角度＝OOF（リーク安全）平滑化ターゲットエンコーディング（target-encode-oof）:** これまでの「新情報源」候補は freq（値の頻度）・missing-count（行の欠損数）で、いずれも目的変数を使わない教師なし特徴だった。今回は初めて**目的変数の情報を使う**古典的強手法を1つ足す: 各カテゴリ列について、5-fold の out-of-fold で「そのカテゴリ水準の平滑化ターゲット平均」を数値列 `<列名>__te` として追加（学習側はfold外平均でリーク回避、test側は全trainの平滑化平均、smoothing=10・未知水準はglobal_mean）。カテゴリ列自体はそのまま残す＝1情報源のみ追加のクリーンな1変更。狙いは全カテゴリの弱点 train_06 や高カーディナリティ列。**ベンチマークは実際の隠し正解（solution.csv）に対してスコアするため、リークがあれば private が悪化して自動的に露見する＝自己検査になっている**点も選定理由。
- **絶対厳守ルールの遵守:** 候補は `coder` サブエージェントに作成させ、書き込みは `experiments/bench_03/candidates/target-encode-oof/system.md` の1ファイルのみ、ベースは `git show HEAD:submissions/03_cv_ensemble/agent/prompts/system.md`、`submissions/` は一切読まない・触らない、`benchmark.py` も触らない、を明示。`diff <(git show HEAD:...) 候補` で `load()` の return 直前への**純粋な挿入1ハンク（75a76,108）のみ**、他は byte-for-byte 同一、静的ゲート（brace=0・ast.parse OK）維持を確認済み。
- **実測（workers=2・スレッド競合なし、[experiments/bench_03/candidates/target-encode-oof/SUMMARY.txt](experiments/bench_03/candidates/target-encode-oof/SUMMARY.txt)）:** 平均delta **+0.0154**（現行03=+0.0143 を +0.0011 上回る）、最悪delta +0.0006（train_06）、最大ステージ35.6秒・全ゲートPASS。**だが現行03との1データセットごとの比較は明確に混在:**
  - 改善（すべて小n・カテゴリ系）: train_13(n=500) +0.0309→**+0.0404**、train_15(n=500) +0.0190→**+0.0231**、train_05(n=1060) +0.0350→**+0.0390**、train_07 +0.0083→+0.0086、train_14 +0.0077→+0.0080。
  - **悪化（弱点含む）: train_06(n=10803, 全カテゴリの弱点) +0.0013→+0.0006、train_01 +0.0065→+0.0059、train_03 +0.0176→+0.0172、train_08 +0.0135→+0.0132。** train_12(天井) は +0.0032→+0.0031 で横ばい。
- **判断: 不採用。** 平均の +0.0011 増は**すべて小n(500〜1060)のカテゴリデータ3件に集中**しており、これらは private 分散が最も大きくノイズと切り分けにくい。一方で**狙った弱点 train_06 はむしろ悪化**し、train_01/03/08 でも小さいながら回帰した。採用基準「平均を明確に上回る、**または**弱点を悪化なく改善」のいずれもクリーンには満たさない（弱点は改善せず複数データで小回帰）。よって規律に従い今回は採用せず記録に留める。
- **本物の知見（次の角度のヒント）:** OOFターゲットエンコーディングは**小n(≲1000)のカテゴリデータで明確に効き（train_13で+0.0095）、逆に中〜大nの全カテゴリ(train_06)では僅かに害**という、n依存の効き方をする。将来の1変更案として「n が閾値未満のときだけ `__te` 列を足す（n-gated target encoding）」を試せば、小nの利得を取りつつ train_06 の回帰を避けられる可能性がある。ゼロから新角度を探すより、この n-gated 版を優先候補として検討してよい。
- **キュー（不変）:** 次アクション＝①次のUTC日（2026-07-13）に `03_cv_ensemble` を提出（申し送りブロックの手順どおり）→ ②実LBスコアが出たらオフライン代理指標との整合を確認 → ③問題なければ `04_missing_count` を提出。target-encode-oof は「n-gated版で再検討する将来候補」として保留（再検証待ちではない）。
- 構成: 候補 [experiments/bench_03/candidates/target-encode-oof/](experiments/bench_03/candidates/target-encode-oof/)、サマリ [SUMMARY.txt](experiments/bench_03/candidates/target-encode-oof/SUMMARY.txt)・全ログ [bench_run.log](experiments/bench_03/candidates/target-encode-oof/bench_run.log)

### 2026-07-12（~17-18 UTC）: オフライン改善ラウンド7（n-gated target encoding＝✅採用候補・05_te_ngated作成）
- **状況:** 実提出枠は本日分（2026-07-12 UTC）を 01_baseline（02:07 UTC）で使い切っており新規提出不可（`kaggle competitions submissions` で確認、当日UTC提出は 01 の1件のみ・COMPLETE 0.787）。現在 2026-07-12 17時台 UTC＝01提出と同じUTC日なので追加提出不可。03_cv_ensemble は次UTC日（2026-07-13）まで提出待ち。よってこの回はオフライン評価のみ。作業ディレクトリ `~/kaggle-autonomous-agent-baseline-auto`、ベンチマーク前に `ps aux | grep benchmark.py` で他プロセス無しを確認、`.venv/bin/python ... --workers 2` で実行。
- **選んだ角度＝ラウンド6が示した「n-gated target encoding」:** ラウンド6でOOFターゲットエンコーディング（TE）が「小n(≲1000)で明確に効き、中〜大nの全カテゴリ(train_06)では僅かに害」というn依存の効き方をすると判明した。今回はその知見どおり、TEブロック全体を `if len(X) < 1500 and len(cats) > 0:` で囲い、**訓練データが小さい(n<1500)ときだけ `__te` 列を足す**1点変更にした。n≥1500 のデータセットはTEブロックを一切通らず、03 と完全に同じコードパスを走る＝弱点 train_06 への回帰リスクを構造的にゼロにする設計。
- **絶対厳守ルールの遵守:** 候補作成は `coder` サブエージェントに委譲。書き込みは `experiments/bench_03/candidates/target-encode-oof-ngated/system.md` のみ、ベースは `git show HEAD:submissions/03_cv_ensemble/agent/prompts/system.md`、`submissions/`・`benchmark.py` は読まない/触らない、を明示。`diff <(git show HEAD:...) 候補` で**ゲート付きTEブロックの純挿入1ハンク（75a76,108、`if len(X) < 1500 and len(cats) > 0:` 付き）のみ**を確認（自己diffクリーン）。サブエージェントがベンチマーク完了前に「実行中」で返したため、オーケストレータ側で他プロセス無しを再確認のうえ再実行し `OVERALL:` 行まで待って結果取得（"実行中"を最終結果にしない教訓の遵守）。
- **クリーンな同条件比較（03ベースラインと候補を両方 workers=2 で実測、CPU競合なし）:**
  - **改善（小n・ゲート対象のみ）:** train_13(n=500) +0.0309→**+0.0404**（+0.0095）、train_05(n=1060) +0.0337→**+0.0390**（+0.0053）、train_15(n=500) +0.0213→**+0.0231**（+0.0018）。train_09(n=1109) は +0.0289 で不変。
  - **n≥1500 の12データセットは 03 とバイト同一（delta完全一致）:** train_01/02/03/04/06/07/08/10/11/12/14/16 すべて 03 と同じ数値。**弱点 train_06 は +0.0004 のまま**＝ラウンド6の非gated版が起こした train_06 回帰（+0.0004→非gatedで悪化）を完全に回避。
  - 平均delta: 03 **+0.0143** → n-gated **+0.0154**（+0.0011）、最悪delta +0.0004（train_06、03と同値）、最大ステージ43.1秒・全ゲートPASS・`validate_submission` PASS。
- **判断: ✅採用候補。** n-gated TE は 03 を**パレート優越**する（全16データで 03 以上、うち3データで厳密に上、悪化ゼロ）。平均の +0.0011 は**大きく再現性のある小nゲイン由来**（ラウンド6でTEが小nで効くことは確認済み）で、下振れは数学的にゼロ（n≥1500はコードパス不変）。非gated TE（train_06に触れる）や 04_missing_count（+0.0144・拡散的）より明確にクリーン。**シンプルさも維持**（03の構造をそのまま・小nだけに列追加の1変更）。
- **成果物:** 提出可能な新ディレクトリ `submissions/05_te_ngated/agent/` を作成（既存の提出物は一切変更せず）。`agent.yaml` は 03 のコピーで `name: cv_ensemble_te_ngated_agent` のみ変更、`system.md` は候補と同一。`validate_submission.py` 合格（ADKコンパイルOK・tools=5）。
- **キュー更新:** 次UTC日（2026-07-13）はまず **03_cv_ensemble を提出（実LB較正）** → その後 **05_te_ngated を最優先で提出**（従来の 04_missing_count・freq-encode-cats はその次に後退）。ただしオフライン代理指標は未較正なので、03のLBスコアが出てから最終判断する。
- 構成: 候補 [experiments/bench_03/candidates/target-encode-oof-ngated/](experiments/bench_03/candidates/target-encode-oof-ngated/)、サマリ [SUMMARY.txt](experiments/bench_03/candidates/target-encode-oof-ngated/SUMMARY.txt)、提出物 [submissions/05_te_ngated/](submissions/05_te_ngated/)

### 2026-07-12（~18 UTC）: オフライン改善ラウンド8（列ごとの欠損フラグ＝不採用・no-op）
- **状況:** 実提出枠は本日分（2026-07-12 UTC）を 01_baseline（02:07 UTC・COMPLETE 0.787）で使い切っており新規提出不可（`kaggle competitions submissions` で確認、当日UTC提出は 01 の1件のみ）。現在 18時台 UTC＝同一UTC日なので追加提出できない。よってこの回はオフライン評価のみ。作業ディレクトリ `~/kaggle-autonomous-agent-baseline-auto`、ベンチマーク前に `ps aux | grep benchmark.py` で他プロセス無しを確認、`.venv/bin/python experiments/bench_03/benchmark.py --system-md experiments/bench_03/candidates/missing-flags-per-col/system.md --workers 2` で実行し `OVERALL:` 行まで待って結果取得。
- **選んだ角度＝「列ごとの欠損インジケータフラグ」:** これまで欠損関連はラウンド1で「行ごとの欠損カウント」1列（missing-count、集約）を試した。今回はそれと直交する「各列の欠損有無を個別の0/1列で表す」を検証。ただしカテゴリ列は `load()` 内で既に `.fillna("nan")` により欠損が明示的な"nan"レベルとしてモデル化されているため、フラグを足す価値があるのは**数値列**のみと判断し、`is_numeric_dtype` の分岐内で「train か test に欠損がある数値列だけ」`<col>__isna`(int 0/1) を X/Xt に追加（`cats` には入れず数値特徴のまま）。train_07(欠損26.5%)・train_04(12.5%・全数値)・train_02(11%・数値26列)など欠損が多く数値主体のデータに効くという仮説。
- **絶対厳守ルールの遵守:** 候補作成は `coder` サブエージェントに委譲。書き込みは `experiments/bench_03/candidates/missing-flags-per-col/system.md` のみ、ベースは `git show HEAD:submissions/03_cv_ensemble/agent/prompts/system.md`、`submissions/`・`benchmark.py` は読まない/触らない、を明示。`diff <(git show HEAD:...) 候補` で**数値分岐への純挿入1ハンク（65a66,70、5行）のみ**を確認（自己diffクリーン、`cats`未変更・4桁prefix coercionや他stageはバイト同一）。サブエージェントがベンチマーク完了前に返したため、オーケストレータ側で PID 完了を待ってから結果取得（"実行中"を最終結果にしない教訓の遵守）。
- **クリーンな実測結果（fresh run のコンソールサマリ、固定シード）:** 平均delta **+0.0143**（03と同値）、最悪delta **+0.0004（train_06）**、最大ステージ68.6秒・全ゲートPASS・`validate_submission` PASS。**16データセット全部が 03 base とdelta一致**（train_01 +0.0055 / train_04 +0.0057 / train_07 +0.0077 / train_02 +0.0152 … いずれも03と同じ、狙った欠損多めのデータでも変化なし）。固定シードのため、真に効く変更なら数値が動くはずだが動かなかった＝実質 no-op。
- **判断: 不採用（no-op）。** GBM各族（XGBoost/LightGBM/CatBoost/HGB）はNaNを「学習的にデフォルト分岐方向を選ぶ」形でネイティブ処理するため、明示的な数値欠損フラグは4桁AUC解像度で追加の分割可能信号を与えなかった。**知見: この16データセットの欠損は概ね非情報的(MCAR/MAR)であり、ネイティブNaN処理が既に取り出せる信号を汲み尽くしている。** よってカテゴリ既定の"nan"レベル同様、数値側も明示フラグは冗長。
- **キュー不変:** 次UTC日（2026-07-13）はまず `03_cv_ensemble` を提出（実LB較正）→ その後 `05_te_ngated` を最優先で提出（申し送りブロック参照）。missing-flags-per-col は将来の再検討リストにも積極的には残さない（no-op のため）。
- 構成: 候補 [experiments/bench_03/candidates/missing-flags-per-col/](experiments/bench_03/candidates/missing-flags-per-col/)

### 2026-07-12（~19 UTC）: オフライン改善ラウンド9 — レアカテゴリ集約(rare-cat-collapse)（不採用・no-op）
- **状況:** 実提出枠は本日分(2026-07-12 UTC)を 01_baseline(02:07 UTC)で使い切り済みのため、この回はオフライン評価のみ（現UTC時刻 19:01）。次UTC日(2026-07-13)まで新規提出不可。
- **選んだ角度＝「レアカテゴリ集約」:** これまでカテゴリ列のデノイズ角度として freq-encode(値の頻度・R5)・missing-count(行の欠損数・R1)・missing-flags-per-col(列の欠損有無・R8) を試し、いずれも no-op〜微差だった。今回はそれらと直交する「train内出現回数が5未満のレア・レベルを単一トークン`__rare__`に畳み込む」を検証。ねらい: 小n・高カーディナリティのカテゴリ列で、ほぼ一意なレア・レベルへの過学習(spurious split)を抑えて汎化を改善。列は増やさない純粋なデノイズで、大nでは閾値が発火せず実質no-op＝下振れ安全。target encoding(05で採用)とも直交(あちらは新情報付与、こちらはレベル削減)。
- **変更点:** `load()` のカテゴリ`else`分岐のみ。`xs=train文字列, ts=test文字列` を作り、`vc=xs.value_counts()` で `rare=set(vc.index[vc<5])` を求め、rareがあれば train/test 双方の該当値を`__rare__`に置換してから `pd.Categorical` を構築。`cats.append(c)` と他stageはバイト同一。ブレース0個(ADK注入ゲート遵守)。
- **絶対厳守ルールの遵守:** 候補作成は `coder` サブエージェントに委譲。書き込みは `experiments/bench_03/candidates/rare-cat-collapse/system.md` のみ、ベースは `git show HEAD:submissions/03_cv_ensemble/agent/prompts/system.md`、`submissions/`は触らない、を明示。自己diffは `else`分岐置換の1ハンクのみ。1個の```pythonフェンス・`braces_in_script:0`・静的ゲート/ `validate_submission` 全PASS。サブエージェントにベンチ完了(~266s, "OVERALL: ALL GATES PASS")を待ってから報告させた。
- **クリーンな実測結果(fresh run・固定シード・workers=2):** 平均delta **+0.0143**（03と同値）、最悪 **+0.0004(train_06)**、最大ステージ 37.8秒。**16データセット全部が 03 base と delta バイト一致**（train_05 +0.0337 / train_13 +0.0309 / train_09 +0.0289 … 小nカテゴリ含め全て03と同一、悪化ゼロ）。固定シードなので真に効けば数値が動くはずだが不動＝実質 no-op。
- **判断: 不採用(no-op)。** GBMのネイティブ・カテゴリ処理（pandas Categorical/CatBoost cat_features/XGB enable_categorical）が、レア・レベルを既に統計的に安定な形で扱っており、閾値5未満のレベルもこの16データには実質存在しないか、明示畳み込みが4桁AUC解像度で分割を変えなかった。**知見: この16データのカテゴリ列にロングテール(train count<5)は実質無く、"頻度/欠損/レア"系のカテゴリ・デノイズ角度は一通り no-op を確認できた。** 今後の探索は(a)新情報を足す系（target encoding=05採用済／カテゴリ交互作用など）か(b)モデル族の多様化系に絞るのが効率的。
- **キュー不変:** 次UTC日(2026-07-13)はまず `03_cv_ensemble` を提出(実LB較正)→ その後 `05_te_ngated` を最優先で提出。rare-cat-collapse は将来の再検討リストに積極的には残さない(no-op)。
- 構成: 候補 [experiments/bench_03/candidates/rare-cat-collapse/](experiments/bench_03/candidates/rare-cat-collapse/)

### 2026-07-12（~20 UTC）: オフライン改善ラウンド10 — カテゴリ交互作用(cat-interactions)（不採用）
- **状況:** 実提出枠は本日分(2026-07-12 UTC)を 01_baseline(02:07 UTC)で使い切り済みのため、この回はオフライン評価のみ（現UTC時刻 ~20:00）。次UTC日(2026-07-13)まで新規提出不可。提出履歴に新規採点結果なし（01_baseline=0.787 のまま・追記不要）。
- **選んだ角度＝「カテゴリ交互作用」:** ラウンド9で「頻度/欠損/レア系デノイズは一通りno-op、今後は(a)新情報付与 か (b)モデル族多様化」と結論づけた。その(a)の未検証角度として、TE(05で採用・小n専用)とは直交する「大nの全カテゴリ弱点 train_06 を狙う交互作用」を検証。ねらい: 全列カテゴリ・n=10803でアンサンブルがベースラインをほぼ超えられない train_06(+0.0004)に、木が直接分割できる2列ジョイントを与えて信号を足す。
- **ベース＝現行ベスト05:** 「05の上でさらに改善するか」を測るため、ベースを `git show HEAD:submissions/05_te_ngated/agent/prompts/system.md` とした（03ではなく05の上に積む差分検証）。
- **変更点:** `load()` のカテゴリ検出ループ直後・既存TE `try:` の直前に独立`try/except`を挿入。`base_cats`のカーディナリティを取り昇順ソート、`card_a*card_b<=100`のペアだけ文字列連結して native `pd.Categorical` 列(`a__x__b`)を**最大3本**追加し`cats`に登録。低カーディナリティ制約でジョイント密度を担保し blow-up をゼロに抑える設計。辞書内包表記はADK注入ゲート(braces_in_script==0)に抵触するため `dict()`+ループで実装。
- **絶対厳守ルールの遵守:** 候補作成は `coder` サブエージェントに委譲。書き込みは `experiments/bench_03/candidates/cat-interactions/system.md` のみ、ベースは `git show HEAD:submissions/05_te_ngated/...`、`submissions/`は不変、を明示。自己diffは挿入1ハンクのみ・```pythonフェンス1個・静的ゲート/`validate_submission` 全PASS(braces=0/ast OK)。ベンチ完走(最大ステージ39.5s)を待ってから報告させた。
- **クリーンな実測結果(fresh run・workers=2):** 平均delta **+0.0152**（05=+0.0154、−0.0002＝ノイズ域）、最大ステージ 39.5秒（3600s枠に余裕）。全16データで delta>0（ベースライン比では回帰なし）だが**対05では train_05 が −0.0047 明確に回帰**（+0.0390→+0.0343）。狙いの train_06 は +0.0004→+0.0008 と微増、train_13 +0.0037改善、train_15/06 は誤差域。
- **判断: 不採用。** 採用基準「05のmeanをクリーンに上回る」も「ゼロ回帰でtrain_06改善」も満たさない（meanは下回り、train_06改善と引き換えにtrain_05が回帰）。**知見: この16データのカテゴリ列は概ね高カーディナリティで `card*card<=100` にほぼ掛からず、交互作用列がほとんど生成されない安全no-opに近い。効果はノイズ域で局所回帰だけが残った。低カーディナリティ制約を緩めれば発火は増えるがジョイントが疎になりブレンド前提が崩れて回帰が増える方向。** 「新情報を足す」系の残り有望角度は target encoding のバリアント(05採用済)に絞られてきたので、次サイクルは **(b)モデル族多様化系**（現行 safety(HGB)/xgb/cat/lgb のブレンド重みの原理的改善など）へ軸足を移す。ただしモデル族の追加は「シンプル1変更」を超える設計変更になりうるため、大きくするならユーザー確認を挟む。
- **キュー不変:** 次UTC日(2026-07-13)はまず `03_cv_ensemble` を提出(実LB較正)→ その後 `05_te_ngated` を最優先。cat-interactions は将来リストに積極的には残さない。
- 構成: 候補 [experiments/bench_03/candidates/cat-interactions/](experiments/bench_03/candidates/cat-interactions/)

### 2026-07-12（~21 UTC）: オフライン改善ラウンド11 — 数値行集約(numeric-row-aggregates)（不採用）
- **状況:** 実提出枠は本日分(2026-07-12 UTC)を 01_baseline(02:07 UTC)で使い切り済みのため、この回はオフライン評価のみ（現UTC時刻 ~21:01、ローカル時計はJST=UTC+9で7/13を表示するが**UTCではまだ2026-07-12**）。次UTC日(2026-07-13)まで新規提出不可。提出履歴に新規採点結果なし（01_baseline=0.787 のまま・追記不要）。
- **選んだ角度＝「数値行集約」:** ラウンド9-10で「カテゴリ列の頻度/欠損/レア/交互作用系は一通りno-op〜微差」と結論づけ、残る有望系は(a)新情報付与のうち**数値側**は未検証だった。今回は初めて数値列に着目し、TE(小nカテゴリ)/cat系(カテゴリ)いずれとも直交する「**行方向に数値特徴量を集約した2列**」を検証。ねらい: 木は1列ずつしか分割できず行方向の線形統計(平均・分散)を近似しにくいので、数値列が多いデータ（特に大nの弱点 train_12）に新しい分割可能信号を与える。全カテゴリの train_06 は数値列<2で発火せず安全no-op(下振れ数学的にゼロ)を設計で担保。
- **ベース＝現行ベスト05:** 「05の上でさらに改善するか」を測るため、ベースを `git show HEAD:submissions/05_te_ngated/agent/prompts/system.md` とした。
- **変更点:** `load()` の `return` 直前に6行を挿入。`num_feats=[c for c in feats if c not in cats]`（＝カテゴリマスク外＝id/target除外済みの数値特徴量）が2列以上のときだけ、train/test 双方に `__num_row_mean`(行平均) と `__num_row_std`(行標準偏差) を追加、NaNは0.0埋め。ラベル不参照＝行内集約でリークなし。TE派生列(`__te`)は集約に含めない(元の`feats`基準)。
- **絶対厳守ルールの遵守:** 候補作成は `coder` サブエージェントに委譲。書き込みは `experiments/bench_03/candidates/numeric-row-aggregates/system.md` のみ、ベースは `git show HEAD:submissions/05_te_ngated/...`、`submissions/`は不変、を明示。自己diffは挿入1ハンク(6行)のみ・```pythonフェンス1個・静的ゲート全PASS(braces=0/ast OK)。**同条件クリーン比較のため、候補と05ベースの両方を同セッション内で `--workers 2 --skip-validate` で走らせ、両ログの完了(mean delta 行)を確認してから判定した**（stale reference 回避）。
- **クリーンな同条件実測（workers=2、候補=`logs/numeric_row_aggregates_w2.log` / 05再計測=`logs/base05_recheck_w2.log`）:** 平均delta **候補+0.0147 vs 05+0.0154（−0.0007）**、最大ステージ38.2秒。**候補は16データ中11データで05より悪化**（train_15 +0.0019・train_16 +0.0016 のみ改善、train_01は+0.0001で誤差）。特に**狙った大nの弱点 train_12 が +0.0031→+0.0019 と悪化**、小nのTEデータも train_13 +0.0404→+0.0378 / train_05 +0.0390→+0.0365 / train_09 +0.0289→+0.0262 と揃って回帰、train_11 +0.0122→+0.0106。設計どおり全カテゴリの train_06 は +0.0004 で05とバイト一致（no-opガード正常動作）。
- **判断: 不採用。** 平均を−0.0007下げ、狙いの弱点を含め過半のデータで回帰。採用基準「05のmeanをクリーンに上回る／弱点を悪化なく改善」をどちらも満たさない。**知見: スケールの異なる生の数値列を行方向に素の平均/分散で集約すると、大きいスケールの列に支配されたノイズ列が2本増えるだけで判別信号にならず、木は低信号の分割候補が増えた分だけフィットが希釈され過半のデータで小幅悪化する。行内集約は列ごとに既にアクセスできる情報の劣化した再表現に留まり、新しい判別情報を足さない。標準化してから集約すれば別だが、それは「1変更」を超える。** これで数値側の素朴なFEも no-op〜微悪と確認でき、**この16データセット suite は素朴な特徴量エンジニアリングに対して概ね飽和**という累積結論がさらに補強された。今後の実効的な伸びしろは (b)モデル族/ブレンドの多様化（設計変更のためユーザー確認要）に絞られる。
- **キュー不変:** 次UTC日(2026-07-13)はまず `03_cv_ensemble` を提出(実LB較正)→ その後 `05_te_ngated` を最優先で提出。numeric-row-aggregates は将来リストに積極的には残さない（素の行集約は劣化再表現のため）。標準化つき行集約は「05のLB較正後・別サイクルの1変更案」として弱い候補に留める。
- 構成: 候補 [experiments/bench_03/candidates/numeric-row-aggregates/](experiments/bench_03/candidates/numeric-row-aggregates/)、ログ [experiments/bench_03/logs/numeric_row_aggregates_w2.log](experiments/bench_03/logs/numeric_row_aggregates_w2.log) / [experiments/bench_03/logs/base05_recheck_w2.log](experiments/bench_03/logs/base05_recheck_w2.log)

### 2026-07-12（~22 UTC）: オフライン改善ラウンド12 — TE平滑化強化(te-smoothing-20)（不採用）
- **状況:** 実提出枠は本日分(2026-07-12 UTC)を 01_baseline(02:07 UTC)で使い切り済みのため、この回もオフライン評価のみ（現UTC時刻 ~22:01、ローカル時計はJST=UTC+9で7/13を表示するが**UTCではまだ2026-07-12**）。次UTC日(2026-07-13)まで新規提出不可。提出履歴に新規採点結果なし（01_baseline=0.787 のまま・追記不要）。
- **選んだ角度＝「TE平滑化強化」:** ラウンド1-11で、カテゴリ系デノイズ(freq/missing/rare)＝no-op、cat交互作用＝ノイズ域、数値行集約＝劣化再表現、と「新情報を足す」系の素朴FEが概ね飽和と確認済み。残るは(b)モデル族多様化(設計変更・要ユーザー確認)だが、その前に**まだ突いていない「採用済みの勝ち技(05のn-gated TE)自体のハイパラ」**を1つ検証。ねらい: 05のBayes平滑化 `smoothing=10.0` は任意に選んだ値であり、小n(≲1000)でレア水準の符号化が過学習気味なら、平滑化を強めれば小nのTEデータをさらに底上げできる可能性。
- **ベース＝現行ベスト05:** ベースを `git show HEAD:submissions/05_te_ngated/agent/prompts/system.md` とし、その上での改善を測定。
- **変更点:** OOFターゲットエンコーディングブロック(`if len(X) < 1500 and len(cats) > 0:` ゲート内)の `smoothing = 10.0` を `20.0` に変える**1数値のみ**。ゲート閾値1500・列追加・モデルコード・ツールはバイト同一。n≥1500の12データはゲート非発火で完全に不変。
- **絶対厳守ルールの遵守:** 候補作成は `coder` サブエージェントに委譲。書き込みは `experiments/bench_03/candidates/te-smoothing-20/system.md` のみ、ベースは `git show HEAD:submissions/05_te_ngated/...`、`submissions/`は不変、を明示。**自己diffは1行(`smoothing 10.0→20.0`)のみのクリーンな差分**を確認。同条件クリーン比較のため候補と05ベースの両方を同セッション内で `--workers 2` で走らせ、両ログの完了を確認してから判定（stale reference 回避）。
- **クリーンな同条件実測（workers=2、ログは候補dir配下 `cand_run.log` / `base05_run.log`、JSONも保存）:** 変化したのはゲート発火する小n3データのみ、他13データはバイト同一。**平均delta 05=+0.01538 vs 候補=+0.01517（−0.00021）で05を上回らず。** 内訳: **train_05(n=1060) が +0.0390→+0.0337 と明確に回帰（−0.0053、oracleも 0.6851→0.6833 と低下＝選択のばらつきでなく真の特徴量劣化）**、train_13(n=500) +0.0404→+0.0421・train_15(n=500) +0.0231→+0.0234 は小幅改善、train_09(n=1109) は不変。大nの全カテゴリ train_06 はゲート非発火で +0.0004 のまま。
- **判断: 不採用。** 平均を−0.00021下げ、しかも最小の悪化(train_05 −0.0053)を出す。採用基準「05のmeanをクリーンに上回る／悪化なし」を両方満たさない。**知見: 平滑化を10→20に強めると、最小n(≲1000)ではレア水準の弱い信号ごと符号化が全体平均へ潰れ、小nでの効き方が非一貫(train_05悪化/train_13,15改善/train_09不変)になる＝過正則化の兆候。05の smoothing=10 は既にこのsuiteのスイートスポット近傍で、単純な定数増は局所最適から外れる。** これで「素朴FE飽和」に加え「勝ち技のハイパラ微調整も伸びしろ無し(局所最適)」が確認でき、実効的な伸びしろは (b)モデル族/ブレンドの多様化に事実上絞られる（設計変更のためユーザー確認要）。
- **キュー不変:** 次UTC日(2026-07-13)はまず `03_cv_ensemble` を提出(実LB較正)→ その後 `05_te_ngated` を最優先で提出。te-smoothing-20 は不採用で将来リストにも残さない。**次サイクル以降の方針メモ:** 素朴FE・勝ち技ハイパラともに飽和が確認できたので、残る大きな伸びしろ（モデル族の多様化・ブレンド重みの原理的改善）は「シンプル1変更」を超える設計変更になりうる。実行前にユーザーへ確認を挟むこと。それまでは低リスクの直交角度（例: TEのゲート閾値の微調整、TEを二値カテゴリ交互作用へ拡張、など未検証の小変更）を1サイクル1件で継続探索する。
- 構成: 候補 [experiments/bench_03/candidates/te-smoothing-20/](experiments/bench_03/candidates/te-smoothing-20/)（`cand_run.log` / `base05_run.log` / `cand_results.json` / `base05_results.json` 同梱）

### 2026-07-12（~23 UTC）: オフライン改善ラウンド13 — n-gated 頻度エンコーディング(ngated-freq-encode)（不採用）
- **状況:** 実提出枠は本日分(2026-07-12 UTC)を 01_baseline(02:07 UTC)で使い切り済み（現UTC時刻 ~23:01、ローカル時計はJST=UTC+9で7/13を表示するが**UTCではまだ2026-07-12**）。次UTC日(2026-07-13)まで新規提出不可。提出履歴に新規採点結果なし（01_baseline=0.787 のまま・追記不要）。よってこの回もオフライン評価のみ。
- **選んだ角度＝「n-gated 頻度(count)エンコーディング」:** ラウンド5でメモした「頻度(値の出現回数)とTE(カテゴリ別の目的平均)は直交する情報源」という着眼を、**05のTEと同じ小nゲート(`len(X)<1500 and cats`)の中で、TE列(`__te`)に加えてcount列(`__cnt`)も足す**形で初めて検証。ねらい: 小n(TEが効くデータ)にだけ、目的とは独立な「カテゴリの珍しさ」信号を1本足して底上げできるか。大nの弱点 train_06 はゲート非発火で完全不変＝下振れゼロという05の良い性質をそのまま継承する低リスク設計。
- **ベース＝現行ベスト05:** `git show HEAD:submissions/05_te_ngated/agent/prompts/system.md` を基準に測定。
- **変更点:** `load()` のTEブロック内(小nゲート内)で、各カテゴリ列 `c` について `vc = X[c].astype(str).value_counts()` を train から計算し、`X[c+"__cnt"]`/`Xt[c+"__cnt"]` に写像(`fillna(0)`)する**3行の追加のみ**。count は目的を使わないためリークなし・OOF不要。ゲート閾値・モデル・ブレンド・ツールはバイト同一。n≥1500の12データはゲート非発火で完全に不変。
- **絶対厳守ルールの遵守:** 候補作成は `coder` サブエージェントに委譲(書き込みは `experiments/bench_03/candidates/ngated-freq-encode/system.md` のみ・ベースは `git show HEAD:...`・`submissions/`不変)。ただし**サブエージェントがベンチマーク完走前に早期リターン(「1/16完了」で停止)した**ため、オーケストレータ側で自己diffがクリーン(意図した3行のみ・`103a104,106`)であることを確認したうえで、05ベースと候補の両方を同セッション・同条件(`--workers 2`)で改めて完走させ、完了マーカー(`NGATEDFREQ_DONE`)確認後に判定した（stale reference / 早期報告の両方を回避）。
- **クリーンな同条件実測（workers=2、両ログ完走確認済み）:** 変化したのはゲート発火する小n3データのみ、他13データはバイト同一。**平均delta 05=+0.01538 vs 候補=+0.01483（−0.00055）で05を下回る。** 内訳（全て悪化・改善ゼロ）: **train_05(n=1060) +0.0390→+0.0365（−0.0025）**、**train_13(n=500) +0.0404→+0.0354（−0.0050）**、**train_15(n=500) +0.0231→+0.0218（−0.0013）**。train_09(n=1109) は不変、大nの全カテゴリ train_06 はゲート非発火で +0.0004 のまま(ガード正常動作)。max_stage 38.7s(train_02)＝3600s枠に余裕。
- **判断: 不採用。** 小n3データが揃って悪化し改善ゼロ、平均も−0.00055下げる。採用基準「05のmeanをクリーンに上回る／悪化なし」を両方満たさない。**知見: この suite では count/頻度エンコーディングはTEに対して直交な追加信号にならず、むしろ最も脆い最小nデータでTEの目的平均信号を低信号列で希釈して一様に(小幅)悪化させる。ラウンド12(TE平滑化強化)と同じ「小nは追加操作に弱い」パターンの再現であり、小nの利得は専ら目的平均(TE)そのものに由来し、上へ別種のカテゴリ符号化を重ねるのは害という累積結論をさらに補強。** 特徴量側(素朴FE・勝ち技への追加符号化・勝ち技ハイパラ)はいずれも伸びしろ無しで飽和と確定的。
- **キュー不変:** 次UTC日(2026-07-13)はまず `03_cv_ensemble` を提出(実LB較正)→ その後 `05_te_ngated` を最優先で提出。ngated-freq-encode は不採用で将来リストにも残さない。**次サイクル以降の方針メモ:** 特徴量側の低リスク直交角度はラウンド8-13でほぼ出尽くした。残る実効的な伸びしろは (b)モデル族/ブレンド重みの多様化に事実上限定されるが、これは「シンプル1変更」を超える設計変更になりうるため**実行前に必ずユーザーへ確認を挟むこと**。それまでの1サイクル1件のオフライン探索は、未検証で残る小変更（例: TEゲート閾値1500の微調整、TEをブレンド前の別モデルへ与える等）に絞って継続する。
- 構成: 候補 [experiments/bench_03/candidates/ngated-freq-encode/](experiments/bench_03/candidates/ngated-freq-encode/)

### 2026-07-13（~00 UTC）: 実提出=03_cv_ensemble ＋ オフライン改善ラウンド14 — TEゲート閾値拡張(te-gate-widen-4000)（不採用）
- **実提出（本日枠を使用）:** UTCが2026-07-13へ変わり本日分の提出枠が空いたため、申し送りブロックの唯一指示どおり `03_cv_ensemble` を提出。`validate_submission.py` 合格を再確認 → zip → `kaggle competitions submit`。submission_id=54625716、00:02 UTC 提出。**⚠️ 採点結果は `SubmissionStatus.ERROR`（スコアなし）。** オフラインベンチ(go.py直接実行)もMBPローカルLLMリハーサルも通っていたのに本番でERROR＝失敗は go.py ロジックでなく**本番エージェント実行環境(gemini-2.5-flash駆動の6ステップ手順)側**の可能性が高い。**ユーザーが今回限り承認した複雑化(CV＋複数モデル族アンサンブル)路線が実グレーダーで動かなかった**ことを意味し、同構造を共有する 05/04 も同様にERRORするリスクが高い。→ **PushNotificationでユーザーに報告し、(A)複雑路線のERROR原因調査・修正 か (B)シンプル路線(01土台＋02_early_stopping等)へ回帰 かの判断を仰いだ。** 詳細と次アクションは冒頭「🚨次回実行への申し送り」ブロック参照。本日 2026-07-13 UTC の提出枠はこれで消費（ERRORでも枠消費した可能性が高い・要確認）。
- **オフライン改善ラウンド14（無制限枠・毎回必須）:** 実提出とは独立に、方針どおり新しい角度を1つ検証。ラウンド13の方針メモで「未検証で残る小変更」として明示した **TEゲート閾値1500の微調整** を実行した（＝新規に角度をゼロ探索せず、記録済みの優先候補を消化）。
  - **選んだ角度＝「TEゲート閾値の拡張」:** 05のOOFターゲットエンコーディングは `if len(X) < 1500 and len(cats) > 0:` で「小規模データセットだけ」にTE列を足す。この**閾値そのもの**を `1500→4000` に上げ、中規模データセットまでTEを効かせるべきかを検証。ラウンド12はTEのもう一方のハイパラ(平滑化定数)を突いたので、これはハイパラ探索の直交な残り一辺。
  - **設計上の外科性:** この16データで n が [1500,4000) に入る唯一のデータは train_16(n=1809, ただしカテゴリ列0で `len(cats)>0` に掛からず非発火) と train_03(n=3501, 7カテゴリ)。よって**新たにTEが発火するのは train_03 のみ**で、他15データはゲート挙動が完全に不変＝バイト同一になるはずという事前予測が立つ（＝「TEが中規模カテゴリ train_03 を助けるか否か」だけを問う純粋な一点実験・他15データへの下振れリスクは数学的にゼロ）。
  - **絶対厳守ルールの遵守:** 候補作成は `coder` サブエージェントに委譲（書き込みは `experiments/bench_03/candidates/te-gate-widen-4000/system.md` のみ・ベースは `git show HEAD:submissions/05_te_ngated/...`・`submissions/`配下は読み書きとも一切触らず）。サブエージェントの自己diffが `77c77`（`1500`→`4000`）の**1行だけ**であることを確認。ベンチマークはオーケストレータが実行し完了マーカー（`round14/DONE`）確認後に判定（サブエージェントの「実行中」報告を最終結果にしない既存ルールを踏襲）。
  - **落とし穴メモ:** 最初のベンチ実行で `--go-python` に**相対パス**を渡したため、ベンチが各データセット用の作業ディレクトリへ `chdir` した先からインタプリタを解決できず、全16データが `FileNotFoundError` で 1秒失敗する偽の全滅が発生。`--go-python` 省略（gbm_venv がデフォルト）で正常化。**教訓: benchmark.py は go.py をデータ別 workdir 内で実行するので、`--go-python` を渡すなら絶対パス（既定のままなら不要）。**
  - **クリーンな同条件実測（workers=2、05ベースと候補を同セッションで両方完走）:** 事前予測どおり**15データが05とバイト同一**（train_06含む）、変化は狙いの train_03 のみ。**train_03(n=3501) +0.0176→+0.0172（−0.0004）と回帰。** 平均delta **05=+0.01538 vs 候補=+0.01536（−0.00002）**。
  - **判断: 不採用。** 唯一動く train_03 が悪化し、平均も下げる（採用基準「05のmeanをクリーンに上回る／悪化なし」を両方満たさず）。**知見: ラウンド6で得た「TEはn依存で小n(≲1100)に効き中〜大では僅かに害」を、ゲートの上側境界で定量確認。ゲートを4000へ広げると中規模カテゴリ train_03 を1件巻き込んで −0.0004 悪化する。ラウンド12(平滑化=10が最適)と合わせ、TEの2ハイパラ(平滑化・ゲート閾値)は両方向とも局所最適で、05の `smoothing=10 / gate<1500` はこのsuiteのTEスイートスポット。** これで「素朴FE飽和(ラウンド8-11)＋勝ち技への追加符号化も害(ラウンド13)＋勝ち技ハイパラも両辺で局所最適(ラウンド12,14)」が出揃い、特徴量・TEチューニング系の低リスク直交角度はほぼ探索し尽くした。
  - **次サイクル以降の方針メモ（更新）:** 実効的な残り伸びしろは (b) モデル族の多様化・ブレンド重みの原理的改善に事実上限定。これらは「シンプル1変更」を超える設計変更になりうるため、**実行前に必ずユーザーへ確認を挟むこと**（自動判断で複雑化しない）。それまでの1サイクル1件のオフライン探索は、まだ残る低リスクの小変更（例: TEを二値/低カーディナリティのカテゴリ交互作用に限定適用、小nでのシード数だけ増やす、ブレンドを rank 平均から OOF-AUC 加重へ、など未検証の一点変更）を順に消化して継続する。
- **キュー（不変）:** ① 03 の採点完了を待つ → 表の03行にスコア記入・前回比記入 → ② 次UTC日(2026-07-14以降)に `05_te_ngated` を提出（申し送りブロックの手順） → ③ 以降 `04_missing_count` / `freq-encode-cats`。
- 構成: 候補 [experiments/bench_03/candidates/te-gate-widen-4000/](experiments/bench_03/candidates/te-gate-widen-4000/)、生ログ・CSV [experiments/bench_03/round14/](experiments/bench_03/round14/)（`base05_results.csv` / `widen4000_results.csv` / 各 `*.log`）
