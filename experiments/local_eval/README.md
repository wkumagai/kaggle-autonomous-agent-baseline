# Local (MBP-LLM) Rehearsal Rig

End-to-end rehearsal of the Kaggle-in-Kaggle harness with **zero cloud cost**: every model
alias in `models.yaml` is routed to a locally-served LLM (llama-swap on the MacBook Pro,
`http://192.168.11.42:9000/v1`, default model `qwen35b-a3b-q6`), and the sandbox uses a
small native linux/arm64 image instead of the amd64-only `gcr.io/kaggle-images/python`.

## Run (from the repo root)

```bash
docker build --platform linux/arm64 -t kk-sandbox-arm64:latest experiments/local_eval  # once
.venv/bin/python experiments/local_eval/run_local_eval_mbp.py \
    --submission-dir submissions/01_baseline/agent \
    --dataset train_03 \
    --output-dir experiments/local_eval/output_smoke
```

If the MBP endpoint is down: `ssh MBP 'nohup ~/llm/bin/llama-swap --config ~/llm/config/llama-swap.yaml --listen :9000 >> ~/llm/llama-swap.log 2>&1 &'`

## Fidelity caveats

- **Weaker LLM**: qwen35b-a3b-q6 is far below the real competition models (Gemini etc.) —
  scores validate *mechanics* (sandbox, tools, submit, trace), not agent quality.
- **Different sandbox image**: python:3.11-slim + pandas/numpy/sklearn/scipy/lightgbm/xgboost/catboost
  on arm64, not the real Kaggle image; some packages an agent assumes may be missing.
- **Slow calls**: 30-60 s per LLM call; llama-swap cold start can take 1-2 min (client timeout 600 s).
- **Cost accounting is meaningless**: litellm reports $0/wrong prices for local models — ignore budget USD.
- Qwen3 thinking is suppressed via `extra_body={"chat_template_kwargs": {"enable_thinking": false}}`.
