#!/usr/bin/env python3
"""Local Evaluation Harness (MBP-local-LLM variant) for Kaggle-in-Kaggle.

Copy-with-modifications of the starter kit's run_local_eval.py that:

1. Routes EVERY model alias in models.yaml to a single locally-served
   OpenAI-compatible model (llama-swap on the MacBook Pro), so whatever
   model an agent.yaml names, it lands on the local model. No API keys,
   no cloud cost.
2. Adds --docker-image (default: kk-sandbox-arm64:latest, a small native
   linux/arm64 sandbox built from experiments/local_eval/Dockerfile)
   instead of the amd64-only gcr.io/kaggle-images/python default.

All other flags and behavior are identical to run_local_eval.py.
Paths resolve relative to the REPO ROOT (two levels above this script),
exactly as the original resolves them relative to itself.

## Usage (from the repo root)
    .venv/bin/python experiments/local_eval/run_local_eval_mbp.py \
        --submission-dir submissions/01_baseline/agent \
        --dataset train_03 \
        --output-dir experiments/local_eval/output_smoke

Note: litellm cost accounting is meaningless for a local model (reports
$0 or wrong prices) — ignore the budget USD numbers.
"""

import argparse
import asyncio
import sys
import warnings
from pathlib import Path

import pandas as pd

# Ensure we can import adk_submission and kaggle_kaggle
from adk_submission import ModelRegistry, compile_submission
from dotenv import load_dotenv
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps._configs import EventsCompactionConfig
from google.adk.models.lite_llm import LiteLlm
from kaggle_kaggle import (
    Evaluation,
    EventDisplay,
    ProblemResult,
    print_results,
    save_trace,
)
from kaggle_kaggle.budget import PricingTable
from kaggle_kaggle.config import BudgetConfig, EvaluationConfig, ProblemConfig

# Load environment variables (kept for parity with run_local_eval.py)
load_dotenv()

# Suppress noise
warnings.filterwarnings("ignore", module=r"authlib\.")
warnings.filterwarnings("ignore", message=r".*PLUGGABLE_AUTH.*", category=UserWarning)
warnings.filterwarnings("ignore", message=r".*Pydantic serializer warnings.*", category=UserWarning)

async def _run_local_eval(
    config: EvaluationConfig,
    metric: str,
    agent_dir: str,
    models: ModelRegistry,
    output_dir: Path,
) -> ProblemResult:
    """Async helper to instantiate Evaluation, compile the agent, and run the loop."""
    async with Evaluation(config, metric=metric) as ev:
        agent = compile_submission(
            agent_dir,
            ev.tools,
            models,
            code_executor=ev.code_executor,
            script_timeout=config.budget.max_exec_seconds,
        )
        display = EventDisplay(
            evaluation=ev,
            problem_id=config.problem.problem_id,
            metric=metric,
        )
        with display:
            result = await ev.run(agent)

        saved_files = save_trace(result, output_dir=output_dir)
        if saved_files:
            print("\nTrace files saved:")
            for f in saved_files:
                print(f"  {f}")
        return result


def main():
    parser = argparse.ArgumentParser(
        description="Run Kaggle-in-Kaggle local evaluation harness against a local (MBP) LLM."
    )
    parser.add_argument(
        "--submission-dir",
        type=str,
        default="sample_submission",
        help="Path to the participant submission directory containing agent.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save evaluation trace files. Defaults to <submission_dir>/../output if using structured layout, or base_dir/output.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="train_01",
        help="Name of the dataset directory in data/ to use for evaluation (default: train_01)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="roc_auc",
        help="Evaluation metric name string (default: roc_auc)",
    )
    parser.add_argument(
        "--docker-image",
        type=str,
        default="kk-sandbox-arm64:latest",
        help="Docker image for the sandbox environment (default: kk-sandbox-arm64:latest, "
        "built from experiments/local_eval/Dockerfile)",
    )
    parser.add_argument(
        "--local-model",
        type=str,
        default="qwen35b-a3b-q6",
        help="Model ID served by the local llama-swap endpoint; every models.yaml alias "
        "is routed to this model (default: qwen35b-a3b-q6)",
    )
    parser.add_argument(
        "--local-api-base",
        type=str,
        default="http://192.168.11.42:9000/v1",
        help="OpenAI-compatible base URL of the local LLM server (default: MBP llama-swap)",
    )
    parser.add_argument(
        "--llm-timeout-seconds",
        type=int,
        default=600,
        help="Per-request client timeout; llama-swap cold-starts can take 1-2 min (default: 600)",
    )
    parser.add_argument(
        "--max-tool-calls",
        type=int,
        default=1000,
        help="Maximum tool executions allowed (default: 1000)",
    )
    parser.add_argument(
        "--max-submissions",
        type=int,
        default=30,
        help="Maximum prediction submissions allowed (default: 30 for local eval)",
    )
    parser.add_argument(
        "--max-selections",
        type=int,
        default=2,
        help="Maximum final submissions selected for private scoring (default: 2)",
    )
    parser.add_argument(
        "--max-exec-seconds",
        type=int,
        default=3600,
        help="Per-command execution timeout in seconds (default: 3600)",
    )
    parser.add_argument(
        "--max-stdout-chars",
        type=int,
        default=5000,
        help="Maximum characters to capture from stdout/stderr per command (default: 5000)",
    )
    parser.add_argument(
        "--max-budget-usd",
        type=float,
        default=2.0,
        help="Maximum LLM token budget in USD (default: 2.0; meaningless for local models)",
    )
    parser.add_argument(
        "--max-llm-calls",
        type=int,
        default=1000,
        help="Hard limit on LLM invocations (default: 1000)",
    )
    parser.add_argument(
        "--max-time-minutes",
        type=int,
        default=60,
        help="Maximum session runtime allowed in minutes (default: 60 for local eval)",
    )
    parser.add_argument(
        "--num-retries",
        type=int,
        default=100,
        help="Number of retries for LiteLLM requests (default: 100)",
    )
    parser.add_argument(
        "--cache-min-tokens",
        type=int,
        default=2048,
        help="Minimum token count to enable context caching (default: 2048)",
    )
    parser.add_argument(
        "--cache-ttl-seconds",
        type=int,
        default=1800,
        help="TTL in seconds for cached context (default: 1800)",
    )
    parser.add_argument(
        "--cache-intervals",
        type=int,
        default=10,
        help="Number of turn intervals between cache updates (default: 10)",
    )
    parser.add_argument(
        "--compaction-interval",
        type=int,
        default=15,
        help="Number of turns between event compactions (default: 15)",
    )
    parser.add_argument(
        "--compaction-overlap-size",
        type=int,
        default=2,
        help="Number of overlapping events preserved during compaction (default: 2)",
    )
    parser.add_argument(
        "--compaction-token-threshold",
        type=int,
        default=16384,
        help="Token threshold to trigger event compaction (default: 16384)",
    )
    parser.add_argument(
        "--compaction-event-retention-size",
        type=int,
        default=5,
        help="Number of recent events retained during compaction (default: 5)",
    )
    args = parser.parse_args()

    # This script lives in experiments/local_eval/; paths must resolve
    # relative to the REPO ROOT, matching the original run_local_eval.py.
    base_dir = Path(__file__).resolve().parent.parent.parent
    submission_dir = base_dir / args.submission_dir

    if not submission_dir.exists():
        print(f"Error: Submission directory not found at {submission_dir}")
        sys.exit(1)

    dataset_dir = base_dir / "data" / args.dataset
    if not dataset_dir.exists():
        print(f"Error: Dataset directory not found at {dataset_dir}")
        sys.exit(1)

    train_path = dataset_dir / "train.csv"
    test_path = dataset_dir / "test.csv"
    sample_submission_path = dataset_dir / "sample_submission.csv"
    solution_path = dataset_dir / "solution.csv"

    for path in [train_path, test_path, sample_submission_path, solution_path]:
        if not path.exists():
            print(f"Error: Required dataset file not found at {path}")
            sys.exit(1)

    # Identify sub-problem ID column
    sub_sample_df = pd.read_csv(sample_submission_path)
    sub_id_col = str(sub_sample_df.columns[0])

    models_yaml_path = base_dir / "models.yaml"
    if args.output_dir:
        output_dir = base_dir / args.output_dir
    else:
        # Detect structured layout: e.g., submissions/01_baseline/agent
        if submission_dir.name == "agent" and submission_dir.parent.parent.name == "submissions":
            output_dir = submission_dir.parent / "output"
        else:
            output_dir = base_dir / "output"

    output_dir.mkdir(parents=True, exist_ok=True)

    if models_yaml_path.exists():
        pricing_table = PricingTable.from_yaml(models_yaml_path)
    else:
        pricing_table = PricingTable.from_yaml()

    # Register EVERY alias from models.yaml against the single local model.
    # Whatever model name an agent.yaml requests, it lands on the MBP model.
    # extra_body suppresses Qwen3 "thinking" output, which otherwise confuses
    # ADK tool-call parsing (verified to pass through llama-swap correctly).
    print(
        f"Local mode: routing all {len(pricing_table.model_ids)} model aliases to "
        f"'{args.local_model}' at {args.local_api_base}"
    )
    models = ModelRegistry()
    for slug in pricing_table.model_ids:
        models.register(
            slug,
            LiteLlm(
                model=f"openai/{args.local_model}",
                api_base=args.local_api_base,
                api_key="dummy",
                num_retries=args.num_retries,
                timeout=args.llm_timeout_seconds,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            ),
        )

    problem = ProblemConfig(
        problem_id=f"kaggle_in_kaggle_{args.dataset}",
        description="Predict the target column for the provided test.csv dataset.",
        id_column=sub_id_col,
        train_path=str(train_path),
        test_path=str(test_path),
        sample_submission_path=str(sample_submission_path),
        solution_path=str(solution_path),
    )

    context_cache_config = ContextCacheConfig(
        min_tokens=args.cache_min_tokens,
        ttl_seconds=args.cache_ttl_seconds,
        cache_intervals=args.cache_intervals,
    )
    events_compaction_config = EventsCompactionConfig(
        compaction_interval=args.compaction_interval,
        overlap_size=args.compaction_overlap_size,
        token_threshold=args.compaction_token_threshold,
        event_retention_size=args.compaction_event_retention_size,
    )

    config = EvaluationConfig(
        problem=problem,
        budget=BudgetConfig(
            max_tool_calls=args.max_tool_calls,
            max_submissions=args.max_submissions,
            max_selections=args.max_selections,
            max_exec_seconds=args.max_exec_seconds,
            max_stdout_chars=args.max_stdout_chars,
            max_budget_usd=args.max_budget_usd,
            max_llm_calls=args.max_llm_calls,
            max_time_minutes=args.max_time_minutes,
            num_retries=args.num_retries,
        ),
        docker_image=args.docker_image,
        models_yaml_path=str(models_yaml_path) if models_yaml_path.exists() else None,
        context_cache_config=context_cache_config,
        events_compaction_config=events_compaction_config,
    )

    print(f"Starting local agent evaluation for {problem.problem_id} with metric {args.metric}...")

    try:
        result = asyncio.run(
            _run_local_eval(
                config,
                args.metric,
                str(submission_dir),
                models,
                output_dir,
            )
        )
        print_results(result)
    except Exception as e:
        print(f"\n>>> Local Evaluation Failed: {e} <<<\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
