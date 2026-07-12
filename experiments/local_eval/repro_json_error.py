#!/usr/bin/env python3
"""Standalone repro for the JSONDecodeError crash in run_local_eval_mbp.py.

kaggle_kaggle.evaluation.Evaluation.run() has a broad `except Exception as e:`
that swallows the traceback and only logs `str(e)`. This script monkeypatches
that module's logger so that, at the moment `logger.error("Agent error: %s", e)`
fires, we print the FULL traceback (sys.exc_info() is still valid inside the
except block) before it gets swallowed. This isolates exactly which file/line
raises the JSONDecodeError, without modifying any site-packages code.

Usage (from repo root):
    .venv/bin/python experiments/local_eval/repro_json_error.py
"""

import asyncio
import logging
import sys
import traceback
import warnings
from pathlib import Path

import pandas as pd

from adk_submission import ModelRegistry, compile_submission
from dotenv import load_dotenv
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps._configs import EventsCompactionConfig
from google.adk.models.lite_llm import LiteLlm
from kaggle_kaggle import Evaluation, EventDisplay, save_trace
from kaggle_kaggle.budget import PricingTable
from kaggle_kaggle.config import BudgetConfig, EvaluationConfig, ProblemConfig

import kaggle_kaggle.evaluation as kk_eval_module

load_dotenv()
warnings.filterwarnings("ignore", module=r"authlib\.")
warnings.filterwarnings("ignore", message=r".*PLUGGABLE_AUTH.*", category=UserWarning)
warnings.filterwarnings("ignore", message=r".*Pydantic serializer warnings.*", category=UserWarning)

# --- The actual repro trick -------------------------------------------------
_real_error = kk_eval_module.logger.error


def _error_with_traceback(msg, *args, **kwargs):
    exc_info = sys.exc_info()
    if exc_info[0] is not None:
        print("\n" + "=" * 80, file=sys.stderr)
        print("FULL TRACEBACK CAPTURED AT kaggle_kaggle.evaluation logger.error()", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        traceback.print_exception(*exc_info, file=sys.stderr)
        print("=" * 80 + "\n", file=sys.stderr)
    return _real_error(msg, *args, **kwargs)


kk_eval_module.logger.error = _error_with_traceback
# -----------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent.parent
SUBMISSION_DIR = BASE_DIR / "submissions/03_cv_ensemble/agent"
DATASET_DIR = BASE_DIR / "data/train_03"
OUTPUT_DIR = BASE_DIR / "experiments/local_eval/output_repro"

LOCAL_MODEL = "qwen35b-a3b-q6"
LOCAL_API_BASE = "http://192.168.11.42:9000/v1"


async def _run():
    sub_sample_df = pd.read_csv(DATASET_DIR / "sample_submission.csv")
    sub_id_col = str(sub_sample_df.columns[0])

    models_yaml_path = BASE_DIR / "models.yaml"
    pricing_table = PricingTable.from_yaml(models_yaml_path)

    models = ModelRegistry()
    for slug in pricing_table.model_ids:
        models.register(
            slug,
            LiteLlm(
                model=f"openai/{LOCAL_MODEL}",
                api_base=LOCAL_API_BASE,
                api_key="dummy",
                num_retries=2,  # small retry buffer for transient LAN blips
                timeout=600,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            ),
        )

    problem = ProblemConfig(
        problem_id="kaggle_in_kaggle_repro",
        description="Predict the target column for the provided test.csv dataset.",
        id_column=sub_id_col,
        train_path=str(DATASET_DIR / "train.csv"),
        test_path=str(DATASET_DIR / "test.csv"),
        sample_submission_path=str(DATASET_DIR / "sample_submission.csv"),
        solution_path=str(DATASET_DIR / "solution.csv"),
    )

    config = EvaluationConfig(
        problem=problem,
        budget=BudgetConfig(
            max_tool_calls=1000,
            max_submissions=30,
            max_selections=2,
            max_exec_seconds=3600,
            max_stdout_chars=5000,
            max_budget_usd=2.0,
            max_llm_calls=1000,
            max_time_minutes=5,  # short: we only need ONE turn to repro
            num_retries=0,
        ),
        docker_image="kk-sandbox-arm64:latest",
        models_yaml_path=str(models_yaml_path),
        context_cache_config=ContextCacheConfig(
            min_tokens=2048, ttl_seconds=1800, cache_intervals=10
        ),
        events_compaction_config=EventsCompactionConfig(
            compaction_interval=15,
            overlap_size=2,
            token_threshold=16384,
            event_retention_size=5,
        ),
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    async with Evaluation(config, metric="roc_auc") as ev:
        agent = compile_submission(
            str(SUBMISSION_DIR),
            ev.tools,
            models,
            code_executor=ev.code_executor,
            script_timeout=config.budget.max_exec_seconds,
        )
        print(f"Compiled agent instruction length (raw template): {len(agent.instruction)} chars")
        display = EventDisplay(evaluation=ev, problem_id=config.problem.problem_id, metric="roc_auc")
        with display:
            result = await ev.run(agent)
        save_trace(result, output_dir=OUTPUT_DIR)
        print(f"end_status={result.end_status}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(_run())
