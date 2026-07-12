#!/usr/bin/env python3
"""Direct repro: build the EXACT LlmRequest ADK would send for submissions/03_cv_ensemble
(full system instruction + real tool schemas), and call LiteLlm.generate_content_async()
directly. No Docker sandbox, no Evaluation() event loop -- just isolates the single LLM
turn that crashes in the full rig, with litellm debug logging turned on so we can see the
raw wire-level response.
"""

import asyncio
import json
import sys
import time
import traceback

from google.adk.models.lite_llm import LiteLlm
from google.adk.models.llm_request import LlmRequest
from google.genai import types as genai_types

import litellm

litellm.set_verbose = True  # dump raw request/response

BASE_DIR_SYSTEM_MD = "/Users/kumacmini/kaggle-autonomous-agent-baseline/submissions/03_cv_ensemble/agent/prompts/system.md"
TASK_PROMPT_JSON = "/Users/kumacmini/kaggle-autonomous-agent-baseline/experiments/local_eval/output_03_rehearsal/trace_kaggle_in_kaggle_train_03.json"


def build_system_instruction() -> str:
    raw = open(BASE_DIR_SYSTEM_MD).read()
    data = json.load(open(TASK_PROMPT_JSON))
    task_prompt = [e for e in data["entries"] if e["type"] == "task_prompt"][0]["content"]
    return raw.format(
        task_prompt=task_prompt,
        metric_name="roc_auc_score",
        metric_direction="higher is better",
        max_submissions=30,
        max_tool_calls=1000,
        max_time_minutes=55,
        max_budget_usd=2.0,
        max_exec_seconds=3600,
        max_stdout_chars=5000,
    )


def build_tools() -> list[genai_types.Tool]:
    decls = [
        genai_types.FunctionDeclaration(
            name="write_file",
            description="Write a file to the sandbox working directory.",
            parameters=genai_types.Schema(
                type="OBJECT",
                properties={
                    "filepath": genai_types.Schema(type="STRING"),
                    "content": genai_types.Schema(type="STRING"),
                },
                required=["filepath", "content"],
            ),
        ),
        genai_types.FunctionDeclaration(
            name="run_command",
            description="Run a shell command in the sandbox.",
            parameters=genai_types.Schema(
                type="OBJECT",
                properties={"command": genai_types.Schema(type="STRING")},
                required=["command"],
            ),
        ),
        genai_types.FunctionDeclaration(
            name="submit_predictions",
            description="Submit a predictions CSV file for public scoring.",
            parameters=genai_types.Schema(
                type="OBJECT",
                properties={"filepath": genai_types.Schema(type="STRING")},
                required=["filepath"],
            ),
        ),
        genai_types.FunctionDeclaration(
            name="select_submission",
            description="Select final submissions for private scoring.",
            parameters=genai_types.Schema(
                type="OBJECT",
                properties={
                    "submission_ids": genai_types.Schema(
                        type="ARRAY", items=genai_types.Schema(type="STRING")
                    )
                },
                required=["submission_ids"],
            ),
        ),
        genai_types.FunctionDeclaration(
            name="get_status",
            description="Check remaining budget and leaderboard standing.",
            parameters=genai_types.Schema(type="OBJECT", properties={}),
        ),
    ]
    return [genai_types.Tool(function_declarations=decls)]


async def main():
    system_instruction = build_system_instruction()
    print(f"system_instruction length: {len(system_instruction)} chars", file=sys.stderr)

    task_prompt = json.load(open(TASK_PROMPT_JSON))
    task_prompt_text = [e for e in task_prompt["entries"] if e["type"] == "task_prompt"][0]["content"]

    llm_request = LlmRequest(
        model="openai/qwen35b-a3b-q6",
        contents=[
            genai_types.Content(
                role="user",
                parts=[genai_types.Part(text=task_prompt_text)],
            )
        ],
        config=genai_types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=build_tools(),
            temperature=0.1,
            max_output_tokens=16384,
        ),
    )

    model = LiteLlm(
        model="openai/qwen35b-a3b-q6",
        api_base="http://192.168.11.42:9000/v1",
        api_key="dummy",
        num_retries=8,
        timeout=280,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

    t0 = time.time()
    try:
        async for resp in model.generate_content_async(llm_request, stream=False):
            print(f"\n--- response chunk at {time.time()-t0:.1f}s ---", file=sys.stderr)
            print(resp.model_dump_json(indent=2)[:4000], file=sys.stderr)
    except Exception:
        print(f"\n=== EXCEPTION after {time.time()-t0:.1f}s ===", file=sys.stderr)
        traceback.print_exc()
    print(f"\nTotal elapsed: {time.time()-t0:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
