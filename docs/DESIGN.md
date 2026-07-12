# Design: `submissions/01_baseline`

## Goal

Provide one simple, robust baseline autonomous agent for the `autonomous-agent-prediction-beta`
meta-competition. The agent should reliably produce a valid, reasonably-scored submission on any
dataset drawn from the competition's family (`data/train_01` .. `train_16`), without per-dataset
tuning, and without spending much of its time/tool-call/token budget.

## Why a single generic `HistGradientBoostingClassifier` pass

The `DATA.md` files across `train_01`..`train_16` show the schema (column count, names, and the mix
of numeric/categorical/ordinal columns) varies per dataset, but two invariants hold everywhere:
`row_id`/`target` are always present and named the same, and categorical/ordinal columns are always
string-encoded (`dtype == object`) while numeric/count columns stay numeric. That's enough structure
to write one script that works unmodified on any dataset in the family:

- Feature columns = everything except `row_id` and `target` — no hardcoded column names, so it
  survives schema drift (12 features in `train_01`, 21 in `train_16`).
- A boolean `dtype == object` mask feeds `categorical_features=` directly into
  `HistGradientBoostingClassifier`, which natively supports categorical splits — no manual
  one-hot/ordinal encoding step is needed, and none of that logic can break on an unexpected
  cardinality or an unseen category at test time.
- The same estimator natively handles missing values (the datasets contain NaNs), so no imputation
  step is needed either.

This keeps the whole training script at roughly 15-20 lines, verified locally against three of the
sixteen sample datasets (12- and 21-feature schemas), producing valid submission files with
plausible AUC (0.71-0.89) with no per-dataset changes.

## Why a minimal tool set

The agent is given only `write_file`, `run_command`, `submit_predictions`, and `get_status` — enough
to write the script, execute it, submit once, and confirm the result. `edit_file` and
`select_submission` are intentionally omitted: this is a single-shot baseline, not an iterative
search, so there is nothing to edit after the first correct script, and `select_submission` is
optional (the harness defaults to the best public score if it's never called). A smaller tool
surface means a smaller prompt, fewer opportunities for the LLM to wander into hyperparameter
tuning or resubmission loops, and lower token spend for a task that doesn't need any of that.

## Prompt structure

`prompts/system.md` follows the structure recommended in
`kaggle-kaggle-skill/resources/agent_instructions.md`: persona, injected `{problem_description}` /
`{metric_name}` / `{metric_direction}`, environment description, injected budget placeholders, and
an explicit numbered workflow (write script -> run it -> submit once -> check status -> stop). The
prompt explicitly tells the agent not to iterate, tune hyperparameters, ensemble, or resubmit, since
the goal here is a cheap, reliable baseline rather than a competitive score.
