# This file is adapted from https://github.com/jennyzzt/dgm.

from prompts.self_improvement_prompt import (find_selfimprove_eval_logs,
                                             get_current_code,
                                             process_selfimprove_eval_logs)
from utils.common_utils import read_file

diagnose_improvement_system_message = """Here is the relevant code for the LLM Coding agent with the model patch applied.

# LLM Coding Agent Code
The current code of LLM coding agent.
----- LLM Coding Agent Code Start -----
{code}
----- LLM Coding Agent Code End -----

# LLM Coding Agent Code Patch
The edits of LLM coding agent from the previous improvement.
----- LLM Coding Agent Code Patch Start -----
{model_patch_text}
----- LLM Coding Agent Code Patch End -----

# Test Patch
SWE-bench's official private tests to detect whether the issue is solved.
----- Test Patch Start -----
{test_patch}
----- Test Patch End -----

# Answer Patch
SWE-bench's official answer patch to the issue.
----- Answer Patch Start -----
{answer_patch}
----- Answer Patch End -----

# Your Task
Your task is to identify:
1. If the model patch has improved the agent's coding capabilities.
2. If the model patch has introduced any new issues or regressions.

Give a detailed analysis of the agent's performance before and after applying the model patch. Focus on the impact of the patch on the agent's problem-solving capabilities and general performance.
"""

diagnose_improvement_prompt = """Here are the logs for the coding agent, before and after applying the model patch, trying to solve the GitHub issues. It will be VERY LONG. Think very hard on the impact of the model patch on the agent's performance.
Note: ignore errors with "create_message_with_backoff" and all API rate limit errors.

# Agent Running Log Before Patch
The coding agent's log before improvement
----- Log Before Patch Start -----
{md_log}
----- Log Before Patch End -----

# Predicted Patch Before Patch
The predicted patch from agent before improvement to try to solve issue.
----- Predicted Patch Before Patch Start -----
{prev_predicted_patch}
----- Predicted Patch Before Patch End -----

# Issue Test Results Before Patch
The test results before improvement from SWE-bench using the above official private tests.
----- Issue Test Results Before Patch Start -----
{eval_log}
----- Issue Test Results Before Patch End -----

# Agent Running Log After Patch
The coding agent's log after improvement
----- Log After Patch Start -----
{new_md_log}
----- Log After Patch End -----

# Predicted Patch After Patch
The predicted patch from agent after improvement to try to solve issue.
----- Predicted Patch After Patch Start -----
{new_predicted_patch}
----- Predicted Patch After Patch End -----

# Issue Test Results After Patch
The test results after improvement from SWE-bench using the above official private tests.
----- Issue Test Results After Patch Start -----
{new_eval_log}
----- Issue Test Results After Patch End -----


# Instruction

Respond precisely in the following format including the JSON start and end markers:

```json
<JSON>
```

In <JSON>, provide a JSON response with the following fields:
- "impact": Analyze the impact of the model patch on the agent's performance. Focus on how the patch has affected the agent's problem-solving capabilities and general performance. This should be a long and thorough analysis.
- "improvements": Identify any improvements introduced by the model patch that enhance the agent's capabilities.
- "regressions": Identify any new issues or regressions introduced by the model patch that affect the agent's effectiveness.
- "score": Provide an overall score for the model patch's impact on the agent's performance. This should be a numerical value between -2 and 2, where -2 indicates a significant negative impact, 0 indicates no impact, and 2 indicates a significant positive impact.

Your response will be automatically parsed, so ensure that the string response is precisely in the correct format. Do NOT include the `<JSON>` tag in your output.
Focus on analyzing the impact of the model patch on the agent's performance, identifying improvements and regressions, and providing an overall score for the patch's impact.
Your thinking should be thorough. Please think very deeply."""


def get_diagnose_improvement_prompt(
    entry_id,
    parent_commit,
    root_dir,
    model_patch_file,
    out_dir,
    run_id,
    dataset,
    patch_files=[],
):
    md_logs, eval_logs, predicted_patches = find_selfimprove_eval_logs(
        entry_id, out_dir, commit_id=parent_commit
    )
    md_log, eval_log, predicted_patch = process_selfimprove_eval_logs(
        md_logs, eval_logs, predicted_patches
    )
    code_files = ["coding_agent.py", "tools/"]
    code_text = get_current_code(root_dir, code_files, patch_files=patch_files)
    model_patch_text = read_file(model_patch_file)
    new_md_logs, new_eval_logs, new_predicted_patches = find_selfimprove_eval_logs(
        entry_id, out_dir, commit_id=run_id
    )
    new_md_log, new_eval_log, new_predicted_patch = process_selfimprove_eval_logs(
        new_md_logs, new_eval_logs, new_predicted_patches
    )

    entry = next((e for e in dataset if e["instance_id"] == entry_id), None)
    answer_patch = entry["patch"]
    test_patch = entry["test_patch"]

    return diagnose_improvement_system_message.format(
        code=code_text,
        model_patch_text=model_patch_text,
        answer_patch=answer_patch,
        test_patch=test_patch,
    ), diagnose_improvement_prompt.format(
        md_log=md_log,
        eval_log=eval_log,
        new_md_log=new_md_log,
        new_eval_log=new_eval_log,
        prev_predicted_patch=predicted_patch,
        new_predicted_patch=new_predicted_patch,
    )
