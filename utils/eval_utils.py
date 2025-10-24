# This file is adapted from https://github.com/jennyzzt/dgm.

import os
import random

from llm import (create_client, extract_json_between_markers,
                 get_response_from_llm)
from llm_withtools import convert_msg_history
from utils.common_utils import load_json_file
from utils.swe_log_parsers import MAP_REPO_TO_PARSER


def get_acc_on_tasks(tasks, commit_path):
    if len(tasks) == 0:
        return []
    metadata = load_json_file(os.path.join(commit_path, "metadata.json"))
    accs = []
    for task in tasks:
        accs.append(
            1 if task in metadata["overall_performance"]["total_resolved_ids"] else 0
        )
    return accs


def parse_eval_output(instance_id, eval_output):
    try:
        if instance_id == "hgm":
            repo = "hgm"
        else:
            # Convert e.g. "scikit-learn__scikit-learn-12421" to "scikit-learn/scikit-learn"
            repo = "-".join(instance_id.replace("__", "/").split("-")[:-1])

        log_parser = MAP_REPO_TO_PARSER[repo]
        # Parse the evaluation output
        return log_parser(eval_output)

    except Exception as e:
        return {}


def msg_history_to_report(instance_id, msg_history, model=None):
    """
    Get test report from the message history.
    """
    # Convert the message history to a generic format
    msg_history = convert_msg_history(msg_history, model=model)

    # Get the test report from the message history
    for msg in reversed(msg_history):
        # Check if the message is from the user
        if msg["role"] == "user":
            # Check if the message contains the tool result
            content = msg["content"]
            if "Tool Result:" in content:
                report = parse_eval_output(instance_id, content)
                # Only return the report if it is not empty
                if report:
                    return report
    return {}


def get_report_score(test_report):
    """
    Get the score from the test report.
    """
    # Percentage of passed tests
    passed_count = sum([1 for v in test_report.values() if v == "PASSED"])
    total_count = len(test_report)
    return passed_count / total_count if total_count > 0 else 0


def score_tie_breaker(
    problem_statement, code_diffs, test_reports, best_score_indices=[], logging=print
):
    """
    Use LLM as a tiebreaker to choose the best code diff.
    """
    best_score_indices = (
        list(range(len(code_diffs))) if not best_score_indices else best_score_indices
    )
    best_score_index = best_score_indices[0]
    try:
        client = create_client("o3")
        proposed_solutions = [
            f"# Proposed solution {i+1}\n\n<code_diff_{i+1}>\n{code_diffs[index]}\n</code_diff{i+1}>\n<test_report_{i+1}>\n{test_reports[index]}\n</test_report_{i+1}>"
            for i, index in enumerate(best_score_indices)
        ]
        proposed_solutions = "\n\n".join(proposed_solutions)
        prompt = f"""Given the following problem statement, proposed solutions, and test reports, provide a summary of the differences between the code diffs and an evaluation of the proposed solutions.

<problem_description>
{problem_statement}
</problem_description>

{proposed_solutions}

Respond precisely in the following format including the JSON start and end markers:

```json
<JSON>
```

In <JSON>, provide a JSON response with the following fields:
- "difference_summary": Summary of the differences between the code diffs.
- "reasoning": Explanation of the reasoning behind the evaluation.
- "scores": List of numerical scores for each proposed solution.

Your response will be automatically parsed, so ensure that the string response is precisely in the correct format. Do NOT include `<JSON>` tag in your output.
"""
        response, msg_history = get_response_from_llm(
            msg=prompt,
            client=client[0],
            model=client[1],
            system_message="You are an excellent software engineer who has been asked to evaluate the proposed solutions to a problem statement.",
            print_debug=True,
            msg_history=None,
        )
        logging(repr(response))
        response_json = extract_json_between_markers(response)
        llm_scores = response_json["scores"]
        llm_best_score_index = random.choice(
            [i for i, score in enumerate(llm_scores) if score == max(llm_scores)]
        )
        best_score_index = best_score_indices[llm_best_score_index]
    except Exception as e:
        logging(f"Error in score_tie_breaker: {e}")
    return best_score_index
