# This file is adapted from https://github.com/jennyzzt/dgm.

import argparse
import datetime
import json
import os
import re
import tempfile
from concurrent.futures import (ProcessPoolExecutor, ThreadPoolExecutor,
                                as_completed)
from enum import Enum
from pathlib import Path

import docker
from datasets import load_dataset

from polyglot.constants import MAP_REPO_VERSION_TO_SPECS, TEST_COMMANDS
from polyglot.docker_build import (build_container, build_env_images,
                                   cleanup_container)
from polyglot.test_spec import make_test_spec
from prompts.testrepo_prompt import get_test_description
from swe_bench.utils import (copy_from_container, copy_to_container,
                             log_container_output, remove_existing_container,
                             safe_log, setup_logger)
from utils.git_utils import filter_patch_by_files, remove_patch_by_files

llm = ""
timeout = 1800  # seconds


def get_eval_script(commands):
    return "\n".join(["#!/bin/bash", "set -uxo pipefail"] + commands) + "\n"


def process_entry(
    entry,
    out_dname,
    model_name_or_path,
    model_patch_paths,
    skip_existing=True,
    init_agent_path=".",
):
    """
    Process a single dataset entry. This function encapsulates the main processing logic
    for each entry to make it suitable for parallel execution.
    """
    instance_id = entry["instance_id"]
    problem_statement = entry["problem_statement"]
    base_commit = entry["base_commit"]
    chat_history_file = out_dname / (instance_id + ".md")
    out_fname = out_dname / (instance_id + ".json")
    eval_file = out_dname / f"{instance_id}_eval.sh"
    eval_result_file = out_dname / f"{instance_id}_eval.md"

    # Skip if output result already exists
    if out_fname.exists() and skip_existing:
        print(f"Skipping existing entry {instance_id}")
        with open(out_fname) as f:
            result = json.loads(f.read())
        return result

    # Initialize container as None to avoid UnboundLocalError
    container = None

    try:
        # Create and start the Docker container
        client = docker.from_env()
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        # Set up thread-specific logger
        logger = setup_logger(str(out_dname / f"{instance_id}_docker.log"))
        nocache = True
        test_spec = make_test_spec(entry)
        # Remove any existing container with the same name
        container_name = test_spec.get_instance_container_name(run_id)
        remove_existing_container(client, container_name)
        # Now create and start the container
        container = build_container(
            test_spec, client, run_id, logger, nocache, force_rebuild=False
        )
        container.start()

        # Copy the necessary files and requirements to the container
        copy_to_container(
            container,
            os.path.join(init_agent_path, "coding_agent_polyglot.py"),
            "/hgm/coding_agent.py",
        )
        copy_to_container(
            container,
            os.path.join(init_agent_path, "requirements.txt"),
            "/hgm/requirements.txt",
        )
        copy_to_container(
            container, os.path.join(init_agent_path, "pytest.ini"), "/hgm/pytest.ini"
        )
        copy_to_container(
            container, os.path.join(init_agent_path, "tools/"), "/hgm/tools/"
        )
        copy_to_container(
            container, os.path.join(init_agent_path, "utils/"), "/hgm/utils/"
        )
        copy_to_container(
            container, os.path.join(init_agent_path, "tests/"), "/hgm/tests/"
        )
        copy_to_container(
            container, os.path.join(init_agent_path, "prompts/"), "/hgm/prompts/"
        )
        copy_to_container(
            container, os.path.join(init_agent_path, "llm.py"), "/hgm/llm.py"
        )
        copy_to_container(
            container,
            os.path.join(init_agent_path, "llm_withtools.py"),
            "/hgm/llm_withtools.py",
        )
        copy_to_container(
            container, os.path.join(init_agent_path, "LICENSE"), "/hgm/LICENSE"
        )
        copy_to_container(
            container, os.path.join(init_agent_path, "README.md"), "/hgm/README.md"
        )
        chat_history_file_container = f"/hgm/{chat_history_file.name}"

        # See the checked repo
        exec_result = container.exec_run("ls -R /testbed", workdir="/")
        log_container_output(exec_result)

        # Get test description
        eval_cmd = MAP_REPO_VERSION_TO_SPECS[entry["language"]]["test_cmd"]
        test_description = get_test_description(eval_cmd, polyglot=True)

        # Apply model patch
        if model_patch_paths:
            safe_log("Applying model patches")
            for model_patch_path in model_patch_paths:
                copy_to_container(container, model_patch_path, "/hgm/parent_patch.txt")
                exec_result = container.exec_run(
                    "/bin/sh -c 'patch -p1 < /hgm/parent_patch.txt'", workdir="/hgm"
                )
                log_container_output(exec_result)
                exec_result = container.exec_run(
                    "rm /hgm/parent_patch.txt", workdir="/hgm"
                )
                log_container_output(exec_result)

        # Install this repo requirements
        safe_log("Installing more requirements")
        exec_result = container.exec_run(
            "python -m pip install -r /hgm/requirements.txt", workdir="/"
        )
        log_container_output(exec_result)

        # Run the agent
        env_vars = {
            "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
            "AWS_REGION": os.getenv("AWS_REGION"),
            "AWS_REGION_NAME": os.getenv("AWS_REGION_NAME"),
            "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID"),
            "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY"),
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            "OpenRouter_API_KEY": os.getenv("OpenRouter_API_KEY"),
        }
        safe_log("Running the agent")
        cmd = [
            "timeout",
            str(timeout),
            "python",
            "/hgm/coding_agent.py",
            "--problem_statement",
            problem_statement,
            "--git_dir",
            "/testbed/",
            "--chat_history_file",
            chat_history_file_container,
            "--base_commit",
            base_commit,
            "--outdir",
            "/hgm/",
            "--test_description",
            test_description,
            "--language",
            entry["language"],
            "--model",
            llm,
            "--timeout",
            str(timeout),
        ]
        exec_result = container.exec_run(cmd, environment=env_vars, workdir="/testbed/")
        log_container_output(exec_result)

        # Copy output files back to host
        logger.info("Copying output files back to host")
        copy_from_container(container, chat_history_file_container, chat_history_file)
        # Additional chat history files
        exec_result = container.exec_run(
            f"find /hgm/ -name '{instance_id}_*.md'", workdir="/"
        )
        chat_history_files_container = exec_result.output.decode().split()
        for chat_history_file_container in chat_history_files_container:
            chat_history_file = out_dname / Path(chat_history_file_container).name
            copy_from_container(
                container, chat_history_file_container, chat_history_file
            )

        # Get model_patch
        logger.info("Getting model_patch")
        exec_result = container.exec_run("cat /hgm/model_patch.diff")
        log_container_output(exec_result)
        model_patch = ""
        model_patch = exec_result.output.decode()

        # Additional proposed model patches
        proposed_model_patches = []

        # Directly do eval
        eval_result = ""
        if not model_patch:
            eval_result = "empty_patch"
            result = {
                "instance_id": instance_id,
                "model_name_or_path": model_name_or_path,
                "model_patch": model_patch,
                "proposed_model_patches": proposed_model_patches,
                "eval_result": eval_result,
                "success": True,
            }
            out_fname.write_text(json.dumps(result, indent=4))
            return {
                "success": True,
                "instance_id": instance_id,
                "eval_result": eval_result,
            }

        exec_result = container.exec_run(
            "git -C /testbed stash push " + " ".join(entry["files"]["solution"]),
            workdir="/",
        )
        log_container_output(exec_result)
        exec_result = container.exec_run(
            f"git -C /testbed reset --hard {entry['test_commit']}", workdir="/"
        )
        log_container_output(exec_result)
        exec_result = container.exec_run(f"git -C /testbed clean -fd", workdir="/")
        log_container_output(exec_result)
        exec_result = container.exec_run("git -C /testbed stash pop", workdir="/")
        log_container_output(exec_result)

        safe_log("Running the eval")
        language = entry["language"]
        test_command = TEST_COMMANDS[language]

        eval_file.write_text(get_eval_script(test_command))

        copy_to_container(container, eval_file, "/testbed/eval.sh")
        exec_result = container.exec_run("ls -R /testbed", workdir="/")
        log_container_output(exec_result)
        exec_result = container.exec_run("chmod +x /testbed/eval.sh", workdir="/")
        log_container_output(exec_result)

        exec_result = container.exec_run("timeout 120 ./eval.sh", workdir="/testbed")
        log_container_output(exec_result, raise_error=False)
        eval_result_file.write_text(exec_result.output.decode())
        if exec_result.exit_code == 0:
            eval_result = "resolved"
        else:
            eval_result = "unresolved"

        # Write result to file
        result = {
            "instance_id": instance_id,
            "model_name_or_path": model_name_or_path,
            "model_patch": model_patch,
            "proposed_model_patches": proposed_model_patches,
            "eval_result": eval_result,
            "success": True,
        }
        out_fname.write_text(json.dumps(result, indent=4))

        return {"success": True, "instance_id": instance_id, "eval_result": eval_result}

    except Exception as e:
        # Check if eval_result exists in local scope
        if "eval_result" not in locals():
            eval_result = "incomplete"
        else:
            eval_result = "error"
        if "model_patch" not in locals():
            model_patch = ""
        if "proposed_model_patches" not in locals():
            proposed_model_patches = ""

        # Write result to file
        result = {
            "instance_id": instance_id,
            "model_name_or_path": model_name_or_path,
            "model_patch": model_patch,
            "proposed_model_patches": proposed_model_patches,
            "eval_result": eval_result,
            "success": False,
        }
        out_fname.write_text(json.dumps(result, indent=4))

        # print(f"Error processing entry {instance_id}: {str(e)}")
        logger.error(f"Error processing entry {instance_id}: {str(e)}")
        return {
            "success": False,
            "instance_id": instance_id,
            "eval_result": eval_result,
        }

    finally:
        # Clean up docker container
        try:
            if container is not None:
                cleanup_container(client, container, logger)
        except Exception as e:
            print(f"Error cleaning up Docker container for {instance_id}: {e}")


def harness(
    dataset_path="polyglot/polyglot_benchmark_metadata.json",
    test_task_list=None,
    max_workers=4,
    model_name_or_path=None,
    model_patch_paths=None,
    pred_dname="./polyglot/predictions",
    output_dir="./polyglot/predictions",
    skip_existing=True,
    init_agent_path=".",
):
    """
    Parallel processing harness using ThreadPoolExecutor.

    Args:
        test_task_list: List of task IDs to process (None for all)
        num_samples: Number of samples to process (-1 for all)
        max_workers: Maximum number of concurrent threads
        model_name_or_path: Model name or path
        model_patch_paths: Paths to the model patches for hgm
        num_evals: Repeated number of swe evaluations
    """
    if model_patch_paths:
        for model_patch_path in model_patch_paths:
            # Read and modify model patch
            with open(model_patch_path, "r") as f:
                patch_content = f.read()
            patch_content = remove_patch_by_files(patch_content)
            # Placeholder for any patch modifications if needed
            with open(model_patch_path, "w") as f:
                f.write(patch_content)

    # Load dataset
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    with open(dataset_path) as f:
        dataset = json.load(f)

    # Ensure that necessary directories exist
    if model_name_or_path is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name_or_path = f"{timestamp}--claude-3-5-sonnet-20241022"
    pred_dname = Path(pred_dname)
    pred_dname.mkdir(exist_ok=True)
    out_dnames = []

    # Prepare the dataset entries
    entries = list(dataset)
    entries = [entry for entry in entries if entry["instance_id"] in test_task_list]

    # Build the environment images
    client = docker.from_env()
    build_env_images(
        client, dataset=entries, max_workers=max_workers, force_rebuild=False
    )

    # Define a function to handle a single evaluation for all specified issues
    model_name_or_path_inst = f"{model_name_or_path}_{0}"
    out_dname = pred_dname / model_name_or_path_inst
    out_dname.mkdir(exist_ok=True)

    print(f"Starting evaluation {0} for model {model_name_or_path}")

    # Process entries in parallel
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_entry = {
            executor.submit(
                process_entry,
                entry,
                out_dname,
                model_name_or_path_inst,
                model_patch_paths,
                skip_existing=skip_existing,
                init_agent_path=init_agent_path,
            ): entry
            for entry in entries
        }

        # Process completed tasks as they finish
        for future in as_completed(future_to_entry):
            result = future.result()
            results.append(future.result())
            if result["success"]:
                print(
                    f"Successfully processed entry {result['instance_id']} for eval {0}"
                )
            else:
                print(
                    f"Failed to process entry {result['instance_id']} for eval {0}: {result.get('error', 'Unknown error')}"
                )
        # Get final results from completed futures

    print(f"All evaluations completed for model {model_name_or_path}")

    # Directly generate report
    # write report to file
    incomplete_ids = [
        result["instance_id"] for result in results if not result["success"]
    ]
    completed_ids = [result["instance_id"] for result in results if result["success"]]
    # Get resolved/unresolved/error/empty patch IDs from results
    resolved_ids = []
    unresolved_ids = []
    error_ids = []
    empty_patch_ids = []
    unstopped_containers = []
    unremoved_images = []

    for result in results:
        if result["success"]:
            if result.get("eval_result") == "resolved":
                resolved_ids.append(result["instance_id"])
            elif result.get("eval_result") == "unresolved":
                unresolved_ids.append(result["instance_id"])
            elif result.get("eval_result") == "empty_patch":
                empty_patch_ids.append(result["instance_id"])
            else:
                error_ids.append(result["instance_id"])

    report = {
        "total_instances": len(dataset),
        "submitted_instances": len(results),
        "completed_instances": len(completed_ids),
        "resolved_instances": len(resolved_ids),
        "unresolved_instances": len(unresolved_ids),
        "empty_patch_instances": len(empty_patch_ids),
        "error_instances": len(error_ids),
        "unstopped_instances": len(unstopped_containers),
        "completed_ids": list(sorted(completed_ids)),
        "incomplete_ids": list(sorted(incomplete_ids)),
        "empty_patch_ids": list(sorted(empty_patch_ids)),
        "submitted_ids": list(sorted(result["instance_id"] for result in results)),
        "resolved_ids": list(sorted(resolved_ids)),
        "unresolved_ids": list(sorted(unresolved_ids)),
        "error_ids": list(sorted(error_ids)),
        "unstopped_containers": list(sorted(unstopped_containers)),
        "unremoved_images": list(sorted(unremoved_images)),
        "schema_version": 2,
    }

    print(report)
    report_file = output_dir / Path(
        model_name_or_path.replace("/", "__")
        + " "
        + datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        + ".json"
    )
    with open(report_file, "w") as f:
        print(json.dumps(report, indent=4), file=f)
    print(f"Report written to {report_file}")

    return out_dnames


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_samples", type=int, default=-1, help="Number of samples to process"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=5,
        help="Maximum number of concurrent threads",
    )
    parser.add_argument(
        "--model_name_or_path", type=str, default=None, help="Model name or path"
    )
    parser.add_argument(
        "--model_patch_paths", type=str, default=None, help="Paths to the model patches"
    )
    parser.add_argument(
        "--num_evals", type=int, default=1, help="Repeated number of swe evaluations"
    )
    parser.add_argument(
        "--num_evals_parallel",
        type=int,
        default=1,
        help="Number of parallel repeated evaluations",
    )
    args = parser.parse_args()

    with open("polyglot/polyglot_benchmark_metadata.json") as f:
        metadata = json.loads(f.read())
        language_task_list = [
            entry["instance_id"]
            for entry in metadata
            if entry["instance_id"].startswith("python")
        ]
        # Create a list of all tasks from metadata
        all_task_list = [entry["instance_id"] for entry in metadata]

    from utils.common_utils import load_json_file

    swe_issues_med = load_json_file("./polyglot/subsets/medium.json")
    model_patch_paths = (
        args.model_patch_paths.split(",")
        if args.model_patch_paths is not None
        else None
    )
    # Run the parallel harness

    harness(
        dataset_path="polyglot/polyglot_benchmark_metadata.json",
        test_task_list=all_task_list,
        num_samples=args.num_samples,
        max_workers=args.max_workers,
        model_name_or_path=args.model_name_or_path,
        model_patch_paths=model_patch_paths,
        num_evals=args.num_evals,
        num_evals_parallel=args.num_evals_parallel,
    )


if __name__ == "__main__":
    main()
