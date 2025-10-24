# This file is adapted from https://github.com/jennyzzt/dgm.

import argparse
import datetime
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import docker
from datasets import load_dataset
from swebench.harness.docker_build import (build_container, build_env_images,
                                           cleanup_container)
from swebench.harness.test_spec import make_test_spec

from prompts.testrepo_prompt import get_test_description
from swe_bench.utils import (copy_from_container, copy_to_container,
                             log_container_output, remove_existing_container,
                             safe_log, setup_logger)
from utils.common_utils import load_json_file

llm = ""  # Global variable to hold the LLM model name or path
timeout = 1800


def process_entry(
    entry, out_dname, model_name_or_path, model_patch_paths, init_agent_path="."
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

    # Skip if output result already exists
    if out_fname.exists():
        print(f"Skipping existing entry {instance_id}")
        return {"success": True, "instance_id": instance_id}

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

        # Check that we are in hgm directory
        tmp_currdir = os.path.abspath(os.getcwd())
        logger.info(f"Current directory: {tmp_currdir}")
        # If not in hgm directory, try to change to it
        if "hgm" not in tmp_currdir.split("/")[-1]:
            try:
                os.chdir("hgm")
                tmp_currdir = os.path.abspath(os.getcwd())
                logger.info(f"Changed directory to: {tmp_currdir}")
            except Exception as e:
                pass
        # If still not in hgm directory, go up until we find it
        count = 0
        while "hgm" not in tmp_currdir.split("/")[-1]:
            os.chdir("..")
            tmp_currdir = os.path.abspath(os.getcwd())
            logger.info(f"Changed directory to: {tmp_currdir}")
            count += 1
            if count > 10:
                raise Exception("Could not find hgm directory after 10 attempts")

        # Copy the necessary files and requirements to the container
        copy_to_container(
            container,
            os.path.join(init_agent_path, "coding_agent.py"),
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

        # Install issue requirements
        logger.info("Setting up environment")
        eval_script = test_spec.eval_script
        eval_file.write_text(eval_script)
        copy_to_container(container, eval_file, "/eval.sh")
        exec_result = container.exec_run(
            "/bin/bash /eval.sh", workdir="/"
        )  # setup environment
        log_container_output(exec_result)
        exec_result = container.exec_run(
            "rm /eval.sh", workdir="/"
        )  # remove eval script
        log_container_output(exec_result)

        # Get test description
        test_description = get_test_description(eval_script=eval_script, swerepo=True)

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
            str(timeout),  # 30min timeout
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
            "--instance_id",
            instance_id,
            "--model",
            llm,  # Use the global llm variable
            "--timeout",
            str(timeout),
        ]
        exec_result = container.exec_run(cmd, environment=env_vars, workdir="/")
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
        model_patch = exec_result.output.decode()
        # Additional proposed model patches
        proposed_model_patches = []
        exec_result = container.exec_run(
            "find /hgm/ -name 'model_patch_*.diff'", workdir="/"
        )
        model_patch_files_container = exec_result.output.decode().split()
        for model_patch_file_container in model_patch_files_container:
            exec_result = container.exec_run(f"cat {model_patch_file_container}")
            log_container_output(exec_result)
            proposed_model_patch = exec_result.output.decode()
            proposed_model_patches.append(proposed_model_patch)

        # Write result to file
        result = {
            "instance_id": instance_id,
            "model_name_or_path": model_name_or_path,
            "model_patch": model_patch,
            "proposed_model_patches": proposed_model_patches,
        }
        out_fname.write_text(json.dumps(result, indent=4))

        return {"success": True, "instance_id": instance_id}

    except Exception as e:
        print(f"Error processing entry {instance_id}: {str(e)}")
        # Create a result file with empty model_patch to mark it as submitted
        try:
            result = {
                "instance_id": instance_id,
                "model_name_or_path": model_name_or_path,
                "model_patch": "",  # Empty patch for failed cases
                "proposed_model_patches": [],
            }
            out_fname.write_text(json.dumps(result, indent=4))
        except Exception as write_error:
            print(f"Error writing empty result for {instance_id}: {str(write_error)}")
        return {"success": False, "instance_id": instance_id, "error": str(e)}

    finally:
        # Clean up docker container
        try:
            if container is not None:
                cleanup_container(client, container, logger)
        except Exception as e:
            print(f"Error cleaning up Docker container for {instance_id}: {e}")


def harness(
    test_task_list=None,
    max_workers=4,
    model_name_or_path=None,
    model_patch_paths=None,
    pred_dname="./swe_bench/predictions",
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
    # Load dataset
    dataset = load_dataset("princeton-nlp/SWE-bench_Verified")
    dataset = dataset["test"]

    # Ensure that necessary directories exist
    if model_name_or_path is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name_or_path = f"{timestamp}--claude-3-5-sonnet-20241022"
    pred_dname = Path(pred_dname)
    pred_dname.mkdir(exist_ok=True)

    # Prepare the dataset entries
    entries = list(dataset)
    entries = [entry for entry in entries if entry["instance_id"] in test_task_list]

    # Build the environment images
    client = docker.from_env()
    build_env_images(
        client, dataset=entries, force_rebuild=False, max_workers=max_workers
    )
    # Get all subdirectories under pred_dname
    subdirs = [d for d in pred_dname.iterdir() if d.is_dir()]
    # Define a function to handle a single evaluation for all specified issues
    model_name_or_path_inst = f"{model_name_or_path}_{len([d for d in subdirs if d.name.startswith(model_name_or_path)])+1}"
    out_dname = pred_dname / model_name_or_path_inst
    out_dname.mkdir(exist_ok=True)

    print(f"Starting evaluation {0} for model {model_name_or_path}")

    # Process entries in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_entry = {
            executor.submit(
                process_entry,
                entry,
                out_dname,
                model_name_or_path_inst,
                model_patch_paths,
                init_agent_path,
            ): entry
            for entry in entries
        }

        # Process completed tasks as they finish
        for future in as_completed(future_to_entry):
            result = future.result()
            if result["success"]:
                print(
                    f"Successfully processed entry {result['instance_id']} for eval {0}"
                )
            else:
                print(
                    f"Failed to process entry {result['instance_id']} for eval {0}: {result.get('error', 'Unknown error')}"
                )

    # Process all evaluations in parallel
    print(f"All evaluations completed for model {model_name_or_path}")
    return [out_dname]


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_samples", type=int, default=-1, help="Number of samples to process"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
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
    parser.add_argument(
        "--pred_dname",
        type=str,
        default="./swe_bench/predictions",
        help="Output directory for predictions",
    )
    parser.add_argument(
        "--test_task_list",
        type=str,
        default=None,
        help="Subset of swe issues to process",
    )
    args = parser.parse_args()

    # Load the test task list
    if args.test_task_list == "small":
        test_task_list = load_json_file("./swe_bench/subsets/small.json")
    elif args.test_task_list == "medium":
        test_task_list = load_json_file("./swe_bench/subsets/medium.json")
    else:
        # Default to None, means all of them
        test_task_list = None

    # Run the parallel harness
    harness(
        test_task_list=test_task_list,
        num_samples=args.num_samples,
        max_workers=args.max_workers,
        model_name_or_path=args.model_name_or_path,
        model_patch_paths=args.model_patch_paths,
        num_evals=args.num_evals,
        num_evals_parallel=args.num_evals_parallel,
        pred_dname=args.pred_dname,
    )


if __name__ == "__main__":
    main()
