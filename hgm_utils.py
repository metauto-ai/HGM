# This file is adapted from https://github.com/jennyzzt/dgm.

import argparse
import datetime
import json
import os
import random
import re
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import stdev

import docker
import numpy as np

import self_improve_step
from polyglot.harness import harness as polyglot_harness
from prompts.self_improvement_prompt import find_selfimprove_eval_logs
from prompts.testrepo_prompt import get_test_description
from self_improve_step import diagnose_problem, save_metadata  # , choose
from swe_bench.harness import harness as swe_harness
from swe_bench.report import make_report
from utils.common_utils import load_json_file
from utils.docker_utils import (build_hgm_container, cleanup_container,
                                copy_from_container, copy_to_container,
                                log_container_output,
                                remove_existing_container, safe_log,
                                setup_logger)
from utils.eval_utils import get_acc_on_tasks
from utils.evo_utils import (get_all_performance, get_model_patch_paths,
                             is_compiled_self_improve)

dataset = None
alpha = 0.5
K = 0.5
bias_factor = 5
nodes = {}
total_tasks = []
output_dir = ""
polyglot = False
n_task_evals = 0
init_evaluated_tasks = []
llm = ""
timeout = 3600

pending_tasks_lock = threading.Lock()


def init(_polyglot, _output_dir, _tasks, _n_task_evals=0, _llm="", _timeout=3600):
    global output_dir, total_tasks, polyglot, n_task_evals, llm, timeout
    output_dir = _output_dir
    timeout = _timeout
    seen = set()
    total_tasks = []
    for item in _tasks:
        if item not in seen:
            seen.add(item)
            total_tasks.append(item)
    polyglot = _polyglot
    n_task_evals = _n_task_evals
    llm = _llm


def any_exceeding_context_length(output_dir, commit_id, instance_ids):
    """
    Check if any of the issues have exceeded the context length.
    """
    for instance_id in instance_ids:
        md_logs, _, _, _ = find_selfimprove_eval_logs(
            instance_id, output_dir, commit_id, filter=False
        )
        error_strs = [
            r"Error in get_response_withtools: Error code: 400 - {'message': 'Input is too long for requested model.'}",
            r"Error in get_response_withtools: Error code: 400 - {'object': 'error', 'message': \"This model's maximum context length is \d+ tokens. However, you requested \d+ tokens in the messages, Please reduce the length of the messages. None\", 'type': 'BadRequestError', 'param': None, 'code': 400}",
            r"Error in get_response_withtools: Error code: 400 - {'error': {'message': 'Your input exceeds the context window of this model. Please adjust your input and try again.', 'type': 'invalid_request_error', 'param': 'input', 'code': 'context_length_exceeded'}}",
        ]
        for md_log in md_logs:
            if any(
                re.search(f"{error_str}\n{error_str}", md_log)
                for error_str in error_strs
            ):
                return True
    return False


def choose_entry(parent_commit, debug=False):
    """
    Choose entry for self-improvement given a parent commit.
    """
    try:
        metadata_path = os.path.join(output_dir, parent_commit, "metadata.json")
        metadata = load_json_file(metadata_path)
        metadata = {
            "accuracy_score": metadata["overall_performance"]["accuracy_score"],
            "total_unresolved_ids": metadata["overall_performance"][
                "total_unresolved_ids"
            ],
            "total_emptypatch_ids": metadata["overall_performance"][
                "total_emptypatch_ids"
            ],
            "total_resolved_ids": metadata["overall_performance"]["total_resolved_ids"],
            "children_count": 0,
        }
    except Exception as e:
        # probably because swe-eval failed, generated code did not compile, etc.
        raise RuntimeError(f"{parent_commit} not eligible for being a parent: {e}")
    if debug:
        safe_log(metadata)

    empty_ids = metadata["total_emptypatch_ids"]
    resolved_ids = metadata["total_resolved_ids"]
    unresolved_ids = metadata["total_unresolved_ids"]

    entry = None

    if polyglot:
        entry_ids = empty_ids + unresolved_ids
        if not entry_ids:
            entry_ids = resolved_ids + empty_ids + unresolved_ids
        entry = random.choice(entry_ids)
    else:
        num_total_ids = len(empty_ids) + len(resolved_ids) + len(unresolved_ids)

        if len(empty_ids) >= 0.1 * num_total_ids and random.random() < 0.25:
            entry = "solve_empty_patches"

        elif random.random() < 0.25:
            entry = "solve_stochasticity"

        elif (
            any_exceeding_context_length(
                output_dir, parent_commit, empty_ids + unresolved_ids
            )
            and random.random() < 0.25
        ):
            entry = "solve_contextlength"

        elif len(unresolved_ids) != 0:
            entry_ids = unresolved_ids
            entry = random.choice(entry_ids)

        else:
            entry = random.choice(resolved_ids + empty_ids + unresolved_ids)
    if entry is None:
        safe_log(metadata)
        raise RuntimeError(
            f"Failed to choose an entry for self-improvement based on {parent_commit}."
        )
    return entry


def eval_agent(
    commit_id,
    tasks=None,
    num_tasks=5,
    max_workers=20,
    pending_tasks=None,
    random_level=0.5,
    skip=True,
    init_agent_path="./",
):
    if commit_id == "failed":
        return [0] * num_tasks
    global n_task_evals, total_tasks
    metadata = load_json_file(os.path.join(output_dir, commit_id, "metadata.json"))
    if tasks is None:
        if commit_id == "initial":
            if len(set(init_evaluated_tasks)) >= len(set(total_tasks)):
                return [metadata["overall_performance"]["accuracy_score"]] * num_tasks
            else:
                if skip:
                    un_evalatued_tasks = [
                        task for task in total_tasks if task not in init_evaluated_tasks
                    ]
                else:
                    un_evalatued_tasks = total_tasks
                order = "random" if random.random() < random_level else "fixed"
                if order == "random":
                    tasks = random.sample(
                        un_evalatued_tasks, min(num_tasks, len(un_evalatued_tasks))
                    )
                else:
                    tasks = un_evalatued_tasks[:num_tasks]
                init_evaluated_tasks.extend(tasks)
                return get_acc_on_tasks(tasks, os.path.join(output_dir, commit_id))
        if pending_tasks is None:
            pending_tasks = []

        with pending_tasks_lock:
            if skip:
                submitted_and_pending = (
                    metadata["overall_performance"]["total_submitted_ids"]
                    + pending_tasks
                )
                un_evalatued_tasks = [
                    task for task in total_tasks if task not in submitted_and_pending
                ]
            else:
                un_evalatued_tasks = total_tasks

            order = "random" if random.random() < random_level else "fixed"
            if len(un_evalatued_tasks) > 0:
                if order == "random":
                    tasks = random.sample(
                        un_evalatued_tasks, min(num_tasks, len(un_evalatued_tasks))
                    )
                else:
                    tasks = un_evalatued_tasks[:num_tasks]
                num_tasks = len(tasks)
            else:
                return [metadata["overall_performance"]["accuracy_score"]] * num_tasks
            pending_tasks.extend(tasks)

    n_task_evals += len(tasks)
    root_dir = os.path.abspath("./")
    metadata = load_json_file(
        os.path.join(root_dir, output_dir, commit_id, "metadata.json")
    )
    if polyglot:
        polyglot_harness(
            test_task_list=tasks,
            max_workers=min(max_workers, len(tasks)),
            model_name_or_path=commit_id,
            model_patch_paths=get_model_patch_paths(root_dir, output_dir, commit_id),
            pred_dname=os.path.join(root_dir, output_dir, commit_id, "predictions"),
            output_dir=os.path.join(root_dir, output_dir, commit_id),
            init_agent_path=init_agent_path,
        )
        overall_performance = get_all_performance(
            commit_id, results_dir=os.path.join(output_dir, commit_id)
        )[1]
    else:
        dnames = swe_harness(
            test_task_list=tasks,
            max_workers=min(max_workers, len(tasks)),
            model_name_or_path=commit_id,
            model_patch_paths=get_model_patch_paths(root_dir, output_dir, commit_id),
            pred_dname=os.path.join(root_dir, output_dir, commit_id, "predictions"),
            init_agent_path=init_agent_path,
        )
        safe_log("Start make_report")
        make_report(
            dnames,
            run_ids=[f"{commit_id}_{i}" for i in range(len(dnames))],
            dataset_name="princeton-nlp/SWE-bench_Verified",
            output_dir=os.path.join(output_dir, commit_id),
            num_eval_procs=min(max_workers, len(tasks)),
        )
        safe_log("Start get_performance")

        _, overall_performance = get_all_performance(
            commit_id, results_dir=os.path.join(output_dir, commit_id)
        )
        safe_log("End of evaluation")
        metadata["swe_dnames"] = [str(dn) for dn in dnames]

    metadata["overall_performance"] = overall_performance
    save_metadata(metadata, os.path.join(root_dir, output_dir, commit_id))
    return get_acc_on_tasks(tasks, os.path.join(root_dir, output_dir, commit_id))


def sample_child(parent_commit, image_name, force_rebuild=False, max_try=1):
    metadata = {}
    root_dir = os.path.abspath("./")  # root_dir should be /hgm
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_dir_base = output_dir  # out_dir_base should be /hgm/output_selfimprove/ or /hgm/output_hgm/{hgm_run_id}/
    run_output_dir = os.path.join(root_dir, f"{output_dir}/{run_id}/")
    os.makedirs(run_output_dir, exist_ok=True)

    try:
        if parent_commit == "failed":
            return "failed"
        if polyglot:
            with open("polyglot/polyglot_benchmark_metadata.json") as f:
                dataset = json.loads(f.read())
        else:
            from datasets import load_dataset

            dataset = load_dataset("princeton-nlp/SWE-bench_Verified")
            dataset = dataset["test"]
        self_improve_step.dataset = dataset

        setup_logger(os.path.join(run_output_dir, "self_improve.log"))
        metadata["run_id"] = run_id
        metadata["parent_commit"] = parent_commit

        container_name = f"hgm-container-{run_id}"
        client = docker.from_env()
        remove_existing_container(client, container_name)
        container = build_hgm_container(
            client,
            root_dir,
            image_name,
            container_name,
            force_rebuild=force_rebuild,
        )
        container.start()
        if polyglot:
            exec_result = container.exec_run("rm /hgm/coding_agent.py", workdir="/")
            log_container_output(exec_result)
            exec_result = container.exec_run(
                "mv /hgm/coding_agent_polyglot.py /hgm/coding_agent.py", workdir="/"
            )
            log_container_output(exec_result)
            exec_result = container.exec_run("rm /hgm/utils/eval_utils.py", workdir="/")
            log_container_output(exec_result)
            exec_result = container.exec_run(
                "rm /hgm/utils/swe_log_parsers.py", workdir="/"
            )
            log_container_output(exec_result)
        else:
            exec_result = container.exec_run(
                "rm /hgm/coding_agent_polyglot.py", workdir="/"
            )

        patch_files = get_model_patch_paths(root_dir, output_dir, parent_commit)
        for patch_file in patch_files:
            copy_to_container(container, patch_file, "/hgm/parent_patch.txt")
            exec_result = container.exec_run(
                "/bin/sh -c 'patch -p1 < /hgm/parent_patch.txt'", workdir="/hgm"
            )
            log_container_output(exec_result)
            exec_result = container.exec_run("rm /hgm/parent_patch.txt", workdir="/hgm")
            log_container_output(exec_result)

        container.exec_run("git init", workdir="/hgm/")
        exec_result = container.exec_run("git add --all", workdir="/hgm/")
        log_container_output(exec_result)
        exec_result = container.exec_run(
            "git -c user.name='user' -c user.email='you@example.com' commit -m 'a nonsense commit message'",
            workdir="/hgm/",
        )

        exec_result = container.exec_run("git log")
        log_container_output(exec_result)
        commit_hash = (
            exec_result.output.decode("utf-8").split("\n")[0].split()[1]
        )  # Get the latest commit hash

        exec_result = container.exec_run(
            "python -m pip install -r /hgm/requirements.txt", workdir="/"
        )
        log_container_output(exec_result)

        safe_log("Getting tasks to improve")
        try:
            entry = choose_entry(parent_commit)
        except Exception as e:
            safe_log(f"Error choosing entry: {e}")
        try:
            safe_log(f"Task to improve: {entry}")
        except Exception as e:
            choose_entry(parent_commit, debug=True)
            raise e
        problem_statement = diagnose_problem(
            entry,
            parent_commit,
            root_dir,
            out_dir_base,
            patch_files=patch_files,
            polyglot=polyglot,
        )
        safe_log(f"problem_statement: {problem_statement}")

        metadata["entry"] = entry
        metadata["problem_statement"] = problem_statement
        if not problem_statement:
            safe_log("Failed to diagnose the problem statement. Exiting.")
            cleanup_container(container)
            save_metadata(metadata, run_output_dir)
            if max_try > 1:
                return sample_child(parent_commit, force_rebuild, max_try - 1)
            else:
                return "failed"

        safe_log("Running self-improvement")
        chat_history_file_container = "/hgm/self_evo.md"
        test_description = get_test_description(swerepo=False)
        env_vars = {
            "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
            "AWS_REGION": os.getenv("AWS_REGION"),
            "AWS_REGION_NAME": os.getenv("AWS_REGION_NAME"),
            "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID"),
            "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY"),
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            "OpenRouter_API_KEY": os.getenv("OpenRouter_API_KEY"),
        }
        cmd = [
            "timeout",
            str(timeout),
            "python",
            "/hgm/coding_agent.py",
            "--problem_statement",
            problem_statement,
            "--git_dir",
            "/hgm/",
            "--chat_history_file",
            chat_history_file_container,
            "--base_commit",
            commit_hash,
            "--outdir",
            "/hgm/",
            "--test_description",
            test_description,
            "--self_improve",
            "--model",
            llm,
            "--timeout",
            str(timeout),
        ]
        exec_result = container.exec_run(cmd, environment=env_vars, workdir="/")
        log_container_output(exec_result, raise_error=False)

        chat_history_file = os.path.join(output_dir, run_id, "self_evo.md")
        copy_from_container(container, chat_history_file_container, chat_history_file)
        model_patch_file = os.path.join(output_dir, run_id, "model_patch.diff")
        copy_from_container(container, "/hgm/model_patch.diff", model_patch_file)

        metadata["overall_performance"] = {
            "accuracy_score": 0.0,
            "total_resolved_instances": 0,
            "total_submitted_instances": 0,
            "files": [],
            "total_submitted_ids": [],
            "total_unresolved_ids": [],
            "total_emptypatch_ids": [],
            "total_resolved_ids": [],
        }
        if not os.path.exists(model_patch_file):
            raise Exception("Model patch file is empty or does not exist")
        with open(model_patch_file, "r") as f:
            patch_content = f.read()
            if not patch_content.strip():
                raise Exception("Model patch file is empty")

    except Exception as e:
        if max_try > 1:
            safe_log(f"Error while sampling a child: {str(e)}. Retrying...")
            safe_log(traceback.format_exc())
            return sample_child(parent_commit, force_rebuild, max_try - 1)
        else:
            safe_log(f"Error while sampling a child: {str(e)}")
            safe_log(traceback.format_exc())
            return "failed"
    finally:
        try:
            cleanup_container(container)
        except Exception as e:
            safe_log(f"Error during container cleanup: {e}")
        save_metadata(metadata, run_output_dir)
    return run_id