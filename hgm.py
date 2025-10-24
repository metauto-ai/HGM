# This file is adapted from https://github.com/jennyzzt/dgm.

import argparse
import datetime
import json
import math
import os
import random
import string
import threading
import time
import traceback
from collections import defaultdict
from concurrent.futures import (ProcessPoolExecutor, ThreadPoolExecutor,
                                TimeoutError, as_completed)
from statistics import stdev

import numpy as np
from datasets import load_dataset
from utils.docker_utils import copy_src_files

import hgm_utils
from config import load_config
from tree import Node
from utils.common_utils import load_json_file
from utils.docker_utils import copy_src_files, setup_logger
from utils.evo_utils import load_hgm_metadata


def update_metadata(output_dir, n_task_evals):
    with open(os.path.join(output_dir, "hgm_metadata.jsonl"), "a") as f:
        f.write(
            json.dumps(
                {
                    "n_task_evals": n_task_evals,
                    "nodes": [
                        node.save_as_dict()
                        for node in hgm_utils.nodes.values()
                        if node.commit_id != "initial"
                    ],
                },
                indent=2,
            )
            + "\n"
        )
    json.dump(
        hgm_utils.init_evaluated_tasks,
        open(os.path.join(output_dir, "init_evaluated_tasks.json"), "w"),
    )


def initialize_run(
    output_dir,
    self_improve_llm,
    tasks,
    initial_agent_name,
    prevrun_dir=None,
    polyglot=False,
    timeout=3600,
    max_workers=20
):
    hgm_utils.init(polyglot, output_dir, tasks, 0, self_improve_llm, timeout)

    # Copy cached initial version into experiment dir
    initial_folder = "initial_swe/" if not polyglot else "initial_polyglot/"
    if not prevrun_dir:
        if not os.path.exists(f"{initial_folder}/{initial_agent_name}"):
            copy_src_files(f"{initial_folder}/{initial_agent_name}/src", build_image=True)
            hgm_utils.output_dir = initial_folder
            hgm_utils.eval_agent(
                initial_agent_name,
                tasks = hgm_utils.total_tasks,
                max_workers=max_workers,
                init_agent_path=f"{initial_folder}/{initial_agent_name}/src",
            )
            hgm_utils.output_dir = output_dir

    os.system(f"cp -r {initial_folder}/{initial_agent_name} {output_dir}/initial")

    Node(commit_id="initial")
    if prevrun_dir:
        # Load previous run's archive
        hgm_utils.init_evaluated_tasks = load_json_file(
            os.path.join(prevrun_dir, "init_evaluated_tasks.json")
        )
        metadata_path = os.path.join(prevrun_dir, "hgm_metadata.jsonl")
        metadata = load_hgm_metadata(metadata_path, last_only=True)
        for node in metadata["nodes"]:
            commit_id = node["commit_id"]
            parent_id = node["parent_id"]
            Node(commit_id, parent_id=parent_id, id=node["id"])
        for node in hgm_utils.nodes.values():
            if node.parent_id is not None:
                parent = hgm_utils.nodes[node.parent_id]
                parent.add_child(node)

    n_task_evals = 0
    submitted_ids = defaultdict(set)  # node_id -> set of submitted task ids
    for node in hgm_utils.nodes.values():
        metadata = load_json_file(
            os.path.join(output_dir, node.commit_id, "metadata.json")
        )
        submitted_ids[node.id] = set(
            metadata["overall_performance"]["total_submitted_ids"]
        )
        node.utility_measures = [
            1
            for _ in range(metadata["overall_performance"]["total_resolved_instances"])
        ] + [
            0
            for _ in range(
                metadata["overall_performance"]["total_submitted_instances"]
                - metadata["overall_performance"]["total_resolved_instances"]
            )
        ]
        if node.commit_id != "initial":
            n_task_evals += metadata["overall_performance"]["total_submitted_instances"]
    hgm_utils.n_task_evals = n_task_evals
    return os.path.join(initial_folder, initial_agent_name, "src"), submitted_ids


def main():
    parser = argparse.ArgumentParser(description="Optimistic Tree Search")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--max_task_evals",
        type=int,
        default=None,
        help="Maximum number of evolution iterations.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=None,
        help="Number of parallel workers for self-improvement attempts.",
    )
    parser.add_argument(
        "--continue_from",
        type=str,
        default=None,
        help="Directory to continue the run from.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for this run (overrides config).",
    )
    parser.add_argument(
        "--polyglot",
        dest="polyglot",
        action="store_true",
        help="Run Polyglot benchmark instead of SWE-bench.",
    )
    parser.add_argument(
        "--no_polyglot",
        dest="polyglot",
        action="store_false",
        help="Disable Polyglot benchmark even if enabled in config.",
    )
    parser.add_argument(
        "--self_improve_llm",
        type=str,
        default=None,
        help="LLM model to use for self-improvement",
    )
    parser.add_argument(
        "--downstream_llm",
        type=str,
        default=None,
        help="LLM model to use for downstream tasks",
    )
    parser.add_argument(
        "--diagnose_llm",
        type=str,
        default=None,
        help="LLM model to use for diagnosis",
    )
    parser.add_argument(
        "--alpha", type=float, default=None, help="Alpha parameter for node expansion."
    )
    parser.add_argument(
        "--cool_down",
        dest="cool_down",
        action="store_true",
        help="Use a decreasing temperature over iterations.",
    )
    parser.add_argument(
        "--no_cool_down",
        dest="cool_down",
        action="store_false",
        help="Disable decreasing temperature over iterations even if enabled in config.",
    )
    parser.add_argument(
        "--beta", type=float, default=None, help="Cooling down factor beta."
    )
    parser.add_argument(
        "--full_eval",
        dest="full_eval",
        action="store_true",
        help="Run full evaluation on SWE even if disabled in config.",
    )

    parser.add_argument(
        "--self_improve_timeout",
        type=int,
        default=None,
        help="Timeout for self-improvement attempts.",
    )
    parser.add_argument(
        "--evaluation_timeout",
        type=int,
        default=None,
        help="Timeout for evaluation attempts.",
    )
    parser.add_argument(
        "--n_pseudo_descendant_evals",
        type=int,
        default=None,
        help="Number of pseudo descendant evaluations.",
    )
    parser.add_argument(
        "--eval_random_level",
        type=float,
        default=None,
        help="Randomness level for evaluation task selection.",
    )
    parser.add_argument(
        "--initial_agent_name",
        type=str,
        default="default_agent",
        help="Name of the initial agent.",
    )

    parser.set_defaults(polyglot=None, cool_down=None, full_eval=None)

    args = parser.parse_args()

    overrides = {}
    if args.max_task_evals is not None:
        overrides["execution.max_task_evals"] = args.max_task_evals
    if args.max_workers is not None:
        overrides["execution.max_workers"] = args.max_workers
    if args.continue_from is not None:
        overrides["paths.continue_from"] = args.continue_from
    if args.output_dir is not None:
        overrides["paths.output_dir"] = args.output_dir
    if args.self_improve_llm is not None:
        overrides["llm.self_improve_llm"] = args.self_improve_llm
    if args.downstream_llm is not None:
        overrides["llm.downstream_llm"] = args.downstream_llm
    if args.diagnose_llm is not None:
        overrides["llm.diagnose_llm"] = args.diagnose_llm
    if args.alpha is not None:
        overrides["optimization.alpha"] = args.alpha
    if args.cool_down is not None:
        overrides["optimization.cool_down"] = args.cool_down
    if args.beta is not None:
        overrides["optimization.beta"] = args.beta
    if args.full_eval is not None:
        overrides["evaluation.full_eval"] = args.full_eval
    if args.self_improve_timeout is not None:
        overrides["execution.self_improve_timeout"] = args.self_improve_timeout
    if args.evaluation_timeout is not None:
        overrides["execution.evaluation_timeout"] = args.evaluation_timeout
    if args.n_pseudo_descendant_evals is not None:
        overrides["optimization.n_pseudo_descendant_evals"] = args.n_pseudo_descendant_evals
    if args.eval_random_level is not None:
        overrides["optimization.eval_random_level"] = args.eval_random_level
    if args.polyglot is not None:
        overrides["evaluation.polyglot"] = args.polyglot
    if args.initial_agent_name is not None:
        overrides["paths.initial_agent_name"] = args.initial_agent_name

    config = load_config(args.config, **overrides)

    if not config.paths.initial_agent_name:
        parser.error(
            "Initial agent name must be provided either in config.yaml or via --initial_agent_name."
        )

    llm_cfg = config.llm
    opt_cfg = config.optimization
    exec_cfg = config.execution
    eval_cfg = config.evaluation
    path_cfg = config.paths

    # Variables for this HGM run
    if path_cfg.output_dir:
        output_dir = os.path.abspath(path_cfg.output_dir)
        run_id = os.path.basename(os.path.normpath(output_dir))
    elif not path_cfg.continue_from:
        run_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S_%f")
        output_dir = os.path.abspath(os.path.join("./output_hgm", run_id))
    else:
        run_id = os.path.basename(os.path.normpath(path_cfg.continue_from))
        output_dir = os.path.abspath(os.path.join("./output_hgm", run_id))

    # Ensure output directory exists and log path info
    os.makedirs(output_dir, exist_ok=True)
    print(f"Working directory: {os.getcwd()}")
    print(f"Using config file: {args.config}")
    print(f"Output directory: {output_dir}")
    print(f"Output directory exists: {os.path.exists(output_dir)}")

    import polyglot.harness
    import self_improve_step

    polyglot.harness.llm = llm_cfg.downstream_llm  # Set the LLM model for downstream tasks
    import swe_bench.harness

    swe_bench.harness.llm = (
        llm_cfg.downstream_llm
    )  # Set the LLM model for downstream tasks
    polyglot.harness.timeout = exec_cfg.evaluation_timeout
    swe_bench.harness.timeout = exec_cfg.evaluation_timeout
    self_improve_step.diagnose_llm = llm_cfg.diagnose_llm
    self_improve_step.self_improve_llm = llm_cfg.self_improve_llm
    # Initialize logger early
    logger = setup_logger(os.path.join(output_dir, "hgm_outer.log"))
    # SWE issues to consider
    if not eval_cfg.polyglot:
        if eval_cfg.full_eval:
            tasks = [
                task["instance_id"]
                for task in load_dataset("princeton-nlp/SWE-bench_Verified")["test"]
            ]
        else:
            tasks = load_json_file("./swe_bench/subsets/small.json") \
                    + load_json_file("./swe_bench/subsets/medium.json") 
        random.seed(42)
        random.shuffle(tasks)
    else:
        tasks = load_json_file("./polyglot/subsets/medium.json") + load_json_file(
            "./polyglot/subsets/small.json"
        )

    src_path, submitted_ids = initialize_run(
        output_dir,
        llm_cfg.self_improve_llm,
        tasks,
        path_cfg.initial_agent_name,
        prevrun_dir=path_cfg.continue_from,
        polyglot=eval_cfg.polyglot,
        timeout=exec_cfg.self_improve_timeout,
        max_workers=exec_cfg.max_workers
    )
    total_num_tasks = len(hgm_utils.total_tasks)

    # Set up logger
    logger.info(
        f"Starting HGM run {run_id} with configuration: {config.to_dict()}"
    )

    def TS_sample(evals):
        alphas = [1 + np.sum(de) for de in evals]
        betas = [1 + len(de) - np.sum(de) for de in evals]
        if opt_cfg.cool_down:
            alphas = np.array(alphas) * (
                10000
                if exec_cfg.max_task_evals == hgm_utils.n_task_evals
                else exec_cfg.max_task_evals**opt_cfg.beta
                / (exec_cfg.max_task_evals - hgm_utils.n_task_evals) ** opt_cfg.beta
            )
            betas = np.array(betas) * (
                10000
                if exec_cfg.max_task_evals == hgm_utils.n_task_evals
                else exec_cfg.max_task_evals**opt_cfg.beta
                / (exec_cfg.max_task_evals - hgm_utils.n_task_evals) ** opt_cfg.beta
            )
        thetas = np.random.beta(alphas, betas)
        return np.argmax(thetas)

    n_pending_expands = 0
    n_pending_measures = 0
    lock = threading.Lock()

    def expand():
        with lock:
            nodes = [
                node
                for node in hgm_utils.nodes.values()
                if np.isfinite(node.mean_utility) and node.mean_utility > 0
            ]
            decendant_evals = [
                node.get_decendant_evals(num_pseudo=opt_cfg.n_pseudo_descendant_evals)
                for node in nodes
            ]
            selected_node = nodes[TS_sample(decendant_evals)]
        child_commit = hgm_utils.sample_child(
            selected_node.commit_id,
            image_name=path_cfg.initial_agent_name + ":latest",
        )
        with lock:
            if child_commit != "failed":
                selected_node.children.append(
                    Node(child_commit, parent_id=selected_node.id)
                )
                update_metadata(output_dir, hgm_utils.n_task_evals)

    def sample():
        time.sleep(random.random())
        with lock:
            nonlocal n_pending_expands, n_pending_measures
            if hgm_utils.n_task_evals >= exec_cfg.max_task_evals:
                return

            if (
                hgm_utils.n_task_evals**opt_cfg.alpha
                >= len(hgm_utils.nodes) - 1 + n_pending_expands
            ):
                n_pending_expands += 1
                is_expand = True
            else:
                is_expand = False
        if is_expand:
            expand()
            with lock:
                n_pending_expands -= 1
                return

        with lock:
            nodes = hgm_utils.nodes[0].get_sub_tree(fn=lambda node: node)
            nodes = [
                node for node in nodes if len(submitted_ids[node.id]) < total_num_tasks
            ]
            evals = [node.utility_measures for node in nodes]
            if len(evals) == 0:
                return
            selected_node = nodes[TS_sample(evals)]
            available_tasks = list(
                [
                    task
                    for task in hgm_utils.total_tasks
                    if task not in submitted_ids[selected_node.id]
                ]
            )
            if len(available_tasks) == 0:
                return
            if random.random() < opt_cfg.eval_random_level:
                selected_node_tasks = random.choice(available_tasks)
            else:
                selected_node_tasks = available_tasks[0]
            submitted_ids[selected_node.id].add(selected_node_tasks)
            n_pending_measures += 1

        evals = hgm_utils.eval_agent(
            selected_node.commit_id,
            tasks=[selected_node_tasks],
            init_agent_path=src_path,
        )
        with lock:
            selected_node.utility_measures += evals
            n_pending_measures -= 1
            update_metadata(output_dir, hgm_utils.n_task_evals)

    try:
        with ThreadPoolExecutor(max_workers=exec_cfg.max_workers) as executor:
            futures = [
                executor.submit(expand)
                for _ in range(
                    len(hgm_utils.nodes) - 1,
                    min(5, int(exec_cfg.max_workers**opt_cfg.alpha)),
                )
            ]
            for future in as_completed(futures):
                future.result()

        with ThreadPoolExecutor(max_workers=exec_cfg.max_workers) as executor:
            futures = [
                executor.submit(sample)
                for _ in range(int(exec_cfg.max_task_evals * 100))
            ]
            for future in as_completed(futures):
                future.result()

    except Exception as e:
        logger.error(f"Error: {e}")
        logger.error(traceback.format_exc())
        print(repr(e))


if __name__ == "__main__":
    main()
