import argparse
import hgm_utils
import os
import swe_bench.harness
from datasets import load_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent_path', type=str, required=True, help='Path to the agent to evaluate')
    parser.add_argument('--results_dir', type=str, default='results', help='Directory to store evaluation results')
    parser.add_argument('--split', type=str, default='Lite', help='Dataset split to use for evaluation')
    parser.add_argument('--llm', type=str, default='gpt-5', help='LLM to use for evaluation')
    parser.add_argument('--hours_per_task', type=int, default=5, help='Hours allocated per task')
    parser.add_argument('--num_workers', type=int, default=10, help='Number of parallel workers for evaluation')
    parser.add_argument('--n_tasks', type=int, default=None, help='Number of tasks to evaluate on')
    args = parser.parse_args()

    id = [task['instance_id'] for task in load_dataset(f'princeton-nlp/SWE-bench_{args.split}')['test']]
    if args.n_tasks is not None:
        id = id[:args.n_tasks]
    hgm_utils.init(False, args.results_dir, id)
    agent_name = os.path.basename(args.agent_path)
    os.makedirs(os.path.join(args.results_dir, agent_name), exist_ok=True)
    swe_bench.harness.llm = args.llm
    swe_bench.harness.timeout = args.hours_per_task * 3600
    hgm_utils.eval_agent(agent_name, max_workers=args.num_workers, init_agent_path=args.agent_path, tasks=id)
