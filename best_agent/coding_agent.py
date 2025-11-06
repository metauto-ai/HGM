# This file is adapted from https://github.com/jennyzzt/dgm.
import argparse
import subprocess
import logging
from logging.handlers import RotatingFileHandler
import os
import threading
from time import time

from llm_withtools import CLAUDE_MODEL, OPENAI_MODEL, chat_with_agent, convert_msg_history
from utils.eval_utils import get_report_score, msg_history_to_report, score_tie_breaker
from utils.git_utils import diff_versus_commit, reset_to_commit, apply_patch

# Thread-local storage for logger instances
thread_local = threading.local()

def get_thread_logger():
    """
    Get the logger instance specific to the current thread.
    Returns None if no logger has been set for this thread.
    """
    return getattr(thread_local, 'logger', None)

def set_thread_logger(logger):
    """
    Set the logger instance for the current thread.
    """
    thread_local.logger = logger

def setup_logger(log_file='./chat_history.md', level=logging.INFO):
    """
    Set up a logger with both file and console handlers.
    """
    # Create logger with a unique name based on thread ID
    logger = logging.getLogger(f'AgenticSystem-{threading.get_ident()}')
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    # Create formatters
    file_formatter = logging.Formatter('%(message)s')
    
    # Create and set up file handler
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
    file_handler.setLevel(level)
    file_handler.setFormatter(file_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    
    # Store logger in thread-local storage
    set_thread_logger(logger)
    
    return logger

def safe_log(message, level=logging.INFO):
    """
    Thread-safe logging function that ensures messages go to the correct logger.
    """
    logger = get_thread_logger()
    if logger:
        logger.log(level, message)
    else:
        print(f"Warning: No logger found for thread {threading.get_ident()}")

class AgenticSystem:
    def __init__(
            self,
            problem_statement,
            git_tempdir,
            base_commit,
            chat_history_file='./chat_history.md',
            test_description=None,
            self_improve=False,
            instance_id=None,
            model=CLAUDE_MODEL
        ):
        self.problem_statement = problem_statement
        self.git_tempdir = git_tempdir
        self.base_commit = base_commit
        self.chat_history_file = chat_history_file
        self.test_description = test_description
        self.self_improve = self_improve
        self.instance_id = instance_id if not self_improve else 'dgm'
        self.code_model = model

        # Initialize logger and store it in thread-local storage
        self.logger = setup_logger(chat_history_file)
        
        # Clear the log file
        with open(chat_history_file, 'w') as f:
            f.write('')

    def get_current_edits(self):
        diff = str(diff_versus_commit(self.git_tempdir, self.base_commit))
        return diff

    def get_regression_tests(self):
        """
        Get the regression tests from the repository.
        """
        instruction = f"""I have uploaded a Python code repository in the directory {self.git_tempdir}.

<problem_description>
{self.problem_statement}
</problem_description>

<test_description>
{self.test_description}
</test_description>

Your task is to identify regression tests in the {self.git_tempdir} directory that should pass both before and after addressing the <problem_description>. I have already taken care of the required dependencies.
At the end, please provide a summary that includes where the regression tests are located, what they are testing, and how they can be executed.
"""

        new_msg_history, _ = chat_with_agent(instruction, model=self.code_model, msg_history=[], logging=safe_log)
        new_msg_history = convert_msg_history(new_msg_history, self.code_model)
        regression_tests_summary = new_msg_history[-1]
        try:
            regression_tests_summary = regression_tests_summary['content']
        except:
            try:
                regression_tests_summary = str(regression_tests_summary)
            except:
                pass
        return regression_tests_summary

    def run_regression_tests(self, regression_tests_summary):
        """
        Run the regression tests and get the test report.
        """
        code_diff = self.get_current_edits()
        instruction = f"""I have uploaded a Python code repository in the directory {self.git_tempdir}. There is an attempt to address the problem statement. Please review the changes and run the regression tests.

<problem_description>
{self.problem_statement}
</problem_description>

<attempted_solution>
{code_diff}
</attempted_solution>

<test_description>
{self.test_description}
</test_description>

<regression_tests_summary>
{regression_tests_summary}
</regression_tests_summary>

Your task is to run the regression tests in the {self.git_tempdir} directory to ensure that the changes made to the code address the <problem_description>.
"""
        new_msg_history, _ = chat_with_agent(instruction, model=self.code_model, msg_history=[], logging=safe_log)
        test_report = msg_history_to_report(self.instance_id, new_msg_history, model=self.code_model)
        return test_report

    def _run_pytest_and_parse(self, specific_tests=None):
        """
        Run pytest with -rA and parse output into a test report dict using eval_utils parsers.
        """
        cmd = ["pytest", "-rA"]
        if specific_tests:
            if isinstance(specific_tests, list):
                cmd += specific_tests
            elif isinstance(specific_tests, str) and specific_tests.strip():
                cmd.append(specific_tests)
        try:
            proc = subprocess.run(
                cmd,
                cwd=self.git_tempdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            full_log = proc.stdout or ""
        except Exception as e:
            full_log = f"Error running pytest: {e}"
        # Lazy import to avoid circulars (keeps top imports clean)
        from utils.eval_utils import parse_eval_output
        instance = self.instance_id if self.instance_id else 'dgm'
        report = parse_eval_output(instance, full_log)
        return report, full_log

    def forward(self, timeout=3600, max_attempts=3, outdir=None):
        timeout -= 60
        start_time = time()
        """
        The forward function for the AgenticSystem.
        """
        instruction = f"""I have uploaded a Python code repository in the directory {self.git_tempdir}. Help solve the following problem.

<problem_description>
{self.problem_statement}
</problem_description>

<test_description>
{self.test_description}
</test_description>

Your task is to make changes to the files in the {self.git_tempdir} directory to address the <problem_description>. I have already taken care of the required dependencies.
"""
        attempts = []  # list of dicts with keys: diff, report, score, log
        os.makedirs(outdir, exist_ok=True) if outdir else None

        for i in range(int(max_attempts)):
            # Reset repo to base commit to start fresh
            try:
                reset_to_commit(self.git_tempdir, self.base_commit)
            except Exception as e:
                safe_log(f"Error resetting to base commit on attempt {i+1}: {e}")

            # Run the agent once to propose and apply a patch
            remaining = max(0, timeout - (time() - start_time))
            try:
                chat_history, n_llm_calls_used = chat_with_agent(
                    instruction,
                    model=self.code_model,
                    msg_history=[],
                    logging=safe_log,
                    timeout=remaining,
                )
                _ = str(chat_history)  # Ensure materialized to avoid lazy behavior
            except Exception as e:
                safe_log(f"chat_with_agent failed on attempt {i+1}: {e}")

            # Capture the diff
            diff = diff_versus_commit(self.git_tempdir, self.base_commit)

            # Run regression tests (plain pytest) and score
            report, full_log = self._run_pytest_and_parse()
            score = get_report_score(report)

            # Save attempt artifacts
            attempts.append({
                'diff': diff,
                'report': report,
                'score': score,
                'log': full_log,
            })
            if outdir:
                # Write diff and report for inspection
                safe_score = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)
                base = os.path.join(outdir, f"attempt_{i+1}_{safe_score}")
                try:
                    with open(base + ".diff", "w") as f:
                        f.write(diff)
                except Exception as e:
                    safe_log(f"Failed writing diff for attempt {i+1}: {e}")
                try:
                    # Write report as a simple text (dict repr) and raw pytest log
                    with open(base + ".report.txt", "w") as f:
                        f.write(str(report))
                    with open(base + ".pytest.log", "w") as f:
                        f.write(full_log)
                except Exception as e:
                    safe_log(f"Failed writing report/log for attempt {i+1}: {e}")

        # Decide best attempt
        if not attempts:
            return  # Nothing to do

        scores = [a['score'] for a in attempts]
        best_score = max(scores)
        best_indices = [idx for idx, s in enumerate(scores) if s == best_score]
        if len(best_indices) == 1:
            best_idx = best_indices[0]
        else:
            # Tiebreaker using eval_utils.score_tie_breaker
            code_diffs = [a['diff'] for a in attempts]
            test_reports = [a['report'] for a in attempts]
            best_idx = score_tie_breaker(self.problem_statement, code_diffs, test_reports, best_score_indices=best_indices, logging=safe_log)

        # Re-apply winning diff and leave working directory on it
        try:
            reset_to_commit(self.git_tempdir, self.base_commit)
            winning_diff = attempts[best_idx]['diff']
            if winning_diff:
                apply_patch(self.git_tempdir, winning_diff)
        except Exception as e:
            safe_log(f"Error applying winning patch: {e}")

        # Done
        return

def main():
    parser = argparse.ArgumentParser(description='Process repository with an agentic system.')
    parser.add_argument('--problem_statement', required=True, help='The problem statement to process')
    parser.add_argument('--git_dir', required=True, help='Path to git repository directory')
    parser.add_argument('--base_commit', required=True, help='Base commit hash to compare against')
    parser.add_argument('--chat_history_file', required=True, help='Path to chat history file')
    parser.add_argument('--outdir', required=False, default="/dgm/", help='Output directory')
    parser.add_argument('--test_description', default=None, required=False, help='Description of how to test the repository')
    parser.add_argument('--self_improve', default=False, action='store_true', help='Whether to self-improve the repository or solving swe')
    parser.add_argument('--instance_id', default=None, help='Instance ID for SWE issue')
    parser.add_argument('--model', required=False, default=OPENAI_MODEL, help='LLM model to use for processing')
    parser.add_argument('--timeout', type=int, default=3600, help='Timeout for LLM calls in seconds')
    parser.add_argument('--max_attempts', type=int, default=3, help='Maximum attempts to generate and evaluate patches')
    args = parser.parse_args()

    # Process the repository
    agentic_system = AgenticSystem(
        problem_statement=args.problem_statement,
        git_tempdir=args.git_dir,
        base_commit=args.base_commit,
        chat_history_file=args.chat_history_file,
        test_description=args.test_description,
        self_improve=args.self_improve,
        instance_id=args.instance_id,
        model=args.model
    )

    # Run the agentic system to try to solve the problem
    agentic_system.forward(args.timeout, max_attempts=args.max_attempts, outdir=args.outdir)

    # Get code diff and save to model_patch.diff
    model_patch = diff_versus_commit(args.git_dir, args.base_commit)
    model_patch_outfile = os.path.join(args.outdir, 'model_patch.diff') if args.outdir else 'model_patch.diff'
    with open(model_patch_outfile, 'w') as f:
        f.write(model_patch)

if __name__ == "__main__":
    main()
