# This file is adapted from https://github.com/jennyzzt/dgm.

import functools
import json
import threading


def read_file(file_path):
    """
    Read a file and return its contents as a string.
    """
    with open(file_path, "r") as f:
        content = f.read().strip()
    return content


def load_json_file(file_path):
    """
    Load a JSON file and return its contents as a dictionary.
    """
    with open(file_path, "r") as file:
        return json.load(file)


class ExecRunTimeoutError(Exception):
    pass


def exec_with_timeout(timeout=60):
    """
    Decorator to wrap a function that calls container.exec_run(),
    ensuring each call has a timeout.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            def run_fn():
                try:
                    result_holder["result"] = func(*args, **kwargs)
                except Exception as e:
                    result_holder["error"] = e

            result_holder = {}
            thread = threading.Thread(target=run_fn)
            thread.start()
            thread.join(timeout)

            if thread.is_alive():
                raise ExecRunTimeoutError(
                    f"[Timeout] function {func.__name__} exceeded {timeout} seconds."
                )

            if "error" in result_holder:
                raise result_holder["error"]

            return result_holder["result"]

        return wrapper

    return decorator


import time


@exec_with_timeout(5)
def mytest():
    print("start")
    for i in range(1, 7):
        time.sleep(1)
        print(f"{i} seconds have passed")
