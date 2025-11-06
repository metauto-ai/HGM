import os
import io
import sys
import traceback
from multiprocessing import Process, Queue
from typing import Optional


MAX_OUTPUT_CHARS = 4000


def tool_info():
    """
    Metadata for the python_exec tool.
    """
    return {
        "name": "python_exec",
        "description": "Run a Python snippet in the repo environment and capture its output.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Optional working directory.",
                    "default": ""
                },
                "code": {
                    "type": "string",
                    "description": "Python code to run."
                },
                "timeout": {
                    "type": "integer",
                    "description": "Max seconds to run.",
                    "default": 60
                }
            },
            "required": ["code"]
        }
    }


def _execute_snippet(code: str, path: Optional[str], q: Queue) -> None:
    """
    Child process target to execute Python code and capture stdout/stderr.
    Puts a dict onto the queue with keys {ok: bool, output: str} or {ok: False, error: str}.
    """
    try:
        if path:
            os.chdir(path)

        buf_out = io.StringIO()
        buf_err = io.StringIO()
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = buf_out, buf_err
        try:
            glb = {"__name__": "__main__"}
            loc = {}
            try:
                exec(code, glb, loc)
            except SystemExit as e:
                print(f"SystemExit: {e}", file=sys.stderr)
            except Exception:  # noqa: E722 - we want full traceback
                traceback.print_exc()
        finally:
            # Restore stdio
            sys.stdout, sys.stderr = old_out, old_err

        out = buf_out.getvalue()
        err = buf_err.getvalue()

        combined = out.strip()
        if err:
            if combined:
                combined += "\n"
            combined += "Error:\n" + err.strip()

        q.put({"ok": True, "output": combined})
    except Exception as ex:  # Fail-safe capture of unexpected errors in child
        q.put({"ok": False, "error": f"Error: {type(ex).__name__}: {ex}"})


def _truncate_output(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    # Keep head and indicate truncation
    truncated_len = len(text) - (limit - 100)
    return text[: limit - 100] + f"\n...[truncated {truncated_len} chars]"


def tool_function(code: str, path: Optional[str] = "", timeout: Optional[int] = 60):
    """
    Execute the provided Python code string optionally in a given working directory.

    Returns stdout and stderr (labeled as Error) combined, truncated to a safe length.
    On timeout or failure, returns a string beginning with "Error:".
    """
    try:
        if not isinstance(code, str) or code.strip() == "":
            return "Error: 'code' must be a non-empty string."
        if path is None:
            path = ""
        if timeout is None:
            timeout = 60
        try:
            timeout = int(timeout)
            if timeout <= 0:
                timeout = 60
        except Exception:
            timeout = 60

        q: Queue = Queue()
        p = Process(target=_execute_snippet, args=(code, path, q))
        p.daemon = True
        p.start()
        p.join(timeout)
        if p.is_alive():
            p.terminate()
            p.join(1)
            return f"Error: Timed out after {timeout} seconds."

        # Retrieve result from queue, if available
        try:
            result = q.get_nowait()
        except Exception:
            # If the child crashed before putting result
            if p.exitcode is not None and p.exitcode != 0:
                return f"Error: Python process exited with code {p.exitcode}."
            return "Error: No output captured."

        if not result.get("ok"):
            return result.get("error", "Error: Unknown failure.")

        output = result.get("output", "")
        return _truncate_output(output)
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}" 
