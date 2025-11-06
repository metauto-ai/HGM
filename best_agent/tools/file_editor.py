import os
from typing import Optional

from utils.common_utils import read_file


def tool_info():
    """
    Metadata for the file_editor tool.

    Operations supported:
    - read: return the full contents of a file
    - write: overwrite a file with provided content
    - replace: search & replace a substring in a file
    """
    return {
        "name": "file_editor",
        "description": "Read, overwrite, or replace file content using simple Python I/O (no shell).",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to operate on."
                },
                "operation": {
                    "type": "string",
                    "enum": ["read", "write", "replace"],
                    "description": "Which operation to perform: read | write | replace."
                },
                "content": {
                    "type": "string",
                    "description": "Content to write (required for write)."
                },
                "target": {
                    "type": "string",
                    "description": "Substring to search for (required for replace)."
                },
                "replacement": {
                    "type": "string",
                    "description": "Replacement string (required for replace)."
                }
            },
            # Path and operation are always required. Other fields depend on the operation
            # and are validated at runtime.
            "required": ["path", "operation"]
        }
    }


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)



def tool_function(
    path: str,
    operation: str,
    content: Optional[str] = None,
    target: Optional[str] = None,
    replacement: Optional[str] = None,
):
    """
    Perform file operations based on the provided arguments.

    Returns:
      - For read: the file contents as a string.
      - For write: the content that was written.
      - For replace: a status message describing the result.
    """
    try:
        if not isinstance(path, str) or not path:
            return "Error: 'path' must be a non-empty string."

        if operation not in {"read", "write", "replace"}:
            return "Error: 'operation' must be one of: read, write, replace."

        if operation == "read":
            # Use the shared utility function for reading
            return read_file(path)

        if operation == "write":
            if content is None:
                return "Error: 'content' is required for write operation."
            _ensure_parent_dir(path)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return content

        if operation == "replace":
            if target is None or replacement is None:
                return "Error: 'target' and 'replacement' are required for replace operation."
            # Read current content
            try:
                current = read_file(path)
            except FileNotFoundError:
                return f"Error: File not found: {path}"

            occurrences = current.count(target)
            if occurrences == 0:
                return "No occurrences found. No changes made."

            updated = current.replace(target, replacement)
            with open(path, "w", encoding="utf-8") as f:
                f.write(updated)
            return f"Replaced {occurrences} occurrence(s)."

        return "Error: Unsupported operation."
    except Exception as e:
        return f"Error: {str(e)}"
