# This file is adapted from https://github.com/jennyzzt/dgm.
import ast
import os
from typing import Any, Dict, List, Optional, Tuple


# Utility: graceful unparse with fallback
try:
    from ast import unparse as ast_unparse  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - fallback import path
    ast_unparse = None  # type: ignore

try:  # optional fallback if available in env
    import astor  # type: ignore
except Exception:  # pragma: no cover - astor may not be installed
    astor = None  # type: ignore


def _to_source(node: ast.AST) -> str:
    """Convert AST back to source using ast.unparse (py>=3.9) or astor if available."""
    if ast_unparse is not None:
        try:
            return ast_unparse(node)
        except Exception:
            pass
    if astor is not None:
        try:
            return astor.to_source(node)
        except Exception:
            pass
    # As an ultimate fallback, use ast.dump (not pretty but at least informative)
    return ast.dump(node, include_attributes=False)


def tool_info() -> Dict[str, Any]:
    return {
        "name": "ast_editor",
        "description": "Perform safe, AST-level edits to Python files (rename, insert, replace).",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File to edit."},
                "operation": {
                    "type": "string",
                    "enum": [
                        "read",
                        "rename_symbol",
                        "insert_import",
                        "replace_node",
                        "append_code",
                    ],
                    "description": "Which AST operation to perform.",
                },
                # rename_symbol
                "symbol": {
                    "type": "string",
                    "description": "Existing identifier (variable/function/class) to rename.",
                },
                "new_name": {
                    "type": "string",
                    "description": "New identifier name for rename_symbol.",
                },
                # insert_import
                "import_stmt": {
                    "type": "string",
                    "description": "Import statement to insert (e.g., 'import os' or 'from x import y').",
                },
                # replace_node
                "target_code": {
                    "type": "string",
                    "description": "Source snippet to locate and replace.",
                },
                "replacement_code": {
                    "type": "string",
                    "description": "Replacement source snippet or code to append.",
                },
                # Optional targeting by location (best-effort)
                "lineno": {
                    "type": "integer",
                    "description": "Line number of node to replace (optional).",
                },
                "col_offset": {
                    "type": "integer",
                    "description": "Column offset of node to replace (optional).",
                },
                # append_code context (optional)
                "function_name": {
                    "type": "string",
                    "description": "If provided, append inside this function; otherwise append at module end.",
                },
            },
            "required": ["path", "operation"],
        },
    }


class _RenameIdentifiers(ast.NodeTransformer):
    def __init__(self, old: str, new: str):
        self.old = old
        self.new = new

    def visit_Name(self, node: ast.Name) -> ast.AST:  # noqa: N802
        if node.id == self.old:
            return ast.copy_location(ast.Name(id=self.new, ctx=node.ctx), node)
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:  # noqa: N802
        # Recurse first
        self.generic_visit(node)
        # If attribute part matches, rename it (e.g., obj.old -> obj.new)
        if node.attr == self.old:
            node.attr = self.new
        return node

    def visit_arg(self, node: ast.arg) -> ast.AST:  # noqa: N802
        if node.arg == self.old:
            node.arg = self.new
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:  # noqa: N802
        if node.name == self.old:
            node.name = self.new
        self.generic_visit(node)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:  # noqa: N802
        if node.name == self.old:
            node.name = self.new
        self.generic_visit(node)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:  # noqa: N802
        if node.name == self.old:
            node.name = self.new
        self.generic_visit(node)
        return node


def _read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _write_file(path: str, src: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)


def _parse_module(src: str, path: str = "<unknown>") -> ast.Module:
    return ast.parse(src, filename=path)


def _module_to_source(module: ast.Module) -> str:
    # Ensure locations are set for unparse
    module = ast.fix_missing_locations(module)
    return _to_source(module)


def _parse_single_stmt(src: str) -> ast.stmt:
    """Parse a single statement from source string."""
    mod = ast.parse(src)
    if not mod.body:
        raise ValueError("No statements found in source snippet.")
    if len(mod.body) != 1:
        # allow multi-statement but pack into a single ast.stmt via ast.parse(...).body list
        # For replace/appends we can return a block by returning a list where needed
        return mod.body[0]
    return mod.body[0]


def _parse_stmts(src: str) -> List[ast.stmt]:
    mod = ast.parse(src)
    return list(mod.body)


def _import_exists(module: ast.Module, new_import: ast.AST) -> bool:
    for node in module.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)) and isinstance(new_import, type(node)):
            # Compare normalized source
            if _to_source(node).strip() == _to_source(new_import).strip():
                return True
    return False


def _prepend_import(module: ast.Module, import_node: ast.AST) -> ast.Module:
    # Place after any module docstring and future imports
    new_body: List[ast.stmt] = []
    idx = 0
    body = list(module.body)

    # Skip module docstring
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) and isinstance(body[0].value.value, str):
        new_body.append(body[0])
        idx = 1

    # Preserve leading __future__ imports ahead of other imports
    while idx < len(body):
        node = body[idx]
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            new_body.append(node)
            idx += 1
        else:
            break

    # Insert our import
    if isinstance(import_node, ast.stmt):
        new_body.append(import_node)
    else:
        raise ValueError("Parsed import is not a statement.")

    # Append the rest
    new_body.extend(body[idx:])
    module.body = new_body
    return module


def _find_node_by_location(module: ast.Module, lineno: Optional[int], col: Optional[int]) -> Optional[ast.AST]:
    if lineno is None and col is None:
        return None
    best: Optional[ast.AST] = None
    for node in ast.walk(module):
        if hasattr(node, "lineno"):
            n_line = getattr(node, "lineno", None)
            n_col = getattr(node, "col_offset", None)
            if lineno is not None and n_line != lineno:
                continue
            if col is not None and n_col != col:
                continue
            best = node
            break
    return best


def _find_first_matching_by_source(module: ast.Module, snippet: str) -> Optional[ast.AST]:
    snippet = snippet.strip()
    if not snippet:
        return None
    # Try to parse snippet; it might be an expression or statement(s)
    try:
        snippet_mod = ast.parse(snippet)
        cand_sources = [
            _to_source(n).strip() for n in snippet_mod.body
        ] or [snippet]
    except Exception:
        cand_sources = [snippet]

    for node in ast.walk(module):
        try:
            src = _to_source(node).strip()
        except Exception:
            continue
        if src in cand_sources or src == snippet:
            return node
    return None


def _replace_node_in_parent(module: ast.Module, target: ast.AST, replacement_nodes: List[ast.AST]) -> bool:
    """Replace target node in its parent with replacement_nodes. Returns True if replaced."""
    class ParentFinder(ast.NodeVisitor):
        def __init__(self, goal: ast.AST):
            self.goal = goal
            self.parent_stack: List[Tuple[ast.AST, Optional[str], Optional[int]]] = []
            self.found: Optional[Tuple[ast.AST, Optional[str], Optional[int]]] = None

        def generic_visit(self, node: ast.AST):
            for field, value in ast.iter_fields(node):
                if isinstance(value, list):
                    for idx, item in enumerate(value):
                        if item is self.goal:
                            self.found = (node, field, idx)
                            return
                elif isinstance(value, ast.AST):
                    if value is self.goal:
                        self.found = (node, field, None)
                        return
            super().generic_visit(node)

    finder = ParentFinder(target)
    finder.visit(module)
    if not finder.found:
        return False
    parent, field, idx = finder.found
    if field is None:
        return False

    container = getattr(parent, field)
    if isinstance(container, list):
        # Replace the single element at idx with replacement_nodes list
        if idx is None:
            return False
        new_list = list(container[:idx]) + replacement_nodes + list(container[idx + 1 :])
        setattr(parent, field, new_list)
        return True
    else:
        # Field is a single child; allow only single replacement
        if len(replacement_nodes) != 1:
            raise ValueError("Cannot replace a single AST field with multiple nodes.")
        setattr(parent, field, replacement_nodes[0])
        return True


def _append_into(module: ast.Module, code: str, function_name: Optional[str]) -> bool:
    nodes = _parse_stmts(code)
    if function_name:
        # Append at end of specified function body
        for node in module.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
                node.body.extend(nodes)
                return True
        return False
    else:
        # Append at module level
        module.body.extend(nodes)
        return True


def _pretty_file(path: str) -> str:
    try:
        src = _read_file(path)
    except Exception as e:
        return f"Error: {e}"
    # Try round-trip through AST for pretty format; fall back to raw
    try:
        mod = _parse_module(src, path)
        pretty = _module_to_source(mod)
        return pretty
    except Exception:
        return src


def tool_function(
    path: str,
    operation: str,
    symbol: Optional[str] = None,
    new_name: Optional[str] = None,
    import_stmt: Optional[str] = None,
    target_code: Optional[str] = None,
    replacement_code: Optional[str] = None,
    lineno: Optional[int] = None,
    col_offset: Optional[int] = None,
    function_name: Optional[str] = None,
):
    """Perform AST-level operations on a Python source file.

    Returns the pretty-printed updated file content, or an error string beginning with 'Error:'.
    """
    try:
        if not isinstance(path, str) or not path:
            return "Error: 'path' must be a non-empty string."
        if operation not in {"read", "rename_symbol", "insert_import", "replace_node", "append_code"}:
            return "Error: Unsupported operation."

        if operation == "read":
            return _read_file(path)

        # Load and parse the module
        try:
            src = _read_file(path)
        except FileNotFoundError:
            return f"Error: File not found: {path}"
        except Exception as e:
            return f"Error: {e}"

        try:
            module = _parse_module(src, path)
        except SyntaxError as e:
            return f"Error: Failed to parse {path}: {e}"

        if operation == "rename_symbol":
            if not symbol or not new_name:
                return "Error: 'symbol' and 'new_name' are required for rename_symbol."
            transformer = _RenameIdentifiers(symbol, new_name)
            module = transformer.visit(module)  # type: ignore
            updated = _module_to_source(module)
            _write_file(path, updated)
            return _pretty_file(path)

        if operation == "insert_import":
            if not import_stmt or not import_stmt.strip():
                return "Error: 'import_stmt' is required for insert_import."
            try:
                import_mod = ast.parse(import_stmt)
            except SyntaxError as e:
                return f"Error: Invalid import syntax: {e}"
            if not import_mod.body or not isinstance(import_mod.body[0], (ast.Import, ast.ImportFrom)):
                return "Error: 'import_stmt' must be an import statement."
            import_node = import_mod.body[0]
            if _import_exists(module, import_node):
                # No change needed
                return _module_to_source(module)
            module = _prepend_import(module, import_node)
            updated = _module_to_source(module)
            _write_file(path, updated)
            return _pretty_file(path)

        if operation == "replace_node":
            if not replacement_code:
                return "Error: 'replacement_code' is required for replace_node."
            replacement_nodes = _parse_stmts(replacement_code)

            target_node: Optional[ast.AST] = None
            # Try by location first if provided
            if lineno is not None or col_offset is not None:
                target_node = _find_node_by_location(module, lineno, col_offset)
            if target_node is None and target_code:
                target_node = _find_first_matching_by_source(module, target_code)
            if target_node is None:
                return "Error: Target node not found for replacement."

            ok = _replace_node_in_parent(module, target_node, replacement_nodes)
            if not ok:
                return "Error: Failed to replace the target node."
            updated = _module_to_source(module)
            _write_file(path, updated)
            return _pretty_file(path)

        if operation == "append_code":
            if not replacement_code:
                return "Error: 'replacement_code' (code to append) is required for append_code."
            appended = _append_into(module, replacement_code, function_name=function_name)
            if not appended:
                return "Error: Failed to append code (function not found?)."
            updated = _module_to_source(module)
            _write_file(path, updated)
            return _pretty_file(path)

        return "Error: Unknown operation."
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"
