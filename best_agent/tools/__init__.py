from pathlib import Path
import importlib

def load_all_tools(logging=print):
    tools_dir = Path(__file__).parent
    tools = []
    
    # Get all Python files in the tools directory (excluding __init__.py)
    tool_files = [f for f in tools_dir.glob("*.py") if f.stem != "__init__"]
    
    for tool_file in tool_files:
        # Import the module
        module_name = f"tools.{tool_file.stem}"
        try:
            module = importlib.import_module(module_name)
            
            # Check if module has required functions
            if hasattr(module, 'tool_info') and hasattr(module, 'tool_function'):
                tools.append({
                    'info': module.tool_info(),
                    'function': module.tool_function,
                    'name': tool_file.stem
                })
            else:
                raise Exception(f"Tool module {module_name} does not have required functions.")
        except Exception as e:
            # Log the error and raise it
            logging(f"Failed to import {module_name}: {e}")
            raise e
    
    return tools
