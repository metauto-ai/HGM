"""
This module provides functionality to load and manage configuration parameters
from YAML files with fallback to default values.
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class LLMConfig:
    """Configuration for Language Model settings."""
    self_improve_llm: str = "gpt-5-mini"
    downstream_llm: str = "gpt-5-mini"
    diagnose_llm: str = "gpt-5-mini"


@dataclass
class OptimizationConfig:
    """Configuration for optimization algorithm parameters."""
    alpha: float = 0.6
    beta: float = 1.0
    cool_down: bool = False
    eval_random_level: float = 1.0
    n_pseudo_descendant_evals: int = 10000


@dataclass
class ExecutionConfig:
    """Configuration for execution and resource management."""
    max_workers: int = 16
    self_improve_timeout: int = 3600
    evaluation_timeout: int = 3600
    max_task_evals: int = 800


@dataclass
class EvaluationConfig:
    """Configuration for evaluation settings."""
    full_eval: bool = False
    polyglot: bool = False


@dataclass
class PathConfig:
    """Configuration for file paths and directories."""
    output_dir: Optional[str] = None
    continue_from: Optional[str] = None
    initial_agent_name: str = ""


@dataclass
class HGMConfig:
    """Main configuration class containing all HGM settings."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    paths: PathConfig = field(default_factory=PathConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'HGMConfig':
        """
        Load configuration from a YAML file.
        
        Args:
            yaml_path: Path to the YAML configuration file
            
        Returns:
            HGMConfig instance with loaded configuration
        """
        if not os.path.exists(yaml_path):
            print(f"Warning: Configuration file {yaml_path} not found. Using defaults.")
            return cls()
        
        try:
            with open(yaml_path, 'r') as f:
                config_data = yaml.safe_load(f) or {}
            
            # Create instances with loaded data
            llm_config = LLMConfig(**config_data.get('llm', {}))
            optimization_config = OptimizationConfig(**config_data.get('optimization', {}))
            execution_config = ExecutionConfig(**config_data.get('execution', {}))
            evaluation_config = EvaluationConfig(**config_data.get('evaluation', {}))
            paths_config = PathConfig(**config_data.get('paths', {}))
            
            return cls(
                llm=llm_config,
                optimization=optimization_config,
                execution=execution_config,
                evaluation=evaluation_config,
                paths=paths_config
            )
        except Exception as e:
            print(f"Error loading configuration from {yaml_path}: {e}")
            print("Using default configuration.")
            return cls()
    
    @classmethod
    def from_yaml_with_overrides(cls, yaml_path: str, **overrides) -> 'HGMConfig':
        """
        Load configuration from YAML and apply command-line overrides.
        
        Args:
            yaml_path: Path to the YAML configuration file
            **overrides: Key-value pairs to override configuration values
            
        Returns:
            HGMConfig instance with loaded and overridden configuration
        """
        config = cls.from_yaml(yaml_path)
        
        # Apply overrides using dot notation (e.g., "llm.self_improve_llm")
        for key, value in overrides.items():
            if value is not None:  # Only override if value is provided
                config._set_nested_attr(key, value)
        
        return config
    
    def _set_nested_attr(self, attr_path: str, value: Any):
        """
        Set a nested attribute using dot notation.
        
        Args:
            attr_path: Dot-separated attribute path (e.g., "llm.self_improve_llm")
            value: Value to set
        """
        parts = attr_path.split('.')
        obj = self
        
        # Navigate to the parent object
        for part in parts[:-1]:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                return  # Invalid path, skip
        
        # Set the final attribute
        final_attr = parts[-1]
        if hasattr(obj, final_attr):
            setattr(obj, final_attr, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary format.
        
        Returns:
            Dictionary representation of the configuration
        """
        return {
            'llm': {
                'self_improve_llm': self.llm.self_improve_llm,
                'downstream_llm': self.llm.downstream_llm,
                'diagnose_llm': self.llm.diagnose_llm,
            },
            'optimization': {
                'alpha': self.optimization.alpha,
                'beta': self.optimization.beta,
                'cool_down': self.optimization.cool_down,
                'eval_random_level': self.optimization.eval_random_level,
                'n_pseudo_descendant_evals': self.optimization.n_pseudo_descendant_evals,
            },
            'execution': {
                'max_workers': self.execution.max_workers,
                'self_improve_timeout': self.execution.self_improve_timeout,
                'evaluation_timeout': self.execution.evaluation_timeout,
                'max_task_evals': self.execution.max_task_evals,
            },
            'evaluation': {
                'full_eval': self.evaluation.full_eval,
                'polyglot': self.evaluation.polyglot,
            },
            'paths': {
                'output_dir': self.paths.output_dir,
                'continue_from': self.paths.continue_from,
                'initial_agent_name': self.paths.initial_agent_name,
            }
        }
    
    def save_to_yaml(self, yaml_path: str):
        """
        Save configuration to a YAML file.
        
        Args:
            yaml_path: Path where to save the YAML configuration file
        """
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)


def load_config(config_path: str = "config.yaml", **overrides) -> HGMConfig:
    """
    Convenience function to load configuration with overrides.
    
    Args:
        config_path: Path to the configuration file
        **overrides: Command-line or programmatic overrides
        
    Returns:
        HGMConfig instance
    """
    return HGMConfig.from_yaml_with_overrides(config_path, **overrides)