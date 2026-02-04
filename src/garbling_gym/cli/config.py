# ABOUTME: Configuration loading from YAML and Python files
# ABOUTME: Validates and normalizes configs into GameConfig objects

import sys
import importlib.util
from pathlib import Path
from typing import Dict, Any, Union
import yaml

from ..core.config import GameConfig
from ..core.types import AssetQuality
from ..core.payoffs import PayoffStructure
from ..core.strategies import GarblingStrategy, strategy_registry


class ConfigLoader:
    """
    Load and validate experiment configurations from YAML or Python files.

    Supports two formats:
    - YAML: Simple declarative configuration
    - Python: Advanced configuration with custom strategies and logic
    """

    @staticmethod
    def load(path: Union[str, Path]) -> GameConfig:
        """
        Load configuration from file.

        Args:
            path: Path to config file (.yaml or .py)

        Returns:
            GameConfig object

        Raises:
            ValueError: If file format is unsupported
            FileNotFoundError: If file doesn't exist
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        if path.suffix in ['.yaml', '.yml']:
            return ConfigLoader._load_yaml(path)
        elif path.suffix == '.py':
            return ConfigLoader._load_python(path)
        else:
            raise ValueError(
                f"Unsupported config format: {path.suffix}. "
                "Use .yaml, .yml, or .py"
            )

    @staticmethod
    def _load_yaml(path: Path) -> GameConfig:
        """
        Load configuration from YAML file.

        YAML format:
        ```yaml
        name: experiment-name
        num_rounds: 20
        use_llm: false
        llm_model: gpt-4o-mini
        prior:
          LOW: 0.3
          MEDIUM: 0.4
          HIGH: 0.3
        ```

        Args:
            path: Path to YAML file

        Returns:
            GameConfig object
        """
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError("YAML config must be a dictionary")

        # Extract and validate fields
        num_rounds = data.get('num_rounds', 20)
        use_llm = data.get('use_llm', False)
        llm_model = data.get('llm_model', 'gpt-4o-mini')

        # Parse prior distribution
        prior = ConfigLoader._parse_prior(data.get('prior', {}))

        # Create config (payoffs use defaults for now)
        config = GameConfig(
            prior=prior,
            num_rounds=num_rounds,
            use_llm=use_llm,
            llm_model=llm_model
        )

        return config

    @staticmethod
    def _load_python(path: Path) -> GameConfig:
        """
        Load configuration from Python file.

        Python files must define a 'config' variable that is either:
        - A GameConfig instance
        - A dictionary with config parameters

        Python format allows custom strategies:
        ```python
        import numpy as np
        from garbling_gym.core.strategies import GarblingStrategy

        custom_strategy = GarblingStrategy(
            matrix=np.array([[...]]),
            name="Custom"
        )

        config = {
            "num_rounds": 50,
            "use_llm": False,
            # Can reference custom strategy
        }
        ```

        Args:
            path: Path to Python file

        Returns:
            GameConfig object
        """
        # Load Python module
        spec = importlib.util.spec_from_file_location("config_module", path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Could not load Python config from {path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module

        try:
            spec.loader.exec_module(module)
        except Exception as e:
            raise ValueError(f"Error executing Python config: {e}")

        # Extract 'config' variable
        if not hasattr(module, 'config'):
            raise ValueError(
                f"Python config must define a 'config' variable. "
                f"Found: {', '.join(dir(module))}"
            )

        config_data = module.config

        # If already a GameConfig, return it
        if isinstance(config_data, GameConfig):
            return config_data

        # If dict, convert to GameConfig
        if isinstance(config_data, dict):
            # Parse prior if present
            if 'prior' in config_data:
                config_data['prior'] = ConfigLoader._parse_prior(config_data['prior'])

            return GameConfig(**config_data)

        raise ValueError(
            f"Config must be GameConfig or dict, got {type(config_data)}"
        )

    @staticmethod
    def _parse_prior(prior_data: Dict) -> Dict[AssetQuality, float]:
        """
        Parse prior distribution from config data.

        Accepts either:
        - String keys: {"LOW": 0.3, "MEDIUM": 0.4, "HIGH": 0.3}
        - AssetQuality keys: {AssetQuality.LOW: 0.3, ...}

        Args:
            prior_data: Prior distribution dictionary

        Returns:
            Dictionary mapping AssetQuality to probabilities
        """
        if not prior_data:
            # Return default
            return {
                AssetQuality.LOW: 0.3,
                AssetQuality.MEDIUM: 0.4,
                AssetQuality.HIGH: 0.3,
            }

        result = {}
        for key, value in prior_data.items():
            if isinstance(key, str):
                # Convert string to enum
                try:
                    quality = AssetQuality[key.upper()]
                except KeyError:
                    raise ValueError(
                        f"Invalid quality '{key}'. Must be LOW, MEDIUM, or HIGH"
                    )
            elif isinstance(key, AssetQuality):
                quality = key
            else:
                raise ValueError(f"Prior keys must be strings or AssetQuality, got {type(key)}")

            result[quality] = float(value)

        return result


def merge_config_overrides(config: GameConfig, overrides: Dict[str, Any]) -> GameConfig:
    """
    Apply CLI overrides to a configuration.

    Args:
        config: Base configuration
        overrides: Dictionary of overrides (e.g., {"num_rounds": 50})

    Returns:
        New GameConfig with overrides applied
    """
    # Convert config to dict
    config_dict = {
        'prior': config.prior,
        'payoffs': config.payoffs,
        'num_rounds': config.num_rounds,
        'use_llm': config.use_llm,
        'llm_model': config.llm_model,
    }

    # Apply overrides
    for key, value in overrides.items():
        if key in config_dict:
            config_dict[key] = value

    return GameConfig(**config_dict)
