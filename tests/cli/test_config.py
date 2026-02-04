"""
Unit tests for configuration loading.

Tests YAML and Python config loading, validation, and overrides.
"""

import pytest
import tempfile
from pathlib import Path

from garbling_gym.cli.config import ConfigLoader, merge_config_overrides
from garbling_gym.core.config import GameConfig
from garbling_gym.core.types import AssetQuality


class TestConfigLoader:
    """Test the ConfigLoader class"""

    def test_load_nonexistent_file(self):
        """Test that loading non-existent file raises FileNotFoundError"""
        with pytest.raises(FileNotFoundError):
            ConfigLoader.load("nonexistent.yaml")

    def test_load_unsupported_format(self):
        """Test that unsupported file format raises ValueError"""
        with tempfile.NamedTemporaryFile(suffix=".txt") as f:
            with pytest.raises(ValueError, match="Unsupported config format"):
                ConfigLoader.load(f.name)

    def test_load_yaml_basic(self):
        """Test loading basic YAML configuration"""
        yaml_content = """
num_rounds: 30
use_llm: false
llm_model: gpt-4o

prior:
  LOW: 0.2
  MEDIUM: 0.5
  HIGH: 0.3
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            try:
                config = ConfigLoader.load(f.name)

                assert config.num_rounds == 30
                assert config.use_llm is False
                assert config.llm_model == "gpt-4o"
                assert config.prior[AssetQuality.LOW] == 0.2
                assert config.prior[AssetQuality.MEDIUM] == 0.5
                assert config.prior[AssetQuality.HIGH] == 0.3
            finally:
                Path(f.name).unlink()

    def test_load_yaml_minimal(self):
        """Test loading YAML with minimal fields (uses defaults)"""
        yaml_content = """
num_rounds: 10
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            try:
                config = ConfigLoader.load(f.name)

                assert config.num_rounds == 10
                assert config.use_llm is False
                # Should have default prior
                assert sum(config.prior.values()) == pytest.approx(1.0)
            finally:
                Path(f.name).unlink()

    def test_load_yaml_invalid_prior_quality(self):
        """Test that invalid quality name raises ValueError"""
        yaml_content = """
prior:
  LOW: 0.3
  INVALID: 0.4
  HIGH: 0.3
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            try:
                with pytest.raises(ValueError, match="Invalid quality"):
                    ConfigLoader.load(f.name)
            finally:
                Path(f.name).unlink()

    def test_load_python_with_dict(self):
        """Test loading Python config with dictionary"""
        py_content = """
config = {
    'num_rounds': 50,
    'use_llm': True,
    'llm_model': 'gpt-4o-mini',
}
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix=".py", delete=False) as f:
            f.write(py_content)
            f.flush()

            try:
                config = ConfigLoader.load(f.name)

                assert config.num_rounds == 50
                assert config.use_llm is True
                assert config.llm_model == "gpt-4o-mini"
            finally:
                Path(f.name).unlink()

    def test_load_python_with_gameconfig(self):
        """Test loading Python config with GameConfig instance"""
        py_content = """
from garbling_gym.core.config import GameConfig
from garbling_gym.core.types import AssetQuality

config = GameConfig(
    num_rounds=100,
    use_llm=False,
    prior={
        AssetQuality.LOW: 0.1,
        AssetQuality.MEDIUM: 0.8,
        AssetQuality.HIGH: 0.1,
    }
)
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix=".py", delete=False) as f:
            f.write(py_content)
            f.flush()

            try:
                config = ConfigLoader.load(f.name)

                assert config.num_rounds == 100
                assert config.prior[AssetQuality.MEDIUM] == 0.8
            finally:
                Path(f.name).unlink()

    def test_load_python_missing_config_variable(self):
        """Test that Python file without 'config' variable raises ValueError"""
        py_content = """
# No config variable defined
some_other_var = 42
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix=".py", delete=False) as f:
            f.write(py_content)
            f.flush()

            try:
                with pytest.raises(ValueError, match="must define a 'config' variable"):
                    ConfigLoader.load(f.name)
            finally:
                Path(f.name).unlink()

    def test_load_python_with_syntax_error(self):
        """Test that Python file with syntax error raises ValueError"""
        py_content = """
config = {
    'num_rounds': 50
    # Missing comma and closing brace
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix=".py", delete=False) as f:
            f.write(py_content)
            f.flush()

            try:
                with pytest.raises(ValueError, match="Error executing"):
                    ConfigLoader.load(f.name)
            finally:
                Path(f.name).unlink()

    def test_load_builtin_yaml_configs(self):
        """Test that built-in YAML configs load correctly"""
        baseline_path = Path("src/garbling_gym/experiments/yaml/baseline.yaml")

        if baseline_path.exists():
            config = ConfigLoader.load(baseline_path)
            assert isinstance(config, GameConfig)
            assert config.num_rounds == 20

    def test_load_builtin_python_configs(self):
        """Test that built-in Python configs load correctly"""
        custom_path = Path("src/garbling_gym/experiments/python/custom_strategy.py")

        if custom_path.exists():
            config = ConfigLoader.load(custom_path)
            assert isinstance(config, GameConfig)
            assert config.num_rounds == 30


class TestMergeConfigOverrides:
    """Test configuration override merging"""

    def test_merge_num_rounds(self):
        """Test overriding num_rounds"""
        base_config = GameConfig(num_rounds=20)
        overrides = {"num_rounds": 50}

        merged = merge_config_overrides(base_config, overrides)

        assert merged.num_rounds == 50
        # Other fields unchanged
        assert merged.use_llm == base_config.use_llm

    def test_merge_multiple_overrides(self):
        """Test overriding multiple fields"""
        base_config = GameConfig(num_rounds=20, use_llm=False)
        overrides = {"num_rounds": 100, "use_llm": True, "llm_model": "gpt-4o"}

        merged = merge_config_overrides(base_config, overrides)

        assert merged.num_rounds == 100
        assert merged.use_llm is True
        assert merged.llm_model == "gpt-4o"

    def test_merge_empty_overrides(self):
        """Test that empty overrides returns equivalent config"""
        base_config = GameConfig(num_rounds=20)
        overrides = {}

        merged = merge_config_overrides(base_config, overrides)

        assert merged.num_rounds == base_config.num_rounds
        assert merged.use_llm == base_config.use_llm

    def test_merge_invalid_override_ignored(self):
        """Test that invalid override keys are ignored"""
        base_config = GameConfig(num_rounds=20)
        overrides = {"invalid_key": 42, "num_rounds": 30}

        # Should not raise, just ignore invalid key
        merged = merge_config_overrides(base_config, overrides)
        assert merged.num_rounds == 30
