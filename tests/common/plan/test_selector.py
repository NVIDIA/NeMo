import pytest
import torch.nn as nn
import dataclasses
from typing import Optional
from nemo.common.plan.plan import Plan
from nemo.common.plan.selector import Selector, MatchContext


# Test fixtures
@pytest.fixture
def simple_model():
    """Create a simple PyTorch model for testing."""
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(10, 32)
            self.linear2 = nn.Linear(32, 32)
            self.attention = nn.MultiheadAttention(32, 4)
            self.mlp = nn.Sequential(
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Linear(64, 32)
            )
    return SimpleModel()


@pytest.fixture
def nested_model():
    """Create a nested PyTorch model for testing."""
    class NestedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(10, 32),
                nn.ReLU(),
                nn.Linear(32, 32)
            )
            self.decoder = nn.Sequential(
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, 10)
            )
            self.attention = nn.MultiheadAttention(32, 4)
    return NestedModel()


@pytest.fixture
def config_dataclass():
    """Create a dataclass for testing dataclass matching."""
    @dataclasses.dataclass(frozen=True)  # Make dataclass immutable and hashable
    class AttentionConfig:
        num_heads: int = 4
        dropout: float = 0.1
        hidden_size: int = 256

    @dataclasses.dataclass(frozen=True)  # Make dataclass immutable and hashable
    class ModelConfig:
        hidden_size: int = 256
        num_layers: int = 6
        attention: AttentionConfig = dataclasses.field(default_factory=AttentionConfig)
        dropout: float = 0.1

    return ModelConfig()


class TestSelector:
    def test_basic_module_matching(self, simple_model):
        """Test basic module matching with exact names."""
        class TestPlan(Plan):
            def execute(self, module):
                return module

        selector = Selector({
            "linear1": TestPlan(),
            "linear2": TestPlan()
        })

        # Check that matches are found
        matches = selector.get_matches(simple_model)
        assert "linear1" in matches
        assert "linear2" in matches
        assert len(matches) == 2

    def test_wildcard_module_matching(self, simple_model):
        """Test module matching with wildcards using fnmatch."""
        class TestPlan(Plan):
            def execute(self, module):
                return module

        selector = Selector({
            "linear*": TestPlan(),  # Match modules starting with 'linear'
            "mlp.*": TestPlan()      # Match children of the mlp module
        })

        # Check that matches are found
        matches = selector.get_matches(simple_model)
        
        # Check that linear1 and linear2 were matched by 'linear*'
        assert "linear1" in matches.keys()
        assert "linear2" in matches.keys()
        
        # Check that modules within mlp were matched by 'mlp.*' (e.g., mlp.0, mlp.1, mlp.2)
        assert any(path.startswith("mlp.") for path in matches.keys())

        # Ensure attention was not matched by these patterns
        assert "attention" not in matches.keys()

    def test_dataclass_matching(self, config_dataclass):
        """Test matching fields in dataclasses."""
        class TestPlan(Plan):
            def execute(self, value):
                return value

        selector = Selector({
            "*.hidden_size": TestPlan(),
            "attention.*": TestPlan()
        })

        # Check that matches are found
        matches = selector.get_matches(config_dataclass)
        assert any("hidden_size" in path for path in matches.keys())
        assert any("attention" in path for path in matches.keys())

    def test_nested_module_matching(self, nested_model):
        """Test matching in nested module structures."""
        class TestPlan(Plan):
            def execute(self, module):
                return module

        selector = Selector({
            "encoder.*": TestPlan(),
            "decoder.*": TestPlan(),
            "attention": TestPlan()
        })

        # Check that matches are found
        matches = selector.get_matches(nested_model)
        assert any("encoder" in path for path in matches.keys())
        assert any("decoder" in path for path in matches.keys())
        assert "attention" in matches

    def test_materialization_caching(self, simple_model):
        """Test that materialization results are cached."""
        class TestPlan(Plan):
            def execute(self, module):
                return module

        selector = Selector({
            "*.attention.*": TestPlan()
        })

        # First materialization
        materialized1 = selector.materialize(simple_model)
        
        # Second materialization should return the same object
        materialized2 = selector.materialize(simple_model)
        assert materialized1 is materialized2

    def test_context_passing(self, simple_model):
        """Test that context is passed to plans that accept it."""
        class ContextPlan(Plan):
            def execute(self, module, context: Optional[MatchContext] = None):
                assert context is not None
                assert context.path == "linear1"
                assert context.root is simple_model
                assert context.target_item is simple_model.linear1
                return module

        selector = Selector({
            "linear1": ContextPlan()
        })

        # Execute should not raise any assertions
        selector(simple_model)

    def test_no_context_plan(self, simple_model):
        """Test that plans without context parameter work correctly."""
        class NoContextPlan(Plan):
            def execute(self, module):
                return module

        selector = Selector({
            "linear1": NoContextPlan()
        })

        # Execute should work without context
        result = selector(simple_model)
        assert result is simple_model

    def test_multiple_patterns_same_module(self, simple_model):
        """Test that later patterns override earlier ones."""
        class Plan1(Plan):
            def execute(self, module):
                return "plan1"

        class Plan2(Plan):
            def execute(self, module):
                return "plan2"

        selector = Selector({
            "linear1": Plan1(),
            "linear1": Plan2()  # Should override Plan1
        })

        matches = selector.get_matches(simple_model)
        assert matches["linear1"] is selector._patterns["linear1"]
        assert isinstance(matches["linear1"], Plan2)

    def test_empty_selector(self, simple_model):
        """Test that an empty selector works correctly."""
        selector = Selector({})
        matches = selector.get_matches(simple_model)
        assert len(matches) == 0
        result = selector(simple_model)
        assert result is simple_model

    def test_invalid_pattern(self, simple_model):
        """Test that invalid patterns are handled gracefully."""
        class TestPlan(Plan):
            def execute(self, module):
                return module

        selector = Selector({
            "nonexistent_module": TestPlan()
        })

        # Should not raise an error, just skip the invalid pattern
        matches = selector.get_matches(simple_model)
        assert len(matches) == 0

    def test_nested_dataclass_matching(self, config_dataclass):
        """Test matching in nested dataclass structures."""
        class TestPlan(Plan):
            def execute(self, value):
                return value

        selector = Selector({
            "attention.num_heads": TestPlan(),
            "attention.dropout": TestPlan(),
            "attention.hidden_size": TestPlan()
        })

        matches = selector.get_matches(config_dataclass)
        assert "attention.num_heads" in matches
        assert "attention.dropout" in matches
        assert "attention.hidden_size" in matches
