import pytest

from nemo.export.utils._mock_import import _mock_import


def test_mock_import_existing_module():
    """Test mocking an existing module."""
    import math as math_org
    with _mock_import("math"):
        import math
        assert math is math_org

def test_mock_import_non_existing_module():
    """Test mocking a non-existing module."""
    with _mock_import("non.existing.module"):
        import non.existing.module

    with pytest.raises(ModuleNotFoundError):
        import non.existing.module
