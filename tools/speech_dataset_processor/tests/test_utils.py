import pytest

from utils.edit_spaces import remove_extra_spaces, add_start_end_spaces


@pytest.mark.parametrize(
    "input,expected_output",
    [
        ("abc xyz   abc xyz", "abc xyz abc xyz"),
        (" abc xyz ", "abc xyz")
    ]
)
def test_remove_extra_spaces(input, expected_output):
    assert remove_extra_spaces(input) == expected_output


@pytest.mark.parametrize(
    "input,expected_output",
    [
        ("abc", " abc "),
        ("abc xyz", " abc xyz ")
    ]
)
def test_add_start_end_spaces(input, expected_output):
    assert add_start_end_spaces(input) == expected_output

