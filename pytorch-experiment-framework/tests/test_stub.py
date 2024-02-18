import pytest

@pytest.mark.parametrize("text", ["Yes", "pytest", "works"])
def test_if_pytest_works(text: str):
    assert type(text) == str
