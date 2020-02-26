import pytest


def pytest_addoption(parser):
    parser.addoption("--cmdopt", action="store", default="type1", help="my option: type1 or type2")


@pytest.fixture
def cmdopt(request):
    return request.config.getoption("--cmdopt")
