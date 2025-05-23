from typing import Any, Generator

import pytest

from .test_utils import create_fake_endpoint_process


@pytest.fixture
def fake_openai_endpoint() -> Generator[Any, Any, Any]:
    """Fixture to create a Flask app with an OpenAI response.

    Being a "proper" fake endpoint, it responds with a payload which can be
    set via app.config.response.
    """
    # Create and run the fake endpoint server
    p = create_fake_endpoint_process()

    yield p  # We only need the process reference for cleanup

    p.terminate()
