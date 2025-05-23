import json

from flask import Request

from nemo.collections.llm.evaluation.adapters.interceptors.system_message_interceptor import SystemMessageInterceptor
from nemo.collections.llm.evaluation.adapters.interceptors.types import AdapterMetadata, AdapterRequest


def test_new_system_injected():
    # Test if the new system message is injected at the beginning

    system_message_interceptor = SystemMessageInterceptor(new_system_message="detailed thinking on")
    data = {
        "model": "gpt-4.1",
        "messages": [
            {"role": "system", "content": "you are a helpful assistant"},
            {"role": "user", "content": "Are semicolons optional in JavaScript?"},
        ],
        "max_tokens": 100,
        "temperature": 0.5,
    }
    request = Request.from_values(
        method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps(data),
    )
    adapter_request = AdapterRequest(
        r=request,
        meta=AdapterMetadata(),
    )
    adapter_response = system_message_interceptor.intercept_request(adapter_request)
    json_output = adapter_response.r.get_json()
    assert json_output["messages"][0]["content"] == "detailed thinking on"
