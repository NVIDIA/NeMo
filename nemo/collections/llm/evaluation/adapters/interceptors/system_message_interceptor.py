import json
from typing import Optional, final

from flask import Request

from .types import AdapterRequest, RequestInterceptor


@final
class SystemMessageInterceptor(RequestInterceptor):
    """Adapter for injecting a custom system message into the payload"""

    _new_system_message: str

    def __init__(self, new_system_message: Optional[str]):
        self._new_system_message = new_system_message

    @final
    def intercept_request(self, ar: AdapterRequest) -> AdapterRequest:
        new_data=json.dumps({
            "messages": [
                {"role": "system", "content": self._new_system_message},
                *json.loads(ar.r.get_data())["messages"]
            ],
            **{k:v for k,v in json.loads(ar.r.get_data()).items() if k != "messages"}
        })

        new_request = Request.from_values(
            path=ar.r.path,
            headers=dict(ar.r.headers),
            data=new_data,
            method=ar.r.method
        )
        return AdapterRequest(
            r=new_request,
            meta=ar.meta,
        )
