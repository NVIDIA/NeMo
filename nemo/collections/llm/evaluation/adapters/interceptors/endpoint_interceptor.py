from typing import final

import flask
import requests

from .types import AdapterRequest, AdapterResponse, RequestInterceptor


@final
class EndpointInterceptor(RequestInterceptor):
    api_url: str

    def __init__(self, api_url: str):
        self.api_url = api_url

    @final
    def intercept_request(self, ar: AdapterRequest) -> AdapterResponse:
        # This is a final interceptor, we'll need the flask_request and api
        resp = AdapterResponse(
            r=requests.request(
                method=ar.r.method,
                url=self.api_url,
                headers={k: v for k, v in ar.r.headers if k.lower() != "host"},
                json=ar.r.json,
                cookies=ar.r.cookies,
                allow_redirects=False,
            ),
            meta=ar.meta,
        )
        return resp
