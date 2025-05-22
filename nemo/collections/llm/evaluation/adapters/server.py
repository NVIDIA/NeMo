"""Main server holding entry-point to all the interceptors.

The arch is as follows:


         ┌─────────────────────┐
         │                     │
         │  core-eval harness  │
         │                     │
         └───▲──────┬──────────┘
             │      │
     returns │      │
             │      │ calls
             │      │
             │      │
         ┌───┼──────┼──────────────────────────────────────────────────┐
         │   │      ▼                                                  │
         │ AdapterServer (@ localhost:3825)                            │
         │                                                             │
         │   ▲      │       chain of RequestInterceptors:              │
         │   │      │       flask.Request                              │
         │   │      │       is passed on the way up                    │
         │   │      │                                                  │   ┌──────────────────────┐
         │   │ ┌────▼───────────────────────────────────────────────┐  │   │                      │
         │   │ │intcptr_1─────►intcptr_2───►...───►intcptr_N────────┼──┼───►                      │
         │   │ │                     │                              │  │   │                      │
         │   │ └─────────────────────┼──────────────────────────────┘  │   │                      │
         │   │                       │(e.g. for caching interceptors,  │   │  upstream endpoint   │
         │   │                       │ this "shortcut" will happen)    │   │   with actual model  │
         │   │                       │                                 │   │                      │
         │   │                       └─────────────┐                   │   │                      │
         │   │                                     │                   │   │                      │
         │ ┌─┼─────────────────────────────────────▼────┐              │   │                      │
         │ │intcptr'_M◄──intcptr'_2◄──...◄───intcptr'_1 ◄──────────────┼───┤                      │
         │ └────────────────────────────────────────────┘              │   └──────────────────────┘
         │                                                             │
         │              Chain of ResponseInterceptors:                 │
         │              requests.Response is passed on the way down    │
         │                                                             │
         │                                                             │
         └─────────────────────────────────────────────────────────────┘

In other words, interceptors are pieces of independent logic which should be
relatively easy to add separately.



"""

import os

import flask
import requests
import werkzeug.serving

from nemo.collections.llm.evaluation.api import AdapterConfig
from nemo.utils import logging

from .interceptors import (
    AdapterMetadata,
    AdapterRequest,
    AdapterResponse,
    EndpointInterceptor,
    RequestInterceptor,
    RequestLoggingInterceptor,
    ResponseInterceptor,
    ResponseLoggingInterceptor,
    ResponseReasoningInterceptor,
    SystemMessageInterceptor,
)


class AdapterServer:
    """Main server which serves on a local port and holds chain of interceptors"""

    DEFAULT_ADAPTER_HOST: str = "localhost"
    DEFAULT_ADAPTER_PORT: int = 3825

    adapter_host: str
    adapter_port: int

    request_interceptors: list[RequestInterceptor]
    response_interceptors: list[ResponseInterceptor]

    app: flask.Flask

    api_url: str
    output_dir: str

    def __init__(
        self,
        api_url: str,
        output_dir: str,
        adapter_config: AdapterConfig,
    ):
        """
        Initializes the app, creates server and adds interceptors

        Args:
            adapter_config: should be obtained from the main framework config, see `adapter_config.py`
        """

        self.request_interceptors = []
        self.response_interceptors = []

        self.app = flask.Flask(__name__)
        self.app.route("/", defaults={"path": ""}, methods=["POST"])(self._handler)
        self.app.route("/<path:path>", methods=["POST"])(self._handler)

        self.adapter_host: str = os.environ.get("ADAPTER_HOST", self.DEFAULT_ADAPTER_HOST)
        self.adapter_port: int = int(os.environ.get("ADAPTER_PORT", self.DEFAULT_ADAPTER_PORT))

        self.api_url = api_url
        self.adapter_config = adapter_config
        self.output_dir = output_dir

        logging.info("Using the following adapter config: %s", adapter_config)

        self._build_interceptor_chains(
            use_reasoning=adapter_config.use_reasoning,
            end_reasoning_token=adapter_config.end_reasoning_token,
            use_custom_system_prompt=adapter_config.use_system_prompt,
            custom_system_prompt=adapter_config.custom_system_prompt,
        )

    def _build_interceptor_chains(
        self,
        use_reasoning: bool,
        end_reasoning_token: str,
        use_custom_system_prompt: bool,
        custom_system_prompt: str,
    ):

        if use_custom_system_prompt:
            self.request_interceptors.append(SystemMessageInterceptor(new_system_message=custom_system_prompt))

        self.request_interceptors.append(EndpointInterceptor(api_url=self.api_url))

        if use_reasoning:
            self.response_interceptors.append(ResponseReasoningInterceptor(end_reasoning_token=end_reasoning_token))

    def run(self) -> None:
        """Start the Flask server."""
        werkzeug.serving.run_simple(
            hostname=self.adapter_host,
            port=self.adapter_port,
            application=self.app,
            threaded=True,
        )

    # The headers we don't want to let out
    _EXCLUDED_HEADERS = [
        "content-encoding",
        "content-length",
        "transfer-encoding",
        "connection",
    ]

    @classmethod
    def _process_response_headers(cls, response: requests.Response) -> list[tuple[str, str]]:
        """Process response headers, removing excluded ones."""
        return [(k, v) for k, v in response.headers.items() if k.lower() not in cls._EXCLUDED_HEADERS]

    def _handler(self, path: str) -> flask.Response:
        adapter_request = AdapterRequest(
            r=flask.request,
            meta=AdapterMetadata(),
        )
        adapter_response = None
        for interceptor in self.request_interceptors:
            output = interceptor.intercept_request(adapter_request)

            if isinstance(output, AdapterResponse):
                adapter_response = output
                break
            if isinstance(output, AdapterRequest):
                adapter_request = output

        # TODO(agronskiy): asserts in prod are bad, make this more elegant.
        assert adapter_response is not None, "There should be a response to process"

        for interceptor in self.response_interceptors:
            try:
                adapter_response = interceptor.intercept_response(adapter_response)
            except Exception as e:
                self._log_response_interceptor_error(interceptor, adapter_response, e)
                raise

        return flask.Response(
            response=adapter_response.r.content,
            status=adapter_response.r.status_code,
            headers=self._process_response_headers(adapter_response.r),
        )

    def _log_response_interceptor_error(
        self,
        interceptor: ResponseInterceptor,
        adapter_response: AdapterResponse,
        error: Exception,
    ) -> None:
        error_message = (
            f"❌ Error in Response Interceptor ❌\n"
            f"Interceptor: {interceptor.__class__.__name__}\n"
            f"Adapter Response Status Code: {adapter_response.r.status_code}\n"
            f"Adapter Response Status Text: {adapter_response.r.reason}\n"
            f"Adapter Response Content: {adapter_response.r.content.decode('utf-8', errors='ignore')}\n"
            f"Error Details: {repr(error)}\n"
        )
        logging.error(error_message)

