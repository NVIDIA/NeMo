from abc import ABC, abstractmethod
from dataclasses import dataclass

import flask
import requests


@dataclass
class AdapterMetadata:
    """This is passed with the _whole_ chain from request back to response.

    Add here any useful information.
    """

    cache_hit: bool = False  # Whether there was a cache hit
    cache_key: str | None = None  # Cache key


@dataclass
class AdapterRequest:
    r: flask.Request
    meta: AdapterMetadata


@dataclass
class AdapterResponse:
    r: requests.Response
    meta: AdapterMetadata


class RequestInterceptor(ABC):
    """Interface for providing interception of requests."""

    @abstractmethod
    def intercept_request(
        self,
        req: AdapterRequest,
    ) -> AdapterRequest | AdapterResponse:
        """Function that will be called by `AdapterServer` on the way upstream.

        If the return type is `Request`, the chain will continue upstream.
        If the reruen type is `Response`, the `AdapterServer` will consider the
        chain finished and start the reverse chain of responses.

        Ex.: the latter case is e.g. how caching interceptor works. For cache miss
        it will continue the chain, passing request unchanged. For cache hit,
        it will go for the response.
        """
        pass


class ResponseInterceptor(ABC):
    """Interface for providing interception of responses."""

    @abstractmethod
    def intercept_response(self, ar: AdapterResponse) -> AdapterResponse:
        pass
