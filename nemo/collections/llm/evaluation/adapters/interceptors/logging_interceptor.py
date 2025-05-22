import json
import os
import threading
from datetime import datetime
from typing import Optional, final

import requests
import structlog

from .types import AdapterRequest, AdapterResponse, RequestInterceptor, ResponseInterceptor


def _get_safe_headers(headers: dict[str, str]) -> dict[str, str]:
    """Create a copy of headers with authorization redacted."""
    safe_headers = dict(headers)
    for header in safe_headers:
        if header.lower() == 'authorization':
            safe_headers[header] = '[REDACTED]'
    return safe_headers


@final
class RequestLoggingInterceptor(RequestInterceptor):
    _logger: structlog.BoundLogger
    _max_requests: Optional[int]
    _logged_requests: int
    _lock: threading.Lock

    def __init__(self, output_dir: str, max_requests: Optional[int] = None, log_failed_requests: bool = False):
        """
        Initialize the request logging interceptor.

        Args:
            output_dir: Directory to store logs in
            max_requests: Maximum number of requests to log. If None, all requests will be logged.
            log_failed_requests: Whether to log failed request-response pairs (status code >= 400)
        """
        self._logger = structlog.get_logger(__name__)
        self._max_requests = max_requests
        self._logged_requests = 0
        self._lock = threading.Lock()

    def _log_request(self, ar: AdapterRequest) -> None:
        """Helper method to log request details"""
        try:
            # Try to parse as JSON first
            payload = ar.r.get_json()
            log_data = {
                "request": {
                    "method": ar.r.method,
                    "url": ar.r.url,
                    "headers": _get_safe_headers(ar.r.headers),
                    "payload": payload,
                }
            }
            self._logger.info("request_log", data=log_data)

        except Exception as e:
            # If JSON parsing fails, log raw data
            log_data = {
                "request": {
                    "method": ar.r.method,
                    "url": ar.r.url,
                    "headers": _get_safe_headers(ar.r.headers),
                    "raw_data": ar.r.get_data().decode('utf-8', errors='ignore'),
                }
            }
            self._logger.info("request_log", data=log_data)

    @final
    def intercept_request(self, ar: AdapterRequest) -> AdapterRequest:
        # Check if we should log this request
        if self._max_requests is not None and self._logged_requests >= self._max_requests:
            return ar

        self._log_request(ar)

        with self._lock:
            self._logged_requests += 1

        return ar


@final
class ResponseLoggingInterceptor(ResponseInterceptor):
    _logger: structlog.BoundLogger

    def __init__(self, use_response_logging: bool = True):
        """
        Initialize the response logging interceptor.

        Args:
            use_response_logging: Whether to log full responses to the console
        """
        self._logger = structlog.get_logger(__name__)
        self._use_response_logging = use_response_logging

    @final
    def intercept_response(self, ar: AdapterResponse) -> AdapterResponse:
        try:
            payload = ar.r.json()

            # Log to structured logger only if response logging is enabled
            if self._use_response_logging:
                log_data = {
                    "response": {"status_code": ar.r.status_code, "payload": payload, "cache_hit": ar.meta.cache_hit}
                }
                self._logger.info("response_log", data=log_data)

        except requests.exceptions.JSONDecodeError as e:
            # For non-JSON responses, only log if response logging is enabled
            if self._use_response_logging:
                log_data = {
                    "response": {
                        "status_code": ar.r.status_code,
                        "raw_content": ar.r.content.decode('utf-8', errors='ignore'),
                        "cache_hit": ar.meta.cache_hit,
                    }
                }
                self._logger.info("response_log", data=log_data)

        return ar
