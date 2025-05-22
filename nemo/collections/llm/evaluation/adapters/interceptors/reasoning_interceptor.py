import json
import re
from typing import final
import logging
import requests

from .types import AdapterResponse, ResponseInterceptor


def _clean_reasoning_tokens(
    response: requests.Response, end_reasoning_token: str
) -> requests.Response:
    """
    Clean up reasoning tokens from the response.

    Args:
        response: The API response object from requests
        end_reasoning_token: Token that marks the end of reasoning section

    Returns:
        Response with reasoning tokens removed
    """
    try:
        if "application/json" not in response.headers.get("Content-Type", ""):
            return response

        status_code = response.status_code
        headers = response.headers
        response_data = response.json()

        # Iterate over the choices and their messages
        for choice in response_data.get("choices", []):
            message = choice.get("message", {})

            if message.get("role") == "assistant":
                content = message.get("content", "")
                if not isinstance(content, str):
                    # particularily, content can be None with function calling
                    continue

                # Remove everything between start and end reasoning tokens
                # Also handle cases where only end token is present
                cleaned_content = re.sub(
                    r".*?" + re.escape(end_reasoning_token),
                    "",
                    content,
                    flags=re.DOTALL,
                ).strip("\n")

                # Update the content of the message with cleaned text
                message["content"] = cleaned_content

                # Log token information
                reasoning_tokens_info = {
                    "total_words": len(content.split()),
                    "cleaned_words": len(cleaned_content.split()),
                }
                print(f"reasoning_tokens_info {json.dumps(reasoning_tokens_info)}")

        modified_response = requests.Response()
        modified_response.status_code = status_code
        modified_response.headers = headers
        modified_response._content = json.dumps(response_data).encode("utf-8")
        return modified_response

    except (ValueError, json.JSONDecodeError) as e:
        # If not JSON or parsing fails, return original response
        print(f"Error parsing JSON response: {e}")
        return response
    except Exception as e:
        print(f"Error cleaning reasoning tokens: {e}")
        logging.exception(e)
        return response


@final
class ResponseReasoningInterceptor(ResponseInterceptor):

    _end_reasoning_token: str

    def __init__(self, end_reasoning_token: str):
        self._end_reasoning_token = end_reasoning_token

    @final
    def intercept_response(self, ar: AdapterResponse) -> AdapterResponse:
        return AdapterResponse(
            r=_clean_reasoning_tokens(ar.r, self._end_reasoning_token),
            meta=ar.meta,
        )
