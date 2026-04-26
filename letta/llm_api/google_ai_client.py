import threading
from typing import List, Optional, Tuple

import httpx
from google import genai
from google.genai.types import HttpOptions

from letta.errors import ErrorCode, LLMAuthenticationError, LLMError, LLMRateLimitError
from letta.llm_api.google_constants import GOOGLE_MODEL_FOR_API_KEY_CHECK
from letta.llm_api.google_vertex_client import GoogleVertexClient
from letta.log import get_logger
from letta.schemas.llm_config import LLMConfig
from letta.settings import model_settings

logger = get_logger(__name__)


class _KeyRotator:
    """Thread-safe round-robin index for Gemini API key selection."""

    def __init__(self) -> None:
        self._index = 0
        self._lock = threading.Lock()

    def next_start(self, num_keys: int) -> int:
        """Return the next starting index and advance the counter."""
        with self._lock:
            idx = self._index
            self._index = (self._index + 1) % num_keys
            return idx


_key_rotator = _KeyRotator()


class GoogleAIClient(GoogleVertexClient):
    provider_label = "Google AI"

    def _get_client(self, llm_config: Optional[LLMConfig] = None):
        timeout_ms = int(model_settings.gemini_timeout_seconds * 1000)
        api_key = None
        if llm_config:
            api_key, _, _ = self.get_byok_overrides(llm_config)
        if not api_key:
            api_key = model_settings.gemini_api_key
        return genai.Client(
            api_key=api_key,
            http_options=HttpOptions(timeout=timeout_ms),
        )

    async def _get_client_async(self, llm_config: Optional[LLMConfig] = None, api_key_override: Optional[str] = None):
        timeout_ms = int(model_settings.gemini_timeout_seconds * 1000)
        api_key = api_key_override
        if not api_key and llm_config:
            api_key, _, _ = await self.get_byok_overrides_async(llm_config)
        if not api_key:
            api_key = model_settings.gemini_api_key
        return genai.Client(
            api_key=api_key,
            http_options=HttpOptions(timeout=timeout_ms),
        )

    async def request_async(self, request_data: dict, llm_config: LLMConfig) -> dict:
        """Wraps GoogleVertexClient.request_async with automatic key rotation on 429."""
        keys = model_settings.all_gemini_api_keys
        if len(keys) <= 1:
            # Single-key path: delegate to parent (preserves all existing retry logic)
            return await super().request_async(request_data, llm_config)

        start = _key_rotator.next_start(len(keys))
        last_error: Optional[LLMRateLimitError] = None
        for attempt in range(len(keys)):
            key = keys[(start + attempt) % len(keys)]
            try:
                return await self._request_async_with_key(request_data, llm_config, key)
            except LLMRateLimitError as e:
                last_error = e
                logger.warning(
                    f"[GoogleAI] Key #{(start + attempt) % len(keys) + 1} rate limited (429), "
                    f"rotating to next key ({attempt + 1}/{len(keys)} tried)"
                )
        assert last_error is not None
        raise last_error

    async def _request_async_with_key(self, request_data: dict, llm_config: LLMConfig, api_key: str) -> dict:
        """Runs the parent request_async logic using the given API key.

        Temporarily patches _get_client_async so the parent's retry loop uses the
        specified key without needing to touch the parent implementation.
        """
        from google.genai import errors
        from letta.helpers.json_helpers import sanitize_unicode_surrogates
        import pydantic_core

        request_data = sanitize_unicode_surrogates(request_data)
        client = await self._get_client_async(llm_config, api_key_override=api_key)

        retry_count = 1
        should_retry = True
        response_data = None
        while should_retry and retry_count <= self.MAX_RETRIES:
            try:
                response = await client.aio.models.generate_content(
                    model=llm_config.model,
                    contents=request_data["contents"],
                    config=request_data["config"],
                )
            except pydantic_core._pydantic_core.ValidationError as e:
                from letta.errors import LLMBadRequestError
                raise LLMBadRequestError(
                    message=f"Invalid tool schema for {self._provider_name()}: {str(e)}",
                    code=ErrorCode.INTERNAL_SERVER_ERROR,
                )
            except errors.APIError as e:
                if e.code == 503 or e.code == 500 or e.code == 504:
                    logger.warning(f"Received {e}, retrying {retry_count}/{self.MAX_RETRIES}")
                    retry_count += 1
                    if retry_count > self.MAX_RETRIES:
                        raise self.handle_llm_error(e, llm_config=llm_config)
                    continue
                raise self.handle_llm_error(e, llm_config=llm_config)
            except Exception as e:
                raise self.handle_llm_error(e, llm_config=llm_config)

            response_data = response.model_dump()
            is_malformed = self.is_malformed_function_call(response_data)
            if is_malformed:
                logger.warning(
                    f"Received MALFORMED_FUNCTION_CALL for {llm_config.model}, "
                    f"retrying {retry_count}/{self.MAX_RETRIES}"
                )
            should_retry = is_malformed
            retry_count += 1

        if response_data is None:
            raise RuntimeError("Failed to get response data after all retries")
        return response_data

    async def stream_async(self, request_data: dict, llm_config: LLMConfig):
        """Wraps GoogleVertexClient.stream_async with automatic key rotation on 429."""
        keys = model_settings.all_gemini_api_keys
        if len(keys) <= 1:
            async for chunk in super().stream_async(request_data, llm_config):
                yield chunk
            return

        start = _key_rotator.next_start(len(keys))
        last_error: Optional[LLMRateLimitError] = None
        for attempt in range(len(keys)):
            key = keys[(start + attempt) % len(keys)]
            try:
                async for chunk in self._stream_async_with_key(request_data, llm_config, key):
                    yield chunk
                return  # stream completed successfully
            except LLMRateLimitError as e:
                last_error = e
                logger.warning(
                    f"[GoogleAI] Key #{(start + attempt) % len(keys) + 1} rate limited (429) during stream, "
                    f"rotating to next key ({attempt + 1}/{len(keys)} tried)"
                )
        assert last_error is not None
        raise last_error

    async def _stream_async_with_key(self, request_data: dict, llm_config: LLMConfig, api_key: str):
        """Runs stream_async logic using the given API key."""
        import json
        from google.genai import errors
        from letta.helpers.json_helpers import sanitize_unicode_surrogates
        import pydantic_core

        request_data = sanitize_unicode_surrogates(request_data)
        client = await self._get_client_async(llm_config, api_key_override=api_key)

        try:
            response = await client.aio.models.generate_content_stream(
                model=llm_config.model,
                contents=request_data["contents"],
                config=request_data["config"],
            )
        except pydantic_core._pydantic_core.ValidationError as e:
            from letta.errors import LLMBadRequestError
            raise LLMBadRequestError(
                message=f"Invalid tool schema for {self._provider_name()}: {str(e)}",
                code=ErrorCode.INTERNAL_SERVER_ERROR,
            )
        except errors.APIError as e:
            raise self.handle_llm_error(e, llm_config=llm_config)
        except Exception as e:
            logger.error(f"Error streaming {self._provider_name()} request: {e} with request data: {json.dumps(request_data)}")
            raise e

        try:
            async for chunk in response:
                yield chunk
        except errors.ClientError as e:
            if e.code == 499:
                logger.info(f"{self._provider_prefix()} Stream cancelled by client (499): {e}")
                return
            raise self.handle_llm_error(e, llm_config=llm_config)
        except errors.APIError as e:
            raise self.handle_llm_error(e, llm_config=llm_config)


def get_gemini_endpoint_and_headers(
    base_url: str, model: Optional[str], api_key: str, key_in_header: bool = True, generate_content: bool = False
) -> Tuple[str, dict]:
    """
    Dynamically generate the model endpoint and headers.
    """
    url = f"{base_url}/v1beta/models"

    # Add the model
    if model is not None:
        url += f"/{model}"

    # Add extension for generating content if we're hitting the LM
    if generate_content:
        url += ":generateContent"

    # Decide if api key should be in header or not
    # Two ways to pass the key: https://ai.google.dev/tutorials/setup
    if key_in_header:
        headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
    else:
        url += f"?key={api_key}"
        headers = {"Content-Type": "application/json"}

    return url, headers


async def google_ai_check_valid_api_key_async(api_key: str):
    """
    Async version to check if Google AI API key is valid without blocking the event loop.
    """
    client = genai.Client(api_key=api_key)
    # use the count token endpoint for a cheap model - as of 5/7/2025 this is slightly faster than fetching the list of models
    try:
        await client.aio.models.count_tokens(
            model=GOOGLE_MODEL_FOR_API_KEY_CHECK,
            contents="",
        )
    except genai.errors.ClientError as e:
        # google api returns 400 invalid argument for invalid api key
        if e.code == 400:
            raise LLMAuthenticationError(message=f"Failed to authenticate with Google AI: {e}", code=ErrorCode.UNAUTHENTICATED)
        raise e
    except Exception as e:
        raise LLMError(message=f"{e}", code=ErrorCode.INTERNAL_SERVER_ERROR)


async def google_ai_get_model_list_async(
    base_url: str, api_key: str, key_in_header: bool = True, client: Optional[httpx.AsyncClient] = None
) -> List[dict]:
    """Asynchronous version to get model list from Google AI API using httpx."""
    from letta.utils import printd

    url, headers = get_gemini_endpoint_and_headers(base_url, None, api_key, key_in_header)

    # Determine if we need to close the client at the end
    close_client = False
    if client is None:
        # Use explicit timeout to prevent httpx.ReadTimeout errors
        client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0))
        close_client = True

    try:
        response = await client.get(url, headers=headers)
        response.raise_for_status()  # Raises HTTPStatusError for 4XX/5XX status
        response_data = response.json()  # convert to dict from string

        # Grab the models out
        model_list = response_data["models"]
        return model_list

    except httpx.HTTPStatusError as http_err:
        # Handle HTTP errors (e.g., response 4XX, 5XX)
        printd(f"Got HTTPError, exception={http_err}")
        logger.error(f"HTTP Error: {http_err.response.status_code}, Message: {http_err.response.text}")
        raise http_err

    except httpx.RequestError as req_err:
        # Handle other httpx-related errors (e.g., connection error)
        printd(f"Got RequestException, exception={req_err}")
        raise req_err

    except Exception as e:
        # Handle other potential errors
        printd(f"Got unknown Exception, exception={e}")
        raise e

    finally:
        # Close the client if we created it
        if close_client:
            await client.aclose()


async def google_ai_get_model_details_async(
    base_url: str, api_key: str, model: str, key_in_header: bool = True, client: Optional[httpx.AsyncClient] = None
) -> dict:
    """Asynchronous version to get model details from Google AI API using httpx."""
    import httpx

    from letta.utils import printd

    url, headers = get_gemini_endpoint_and_headers(base_url, model, api_key, key_in_header)

    # Determine if we need to close the client at the end
    close_client = False
    if client is None:
        # Use explicit timeout to prevent httpx.ReadTimeout errors
        client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0))
        close_client = True

    try:
        response = await client.get(url, headers=headers)
        printd(f"response = {response}")
        response.raise_for_status()  # Raises HTTPStatusError for 4XX/5XX status
        response_data = response.json()  # convert to dict from string
        printd(f"response.json = {response_data}")

        # Return the model details
        return response_data

    except httpx.HTTPStatusError as http_err:
        # Handle HTTP errors (e.g., response 4XX, 5XX)
        printd(f"Got HTTPError, exception={http_err}")
        logger.error(f"HTTP Error: {http_err.response.status_code}, Message: {http_err.response.text}")
        raise http_err

    except httpx.RequestError as req_err:
        # Handle other httpx-related errors (e.g., connection error)
        printd(f"Got RequestException, exception={req_err}")
        raise req_err

    except Exception as e:
        # Handle other potential errors
        printd(f"Got unknown Exception, exception={e}")
        raise e

    finally:
        # Close the client if we created it
        if close_client:
            await client.aclose()


async def google_ai_get_model_context_window_async(base_url: str, api_key: str, model: str, key_in_header: bool = True) -> int:
    model_details = await google_ai_get_model_details_async(base_url=base_url, api_key=api_key, model=model, key_in_header=key_in_header)
    # TODO should this be:
    # return model_details["inputTokenLimit"] + model_details["outputTokenLimit"]
    return int(model_details["inputTokenLimit"])
