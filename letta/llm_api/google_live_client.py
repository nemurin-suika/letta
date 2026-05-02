"""
Google Gemini Live API client using bidiGenerateContent (WebSocket).

Supports text-only input/output. Audio output parts are silently ignored.
Each request_async / stream_async call opens a fresh WebSocket session,
which matches Letta's turn-by-turn agent loop model.

Error handling strategy:
  - LLMRateLimitError / LLMServerError per key → rotate to next key
  - All keys exhausted → fallback to `claude -p` subprocess
"""

import asyncio
import json
import random
import shutil
from typing import AsyncIterator, Optional

from letta.errors import ErrorCode, LLMError, LLMRateLimitError, LLMServerError
from letta.llm_api.google_ai_client import GoogleAIClient
from letta.log import get_logger
from letta.schemas.llm_config import LLMConfig
from letta.settings import model_settings

logger = get_logger(__name__)

# Errors that warrant trying the next API key
_ROTATABLE_ERRORS = (LLMRateLimitError, LLMServerError)


class GoogleLiveClient(GoogleAIClient):
    """
    LLM client for Google Gemini Live API (bidiGenerateContent over WebSocket).
    Inherits request building and response parsing from GoogleAIClient /
    GoogleVertexClient; only the transport layer is replaced.
    """

    provider_label = "Google Live"

    # ------------------------------------------------------------------ #
    #  Public interface
    # ------------------------------------------------------------------ #

    async def request_async(self, request_data: dict, llm_config: LLMConfig) -> dict:
        keys = model_settings.all_gemini_api_keys or (
            [model_settings.gemini_api_key] if model_settings.gemini_api_key else []
        )
        shuffled = random.sample(keys, len(keys)) if keys else []

        last_error: Optional[Exception] = None
        for attempt, key in enumerate(shuffled):
            try:
                return await self._request_live_async(request_data, llm_config, key)
            except _ROTATABLE_ERRORS as e:
                last_error = e
                logger.warning(
                    f"[GoogleLive] Key #{attempt + 1} failed ({type(e).__name__}), "
                    f"rotating ({attempt + 1}/{len(shuffled)} tried)"
                )

        # All Gemini keys exhausted — fall back to claude -p
        if last_error is not None:
            logger.warning(
                f"[GoogleLive] All {len(shuffled)} key(s) failed. "
                "Falling back to claude -p"
            )
        else:
            logger.warning("[GoogleLive] No API keys configured. Falling back to claude -p")

        return await self._fallback_to_claude_cli(request_data)

    async def stream_async(
        self, request_data: dict, llm_config: LLMConfig
    ) -> AsyncIterator:
        """
        Yield GenerateContentResponse chunks from the Live API stream.
        Text parts are yielded as they arrive; function calls are yielded
        as a single final chunk once the turn is complete.
        Falls back to claude -p (as a single text chunk) if all keys fail.
        """
        from google.genai.types import (
            Candidate,
            Content,
            FinishReason,
            GenerateContentResponse,
            Part,
        )

        keys = model_settings.all_gemini_api_keys or (
            [model_settings.gemini_api_key] if model_settings.gemini_api_key else []
        )
        shuffled = random.sample(keys, len(keys)) if keys else []

        last_error: Optional[Exception] = None
        for attempt, key in enumerate(shuffled):
            try:
                async for chunk in self._stream_live_async(request_data, llm_config, key):
                    yield chunk
                return  # success
            except _ROTATABLE_ERRORS as e:
                last_error = e
                logger.warning(
                    f"[GoogleLive] Stream key #{attempt + 1} failed ({type(e).__name__}), "
                    f"rotating ({attempt + 1}/{len(shuffled)} tried)"
                )

        # All keys exhausted — yield Claude CLI response as single chunk
        if last_error is not None:
            logger.warning(
                f"[GoogleLive] All {len(shuffled)} stream key(s) failed. "
                "Falling back to claude -p"
            )
        else:
            logger.warning("[GoogleLive] No API keys configured. Falling back to claude -p")

        cli_response = await self._fallback_to_claude_cli(request_data)
        text = cli_response["candidates"][0]["content"]["parts"][0].get("text", "")
        yield GenerateContentResponse(
            candidates=[
                Candidate(
                    content=Content(
                        role="model",
                        parts=[Part(text=text)],
                    ),
                    finish_reason=FinishReason.STOP,
                    index=0,
                )
            ],
        )

    # ------------------------------------------------------------------ #
    #  Internal: Live API per-key helpers
    # ------------------------------------------------------------------ #

    async def _request_live_async(
        self, request_data: dict, llm_config: LLMConfig, api_key: str
    ) -> dict:
        from google import genai
        from google.genai import errors as genai_errors
        from google.genai.types import HttpOptions

        timeout_ms = int(model_settings.gemini_timeout_seconds * 1000)
        client = genai.Client(
            api_key=api_key,
            http_options=HttpOptions(timeout=timeout_ms),
        )
        live_config = self._build_live_config(request_data["config"])
        contents = request_data["contents"]

        text_parts: list[str] = []
        function_calls = []
        usage_metadata = None

        try:
            async with client.aio.live.connect(
                model=llm_config.model, config=live_config
            ) as session:
                await session.send_client_content(turns=contents, turn_complete=True)

                async for response in session.receive():
                    if hasattr(response, "usage_metadata") and response.usage_metadata:
                        usage_metadata = response.usage_metadata

                    if response.server_content and response.server_content.model_turn:
                        for part in response.server_content.model_turn.parts:
                            if part.text:
                                text_parts.append(part.text)
                            elif part.function_call:
                                function_calls.append(part.function_call)

                    elif response.tool_call:
                        for fc in response.tool_call.function_calls:
                            function_calls.append(fc)
                        break

                    if response.server_content and response.server_content.turn_complete:
                        break

        except genai_errors.APIError as e:
            raise self.handle_llm_error(e, llm_config=llm_config)
        except LLMError:
            raise
        except Exception as e:
            # WebSocket 연결 오류(1011 internal error 등)는 서버 측 일시 오류이므로
            # LLMServerError로 매핑해 키 로테이션 → Claude CLI fallback 흐름을 탄다
            try:
                from websockets.exceptions import WebSocketException
                if isinstance(e, WebSocketException):
                    raise LLMServerError(
                        message=f"[GoogleLive] WebSocket error: {e}",
                        code=ErrorCode.INTERNAL_SERVER_ERROR,
                    ) from e
            except ImportError:
                pass
            raise self.handle_llm_error(e, llm_config=llm_config)

        parts = []
        if text_parts:
            parts.append({"text": "".join(text_parts)})
        for fc in function_calls:
            parts.append(
                {
                    "function_call": {
                        "name": fc.name,
                        "args": dict(fc.args) if fc.args else {},
                    }
                }
            )
        if not parts:
            parts = [{"text": ""}]

        return {
            "candidates": [
                {
                    "content": {"role": "model", "parts": parts},
                    "finish_reason": "STOP",
                    "index": 0,
                }
            ],
            "usage_metadata": {
                "prompt_token_count": getattr(usage_metadata, "prompt_token_count", 0) or 0,
                "candidates_token_count": getattr(usage_metadata, "candidates_token_count", 0) or 0,
                "total_token_count": getattr(usage_metadata, "total_token_count", 0) or 0,
            },
        }

    async def _stream_live_async(
        self, request_data: dict, llm_config: LLMConfig, api_key: str
    ) -> AsyncIterator:
        from google import genai
        from google.genai import errors as genai_errors
        from google.genai.types import (
            Candidate,
            Content,
            FinishReason,
            GenerateContentResponse,
            HttpOptions,
            Part,
        )

        timeout_ms = int(model_settings.gemini_timeout_seconds * 1000)
        client = genai.Client(
            api_key=api_key,
            http_options=HttpOptions(timeout=timeout_ms),
        )
        live_config = self._build_live_config(request_data["config"])
        contents = request_data["contents"]

        pending_fcs = []

        try:
            async with client.aio.live.connect(
                model=llm_config.model, config=live_config
            ) as session:
                await session.send_client_content(turns=contents, turn_complete=True)

                async for response in session.receive():
                    if response.server_content and response.server_content.model_turn:
                        text_parts = []
                        for part in response.server_content.model_turn.parts:
                            if part.text:
                                text_parts.append(part.text)
                            elif part.function_call:
                                pending_fcs.append(part.function_call)

                        if text_parts:
                            yield GenerateContentResponse(
                                candidates=[
                                    Candidate(
                                        content=Content(
                                            role="model",
                                            parts=[Part(text="".join(text_parts))],
                                        ),
                                        finish_reason=FinishReason.UNSPECIFIED,
                                        index=0,
                                    )
                                ],
                            )

                    elif response.tool_call:
                        for fc in response.tool_call.function_calls:
                            pending_fcs.append(fc)

                    if response.server_content and response.server_content.turn_complete:
                        if pending_fcs:
                            yield GenerateContentResponse(
                                candidates=[
                                    Candidate(
                                        content=Content(
                                            role="model",
                                            parts=[Part(function_call=fc) for fc in pending_fcs],
                                        ),
                                        finish_reason=FinishReason.STOP,
                                        index=0,
                                    )
                                ],
                            )
                        break

                    elif response.tool_call and pending_fcs:
                        yield GenerateContentResponse(
                            candidates=[
                                Candidate(
                                    content=Content(
                                        role="model",
                                        parts=[Part(function_call=fc) for fc in pending_fcs],
                                    ),
                                    finish_reason=FinishReason.STOP,
                                    index=0,
                                )
                            ],
                        )
                        break

        except genai_errors.APIError as e:
            raise self.handle_llm_error(e, llm_config=llm_config)
        except LLMError:
            raise
        except Exception as e:
            try:
                from websockets.exceptions import WebSocketException
                if isinstance(e, WebSocketException):
                    raise LLMServerError(
                        message=f"[GoogleLive] WebSocket error: {e}",
                        code=ErrorCode.INTERNAL_SERVER_ERROR,
                    ) from e
            except ImportError:
                pass
            raise self.handle_llm_error(e, llm_config=llm_config)

    # ------------------------------------------------------------------ #
    #  Internal: Claude CLI fallback
    # ------------------------------------------------------------------ #

    async def _fallback_to_claude_cli(self, request_data: dict) -> dict:
        """
        Last-resort fallback: calls `claude -p` subprocess with the conversation
        text extracted from request_data["contents"].
        Returns a GenerateContentResponse-compatible dict (text only).
        """
        if not shutil.which("claude"):
            raise RuntimeError(
                "[GoogleLive] Fallback failed: 'claude' CLI not found in PATH. "
                "Install Claude Code CLI or add more Gemini API keys."
            )

        prompt = self._build_cli_prompt(request_data)
        cmd = ["claude", "-p", prompt, "--output-format", "json"]

        logger.info(f"[GoogleLive] claude -p fallback, prompt length={len(prompt)}")
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
        except OSError as e:
            raise RuntimeError(f"[GoogleLive] claude CLI 실행 실패: {e}") from e

        stdout_text = stdout.decode().strip()

        if proc.returncode != 0:
            error_detail = stderr.decode().strip() or stdout_text
            try:
                error_json = json.loads(stdout_text)
                error_detail = error_json.get("result") or error_detail
            except json.JSONDecodeError:
                pass
            raise RuntimeError(f"[GoogleLive] claude -p 오류 (exit {proc.returncode}): {error_detail}")

        text = stdout_text
        try:
            result = json.loads(stdout_text)
            if result.get("is_error"):
                raise RuntimeError(
                    f"[GoogleLive] claude -p is_error=true: {result.get('result', '')}"
                )
            text = result.get("result", stdout_text)
        except json.JSONDecodeError:
            pass

        logger.info(f"[GoogleLive] claude -p fallback 성공, response length={len(text)}")
        return {
            "candidates": [
                {
                    "content": {"role": "model", "parts": [{"text": text}]},
                    "finish_reason": "STOP",
                    "index": 0,
                }
            ],
            "usage_metadata": {
                "prompt_token_count": 0,
                "candidates_token_count": 0,
                "total_token_count": 0,
            },
        }

    @staticmethod
    def _build_cli_prompt(request_data: dict) -> str:
        """Google-format request_data에서 대화 텍스트를 추출해 프롬프트를 구성합니다."""
        config = request_data.get("config", {})
        contents = request_data.get("contents", [])

        lines: list[str] = []

        system = config.get("system_instruction")
        if system:
            if isinstance(system, dict):
                # Content 형식인 경우 parts에서 텍스트 추출
                for p in system.get("parts", []):
                    if p.get("text"):
                        lines.append(f"[System]: {p['text']}")
            elif isinstance(system, str):
                lines.append(f"[System]: {system}")

        for turn in contents:
            role = turn.get("role", "unknown")
            parts = turn.get("parts", [])
            texts = [p["text"] for p in parts if isinstance(p, dict) and p.get("text")]
            if texts:
                lines.append(f"[{role}]: {' '.join(texts)}")

        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  Internal: LiveConnectConfig builder
    # ------------------------------------------------------------------ #

    def _build_live_config(self, config: dict):
        from google.genai import types

        kwargs: dict = {"response_modalities": ["TEXT"]}

        if config.get("temperature") is not None:
            kwargs["temperature"] = config["temperature"]
        if config.get("max_output_tokens"):
            kwargs["max_output_tokens"] = config["max_output_tokens"]
        if config.get("system_instruction"):
            kwargs["system_instruction"] = config["system_instruction"]
        if config.get("tools"):
            kwargs["tools"] = config["tools"]

        return types.LiveConnectConfig(**kwargs)
