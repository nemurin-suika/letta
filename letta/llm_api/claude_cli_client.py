"""LLM client that delegates to the `claude -p` CLI subprocess.

Gemini rate limit 시 fallback으로 사용. API 키 없이 Claude CLI 구독을 통해 응답.
"""

import asyncio
import json
import shutil
import time
from typing import TYPE_CHECKING, List, Optional

from letta.errors import ErrorCode, LLMBadRequestError, LLMConnectionError, LLMError, LLMServerError
from letta.llm_api.llm_client_base import LLMClientBase
from letta.log import get_logger
from letta.schemas.enums import AgentType
from letta.schemas.openai.chat_completion_response import (
    ChatCompletionResponse,
    Choice,
    Message,
    UsageStatistics,
)

if TYPE_CHECKING:
    from letta.schemas.embedding_config import EmbeddingConfig
    from letta.schemas.llm_config import LLMConfig

logger = get_logger(__name__)


class ClaudeCLIClient(LLMClientBase):
    """claude -p 서브프로세스를 LLM 백엔드로 사용하는 클라이언트."""

    # ---------------------------------------------------------------------------
    # Request building
    # ---------------------------------------------------------------------------

    def build_request_data(
        self,
        agent_type: AgentType,
        messages: List[Message],
        llm_config: "LLMConfig",
        tools: List[dict],
        force_tool_call: Optional[str] = None,
        requires_subsequent_tool_call: bool = False,
        tool_return_truncation_chars: Optional[int] = None,
        system: Optional[str] = None,
    ) -> dict:
        prompt = self._format_prompt(messages, system)
        return {"prompt": prompt, "model": llm_config.model or ""}

    # ---------------------------------------------------------------------------
    # Sync/async request
    # ---------------------------------------------------------------------------

    def request(self, request_data: dict, llm_config: "LLMConfig") -> dict:
        return asyncio.get_event_loop().run_until_complete(self.request_async(request_data, llm_config))

    async def request_async(self, request_data: dict, llm_config: "LLMConfig") -> dict:
        if not shutil.which("claude"):
            raise LLMConnectionError(
                message="'claude' CLI를 PATH에서 찾을 수 없습니다. Claude Code CLI를 설치하세요.",
                code=ErrorCode.INTERNAL_SERVER_ERROR,
            )

        prompt = request_data["prompt"]
        model = request_data.get("model", "")

        # 프롬프트는 -p의 인자로 직접 전달 (stdin은 claude CLI가 지원하지 않음)
        cmd = ["claude", "-p", prompt, "--output-format", "json"]
        if model and model != "claude-cli":
            cmd.extend(["--model", model])

        logger.info(f"[ClaudeCLI] 실행: claude -p <prompt({len(prompt)}chars)> --output-format json")
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
        except OSError as e:
            raise LLMConnectionError(
                message=f"claude CLI 실행 실패: {e}",
                code=ErrorCode.INTERNAL_SERVER_ERROR,
            )

        stdout_text = stdout.decode().strip()
        stderr_text = stderr.decode().strip()

        if proc.returncode != 0:
            # JSON 응답이 있으면 result 필드에서 실제 오류 메시지 추출
            error_detail = stderr_text or stdout_text
            try:
                error_json = json.loads(stdout_text)
                error_detail = error_json.get("result") or error_detail
            except json.JSONDecodeError:
                pass
            logger.error(f"[ClaudeCLI] 실패 (exit {proc.returncode}): {error_detail}")
            raise LLMServerError(
                message=f"claude CLI 오류: {error_detail}",
                code=ErrorCode.INTERNAL_SERVER_ERROR,
            )

        try:
            result = json.loads(stdout_text)
        except json.JSONDecodeError:
            return {"result": stdout_text, "usage": {}}

        # is_error: true 는 exit code 0이어도 내용이 오류인 경우
        if result.get("is_error"):
            error_msg = result.get("result") or "claude CLI에서 오류 반환"
            logger.error(f"[ClaudeCLI] is_error=true: {error_msg}")
            raise LLMServerError(
                message=f"claude CLI 오류: {error_msg}",
                code=ErrorCode.INTERNAL_SERVER_ERROR,
            )

        return result

    # ---------------------------------------------------------------------------
    # Response conversion
    # ---------------------------------------------------------------------------

    async def convert_response_to_chat_completion(
        self,
        response_data: dict,
        input_messages: List[Message],
        llm_config: "LLMConfig",
    ) -> ChatCompletionResponse:
        text = ""
        usage_raw: dict = {}

        if isinstance(response_data, dict):
            text = response_data.get("result") or ""
            usage_raw = response_data.get("usage") or {}

        prompt_tokens = int(usage_raw.get("input_tokens", 0))
        completion_tokens = int(usage_raw.get("output_tokens", 0))

        return ChatCompletionResponse(
            id=f"claude-cli-{int(time.time())}",
            object="chat.completion",
            created=int(time.time()),
            model=llm_config.model or "claude-cli",
            choices=[
                Choice(
                    index=0,
                    message=Message(role="assistant", content=text),
                    finish_reason="stop",
                )
            ],
            usage=UsageStatistics(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    # ---------------------------------------------------------------------------
    # Streaming (미지원 — SimpleLLMRequestAdapter로 교체됨)
    # ---------------------------------------------------------------------------

    async def stream_async(self, request_data: dict, llm_config: "LLMConfig"):
        raise NotImplementedError("ClaudeCLIClient는 스트리밍을 지원하지 않습니다.")

    # ---------------------------------------------------------------------------
    # Embeddings (미지원)
    # ---------------------------------------------------------------------------

    async def request_embeddings(self, texts: List[str], embedding_config: "EmbeddingConfig") -> List[List[float]]:
        raise NotImplementedError("ClaudeCLIClient는 임베딩을 지원하지 않습니다.")

    # ---------------------------------------------------------------------------
    # Misc
    # ---------------------------------------------------------------------------

    def is_reasoning_model(self, llm_config: "LLMConfig") -> bool:
        return False

    def handle_llm_error(self, e: Exception, llm_config: Optional["LLMConfig"] = None) -> Exception:
        if isinstance(e, LLMError):
            return e
        return LLMBadRequestError(
            message=f"claude CLI 오류: {e}",
            code=ErrorCode.INTERNAL_SERVER_ERROR,
        )

    # ---------------------------------------------------------------------------
    # Message formatting
    # ---------------------------------------------------------------------------

    def _format_prompt(self, messages: List[Message], system: Optional[str] = None) -> str:
        """Letta 메시지를 claude -p에 전달할 텍스트 프롬프트로 변환."""
        parts: List[str] = []

        # 시스템 프롬프트 추출
        sys_content = system or ""
        for msg in messages:
            role = self._get_field(msg, "role")
            if role == "system":
                sys_content = self._get_content(msg) or sys_content
                break

        if sys_content:
            parts.append(f"<system>\n{sys_content}\n</system>")

        for msg in messages:
            role = self._get_field(msg, "role")
            if role == "system":
                continue

            content = self._get_content(msg)
            tool_calls = self._get_field(msg, "tool_calls") or []

            if role == "user":
                parts.append(f"Human: {content}")
            elif role == "assistant":
                if tool_calls:
                    call_strs = []
                    for tc in tool_calls:
                        fn = self._get_field(tc, "function") or {}
                        name = self._get_field(fn, "name") or ""
                        args = self._get_field(fn, "arguments") or ""
                        call_strs.append(f"[Tool call: {name}({args})]")
                    text = f"Assistant: {content}\n" + "\n".join(call_strs) if content else "Assistant: " + "\n".join(call_strs)
                    parts.append(text)
                else:
                    parts.append(f"Assistant: {content}")
            elif role == "tool":
                parts.append(f"[Tool result]: {content}")

        return "\n\n".join(parts)

    @staticmethod
    def _get_field(obj, field: str):
        """dict 또는 Pydantic 모델에서 필드 값을 가져옴."""
        if isinstance(obj, dict):
            return obj.get(field)
        return getattr(obj, field, None)

    @staticmethod
    def _get_content(msg) -> str:
        content = ClaudeCLIClient._get_field(msg, "content") or ""
        if isinstance(content, list):
            content = "\n".join(
                c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"
            )
        return str(content)
