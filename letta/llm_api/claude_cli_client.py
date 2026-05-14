"""LLM client that delegates to the `claude -p` CLI subprocess.

Gemini rate limit 시 fallback으로 사용. API 키 없이 Claude CLI 구독을 통해 응답.
"""

import asyncio
import glob
import json
import os
import re
import shutil
import time
import uuid
from typing import TYPE_CHECKING, List, Optional, Tuple

from letta.errors import ErrorCode, LLMBadRequestError, LLMConnectionError, LLMError, LLMServerError
from letta.llm_api.llm_client_base import LLMClientBase
from letta.log import get_logger
from letta.schemas.enums import AgentType
from letta.schemas.openai.chat_completion_response import (
    ChatCompletionResponse,
    Choice,
    FunctionCall,
    Message,
    ToolCall,
    UsageStatistics,
)

if TYPE_CHECKING:
    from letta.schemas.embedding_config import EmbeddingConfig
    from letta.schemas.llm_config import LLMConfig

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Core memory 태그 형식 (Claude가 응답 끝에 출력)
#
# 추가: <letta_memory_append label="human">내용</letta_memory_append>
# 교체: <letta_memory_replace label="human"><old>기존</old><new>새것</new></letta_memory_replace>
# ---------------------------------------------------------------------------

_APPEND_RE = re.compile(
    r'<letta_memory_append\s+label=["\']?([^"\'>\s]+)["\']?>(.*?)</letta_memory_append>',
    re.DOTALL,
)
_REPLACE_RE = re.compile(
    r'<letta_memory_replace\s+label=["\']?([^"\'>\s]+)["\']?>'
    r'\s*<old>(.*?)</old>\s*<new>(.*?)</new>\s*</letta_memory_replace>',
    re.DOTALL,
)
_ARCHIVAL_INSERT_RE = re.compile(
    r'<letta_archival_insert([^>]*)>(.*?)</letta_archival_insert>',
    re.DOTALL,
)
_ARCHIVAL_SEARCH_RE = re.compile(
    r'<letta_archival_search([^>]*)>(.*?)</letta_archival_search>',
    re.DOTALL,
)
_ATTR_RE = re.compile(r'(\w+)=["\']?([^"\'>\s]*)["\']?')

# [유튜브 링크]URL[/유튜브 링크] 패턴 — Open-LLM-VTuber에서 이미 영상 URL로 교체된 상태로 전달됨
_YOUTUBE_RE = re.compile(r"\[유튜브 링크\](.*?)\[/유튜브 링크\]", re.DOTALL)
_IMAGE_LINK_RE = re.compile(r"\[이미지 링크\](.*?)\[/이미지 링크\]", re.DOTALL)
_WEB_LINK_RE = re.compile(r"\[웹 링크\](.*?)\[/웹 링크\]", re.DOTALL)
_SCREENSHOT_RE = re.compile(r"\[스크린샷([^\]]*)\](.*?)\[/스크린샷\]", re.DOTALL)


def _expand_youtube_links(text: str) -> str:
    """[유튜브 링크]영상URL[/유튜브 링크]를 WebFetch 지시문으로 변환."""
    def replace(m: re.Match) -> str:
        url = m.group(1).strip()
        return (
            f"[미즈키가 시청한 유튜브 영상]\n"
            f"URL: {url}\n"
            f"WebFetch로 위 URL을 직접 방문해서 영상 내용을 파악하세요. 유튜브 외 다른 사이트는 방문하지 마세요.\n"
            f"미즈키가 이 영상을 실제로 시청한 것처럼 자연스럽게 반응하세요.\n"
            f"[/미즈키가 시청한 유튜브 영상]"
        )
    return _YOUTUBE_RE.sub(replace, text)


def _expand_image_links(text: str) -> str:
    """[이미지 링크]URL[/이미지 링크]를 WebFetch 이미지 감상 지시문으로 변환."""
    def replace(m: re.Match) -> str:
        url = m.group(1).strip()
        return (
            f"[채팅에 이미지 링크 공유됨]\n"
            f"URL: {url}\n"
            f"WebFetch로 위 이미지 URL을 방문해서 이미지를 직접 보고 자연스럽게 감상/반응을 남겨주세요.\n"
            f"[/채팅에 이미지 링크 공유됨]"
        )
    return _IMAGE_LINK_RE.sub(replace, text)


def _expand_web_links(text: str) -> str:
    """[웹 링크]URL[/웹 링크]를 WebFetch 감상 지시문으로 변환."""
    def replace(m: re.Match) -> str:
        url = m.group(1).strip()
        return (
            f"[채팅에 웹 링크 공유됨]\n"
            f"URL: {url}\n"
            f"WebFetch로 위 URL을 방문해서 내용을 파악하고 자연스럽게 반응하세요.\n"
            f"[/채팅에 웹 링크 공유됨]"
        )
    return _WEB_LINK_RE.sub(replace, text)


def _expand_screenshots(text: str) -> str:
    """[스크린샷] 태그를 Read 지시문으로 변환. 파일이 없으면 '이전 스크린샷'으로 대체."""
    def replace(m: re.Match) -> str:
        attrs = m.group(1)
        file_path = m.group(2).strip()
        source_m = re.search(r"source=(\w+)", attrs)
        source = source_m.group(1) if source_m else "screen"
        label = {"screen": "화면", "camera": "카메라", "youtube_frame": "유튜브 영상 프레임"}.get(source, "화면")

        if file_path and os.path.exists(file_path):
            if source == "youtube_frame":
                desc = "유튜브 영상의 특정 시점 프레임입니다. 영상 내용을 파악하는 데 활용하세요."
            else:
                desc = "화면에 무엇이 보이는지 파악하세요."
            return (
                f"[현재 {label} 첨부됨]\n"
                f"Read 툴로 {file_path} 파일을 읽어 {desc}\n"
                f"[/현재 {label} 첨부됨]"
            )
        return f"[이전 {label} — 내용은 이미 처리됨]"

    return _SCREENSHOT_RE.sub(replace, text)

MEMORY_INSTRUCTIONS = """\

<letta_memory_instructions>
위의 <persona>와 <human> 섹션이 당신의 현재 장기 기억(core memory)입니다. 이 내용을 신뢰하고 참고하세요.
시스템 프롬프트에 표시된 archival memory 개수도 확인하고, 관련 내용이 있을 것 같으면 검색하세요.

대화 중 **당신의 캐릭터(위 시스템 프롬프트 참조)로서 자연스럽게 "이건 기억해둬야겠다"고 느끼는 순간**이 있다면, 응답 맨 끝에 아래 태그로 저장하세요.

어떤 걸 저장할지는 당신 스스로 판단하세요. 기준이 되는 질문:
- **human 블록**: "이 사람에 대해 새로 알게 된 건데, 나중에 잊으면 어색하거나 실례가 될까?"
- **persona 블록**: "나한테 일어난 일인데, 내가 어떤 AI인지를 나타내는 중요한 경험이나 변화일까?"
- **archival memory**: "너무 상세해서 core 메모리에 담기엔 길지만, 나중에 검색해서 찾을 수 있어야 하는 내용일까?"

특히 아래는 반드시 archival memory에 기록하세요 (persona 블록이 아님 — 상세 기록은 archival에):
- 후원/구독 이벤트: 누가, 얼마를, 어떤 말과 함께, 몇 번째 후원인지 (예: "몽팬님 첫 후원 1000원 — 채팅/후원 테스트 겸")
- 인상적인 시청자: 자주 오거나 특별한 교류가 있었던 사람과 그 이유

저장 안 해도 되는 것: 지금 이 순간만의 상황, 이미 블록에 있는 내용의 반복.

=== Core Memory 글자 수 제한 ===

각 블록의 메타데이터에 표시된 `chars_current`와 `chars_limit`은 **반드시 지켜야 하는 제한**입니다.

- `chars_current`: 현재 블록에 저장된 글자 수
- `chars_limit`: 허용되는 최대 글자 수

**규칙:**
1. 새 내용을 추가하기 전에 항상 확인하세요: `chars_current + 추가할 내용 길이 ≤ chars_limit`
2. 초과할 것 같으면 `letta_memory_replace`로 덜 중요한 기존 내용을 지운 뒤 추가하세요.
3. `chars_current`가 이미 `chars_limit`을 넘었다면, 추가하기 전에 반드시 먼저 내용을 줄여야 합니다.
4. 추가 대신 **전체를 간결하게 재작성**하는 것도 좋은 방법입니다 (`<old>전체 내용</old><new>요약된 내용+새 정보</new>`).

=== Core Memory (즉시 참조, 핵심 정보) ===

새 정보 추가:
<letta_memory_append label="LABEL">기억할 내용</letta_memory_append>

기존 정보 교체/수정:
<letta_memory_replace label="LABEL"><old>교체할 기존 텍스트(정확히 일치)</old><new>새로운 텍스트(삭제 시 빈 값)</new></letta_memory_replace>

=== Archival Memory (대용량 장기 저장, 검색으로 조회) ===

저장 (tags는 선택사항, 쉼표 구분):
<letta_archival_insert tags="태그1,태그2">저장할 내용</letta_archival_insert>

검색 (관련 기억이 archival에 있을 것 같을 때):
<letta_archival_search>검색 쿼리</letta_archival_search>

메모리 태그는 응답 맨 끝에만 추가하세요.
</letta_memory_instructions>"""


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
        # 프롬프트에서 스크린샷 temp 파일 경로 추출 (처리 후 클린업용)
        screenshot_files = re.findall(
            r"Read 툴로 (/tmp/claude_(?:screenshot|yt_frame)_[a-f0-9_]+\.jpg) 파일을", prompt
        )
        return {"prompt": prompt, "model": llm_config.model or "", "screenshot_files": screenshot_files}

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
        screenshot_files: list = request_data.get("screenshot_files", [])

        # 스크린샷 파일이 있으면 Read 툴도 허용
        allowed_tools = "WebFetch,Read(/tmp/claude_screenshot*),Read(/tmp/claude_yt_frame*)" if screenshot_files else "WebFetch"
        cmd = ["claude", "-p", prompt, "--output-format", "json", "--allowedTools", allowed_tools]
        if model and model != "claude-cli":
            cmd.extend(["--model", model])

        logger.info(
            f"[ClaudeCLI] 실행: claude -p <prompt({len(prompt)}chars)> "
            f"--output-format json (screenshots={len(screenshot_files)})"
        )
        TIMEOUT_SECONDS = 300  # 5분
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd="/tmp",  # 프로젝트 CLAUDE.md 컨텍스트 격리
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=TIMEOUT_SECONDS)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()  # 버퍼 정리
                logger.error(f"[ClaudeCLI] {TIMEOUT_SECONDS}초 타임아웃 — 프로세스 강제 종료")
                raise LLMServerError(
                    message=f"claude CLI 타임아웃: {TIMEOUT_SECONDS}초 초과",
                    code=ErrorCode.INTERNAL_SERVER_ERROR,
                )
        except OSError as e:
            raise LLMConnectionError(
                message=f"claude CLI 실행 실패: {e}",
                code=ErrorCode.INTERNAL_SERVER_ERROR,
            )
        finally:
            # 처리 완료 후 스크린샷 temp 파일 삭제
            for f in screenshot_files:
                try:
                    os.unlink(f)
                    logger.debug(f"[ClaudeCLI] temp 파일 삭제: {f}")
                except Exception:
                    pass

        stdout_text = stdout.decode().strip()
        stderr_text = stderr.decode().strip()

        self._dump_debug_log(prompt, stdout_text, stderr_text, proc.returncode)

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

    @staticmethod
    def _dump_debug_log(prompt: str, stdout: str, stderr: str, returncode: Optional[int]) -> None:
        log_dir = os.environ.get("LETTA_CLAUDE_CLI_DEBUG_DIR", "/app/logs/claude_cli_debug")
        try:
            os.makedirs(log_dir, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.join(log_dir, f"{ts}_{int(time.time()*1000) % 100000}.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write("=== PROMPT ===\n")
                f.write(prompt)
                f.write(f"\n\n=== STDOUT (exit={returncode}) ===\n")
                f.write(stdout)
                if stderr:
                    f.write("\n\n=== STDERR ===\n")
                    f.write(stderr)
            msg = f"[ClaudeCLI] 디버그 로그 저장: {path}"
            print(msg, flush=True)
            logger.info(msg)
        except Exception as e:
            msg = f"[ClaudeCLI] 디버그 로그 저장 실패: {type(e).__name__}: {e} (log_dir={log_dir})"
            print(msg, flush=True)
            logger.error(msg)

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

        # claude CLI는 캐시 토큰을 input_tokens와 별도로 반환하므로 모두 합산
        # 합산하지 않으면 Letta가 실제 컨텍스트 사용량을 극소로 인식해 압축이 트리거되지 않음
        prompt_tokens = (
            int(usage_raw.get("input_tokens", 0))
            + int(usage_raw.get("cache_creation_input_tokens", 0))
            + int(usage_raw.get("cache_read_input_tokens", 0))
        )
        completion_tokens = int(usage_raw.get("output_tokens", 0))

        # 환각 H:/A: 대화 잘라내기 (LLM이 미래 입력을 상상해서 생성하는 패턴 차단)
        text = self._strip_hallucinated_conversation(text)

        # 응답에서 core memory 태그 파싱 및 제거
        clean_text, tool_calls = self._extract_memory_ops(text)

        finish_reason = "tool_calls" if tool_calls else "stop"

        return ChatCompletionResponse(
            id=f"claude-cli-{int(time.time())}",
            object="chat.completion",
            created=int(time.time()),
            model=llm_config.model or "claude-cli",
            choices=[
                Choice(
                    index=0,
                    message=Message(
                        role="assistant",
                        content=clean_text or None,
                        tool_calls=tool_calls if tool_calls else None,
                    ),
                    finish_reason=finish_reason,
                )
            ],
            usage=UsageStatistics(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    @staticmethod
    def _strip_hallucinated_conversation(text: str) -> str:
        """LLM이 응답 안에 상상으로 생성한 H: / A: 대화 형식을 잘라냄.

        LLM이 학습 데이터 패턴을 바탕으로 아직 오지 않은 시청자 채팅 등
        미래 입력을 'H: [채팅] ...', 'A: ...' 형식으로 스스로 만들어내는
        환각이 발생한다. 'H: [' 패턴이 나타나는 지점에서 응답을 잘라낸다.
        """
        import re as _re
        m = _re.search(r'\nH: \[', text)
        if m:
            stripped = text[:m.start()].rstrip()
            logger.warning(
                f"[ClaudeCLI] 환각 H:/A: 대화 감지 — 응답 {len(text)}자 중 "
                f"{len(text) - m.start()}자 잘라냄. 미리보기: {repr(text[m.start():m.start()+80])}"
            )
            return stripped
        return text

    @staticmethod
    def _extract_memory_ops(text: str) -> Tuple[str, List[ToolCall]]:
        """응답 텍스트에서 letta_memory / letta_archival 태그를 추출하고 ToolCall 목록으로 변환."""
        tool_calls: List[ToolCall] = []

        def _make_id() -> str:
            return f"call_{uuid.uuid4().hex[:24]}"

        # core_memory_append 태그 처리
        for match in _APPEND_RE.finditer(text):
            label = match.group(1).strip()
            content = match.group(2).strip()
            tool_calls.append(
                ToolCall(
                    id=_make_id(),
                    type="function",
                    function=FunctionCall(
                        name="core_memory_append",
                        arguments=json.dumps({"label": label, "content": content}, ensure_ascii=False),
                    ),
                )
            )

        # core_memory_replace 태그 처리
        for match in _REPLACE_RE.finditer(text):
            label = match.group(1).strip()
            old_content = match.group(2).strip()
            new_content = match.group(3).strip()
            tool_calls.append(
                ToolCall(
                    id=_make_id(),
                    type="function",
                    function=FunctionCall(
                        name="core_memory_replace",
                        arguments=json.dumps(
                            {"label": label, "old_content": old_content, "new_content": new_content},
                            ensure_ascii=False,
                        ),
                    ),
                )
            )

        # archival_memory_insert 태그 처리
        for match in _ARCHIVAL_INSERT_RE.finditer(text):
            attrs = {m.group(1): m.group(2) for m in _ATTR_RE.finditer(match.group(1))}
            content = match.group(2).strip()
            tags_str = attrs.get("tags", "")
            tags = [t.strip() for t in tags_str.split(",") if t.strip()] if tags_str else None
            args: dict = {"content": content}
            if tags:
                args["tags"] = tags
            tool_calls.append(
                ToolCall(
                    id=_make_id(),
                    type="function",
                    function=FunctionCall(
                        name="archival_memory_insert",
                        arguments=json.dumps(args, ensure_ascii=False),
                    ),
                )
            )

        # archival_memory_search 태그 처리
        for match in _ARCHIVAL_SEARCH_RE.finditer(text):
            attrs = {m.group(1): m.group(2) for m in _ATTR_RE.finditer(match.group(1))}
            query = match.group(2).strip()
            top_k_str = attrs.get("top_k", "")
            tags_str = attrs.get("tags", "")
            top_k = int(top_k_str) if top_k_str.isdigit() else None
            tags = [t.strip() for t in tags_str.split(",") if t.strip()] if tags_str else None
            args = {"query": query}
            if top_k:
                args["top_k"] = top_k
            if tags:
                args["tags"] = tags
            tool_calls.append(
                ToolCall(
                    id=_make_id(),
                    type="function",
                    function=FunctionCall(
                        name="archival_memory_search",
                        arguments=json.dumps(args, ensure_ascii=False),
                    ),
                )
            )

        if tool_calls:
            # 태그 전체를 응답 텍스트에서 제거
            clean = _APPEND_RE.sub("", text)
            clean = _REPLACE_RE.sub("", clean)
            clean = _ARCHIVAL_INSERT_RE.sub("", clean)
            clean = _ARCHIVAL_SEARCH_RE.sub("", clean)
            clean = clean.rstrip()
            logger.info(f"[ClaudeCLI] 메모리 작업 {len(tool_calls)}개 파싱: {[tc.function.name for tc in tool_calls]}")
        else:
            clean = text

        return clean, tool_calls

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
            # 메모리 지시문을 시스템 프롬프트 끝에 붙여 강조
            parts.append(f"<system>\n{sys_content}{MEMORY_INSTRUCTIONS}\n</system>")

        # 마지막 user 메시지 인덱스를 미리 찾기 (캐시 무효화 + 강조 마커 부착 대상)
        last_user_idx = -1
        for i, msg in enumerate(messages):
            if self._get_field(msg, "role") == "user":
                content_check = self._get_content(msg)
                if content_check and content_check.strip():
                    last_user_idx = i

        for i, msg in enumerate(messages):
            role = self._get_field(msg, "role")
            if role == "system":
                continue

            content = self._get_content(msg)
            tool_calls = self._get_field(msg, "tool_calls") or []

            if role == "user":
                content = _expand_screenshots(content)
                content = _expand_youtube_links(content)
                content = _expand_image_links(content)
                content = _expand_web_links(content)
                if not content.strip():
                    logger.warning("[ClaudeCLI] 빈 user 메시지 건너뜀 — LLM 상상 입력 방지")
                    continue
                if i == last_user_idx:
                    # 마지막 user 입력: 캐시 무효화 nonce + 강조 마커로 감싼다
                    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
                    nonce = uuid.uuid4().hex[:12]
                    parts.append(
                        f"Human: <latest_user_message timestamp=\"{ts}\" id=\"{nonce}\">\n"
                        f"위 시스템 지침에 따라 이 메시지 한 건에만 응답하세요.\n"
                        f"- 미래 채팅/네무린 메시지를 응답 안에서 만들지 마세요"
                        f"(H:, A:, [채팅] 누구:, [네무린] 등 가상 턴 금지).\n"
                        f"- 임의로 방종 멘트를 끼워 넣지 마세요. "
                        f"아직 일어나지 않은 방송 종료를 archival에 저장하지 마세요.\n\n"
                        f"{content}\n"
                        f"</latest_user_message>"
                    )
                else:
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
                    if not content.strip():
                        continue
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
        from letta.schemas.letta_message_content import (
            TextContent,
            ImageContent,
            UrlImage,
            Base64Image,
        )

        content = ClaudeCLIClient._get_field(msg, "content") or ""

        if isinstance(content, list):
            texts = []
            for c in content:
                if isinstance(c, TextContent):
                    texts.append(c.text)
                elif isinstance(c, dict) and c.get("type") == "text":
                    texts.append(c.get("text", ""))
                elif isinstance(c, ImageContent):
                    src = c.source
                    if isinstance(src, UrlImage):
                        texts.append(f"[이미지 링크]{src.url}[/이미지 링크]")
                    elif isinstance(src, Base64Image):
                        pass  # letta_agent에서 [스크린샷] 태그로 텍스트에 이미 포함됨
                elif isinstance(c, dict) and c.get("type") == "image":
                    src = c.get("source", {})
                    if isinstance(src, dict):
                        if src.get("type") == "url":
                            texts.append(f"[이미지 링크]{src.get('url', '')}[/이미지 링크]")
                        elif src.get("type") == "base64":
                            pass  # letta_agent에서 [스크린샷] 태그로 텍스트에 이미 포함됨
            return "\n".join(texts)

        if isinstance(content, TextContent):
            return content.text

        # Pydantic 모델이 통째로 넘어오는 경우 — 텍스트 필드만 추출
        if hasattr(content, "type") and getattr(content, "type", None) != "text":
            return ""

        return str(content)
