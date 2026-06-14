"""LLM client that delegates to the `codex exec` CLI subprocess.

API 키 기반 LLM 호출 대신 Codex CLI 로그인/구독을 통해 응답.
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
# Core memory 태그 형식 (Codex가 응답 끝에 출력)
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
# 두 형식 모두 허용: <letta_archival_delete id="passage-..." /> 또는 <letta_archival_delete>passage-...</letta_archival_delete>
_ARCHIVAL_DELETE_RE = re.compile(
    r'<letta_archival_delete(?:\s+id=["\']?([^"\'>\s]+)["\']?\s*/?>|>(.*?)</letta_archival_delete>)',
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
    """[스크린샷] 태그를 Read 지시문으로 변환. 파일이 없으면 '이전 스크린샷'으로 대체.

    ago=Ns 속성이 있으면 시간 라벨 적용:
    - ago=0s → "현재 화면"
    - ago>0  → "N초 전 화면"
    """
    def replace(m: re.Match) -> str:
        attrs = m.group(1)
        file_path = m.group(2).strip()
        source_m = re.search(r"source=(\w+)", attrs)
        source = source_m.group(1) if source_m else "screen"
        ago_m = re.search(r"ago=([\d.]+)s", attrs)
        ago = float(ago_m.group(1)) if ago_m else None

        source_label = {
            "screen": "화면",
            "camera": "카메라",
            "youtube_frame": "유튜브 영상 프레임",
        }.get(source, "화면")

        # 시간 라벨 결정
        if ago is None or source == "camera":
            time_prefix = "현재"
        elif ago <= 0.5:
            time_prefix = "현재"
        else:
            # 정수면 "6초 전", 소수면 "1.5초 전"
            ago_str = f"{int(ago)}" if ago == int(ago) else f"{ago:g}"
            time_prefix = f"{ago_str}초 전"

        full_label = f"{time_prefix} {source_label}"

        if file_path and os.path.exists(file_path):
            if source == "youtube_frame":
                desc = "유튜브 영상의 특정 시점 프레임입니다. 영상 내용을 파악하는 데 활용하세요."
            elif time_prefix == "현재":
                desc = "지금 화면에 무엇이 보이는지 파악하세요."
            else:
                desc = f"{time_prefix} 화면입니다. 현재 화면과 비교해서 진행 흐름/변화를 파악하세요."
            return (
                f"[{full_label} 첨부됨]\n"
                f"Read 툴로 {file_path} 파일을 읽어 {desc}\n"
                f"[/{full_label} 첨부됨]"
            )
        return f"[{full_label} — 내용은 이미 처리됨]"

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
- **내 방송**의 후원/구독 이벤트: 누가, 얼마를, 어떤 말과 함께, 몇 번째 후원인지 (예: "몽팬님 첫 후원 1000원 — 채팅/후원 테스트 겸")
- **내 방송**의 인상적인 시청자: 자주 오거나 특별한 교류가 있었던 사람과 그 이유

**[치지직 방송 보기] 섹션이 시스템 프롬프트에 포함된 상태에서는 내가 다른 스트리머 방송을 시청 중이라는 뜻입니다. 이 모드에서는 후원/구독/시청자 정보를 절대 archival에 저장하지 마세요. 그 후원·구독·시청자는 모두 다른 방송의 것이고 내 기록 대상이 아닙니다.**

저장 안 해도 되는 것: 지금 이 순간만의 상황, 이미 블록에 있는 내용의 반복

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

삭제 — 잘못된 기억이나 환각 기록을 정리할 때 사용.

**중요한 사용 규칙:**
1. **반드시 먼저 `letta_archival_search`로 검색해서 실제 passage ID를 확보**한 뒤 삭제하세요.
2. 검색 결과 JSON 안의 `"id"` 필드 값을 정확히 복사해서 사용하세요 (예: `passage-323de23a-2b4c-4a73-89f0-88f8c6be2929`).
3. **절대로 `...`, `passage-...`, `passage-xxx` 같은 플레이스홀더로 호출하지 마세요.** 가짜 ID는 무조건 삭제 실패합니다.
4. 한 번에 여러 개 삭제하려면 태그를 여러 줄 나란히 적으면 됩니다.

올바른 사용 예시:
<letta_archival_delete id="passage-323de23a-2b4c-4a73-89f0-88f8c6be2929"/>
<letta_archival_delete id="passage-a6a1c2ed-f97d-461d-9720-2faad8cf3db2"/>

=== 대화 압축(Compact) 시 주의 ===

대화가 압축될 때 생성되는 요약에 반드시 아래 내용을 포함하세요:
1. `letta_memory_replace`가 실패로 끝났다면, 이후에 성공했는지 여부를 명확히 기록하세요. "Failed" 기록만 남기면 압축 후 불필요한 재시도를 유발합니다.
2. 현재 core memory 블록(persona, human)의 실제 내용을 verbatim으로 포함하세요. 압축 후 `letta_memory_replace` 사용 시 정확한 old 값 제공에 필요합니다.

메모리 태그는 응답 맨 끝에만 추가하세요.
</letta_memory_instructions>"""


# 스크린샷/유튜브 프레임 임시 파일 보존 시간(초).
# 한 대화 턴에서 Letta가 메모리 작업 등으로 codex exec를 여러 번 호출할 수 있기 때문에
# 즉시 삭제하면 후속 호출에서 파일이 사라져 "이전 화면 — 내용은 이미 처리됨"으로 표시된다.
# TTL을 두고 매 호출 시작 시 stale 파일만 정리한다.
_SCREENSHOT_FILE_TTL = 600  # 10분


def _cleanup_stale_screenshot_files() -> None:
    """TTL을 초과한 스크린샷/유튜브 프레임 임시 파일 삭제."""
    now = time.time()
    deleted = 0
    for pattern in ("/tmp/claude_screenshot_*", "/tmp/claude_yt_frame_*"):
        for f in glob.glob(pattern):
            try:
                if now - os.path.getmtime(f) > _SCREENSHOT_FILE_TTL:
                    os.unlink(f)
                    deleted += 1
            except Exception:
                pass
    if deleted:
        logger.debug(f"[CodexCLI] stale 스크린샷 {deleted}개 삭제")


class CodexCLIClient(LLMClientBase):
    """codex exec 서브프로세스를 LLM 백엔드로 사용하는 클라이언트."""

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
        system_prompt, user_content = self._format_request(messages, system)
        combined_for_scan = system_prompt + "\n" + user_content
        screenshot_files = re.findall(
            r"Read 툴로 (/tmp/claude_(?:screenshot|yt_frame)_[a-f0-9_]+\.jpg) 파일을", combined_for_scan
        )
        return {
            "system_prompt": system_prompt,
            "user_content": user_content,
            "model": llm_config.model or "",
            "screenshot_files": screenshot_files,
        }

    # ---------------------------------------------------------------------------
    # Sync/async request
    # ---------------------------------------------------------------------------

    def request(self, request_data: dict, llm_config: "LLMConfig") -> dict:
        return asyncio.get_event_loop().run_until_complete(self.request_async(request_data, llm_config))

    async def request_async(self, request_data: dict, llm_config: "LLMConfig") -> dict:
        if not shutil.which("codex"):
            raise LLMConnectionError(
                message="'codex' CLI를 PATH에서 찾을 수 없습니다. Codex CLI를 설치하세요.",
                code=ErrorCode.INTERNAL_SERVER_ERROR,
            )

        # 한 대화 턴 안에서 여러 LLM 호출이 같은 스크린샷 파일을 참조할 수 있도록
        # 즉시 삭제하지 않고, 매 호출 시작 시 TTL 초과 파일만 정리한다.
        _cleanup_stale_screenshot_files()

        system_prompt = request_data["system_prompt"]
        user_content = request_data["user_content"]
        model = request_data.get("model", "")
        screenshot_files: list = request_data.get("screenshot_files", [])

        output_path = f"/tmp/codex_cli_last_message_{uuid.uuid4().hex}.txt"
        prompt_text = (
            "<system_prompt>\n"
            f"{system_prompt}\n"
            "</system_prompt>\n\n"
            "<user_content>\n"
            f"{user_content}\n"
            "</user_content>\n\n"
            "위 system_prompt를 상위 지시로 따르고, user_content에만 답하세요. "
            "최종 답변만 출력하세요."
        )
        cmd = [
            "codex",
            "--search",
            "--ask-for-approval", "never",
            "exec",
            "--sandbox", "read-only",
            "--skip-git-repo-check",
            "--ephemeral",
            "--color", "never",
            "-C", "/tmp",
            "-o", output_path,
        ]
        if model and model != "codex-cli":
            cmd.extend(["--model", model])
        cmd.append("-")

        logger.info(
            f"[CodexCLI] 실행: codex exec <stdin({len(prompt_text)}chars)> "
            f"--model {model or '<default>'} (screenshots={len(screenshot_files)})"
        )
        TIMEOUT_SECONDS = 300  # 5분
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd="/tmp",  # 프로젝트 AGENTS.md/CLAUDE.md 컨텍스트 격리
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(prompt_text.encode("utf-8")),
                    timeout=TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()  # 버퍼 정리
                logger.error(f"[CodexCLI] {TIMEOUT_SECONDS}초 타임아웃 — 프로세스 강제 종료")
                raise LLMServerError(
                    message=f"codex CLI 타임아웃: {TIMEOUT_SECONDS}초 초과",
                    code=ErrorCode.INTERNAL_SERVER_ERROR,
                )
        except OSError as e:
            raise LLMConnectionError(
                message=f"codex CLI 실행 실패: {e}",
                code=ErrorCode.INTERNAL_SERVER_ERROR,
            )

        stdout_text = stdout.decode().strip()
        stderr_text = stderr.decode().strip()

        debug_prompt = f"=== SYSTEM_PROMPT ===\n{system_prompt}\n\n=== USER_CONTENT ===\n{user_content}"
        self._dump_debug_log(debug_prompt, stdout_text, stderr_text, proc.returncode)

        if proc.returncode != 0:
            error_detail = stderr_text or stdout_text
            logger.error(f"[CodexCLI] 실패 (exit {proc.returncode}): {error_detail}")
            raise LLMServerError(
                message=f"codex CLI 오류: {error_detail}",
                code=ErrorCode.INTERNAL_SERVER_ERROR,
            )

        try:
            with open(output_path, "r", encoding="utf-8") as f:
                result_text = f.read().strip()
        except FileNotFoundError:
            result_text = stdout_text.strip()
        finally:
            try:
                os.unlink(output_path)
            except FileNotFoundError:
                pass

        usage = {
            "input_tokens": max(1, len(prompt_text) // 4),
            "output_tokens": max(1, len(result_text) // 4) if result_text else 0,
        }
        return {"result": result_text, "usage": usage}

    @staticmethod
    def _dump_debug_log(prompt: str, stdout: str, stderr: str, returncode: Optional[int]) -> None:
        log_dir = os.environ.get("LETTA_CODEX_CLI_DEBUG_DIR", "/app/logs/codex_cli_debug")
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
            msg = f"[CodexCLI] 디버그 로그 저장: {path}"
            print(msg, flush=True)
            logger.info(msg)
        except Exception as e:
            msg = f"[CodexCLI] 디버그 로그 저장 실패: {type(e).__name__}: {e} (log_dir={log_dir})"
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

        prompt_tokens = (
            int(usage_raw.get("input_tokens", 0))
            + int(usage_raw.get("cache_creation_input_tokens", 0))
            + int(usage_raw.get("cache_read_input_tokens", 0))
        )
        completion_tokens = int(usage_raw.get("output_tokens", 0))

        # 응답에서 core memory 태그 파싱 및 제거
        clean_text, tool_calls = self._extract_memory_ops(text)

        finish_reason = "tool_calls" if tool_calls else "stop"

        return ChatCompletionResponse(
            id=f"codex-cli-{int(time.time())}",
            object="chat.completion",
            created=int(time.time()),
            model=llm_config.model or "codex-cli",
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

        # archival_memory_delete 태그 처리 (속성 id="..." 또는 본문에 passage_id)
        # 명백한 플레이스홀더(요약·예시 인용)는 건너뜀 — 진짜 UUID 형식만 허용.
        _passage_uuid_re = re.compile(r"^passage-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
        for match in _ARCHIVAL_DELETE_RE.finditer(text):
            passage_id = (match.group(1) or match.group(2) or "").strip()
            if not passage_id:
                continue
            if not _passage_uuid_re.match(passage_id):
                logger.warning(
                    f"[CodexCLI] archival_memory_delete 무시 — 유효한 passage ID 형식 아님: {passage_id!r}"
                )
                continue
            tool_calls.append(
                ToolCall(
                    id=_make_id(),
                    type="function",
                    function=FunctionCall(
                        name="archival_memory_delete",
                        arguments=json.dumps({"passage_id": passage_id}, ensure_ascii=False),
                    ),
                )
            )

        if tool_calls:
            # 태그 전체를 응답 텍스트에서 제거
            clean = _APPEND_RE.sub("", text)
            clean = _REPLACE_RE.sub("", clean)
            clean = _ARCHIVAL_INSERT_RE.sub("", clean)
            clean = _ARCHIVAL_SEARCH_RE.sub("", clean)
            clean = _ARCHIVAL_DELETE_RE.sub("", clean)
            clean = clean.rstrip()
            logger.info(f"[CodexCLI] 메모리 작업 {len(tool_calls)}개 파싱: {[tc.function.name for tc in tool_calls]}")
        else:
            clean = text

        return clean, tool_calls

    # ---------------------------------------------------------------------------
    # Streaming (미지원 — SimpleLLMRequestAdapter로 교체됨)
    # ---------------------------------------------------------------------------

    async def stream_async(self, request_data: dict, llm_config: "LLMConfig"):
        raise NotImplementedError("CodexCLIClient는 스트리밍을 지원하지 않습니다.")

    # ---------------------------------------------------------------------------
    # Embeddings (미지원)
    # ---------------------------------------------------------------------------

    async def request_embeddings(self, texts: List[str], embedding_config: "EmbeddingConfig") -> List[List[float]]:
        raise NotImplementedError("CodexCLIClient는 임베딩을 지원하지 않습니다.")

    # ---------------------------------------------------------------------------
    # Misc
    # ---------------------------------------------------------------------------

    def is_reasoning_model(self, llm_config: "LLMConfig") -> bool:
        return False

    def handle_llm_error(self, e: Exception, llm_config: Optional["LLMConfig"] = None) -> Exception:
        if isinstance(e, LLMError):
            return e
        return LLMBadRequestError(
            message=f"codex CLI 오류: {e}",
            code=ErrorCode.INTERNAL_SERVER_ERROR,
        )

    # ---------------------------------------------------------------------------
    # Message formatting
    # ---------------------------------------------------------------------------

    def _format_request(
        self, messages: List[Message], system: Optional[str] = None
    ) -> Tuple[str, str]:
        """Letta 메시지를 (system_prompt, user_content) 튜플로 변환.

        system_prompt: Codex 프롬프트 안에 상위 지시 블록으로 전달될 시스템 텍스트
        user_content: Codex 프롬프트 안에 전달될 사용자 메시지 (대화 히스토리 + 현재 입력 포함, XML 태그 구조)

        H:/A: transcript 패턴을 쓰지 않아 LLM의 환각 대화 생성 경향을 줄임.
        """
        # 시스템 프롬프트 추출
        sys_content = system or ""
        for msg in messages:
            role = self._get_field(msg, "role")
            if role == "system":
                sys_content = self._get_content(msg) or sys_content
                break

        system_prompt = f"{sys_content}{MEMORY_INSTRUCTIONS}" if sys_content else ""

        # 시스템 외 메시지를 history와 current_input으로 분리
        non_system: List[Tuple[str, str, list]] = []
        for msg in messages:
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
            non_system.append((role, content, tool_calls))

        # 마지막 user 메시지를 current_input으로 분리
        last_user_idx = -1
        for i in range(len(non_system) - 1, -1, -1):
            if non_system[i][0] == "user" and non_system[i][1].strip():
                last_user_idx = i
                break

        history_msgs = non_system[:last_user_idx] if last_user_idx >= 0 else non_system
        current_msg = non_system[last_user_idx] if last_user_idx >= 0 else None

        # history 직렬화 — XML 태그로 감싸서 transcript 패턴 회피
        history_parts: List[str] = []
        for role, content, tool_calls in history_msgs:
            if role == "user":
                if not content.strip():
                    continue
                history_parts.append(f"<user_message>\n{content}\n</user_message>")
            elif role == "assistant":
                if tool_calls:
                    call_strs = []
                    for tc in tool_calls:
                        fn = self._get_field(tc, "function") or {}
                        name = self._get_field(fn, "name") or ""
                        args = self._get_field(fn, "arguments") or ""
                        call_strs.append(f"<tool_call name=\"{name}\">{args}</tool_call>")
                    body = (content + "\n" if content else "") + "\n".join(call_strs)
                    history_parts.append(f"<assistant_message>\n{body}\n</assistant_message>")
                else:
                    if not content.strip():
                        continue
                    history_parts.append(f"<assistant_message>\n{content}\n</assistant_message>")
            elif role == "tool":
                history_parts.append(f"<tool_result>\n{content}\n</tool_result>")

        user_content_parts: List[str] = []
        if history_parts:
            user_content_parts.append("<conversation_history>")
            user_content_parts.extend(history_parts)
            user_content_parts.append("</conversation_history>")

        if current_msg is not None:
            user_content_parts.append(f"<current_input>\n{current_msg[1]}\n</current_input>")

        user_content = "\n\n".join(user_content_parts)
        return system_prompt, user_content

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

        content = CodexCLIClient._get_field(msg, "content") or ""

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
