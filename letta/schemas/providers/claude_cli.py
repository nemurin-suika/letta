import shutil
from typing import Literal

from pydantic import Field

from letta.schemas.enums import ProviderCategory, ProviderType
from letta.schemas.llm_config import LLMConfig
from letta.schemas.providers.base import Provider

# claude -p로 사용 가능한 모델 목록 (handle 형식: claude_cli/<model>)
# 모델명은 claude CLI의 --model 플래그에 그대로 전달됨.
# "claude-cli"는 --model 없이 CLI 기본값을 사용하는 특수 값.
CLAUDE_CLI_MODELS = {
    "claude-cli": 200000,                    # CLI 기본 모델 (플래그 없음)
    "claude-opus-4-5": 200000,
    "claude-sonnet-4-5": 200000,
    "claude-haiku-4-5": 200000,
}


class ClaudeCLIProvider(Provider):
    """
    `claude -p` 서브프로세스를 LLM 백엔드로 사용하는 프로바이더.
    API 키 불필요 — Claude Code CLI 구독으로 동작.
    """

    provider_type: Literal[ProviderType.claude_cli] = Field(
        ProviderType.claude_cli, description="The type of the provider."
    )
    provider_category: ProviderCategory = Field(
        ProviderCategory.base, description="The category of the provider."
    )

    async def check_api_key(self):
        if not shutil.which("claude"):
            raise RuntimeError("'claude' CLI를 PATH에서 찾을 수 없습니다. Claude Code CLI를 설치하세요.")

    def get_default_max_output_tokens(self, model_name: str) -> int:
        return 8192

    async def list_llm_models_async(self) -> list[LLMConfig]:
        configs = []
        for model, context_window in CLAUDE_CLI_MODELS.items():
            configs.append(
                LLMConfig(
                    model=model,
                    model_endpoint_type="claude_cli",
                    model_endpoint="",
                    context_window=context_window,
                    handle=self.get_handle(model),
                    max_tokens=self.get_default_max_output_tokens(model),
                    provider_name=self.name,
                    provider_category=self.provider_category,
                )
            )
        return configs

    async def list_embedding_models_async(self):
        return []

    async def get_model_context_window_async(self, model_name: str) -> int:
        return CLAUDE_CLI_MODELS.get(model_name, 200000)
