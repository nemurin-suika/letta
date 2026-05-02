from typing import Literal

from pydantic import Field

from letta.llm_api.google_constants import GOOGLE_LIVE_MODEL_TO_CONTEXT_LENGTH
from letta.schemas.enums import ProviderCategory, ProviderType
from letta.schemas.llm_config import LLMConfig
from letta.schemas.providers.base import Provider


class GoogleLiveProvider(Provider):
    """
    Provider for Google Gemini Live API (bidiGenerateContent).
    Uses a static model list because Live models only support bidiGenerateContent
    and are filtered out by the standard generateContent discovery endpoint.
    """

    provider_type: Literal[ProviderType.google_live] = Field(
        ProviderType.google_live, description="The type of the provider."
    )
    provider_category: ProviderCategory = Field(
        ProviderCategory.base, description="The category of the provider (base or byok)"
    )
    base_url: str = "https://generativelanguage.googleapis.com"

    async def check_api_key(self):
        from letta.llm_api.google_ai_client import google_ai_check_valid_api_key_async

        api_key = await self.api_key_enc.get_plaintext_async() if self.api_key_enc else None
        await google_ai_check_valid_api_key_async(api_key)

    def get_default_max_output_tokens(self, model_name: str) -> int:
        return 65536

    async def list_llm_models_async(self) -> list[LLMConfig]:
        configs = []
        for model, context_length in GOOGLE_LIVE_MODEL_TO_CONTEXT_LENGTH.items():
            configs.append(
                LLMConfig(
                    model=model,
                    model_endpoint_type="google_live",
                    model_endpoint=self.base_url,
                    context_window=context_length,
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
        return GOOGLE_LIVE_MODEL_TO_CONTEXT_LENGTH.get(model_name, 131072)
