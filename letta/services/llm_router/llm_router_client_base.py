"""Base LLM routing client.

This is the base (OSS/self-hosted) implementation without Redis support.
The Redis-backed implementation lives in llm_router_client.py.
"""

import json
import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from letta.schemas.llm_config import LLMConfig
    from letta.schemas.user import User

logger = logging.getLogger(__name__)

_STATIC_FALLBACK_HANDLE = "local/fallback"


class LLMRoutingClient:
    """Base LLM routing client.

    Used when Redis is not configured (OSS/self-hosted).
    Auto mode is not supported without Redis.
    """

    async def resolve_auto_mode_config(
        self,
        stored_llm_config: "LLMConfig",
        actor: "User",
    ) -> tuple["LLMConfig", bool, str]:
        """Resolve an auto mode handle to an actual model config.

        Args:
            stored_llm_config: The agent's stored LLM config (with auto mode handle).
            actor: The user actor for provider lookups.

        Returns:
            Tuple of (resolved_config, is_primary, primary_handle).

        Raises:
            RuntimeError: Auto mode requires Redis for circuit breaker support.
        """
        raise RuntimeError(
            "Auto mode requires Redis for circuit breaker support. Configure Redis or disable auto_mode_enabled in settings."
        )

    async def record_failure(self, handle: str) -> None:
        """Record a failure for a model handle (noop in base)."""

    async def record_success(self, handle: str) -> None:
        """Record a success for a model handle (noop in base)."""

    async def get_fallback_config(
        self,
        stored_llm_config: "LLMConfig",
        actor: "User",
    ) -> "LLMConfig":
        """Get the fallback config for auto mode.

        Args:
            stored_llm_config: The agent's stored LLM config.
            actor: The user actor for provider lookups.

        Raises:
            RuntimeError: Auto mode requires Redis for circuit breaker support.
        """
        raise RuntimeError(
            "Auto mode requires Redis for circuit breaker support. Configure Redis or disable auto_mode_enabled in settings."
        )

    def get_fallback_handle(self, handle: Optional[str]) -> Optional[str]:
        """Get the fallback handle for a given primary handle.

        Reads LLM_FALLBACK_CONFIG and LLM_FALLBACK_TRIGGER_PROVIDERS from settings.
        Returns a static fallback handle when the primary handle's provider prefix matches
        a configured trigger provider.
        """
        from letta.settings import model_settings

        if not model_settings.llm_fallback_config:
            logger.debug("[LLM ROUTER] llm_fallback_config 미설정, fallback 비활성화")
            return None

        if not handle:
            logger.warning("[LLM ROUTER] handle이 None/빈값, fallback 매칭 불가")
            return None

        trigger_providers = {p.strip() for p in model_settings.llm_fallback_trigger_providers.split(",") if p.strip()}
        provider_prefix = handle.split("/")[0] if "/" in handle else handle
        logger.debug("[LLM ROUTER] fallback 체크: handle=%s prefix=%s triggers=%s", handle, provider_prefix, trigger_providers)
        if provider_prefix in trigger_providers:
            return _STATIC_FALLBACK_HANDLE
        return None

    async def get_fallback_config_for_handle(
        self,
        fallback_handle: str,
        stored_llm_config: "LLMConfig",
        actor: "User",
    ) -> "LLMConfig":
        """Get a fallback config for the static local fallback handle.

        Parses the JSON stored in LETTA_LLM_FALLBACK_CONFIG and returns it as an LLMConfig.
        """
        from letta.schemas.llm_config import LLMConfig
        from letta.settings import model_settings

        if fallback_handle == _STATIC_FALLBACK_HANDLE and model_settings.llm_fallback_config:
            try:
                config_dict = json.loads(model_settings.llm_fallback_config)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"LETTA_LLM_FALLBACK_CONFIG is not valid JSON: {e}") from e
            if "handle" not in config_dict:
                config_dict["handle"] = _STATIC_FALLBACK_HANDLE
            logger.info("Loading static fallback LLM config: model=%s", config_dict.get("model"))
            return LLMConfig(**config_dict)

        raise RuntimeError(
            "Fallback routing requires Redis for circuit breaker support, "
            "or LETTA_LLM_FALLBACK_CONFIG to be set for static fallback."
        )

    def apply_reroute_rules(self, resolved_config, messages, stored_llm_config, agent_state):
        """Apply content-based reroute rules (noop in base)."""
        return resolved_config
