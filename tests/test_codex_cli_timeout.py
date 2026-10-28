import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from letta.llm_api.codex_cli_client import CodexCLIClient


@pytest.mark.asyncio
async def test_timeout_does_not_read_stdout_twice(tmp_path):
    process = SimpleNamespace(
        returncode=None,
        kill=Mock(),
        wait=AsyncMock(),
        communicate=AsyncMock(),
    )
    request_data = {
        "system_prompt": "system",
        "user_content": "user",
        "model": "gpt-6-sol",
        "screenshot_files": [],
        "session_key": "test-timeout",
    }

    with (
        patch("letta.llm_api.codex_cli_client.shutil.which", return_value="/usr/bin/codex"),
        patch("letta.llm_api.codex_cli_client._prepare_codex_home", return_value=str(tmp_path)),
        patch("letta.llm_api.codex_cli_client._codex_user_environment", return_value={}),
        patch("letta.llm_api.codex_cli_client._write_codex_usage") as write_usage,
        patch("letta.llm_api.codex_cli_client.asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=process),
        patch.object(CodexCLIClient, "_communicate_streaming_codex", new_callable=AsyncMock, side_effect=asyncio.TimeoutError),
    ):
        with pytest.raises(Exception, match="타임아웃"):
            await CodexCLIClient().request_async(request_data, SimpleNamespace())

    process.kill.assert_called_once_with()
    process.wait.assert_awaited_once_with()
    process.communicate.assert_not_awaited()
    write_usage.assert_called_once_with(str(tmp_path), request_data["model"], None, completed=False)
