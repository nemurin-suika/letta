import json
import os
import shutil
import subprocess
import tomllib
from unittest.mock import patch

from letta.llm_api.codex_cli_client import (
    CodexCLIClient,
    _codex_user_environment,
    _prepare_codex_home,
    _requested_mcp_servers,
)


def test_codex_uses_real_home_and_user_shell_path():
    shell_output = b"HOME=/Users/nemurin\0PATH=/Users/nemurin/.rd/bin:/usr/bin\0JAVA_HOME=/opt/java\0LGF_GH_TOKEN=secret\0"
    with patch.dict(os.environ, {"HOME": "/Users/nemurin/ai", "MIZUKI_REAL_HOME": "/Users/nemurin"}):
        with patch("letta.llm_api.codex_cli_client.subprocess.run") as run:
            run.return_value.stdout = shell_output
            environment = _codex_user_environment()

    assert environment == {
        "HOME": "/Users/nemurin",
        "PATH": "/Users/nemurin/.rd/bin:/usr/bin",
        "JAVA_HOME": "/opt/java",
    }
    assert run.call_args.kwargs["env"]["HOME"] == "/Users/nemurin"
    assert run.call_args.args[0] == ["/bin/zsh", "-lic", "/usr/bin/env -0"]


def test_codex_keeps_inherited_path_if_user_shell_fails():
    with patch.dict(os.environ, {"HOME": "/Users/nemurin/ai", "MIZUKI_REAL_HOME": "/Users/nemurin"}):
        with patch("letta.llm_api.codex_cli_client.subprocess.run", side_effect=subprocess.TimeoutExpired("zsh", 10)):
            environment = _codex_user_environment()

    assert environment == {"HOME": "/Users/nemurin"}


def test_mcp_selection_uses_only_latest_user_message():
    messages = [
        {"role": "user", "content": "wiki.example.com"},
        {"role": "tool", "content": "workspace.slack.com"},
        {"role": "user", "content": "오늘은 슬랙을 확인해줘"},
    ]
    assert _requested_mcp_servers(messages) == ("slack",)
    assert _requested_mcp_servers([{"role": "user", "content": "다른 작업"}]) == ()
    assert _requested_mcp_servers([{"role": "user", "content": "wiki.example.com 과 workspace.slack.com"}]) == (
        "confluence",
        "slack",
    )
    assert _requested_mcp_servers([{"role": "user", "content": "fakewiki.example.com"}]) == ()


def test_build_request_selects_mcp_from_user_not_system_or_tools():
    client = CodexCLIClient()
    messages = [
        {"role": "system", "content": "wiki.example.com"},
        {"role": "user", "content": "슬랙 확인"},
        {"role": "tool", "content": "wiki.example.com"},
    ]
    config = type("Config", (), {"model": "codex-cli"})()
    request = client.build_request_data(None, messages, config, [])
    assert request["mcp_servers"] == ("slack",)


def test_temporary_codex_home_contains_only_requested_mcp(tmp_path):
    claude_config = tmp_path / "claude.json"
    claude_config.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "confluence": {
                        "type": "stdio",
                        "command": "confluence-server",
                        "args": ["--api-key", "fake-secret"],
                        "env": {"TOKEN": "test-token"},
                        "timeout": 300,
                    },
                    "slack": {"type": "stdio", "command": "slack-server", "args": []},
                    "google-drive": {"type": "stdio", "command": "drive-server", "args": []},
                }
            }
        )
    )
    codex_home = tmp_path / "real_codex"
    codex_home.mkdir()
    (codex_home / "config.toml").write_text(
        '[mcp_servers.inherited]\ncommand = "inherited-server"\n'
        '[mcp_servers.inherited.env]\nTOKEN = "inherited-token"\n'
        "[features]\nhooks = true\n"
    )
    with (
        patch.dict(os.environ, {"CODEX_HOME": str(codex_home)}),
        patch("letta.llm_api.codex_cli_client._CLAUDE_MCP_CONFIG", str(claude_config)),
    ):
        for requested, expected in [((), set()), (("confluence",), {"confluence"}), (("slack",), {"slack"})]:
            temp_home = _prepare_codex_home(requested)
            try:
                config_path = os.path.join(temp_home, "config.toml")
                with open(config_path, "rb") as config_file:
                    config = tomllib.load(config_file)
                assert set(config.get("mcp_servers", {})) == expected
                assert config["features"]["hooks"] is True
                assert os.stat(temp_home).st_mode & 0o777 == 0o700
                assert os.stat(config_path).st_mode & 0o777 == 0o600
                if requested == ("confluence",):
                    assert config["mcp_servers"]["confluence"]["env"]["TOKEN"] == "test-token"
                    assert config["mcp_servers"]["confluence"]["startup_timeout_sec"] == 300
            finally:
                shutil.rmtree(temp_home)


def test_unavailable_mcp_does_not_get_added(tmp_path):
    claude_config = tmp_path / "claude.json"
    claude_config.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "confluence": {"disabled": True, "type": "stdio", "command": "server"},
                    "slack": {"type": "stdio", "command": "server", "env": {"TOKEN": 123}},
                }
            }
        )
    )
    with patch("letta.llm_api.codex_cli_client._CLAUDE_MCP_CONFIG", str(claude_config)):
        temp_home = _prepare_codex_home(("confluence", "slack"))
        try:
            with open(os.path.join(temp_home, "config.toml"), "rb") as config_file:
                assert "mcp_servers" not in tomllib.load(config_file)
        finally:
            shutil.rmtree(temp_home)
