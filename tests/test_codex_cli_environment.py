import os
import subprocess
from unittest.mock import patch

from letta.llm_api.codex_cli_client import _codex_user_environment


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
