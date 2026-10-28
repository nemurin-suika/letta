import json
import stat

import pytest

from letta.llm_api.codex_cli_client import _write_codex_usage


@pytest.fixture
def usage_paths(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "config.toml").write_text('model_provider = "example"\nmodel = "example-model"\n')
    home = tmp_path / "persistent"
    monkeypatch.setenv("VTUBER_CODEX_USAGE_HOME", str(home))
    return runtime, home


def read_record(home):
    paths = list((home / "usage").glob("*/*/*/*.json"))
    assert len(paths) == 1
    assert stat.S_IMODE(paths[0].stat().st_mode) == 0o600
    assert not list((home / "usage").rglob("*.tmp"))
    return json.loads(paths[0].read_text())


def test_reported_usage_is_private_and_excludes_content(usage_paths):
    runtime, home = usage_paths
    _write_codex_usage(str(runtime), "selected-model", {
        "input_tokens": 100, "cached_input_tokens": 40, "output_tokens": 25,
        "prompt": "not for storage", "answer": "not for storage",
    }, True)
    record = read_record(home)
    assert record["model"] == "selected-model"
    assert record["model_provider"] == "example"
    assert record["usage_status"] == "complete"
    assert record["usage"]["total_tokens"] == 125
    assert record["usage"]["cached_input_tokens"] == 40
    assert "not for storage" not in json.dumps(record)


@pytest.mark.parametrize("usage", [None, {}, {"input_tokens": 1},
    {"input_tokens": -1, "output_tokens": 2},
    {"input_tokens": True, "output_tokens": 2},
    {"input_tokens": 10, "output_tokens": 2, "cached_input_tokens": 11},
    {"input_tokens": 10, "output_tokens": 2, "reasoning_output_tokens": 3},
    {"input_tokens": 0, "output_tokens": 0, "total_tokens": 100},
])
def test_missing_or_invalid_usage_is_not_free(usage_paths, usage):
    runtime, home = usage_paths
    _write_codex_usage(str(runtime), "codex-cli", usage, True)
    record = read_record(home)
    assert record["usage_status"] == "unavailable"
    assert record["usage"] is None
    assert record["model"] == "example-model"


def test_failed_call_with_tokens_is_partial(usage_paths):
    runtime, home = usage_paths
    _write_codex_usage(str(runtime), "selected-model", {"input_tokens": 10, "output_tokens": 2}, False)
    assert read_record(home)["usage_status"] == "partial"


def test_zero_reported_usage_is_valid(usage_paths):
    runtime, home = usage_paths
    _write_codex_usage(str(runtime), "selected-model", {"input_tokens": 0, "output_tokens": 0}, True)
    record = read_record(home)
    assert record["usage_status"] == "complete"
    assert record["usage"]["total_tokens"] == 0


def test_recording_failure_does_not_break_chat(usage_paths):
    runtime, home = usage_paths
    (runtime / "config.toml").unlink()
    _write_codex_usage(str(runtime), "selected-model", {}, False)
    assert not (home / "usage").exists()


def test_home_override_is_not_temporary_home(usage_paths, monkeypatch):
    runtime, home = usage_paths
    monkeypatch.setenv("CODEX_HOME", str(runtime))
    _write_codex_usage(str(runtime), "selected-model", {"input_tokens": 1, "output_tokens": 2}, True)
    assert read_record(home)["usage"]["total_tokens"] == 3
    assert not (runtime / "usage").exists()
