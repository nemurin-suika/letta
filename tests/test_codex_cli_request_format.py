from letta.llm_api.codex_cli_client import CodexCLIClient


def test_tool_results_after_last_user_are_included_in_codex_prompt():
    client = CodexCLIClient()
    messages = [
        {"role": "system", "content": "system instructions"},
        {"role": "user", "content": "earlier request"},
        {"role": "assistant", "content": "earlier answer"},
        {"role": "user", "content": "search for the records"},
        {
            "role": "assistant",
            "content": "searching",
            "tool_calls": [{"function": {"name": "archival_memory_search", "arguments": '{"query":"record"}'}}],
        },
        {"role": "tool", "content": "first search result"},
        {"role": "assistant", "content": "searching again"},
        {"role": "tool", "content": "second search result"},
    ]

    _, prompt = client._format_request(messages)

    assert prompt.index("earlier request") < prompt.index("earlier answer")
    assert prompt.index("earlier answer") < prompt.index("<current_input>")
    assert prompt.index("search for the records") < prompt.index("<continuation>")
    assert prompt.index('name="archival_memory_search"') < prompt.index("first search result")
    assert prompt.index("first search result") < prompt.index("searching again") < prompt.index("second search result")


def test_last_user_without_following_tools_has_no_continuation():
    client = CodexCLIClient()
    _, prompt = client._format_request([
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answered"},
        {"role": "user", "content": "next"},
    ])

    assert "<continuation>" not in prompt
    assert "<current_input>\nnext\n</current_input>" in prompt


def test_memory_instructions_keep_memory_xml_out_of_progress_messages():
    client = CodexCLIClient()

    system_prompt, _ = client._format_request([
        {"role": "system", "content": "system instructions"},
        {"role": "user", "content": "next"},
    ])

    assert "중간 메시지 태그 금지" in system_prompt
    assert "도구 실행 전 설명" in system_prompt
    assert "최종 응답 맨 끝에서만 변환됩니다" in system_prompt
    assert "중간 메시지에 검색 태그를 출력하지 말고" in system_prompt
