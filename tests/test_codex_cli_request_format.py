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
