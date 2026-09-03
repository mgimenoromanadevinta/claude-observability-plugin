from __future__ import annotations

import json


def _write_jsonl(path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_subagent_observations_recurse_into_grandchild_agent(
    hook_module,
    fake_langfuse,
    tmp_path,
):
    """A subagent that itself launches an Agent tool_use (a 3rd-level, or
    "grandchild", agent) must have that nested agent's own generations
    emitted too — otherwise its tokens/cost/tool spans never reach Langfuse.

    Builds a synthetic child-agent transcript (agent-child.jsonl) that
    launches a grandchild agent, resolved via a subagents/ dir sitting next
    to agent-child's own jsonl — the same layout Claude Code uses one level
    up (transcript.jsonl -> transcript/subagents/).
    """
    child_jsonl = tmp_path / "subagents" / "agent-child.jsonl"
    _write_jsonl(child_jsonl, [
        {
            "type": "user",
            "timestamp": "2026-01-01T00:02:01.500Z",
            "sessionId": "session-child",
            "agentId": "child",
            "uuid": "child-user-1",
            "message": {"role": "user", "content": "Summarize the docs"},
        },
        {
            "type": "assistant",
            "timestamp": "2026-01-01T00:02:05.000Z",
            "sessionId": "session-child",
            "agentId": "child",
            "uuid": "child-assistant-1",
            "requestId": "req-child-1",
            "message": {
                "id": "msg-child-1",
                "role": "assistant",
                "model": "claude-test",
                "content": [{
                    "type": "tool_use",
                    "id": "toolu_grandchild_agent",
                    "name": "Agent",
                    "input": {"description": "Grandchild work"},
                }],
            },
        },
        {
            "type": "user",
            "timestamp": "2026-01-01T00:02:06.000Z",
            "sessionId": "session-child",
            "agentId": "child",
            "uuid": "child-tool-result-1",
            "message": {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "toolu_grandchild_agent",
                    "content": "Grandchild agent launched.",
                }],
            },
        },
        {
            "type": "assistant",
            "timestamp": "2026-01-01T00:02:07.000Z",
            "sessionId": "session-child",
            "agentId": "child",
            "uuid": "child-assistant-2",
            "requestId": "req-child-2",
            "message": {
                "id": "msg-child-2",
                "role": "assistant",
                "model": "claude-test",
                "content": [{"type": "text", "text": "Done, grandchild finished too."}],
            },
        },
    ])

    # Grandchild agent transcript, nested under the child's own subagents/ dir.
    grandchild_jsonl = tmp_path / "subagents" / "agent-child" / "subagents" / "agent-grandchild.jsonl"
    grandchild_jsonl.parent.mkdir(parents=True)
    (grandchild_jsonl.parent / "agent-grandchild.meta.json").write_text(
        json.dumps({
            "agentType": "general-purpose",
            "description": "Grandchild work",
            "toolUseId": "toolu_grandchild_agent",
            "spawnDepth": 2,
        }),
        encoding="utf-8",
    )
    _write_jsonl(grandchild_jsonl, [
        {
            "type": "user",
            "timestamp": "2026-01-01T00:02:05.500Z",
            "sessionId": "session-grandchild",
            "agentId": "grandchild",
            "uuid": "grandchild-user-1",
            "message": {"role": "user", "content": "Do the grandchild work"},
        },
        {
            "type": "assistant",
            "timestamp": "2026-01-01T00:02:06.500Z",
            "sessionId": "session-grandchild",
            "agentId": "grandchild",
            "uuid": "grandchild-assistant-1",
            "requestId": "req-grandchild-1",
            "message": {
                "id": "msg-grandchild-1",
                "role": "assistant",
                "model": "claude-test",
                "content": [{"type": "text", "text": "Grandchild is done."}],
                "usage": {"input_tokens": 11, "output_tokens": 7},
            },
        },
    ])

    hook_module.emit_subagent_observations(
        fake_langfuse,
        None,
        {
            "path": child_jsonl,
            "description": "Summarize docs",
            "agent_type": "general-purpose",
        },
        None,
    )

    names = [observation.name for observation in fake_langfuse.observations]
    assert "Subagent: Grandchild work" in names, (
        "grandchild agent span missing: nested Agent/Task launches inside a "
        "subagent transcript were not resolved to their own subagent transcript"
    )

    grandchild_generations = [
        observation for observation in fake_langfuse.observations
        if observation.name == "Subagent LLM Call"
        and observation.kwargs.get("usage_details") == {"input": 11, "output": 7}
    ]
    assert grandchild_generations, (
        "grandchild agent's LLM generation (tokens/cost) was not emitted"
    )
