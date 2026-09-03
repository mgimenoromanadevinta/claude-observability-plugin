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

    Real Claude Code stores every agent's meta/jsonl flat in the top-level
    session's single subagents/ dir, keyed by globally-unique toolUseId
    (with spawnDepth/parentAgentId marking nesting) — not in a per-agent
    nested subagents/ dir. So the caller must pass the flat
    subagent_transcripts_by_tool_use_id map explicitly; it is threaded
    unchanged through every recursion level rather than re-resolved per
    child.
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

    # Grandchild agent transcript. Real Claude Code stores every nested
    # agent's meta/jsonl flat in the *top-level* session's subagents/ dir,
    # keyed by its globally-unique toolUseId — never in a subagents/ dir
    # nested under the parent agent's own transcript.
    grandchild_jsonl = tmp_path / "subagents" / "agent-grandchild.jsonl"
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

    grandchild_transcripts_by_tool_use_id = {
        "toolu_grandchild_agent": {
            "path": grandchild_jsonl,
            "description": "Grandchild work",
            "agent_type": "general-purpose",
        },
    }

    hook_module.emit_subagent_observations(
        fake_langfuse,
        None,
        {
            "path": child_jsonl,
            "description": "Summarize docs",
            "agent_type": "general-purpose",
        },
        None,
        subagent_transcripts_by_tool_use_id=grandchild_transcripts_by_tool_use_id,
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
