"""Focused tests for the opt-in task-local skill-state projection.

Covers: activation, path/file refusal, canonical state and merge-patch
behavior, request-only projection, full-history fallback, and the
system-prompt opt-in instruction.
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import agent.skill_state as ss


# ── helpers ─────────────────────────────────────────────────────────


def _home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _write_doc(path, state=None, revision=0):
    doc = {"version": 1, "revision": revision, "state": state if state is not None else {}}
    path.write_text(json.dumps(doc), encoding="utf-8")
    os.chmod(path, 0o600)
    return doc


def _patch_content(patch, revision=0):
    payload = json.dumps({"revision": revision, "patch": patch})
    return (
        "Working on the task.\n"
        + ss.BEGIN_MARKER
        + payload
        + ss.END_MARKER
        + "\nDone for now."
    )


def _conversation(content=None):
    msgs = [{"role": "user", "content": "Do the task."}]
    if content is not None:
        msgs.append(
            {
                "role": "assistant",
                "content": content,
                "tool_calls": [
                    {"id": "call_1", "type": "function",
                     "function": {"name": "execute_code", "arguments": "{}"}}
                ],
            }
        )
        msgs.append({"role": "tool", "tool_call_id": "call_1", "content": "ok"})
    return msgs


def _agent():
    return SimpleNamespace()


# ── activation ──────────────────────────────────────────────────────


def test_inactive_when_env_unset(tmp_path, monkeypatch):
    _home(tmp_path, monkeypatch)
    monkeypatch.delenv(ss.ENV_VAR, raising=False)
    assert ss.resolve_state_path() is None
    msgs = _conversation(_patch_content({"a": 1}))
    assert ss.apply_skill_state_projection(_agent(), msgs) is False
    assert len(msgs) == 3  # untouched


def test_active_with_valid_setup(tmp_path, monkeypatch):
    home = _home(tmp_path, monkeypatch)
    doc_path = home / "skill-state.json"
    _write_doc(doc_path)
    monkeypatch.setenv(ss.ENV_VAR, str(doc_path))
    assert ss.resolve_state_path() == doc_path.resolve()
    msgs = _conversation(_patch_content({"a": 1}))
    assert ss.apply_skill_state_projection(_agent(), msgs) is True


# ── path/file refusal ───────────────────────────────────────────────


def test_refuses_relative_path(tmp_path, monkeypatch):
    _home(tmp_path, monkeypatch)
    doc = tmp_path / "state.json"
    _write_doc(doc)
    monkeypatch.setenv(ss.ENV_VAR, "state.json")
    assert ss.resolve_state_path() is None


def test_refuses_path_outside_hermes_home(tmp_path, monkeypatch):
    _home(tmp_path, monkeypatch)
    outside = tmp_path / "elsewhere" / "state.json"
    outside.parent.mkdir()
    _write_doc(outside)
    monkeypatch.setenv(ss.ENV_VAR, str(outside))
    assert ss.resolve_state_path() is None


def test_refuses_symlink(tmp_path, monkeypatch):
    home = _home(tmp_path, monkeypatch)
    target = home / "real.json"
    _write_doc(target)
    link = home / "link.json"
    link.symlink_to(target)
    monkeypatch.setenv(ss.ENV_VAR, str(link))
    assert ss.resolve_state_path() is None


def test_refuses_group_or_world_permissions(tmp_path, monkeypatch):
    home = _home(tmp_path, monkeypatch)
    doc = home / "state.json"
    _write_doc(doc)
    os.chmod(doc, 0o644)
    monkeypatch.setenv(ss.ENV_VAR, str(doc))
    assert ss.resolve_state_path() is None


def test_refuses_missing_file(tmp_path, monkeypatch):
    home = _home(tmp_path, monkeypatch)
    monkeypatch.setenv(ss.ENV_VAR, str(home / "nope.json"))
    assert ss.resolve_state_path() is None


# ── canonical document & merge patch ────────────────────────────────


def test_read_refuses_noncanonical_document(tmp_path, monkeypatch):
    home = _home(tmp_path, monkeypatch)
    doc_path = home / "state.json"
    _write_doc(doc_path)
    monkeypatch.setenv(ss.ENV_VAR, str(doc_path))

    for bad in [
        {"version": 2, "revision": 0, "state": {}},
        {"version": 1, "revision": -1, "state": {}},
        {"version": 1, "revision": "0", "state": {}},
        {"version": 1, "revision": 0, "state": []},
        {"version": 1, "revision": 0},
        {"version": 1, "revision": 0, "state": {}, "extra": 1},
    ]:
        doc_path.write_text(json.dumps(bad))
        assert ss.read_state_document(doc_path) is None

    doc_path.write_text("not json")
    assert ss.read_state_document(doc_path) is None


def test_write_is_canonical_and_atomic(tmp_path, monkeypatch):
    home = _home(tmp_path, monkeypatch)
    doc_path = home / "state.json"
    _write_doc(doc_path)
    monkeypatch.setenv(ss.ENV_VAR, str(doc_path))

    doc = ss.read_state_document(doc_path)
    new_doc = ss.accept_patch(json.dumps({"revision": 0, "patch": {"x": 1}}), doc)
    assert new_doc["revision"] == 1
    assert ss.write_state_document(doc_path, new_doc)
    raw = doc_path.read_text()
    assert raw == json.dumps(new_doc, sort_keys=True, separators=(",", ":"))
    assert ss.read_state_document(doc_path)["state"] == {"x": 1}
    leftovers = [p for p in home.iterdir() if p.name.startswith(".skill-state-")]
    assert leftovers == []


def test_merge_patch_semantics():
    base = {"a": 1, "b": {"c": 2, "d": 3}, "e": 5}
    out = ss.merge_patch(base, {"a": 9, "b": {"c": None, "z": 0}, "e": None})
    assert out == {"a": 9, "b": {"d": 3, "z": 0}}
    # input not mutated
    assert base == {"a": 1, "b": {"c": 2, "d": 3}, "e": 5}


def test_accept_patch_rejects_malformed_and_stale():
    doc = {"version": 1, "revision": 4, "state": {"k": "v"}}
    assert ss.accept_patch("{bad json", doc) is None
    assert ss.accept_patch(json.dumps([1, 2]), doc) is None
    assert ss.accept_patch(json.dumps({"patch": {}}), doc) is None
    assert ss.accept_patch(json.dumps({"revision": 3, "patch": {}}), doc) is None
    assert ss.accept_patch(json.dumps({"revision": 5, "patch": {}}), doc) is None
    assert ss.accept_patch(json.dumps({"revision": 4, "patch": "no"}), doc) is None
    ok = ss.accept_patch(json.dumps({"revision": 4, "patch": {"n": 1}}), doc)
    assert ok == {"version": 1, "revision": 5, "state": {"k": "v", "n": 1}}


# ── evidence: missing / duplicate / incomplete ──────────────────────


def _armed(tmp_path, monkeypatch, state=None):
    home = _home(tmp_path, monkeypatch)
    doc_path = home / "skill-state.json"
    _write_doc(doc_path, state=state)
    monkeypatch.setenv(ss.ENV_VAR, str(doc_path))
    return doc_path


def test_missing_tool_result_preserves_history(tmp_path, monkeypatch):
    _armed(tmp_path, monkeypatch)
    msgs = _conversation(_patch_content({"a": 1}))
    msgs = msgs[:-1]  # drop the tool result
    assert ss.apply_skill_state_projection(_agent(), msgs) is False
    assert len(msgs) == 2


def test_duplicate_tool_result_preserves_history(tmp_path, monkeypatch):
    _armed(tmp_path, monkeypatch)
    msgs = _conversation(_patch_content({"a": 1}))
    msgs.append(dict(msgs[-1]))  # duplicate evidence
    assert ss.apply_skill_state_projection(_agent(), msgs) is False
    assert len(msgs) == 4


def test_incomplete_tool_result_preserves_history(tmp_path, monkeypatch):
    _armed(tmp_path, monkeypatch)
    msgs = _conversation(_patch_content({"a": 1}))
    msgs[-1]["content"] = None
    assert ss.apply_skill_state_projection(_agent(), msgs) is False


def test_duplicate_marker_blocks_preserve_history(tmp_path, monkeypatch):
    _armed(tmp_path, monkeypatch)
    payload = json.dumps({"revision": 0, "patch": {"a": 1}})
    content = ss.BEGIN_MARKER + payload + ss.END_MARKER + " again " + ss.BEGIN_MARKER + payload + ss.END_MARKER
    msgs = _conversation(content)
    assert ss.apply_skill_state_projection(_agent(), msgs) is False


def test_unterminated_marker_preserves_history(tmp_path, monkeypatch):
    _armed(tmp_path, monkeypatch)
    msgs = _conversation(ss.BEGIN_MARKER + json.dumps({"revision": 0, "patch": {}}))
    assert ss.apply_skill_state_projection(_agent(), msgs) is False


def test_oversized_patch_preserves_history(tmp_path, monkeypatch):
    _armed(tmp_path, monkeypatch)
    big = {"revision": 0, "patch": {"blob": "x" * (ss.MAX_PATCH_BYTES)}}
    msgs = _conversation(ss.BEGIN_MARKER + json.dumps(big) + ss.END_MARKER)
    assert ss.apply_skill_state_projection(_agent(), msgs) is False


def test_stale_patch_preserves_history(tmp_path, monkeypatch):
    _armed(tmp_path, monkeypatch, revision=7)
    msgs = _conversation(_patch_content({"a": 1}, revision=6))
    assert ss.apply_skill_state_projection(_agent(), msgs) is False
    assert len(msgs) == 3


# ── request-only projection ─────────────────────────────────────────


def test_projection_shape_and_durable_history(tmp_path, monkeypatch):
    path = _armed(tmp_path, monkeypatch, state={"done": []})
    msgs = [
        {"role": "user", "content": "Do the task."},
        {"role": "assistant", "content": "starting", "tool_calls": [
            {"id": "call_0", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_0", "content": "earlier"},
        {"role": "assistant", "content": _patch_content({"done": ["step1"]}),
         "tool_calls": [{"id": "call_1", "type": "function",
                         "function": {"name": "execute_code", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
    ]
    durable = [json.loads(json.dumps(m)) for m in msgs]
    agent = _agent()

    assert ss.apply_skill_state_projection(agent, msgs) is True
    # projected request: user (task + canonical state), assistant, tool result
    assert [m["role"] for m in msgs] == ["user", "assistant", "tool"]
    assert msgs[0]["content"].startswith("Do the task.")
    assert json.dumps({"version": 1, "revision": 1, "state": {"done": ["step1"]}},
                      sort_keys=True, separators=(",", ":")) in msgs[0]["content"]
    assert msgs[1]["tool_calls"][0]["id"] == "call_1"
    assert msgs[2]["tool_call_id"] == "call_1"
    # durable transcript untouched (caller's history list here stands in)
    assert durable[0] is not msgs[0]
    # document advanced monotonically
    doc = ss.read_state_document(path)
    assert doc["revision"] == 1 and doc["state"] == {"done": ["step1"]}


def test_no_patch_in_history_is_full_history(tmp_path, monkeypatch):
    _armed(tmp_path, monkeypatch)
    msgs = _conversation()  # plain user message only
    msgs.append({"role": "assistant", "content": "all done"})
    before = [json.loads(json.dumps(m)) for m in msgs]
    assert ss.apply_skill_state_projection(_agent(), msgs) is False
    assert msgs == before


def test_next_request_without_patch_is_full_history(tmp_path, monkeypatch):
    path = _armed(tmp_path, monkeypatch)
    msgs = _conversation(_patch_content({"a": 1}))
    assert ss.apply_skill_state_projection(_agent(), msgs) is True
    # The request AFTER the projected one carries full history and no
    # re-application (revision already advanced; patch is now stale).
    msgs2 = _conversation(_patch_content({"a": 1}))
    assert ss.apply_skill_state_projection(_agent(), msgs2) is False
    assert len(msgs2) == 3
    assert ss.read_state_document(path)["revision"] == 1


# ── system-prompt opt-in ────────────────────────────────────────────


def _make_agent(**overrides):
    base = dict(
        load_soul_identity=False,
        skip_context_files=True,
        valid_tool_names=[],
        _task_completion_guidance=False,
        _tool_use_enforcement=False,
        _execution_guidance=False,
        _environment_probe=False,
        _kanban_worker_guidance="",
        _memory_store=None,
        _memory_manager=None,
        _bot_mode_protocol=False,
        model="",
        provider="",
        platform="",
        pass_session_id=False,
        session_id="",
        _emit_status=lambda *a, **k: None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _stable(agent):
    from unittest.mock import patch

    with (
        patch("run_agent.load_soul_md", return_value=""),
        patch("run_agent.build_environment_hints", return_value=""),
    ):
        from agent.system_prompt import build_system_prompt_parts

        return build_system_prompt_parts(agent)["stable"]


def test_system_prompt_instruction_only_when_active(tmp_path, monkeypatch):
    home = _home(tmp_path, monkeypatch)
    doc_path = home / "skill-state.json"
    _write_doc(doc_path)

    monkeypatch.delenv(ss.ENV_VAR, raising=False)
    assert ss.PROTOCOL_INSTRUCTION not in _stable(_make_agent())

    monkeypatch.setenv(ss.ENV_VAR, str(doc_path))
    stable = _stable(_make_agent())
    assert ss.PROTOCOL_INSTRUCTION in stable
    # added exactly once, carries no state content or absolute path
    assert stable.count(ss.PROTOCOL_INSTRUCTION) == 1
    assert str(doc_path) not in stable
    assert str(home) not in stable


def test_invalid_path_gives_no_instruction(tmp_path, monkeypatch):
    _home(tmp_path, monkeypatch)
    monkeypatch.setenv(ss.ENV_VAR, str(tmp_path / "not-there.json"))
    assert ss.PROTOCOL_INSTRUCTION not in _stable(_make_agent())
