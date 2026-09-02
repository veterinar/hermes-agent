"""Focused behavior tests for the opt-in task-local skill_state route.

Covers: inactive parity, one valid patch + projected request, malformed
patch full-history fallback, atomic bounded state update, and full
transcript persistence (projection never mutates the source sequence).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agent import skill_state as ss


class _Agent:
    """Minimal agent double — only carries the patch signature memo."""

    _skill_state_applied_sig = None


def _history():
    return [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "Do the vetclub task"},
        {
            "role": "assistant",
            "content": "working",
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "r1"},
        {
            "role": "assistant",
            "content": (
                "progress\n"
                "<skill-state-patch>\n"
                '{"done": ["step1"]}\n'
                "</skill-state-patch>"
            ),
            "tool_calls": [
                {"id": "c2", "type": "function", "function": {"name": "g", "arguments": "{}"}},
                {"id": "c3", "type": "function", "function": {"name": "h", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "c3", "content": "r3"},
        {"role": "tool", "tool_call_id": "c2", "content": "r2"},
    ]


@pytest.fixture
def state_env(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    state_path = home / "skill-state.json"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv(ss.SKILL_STATE_PATH_ENV, str(state_path))
    return home, state_path


# ---------------------------------------------------------------------------
# 1. Inactive parity
# ---------------------------------------------------------------------------

def test_inactive_mode_is_identity(monkeypatch):
    monkeypatch.delenv(ss.SKILL_STATE_PATH_ENV, raising=False)
    hist = _history()
    agent = _Agent()
    out = ss.maybe_project_skill_state(agent, hist)
    assert out is hist  # same object — zero behavior change


def test_activation_rejects_path_escape(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv(ss.SKILL_STATE_PATH_ENV, str(tmp_path / "outside.json"))
    assert ss.resolve_skill_state_path() is None


# ---------------------------------------------------------------------------
# 2. One valid patch → projected request
# ---------------------------------------------------------------------------

def test_valid_patch_projects_request(state_env):
    home, state_path = state_env
    hist = _history()
    agent = _Agent()
    out = ss.maybe_project_skill_state(agent, hist)

    assert len(out) == 5
    assert out[0] == {"role": "system", "content": "SYS"}
    assert out[1]["role"] == "user"
    assert out[1]["content"].startswith("Do the vetclub task")
    assert "<skill-state>" in out[1]["content"]
    canonical_state = json.loads(out[1]["content"].split("<skill-state>")[1].split("</skill-state>")[0])
    assert canonical_state == {"done": ["step1"]}
    assert [c["id"] for c in out[2]["tool_calls"]] == ["c2", "c3"]
    assert [m["tool_call_id"] for m in out[3:]] == ["c3", "c2"] or [
        m["tool_call_id"] for m in out[3:]
    ] == ["c2", "c3"]
    # No older reasoning/tool replay: nothing from the c1 round remains.
    assert all(m.get("tool_call_id") != "c1" for m in out)
    assert "working" not in json.dumps(out)

    # State file published with monotonic revision, owner-only, valid schema.
    doc = ss.read_state(state_path)
    assert doc == {"version": 1, "revision": 1, "state": {"done": ["step1"]}}
    assert (os.stat(state_path).st_mode & 0o077) == 0

    # Second projection of the same assistant content does NOT bump revision.
    out2 = ss.maybe_project_skill_state(agent, _history())
    assert ss.read_state(state_path)["revision"] == 1
    assert out2[1] == out[1]


# ---------------------------------------------------------------------------
# 3. Malformed / missing / duplicate patch → full-history fallback
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "content",
    [
        "no block at all",
        "<skill-state-patch>{broken json}</skill-state-patch>",
        "<skill-state-patch>{}</skill-state-patch><skill-state-patch>{}</skill-state-patch>",
        "<skill-state-patch>[1,2]</skill-state-patch>",
        "<skill-state-patch>" + '{"k": "' + "x" * (ss.MAX_PATCH_BYTES + 10) + '"}' + "</skill-state-patch>",
    ],
)
def test_bad_patch_falls_back_to_full_history(state_env, content):
    hist = _history()
    hist[4]["content"] = content
    out = ss.maybe_project_skill_state(_Agent(), hist)
    assert out == hist
    assert len(out) == 7


def test_incomplete_tool_results_fall_back(state_env):
    hist = _history()
    del hist[6]  # drop one of the two matching results
    out = ss.maybe_project_skill_state(_Agent(), hist)
    assert out == hist


def test_no_tool_calls_no_projection(state_env):
    hist = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "<skill-state-patch>{\"a\":1}</skill-state-patch>"},
    ]
    out = ss.maybe_project_skill_state(_Agent(), hist)
    assert out == hist


# ---------------------------------------------------------------------------
# 4. Atomic bounded state update
# ---------------------------------------------------------------------------

def test_atomic_update_preserves_previous_on_failure(state_env, tmp_path):
    home, state_path = state_env
    ss.write_state_atomic(state_path, {"version": 1, "revision": 5, "state": {"keep": 1}})
    before = state_path.read_bytes()

    # Oversized state → rejected, previous file intact.
    big = {"blob": "x" * (ss.MAX_STATE_BYTES + 100)}
    with pytest.raises(ss.SkillStateError):
        ss.write_state_atomic(state_path, {"version": 1, "revision": 6, "state": big})
    assert state_path.read_bytes() == before

    # Wrong schema → rejected.
    with pytest.raises(ss.SkillStateError):
        ss.write_state_atomic(state_path, {"version": 99, "revision": 6, "state": {}})
    with pytest.raises(ss.SkillStateError):
        ss.write_state_atomic(state_path, {"version": 1, "revision": -1, "state": {}})
    assert state_path.read_bytes() == before

    # No leftover temp files.
    assert [p for p in home.iterdir() if p.name.startswith(".skill-state-")] == []


def test_read_rejects_symlink_and_bad_files(state_env):
    home, state_path = state_env
    ss.write_state_atomic(state_path, {"version": 1, "revision": 0, "state": {}})
    link = home / "link.json"
    link.symlink_to(state_path)
    with pytest.raises(ss.SkillStateError):
        ss.read_state(link)

    bad = home / "bad.json"
    bad.write_text('{"version": 1, "revision": 0, "state": []}', encoding="utf-8")
    os.chmod(bad, 0o600)
    with pytest.raises(ss.SkillStateError):
        ss.read_state(bad)

    loose = home / "loose.json"
    loose.write_text('{"version": 1, "revision": 0, "state": {}}', encoding="utf-8")
    os.chmod(loose, 0o644)
    with pytest.raises(ss.SkillStateError):
        ss.read_state(loose)


def test_merge_patch_semantics():
    state = {"a": 1, "b": {"x": 1, "y": 2}, "c": 3}
    patch = {"b": {"x": None, "z": 4}, "c": None}
    assert ss.apply_merge_patch(state, patch) == {"a": 1, "b": {"y": 2, "z": 4}}
    assert state == {"a": 1, "b": {"x": 1, "y": 2}, "c": 3}  # input untouched


# ---------------------------------------------------------------------------
# 5. Full transcript persistence: projection never mutates the source
# ---------------------------------------------------------------------------

def test_projection_does_not_mutate_history(state_env):
    hist = _history()
    snapshot = json.dumps(hist, sort_keys=True)
    out = ss.maybe_project_skill_state(_Agent(), hist)
    assert out is not hist
    assert json.dumps(hist, sort_keys=True) == snapshot
    # The projected request itself is API-valid: strict role order and the
    # tool results exactly satisfy the assistant's (parallel) tool calls.
    assert [m["role"] for m in out] == ["system", "user", "assistant", "tool", "tool"]
    assert {m["tool_call_id"] for m in out[3:]} == {c["id"] for c in out[2]["tool_calls"]}


def test_projection_survives_parallel_tool_calls(state_env):
    hist = _history()
    out = ss.maybe_project_skill_state(_Agent(), hist)
    ids = [c["id"] for c in out[2]["tool_calls"]]
    results = [m["tool_call_id"] for m in out[3:]]
    assert sorted(ids) == sorted(results) and len(ids) == 2
