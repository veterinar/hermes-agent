"""Behavioral coverage for the operator-controlled unrestricted policy.

The policy removes Hermes guardrails, including subprocess environment
filtering and context-based rewrites. Explicit per-call environment values
still override the inherited process environment.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _enable_unrestricted(monkeypatch):
    from hermes_cli.runtime_policy import reset_unrestricted_for_tests

    monkeypatch.setenv("HERMES_UNRESTRICTED", "1")
    reset_unrestricted_for_tests()
    yield
    reset_unrestricted_for_tests()


def _terminal_config() -> dict:
    return {
        "env_type": "local",
        "timeout": 180,
        "cwd": "/tmp",
        "host_cwd": None,
        "modal_mode": "auto",
        "docker_image": "",
        "singularity_image": "",
        "modal_image": "",
        "daytona_image": "",
    }


def _run_terminal(command: str, *, timeout: int | None = None) -> tuple[dict, MagicMock]:
    from tools.terminal_tool import terminal_tool

    mock_env = MagicMock()
    mock_env.execute.return_value = {"output": "done", "returncode": 0}
    with (
        patch("tools.terminal_tool._get_env_config", return_value=_terminal_config()),
        patch("tools.terminal_tool._start_cleanup_thread"),
        patch("tools.terminal_tool._active_environments", {"default": mock_env}),
        patch("tools.terminal_tool._last_activity", {"default": 0}),
        patch(
            "tools.terminal_tool._check_all_guards",
            return_value={"approved": True},
        ),
    ):
        result = json.loads(terminal_tool(command=command, timeout=timeout))
    return result, mock_env


def test_unrestricted_bypasses_file_read_write_and_safe_root_guards(
    tmp_path: Path, monkeypatch
):
    from agent import file_safety

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setattr(file_safety, "_hermes_home_path", lambda: hermes_home)
    monkeypatch.setattr(file_safety, "_hermes_root_path", lambda: hermes_home)
    monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(tmp_path / "safe"))

    assert file_safety.get_write_denied_error("/etc/shadow") is None
    assert file_safety.get_write_denied_error(str(tmp_path / "outside.txt")) is None
    assert file_safety.get_read_block_error(str(tmp_path / ".env")) is None


def test_unrestricted_allows_session_state_paths(tmp_path: Path, monkeypatch):
    from agent import file_safety
    from tools import file_tools

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setattr(file_safety, "_hermes_home_path", lambda: hermes_home)
    monkeypatch.setattr(file_safety, "_hermes_root_path", lambda: hermes_home)

    assert file_safety.is_write_denied(str(hermes_home / "state.db")) is False
    assert file_safety.is_write_denied(str(hermes_home / "sessions" / "turn.json")) is False

    state_db = hermes_home / "state.db"
    result = json.loads(file_tools.write_file_tool(str(state_db), "operator bytes"))
    assert result.get("error") is None
    assert state_db.read_text(encoding="utf-8") == "operator bytes"


def test_unrestricted_bypasses_high_level_sensitive_and_profile_guards(
    tmp_path: Path, monkeypatch
):
    from agent import file_safety
    from tools import file_tools

    root = tmp_path / ".hermes"
    active = root / "profiles" / "active"
    active.mkdir(parents=True)
    monkeypatch.setattr(file_safety, "_hermes_root_path", lambda: root)
    monkeypatch.setattr(file_safety, "_hermes_home_path", lambda: active)

    assert file_tools._check_sensitive_path("/etc/hosts") is None
    assert file_tools._check_cross_profile_path(
        str(root / "profiles" / "other" / "skills" / "x"),
    ) is None


def test_unrestricted_allows_large_foreground_timeout():
    result, mock_env = _run_terminal("echo ok", timeout=9999)

    assert result.get("error") is None
    assert mock_env.execute.call_args.kwargs["timeout"] == 9999


def test_unrestricted_zero_terminal_timeout_means_unlimited():
    result, mock_env = _run_terminal("echo ok", timeout=0)

    assert result.get("error") is None
    assert mock_env.execute.call_args.kwargs["timeout"] == 0


def test_unrestricted_terminal_schema_exposes_zero_as_unlimited():
    from tools.terminal_tool import build_terminal_schema

    timeout_schema = build_terminal_schema()["parameters"]["properties"]["timeout"]

    assert timeout_schema["minimum"] == 0
    assert "unlimited" in timeout_schema["description"].lower()


def test_unrestricted_allows_self_repository_git_mutation(tmp_path: Path, monkeypatch):
    import tools.self_repo_guard as self_repo_guard

    repo = tmp_path / "hermes-agent"
    (repo / ".git").mkdir(parents=True)
    monkeypatch.setattr(
        self_repo_guard,
        "get_running_source_root",
        lambda: repo.resolve(),
    )

    config = _terminal_config()
    config["cwd"] = str(repo)
    mock_env = MagicMock()
    mock_env.cwd = str(repo)
    mock_env.execute.return_value = {"output": "done", "returncode": 0}
    with (
        patch("tools.terminal_tool._get_env_config", return_value=config),
        patch("tools.terminal_tool._start_cleanup_thread"),
        patch("tools.terminal_tool._active_environments", {"default": mock_env}),
        patch("tools.terminal_tool._last_activity", {"default": 0}),
        patch(
            "tools.terminal_tool._check_all_guards",
            return_value={"approved": True},
        ),
    ):
        from tools.terminal_tool import terminal_tool

        result = json.loads(terminal_tool(command="git checkout main"))

    assert result.get("status") != "blocked"
    mock_env.execute.assert_called_once()


def test_unrestricted_allows_long_lived_foreground_command():
    result, mock_env = _run_terminal("npm run dev")

    assert result.get("error") is None
    assert mock_env.execute.call_args.args[0] == "npm run dev"


def test_unrestricted_allows_gateway_lifecycle_command(monkeypatch):
    monkeypatch.setenv("_HERMES_GATEWAY", "1")

    result, mock_env = _run_terminal("hermes gateway restart")

    assert result.get("error") is None
    assert mock_env.execute.call_args.args[0] == "hermes gateway restart"


def test_unrestricted_gateway_bypasses_persistent_launchctl_and_referenced_script(
    tmp_path: Path, monkeypatch
):
    script = tmp_path / "restart.sh"
    script.write_text(
        "#!/bin/sh\nlaunchctl submit -l ai.hermes.gateway-restart -- /bin/true\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("_HERMES_GATEWAY", "1")

    direct_result, direct_env = _run_terminal(
        "launchctl submit -l ai.hermes.gateway-restart -- /bin/true"
    )
    script_result, script_env = _run_terminal(f"sh {script}")

    assert direct_result.get("error") is None
    assert script_result.get("error") is None
    direct_env.execute.assert_called_once()
    script_env.execute.assert_called_once()


def test_unrestricted_allows_cron_gateway_lifecycle_command():
    from cron.lifecycle_guard import check_gateway_lifecycle

    check_gateway_lifecycle("run hermes gateway restart nightly")


def test_unrestricted_subprocess_envs_inherit_operator_environment(monkeypatch):
    from tools.environments.local import (
        _make_run_env,
        _sanitize_subprocess_env,
        hermes_subprocess_env,
    )

    inherited = {
        "HERMES_UNRESTRICTED": "1",
        "OPENAI_API_KEY": "fake-provider-key",
        "GH_TOKEN": "fake-github-token",
        "AUXILIARY_VISION_API_KEY": "fake-aux-key",
        "VIRTUAL_ENV": "/tmp/operator-venv",
        "PATH": "/usr/bin:/bin",
    }
    with patch.dict(os.environ, inherited, clear=True):
        foreground = _make_run_env({})
        # Some child surfaces supply a purpose-built base env. Unrestricted
        # mode still inherits the full operator env before those overrides.
        background = _sanitize_subprocess_env({"PATH": inherited["PATH"]})
        child = hermes_subprocess_env()

    for env in (foreground, background, child):
        for key, value in inherited.items():
            if key == "PATH":
                assert value in env.get(key, "")
            else:
                assert env.get(key) == value


def test_unrestricted_env_inherits_process_identity_without_context_rewrites(monkeypatch):
    import gateway.session_context as session_context
    from tools.environments.local import _make_run_env

    saved_engaged = session_context._session_context_engaged
    saved_values = {name: var.get() for name, var in session_context._VAR_MAP.items()}
    try:
        session_context._session_context_engaged = True
        for var in session_context._VAR_MAP.values():
            var.set(session_context._UNSET)
        inherited = {
            "HERMES_SESSION_KEY": "profile-a-session",
            "HERMES_SESSION_PROFILE": "profile-a",
            "PROFILE_A_API_KEY": "profile-a-secret",
            "HERMES_KANBAN_DB": "/tmp/profile-a-kanban.db",
        }
        for key, value in inherited.items():
            monkeypatch.setenv(key, value)

        env = _make_run_env({"HERMES_SESSION_KEY": "explicit-call-session"})
    finally:
        session_context._session_context_engaged = saved_engaged
        for name, var in session_context._VAR_MAP.items():
            var.set(saved_values[name])

    assert env["HERMES_SESSION_KEY"] == "explicit-call-session"
    assert env["HERMES_SESSION_PROFILE"] == "profile-a"
    assert env["PROFILE_A_API_KEY"] == "profile-a-secret"
    assert env["HERMES_KANBAN_DB"] == "/tmp/profile-a-kanban.db"
