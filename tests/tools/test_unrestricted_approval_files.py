"""Unrestricted-mode behavior for file and approval policy gates."""

from __future__ import annotations

import json
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


def test_protected_instruction_write_does_not_prompt(tmp_path: Path, monkeypatch):
    from tools import file_tools

    target = tmp_path / "AGENTS.md"
    monkeypatch.setattr(
        file_tools,
        "_request_protected_instruction_approval",
        lambda *_args, **_kwargs: pytest.fail("unrestricted mode prompted"),
    )

    result = json.loads(file_tools.write_file_tool(str(target), "operator owned\n"))

    assert result.get("error") is None
    assert target.read_text(encoding="utf-8") == "operator owned\n"


def test_ssh_config_write_does_not_request_approval(tmp_path: Path, monkeypatch):
    from tools import approval, file_tools

    home = tmp_path / "home"
    target = home / ".ssh" / "config"
    target.parent.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(
        approval,
        "_run_approval_gate",
        lambda **_kwargs: pytest.fail("unrestricted mode requested approval"),
    )

    result = json.loads(file_tools.write_file_tool(str(target), "Host *\n"))

    assert result.get("error") is None
    assert target.read_text(encoding="utf-8") == "Host *\n"


def test_v4a_parent_traversal_reaches_file_backend():
    from tools import file_tools

    mock_ops = MagicMock()
    result_obj = MagicMock()
    result_obj.to_dict.return_value = {"status": "ok", "operations": 1}
    mock_ops.patch_v4a.return_value = result_obj
    patch_text = (
        "*** Begin Patch\n"
        "*** Add File: ../operator-target.txt\n"
        "+owned\n"
        "*** End Patch\n"
    )

    with patch("tools.file_tools._get_file_ops", return_value=mock_ops):
        result = json.loads(file_tools.patch_tool(mode="patch", patch=patch_text))

    assert result.get("error") is None
    mock_ops.patch_v4a.assert_called_once_with(patch_text)
