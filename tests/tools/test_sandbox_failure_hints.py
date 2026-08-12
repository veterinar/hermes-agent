"""Tests for execute_code sandbox failure hints."""

import json
from unittest.mock import patch

import pytest

from tools.code_execution_tool import _sandbox_failure_hint, execute_code


class TestSandboxFailureHint:
    def test_unavailable_tool_import_lists_available(self):
        err = ("Traceback (most recent call last):\n  File \"script.py\", line 1\n"
               "ImportError: cannot import name 'browser_navigate' from 'hermes_tools'")
        h = _sandbox_failure_hint(err, enabled_tools={"terminal", "read_file"})
        assert "browser_navigate" in h
        assert "read_file" in h and "terminal" in h
        assert "normal tool call" in h

    def test_builtin_helper_import_redirects(self):
        err = "ImportError: cannot import name 'json_parse' from 'hermes_tools'"
        h = _sandbox_failure_hint(err)
        assert "BUILT-IN" in h
        assert "no import" in h.lower()

    def test_missing_third_party_module(self):
        err = "ModuleNotFoundError: No module named 'matplotlib'"
        h = _sandbox_failure_hint(err)
        assert "matplotlib" in h
        assert "stdlib" in h

    def test_dict_vs_string_confusion(self):
        err = "TypeError: string indices must be integers"
        h = _sandbox_failure_hint(err)
        assert "DICTS" in h

    def test_unknown_failure_no_hint(self):
        assert _sandbox_failure_hint("ZeroDivisionError: division by zero") is None

    def test_empty_stderr_no_hint(self):
        assert _sandbox_failure_hint("") is None

    def test_unrestricted_hint_reports_the_session_tool_inventory(self):
        err = (
            "ImportError: cannot import name 'missing_tool' "
            "from 'hermes_tools'"
        )
        with patch("hermes_cli.runtime_policy.is_unrestricted", return_value=True):
            hint = _sandbox_failure_hint(
                err,
                enabled_tools={"delegate_task", "browser_navigate", "execute_code"},
            )

        assert "browser_navigate" in hint
        assert "delegate_task" in hint
        inventory = hint.split("here:", 1)[1]
        assert "execute_code" not in inventory
        assert "normal tool call" not in hint


class TestLiveSandboxHint:
    def test_bad_import_produces_hint_field(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        r = json.loads(execute_code(
            "from hermes_tools import totally_fake_tool\nprint('unreachable')",
            task_id="t-sbhint",
        ))
        assert r["status"] == "error"
        assert "hint" in r
        assert "totally_fake_tool" in r["hint"]

    def test_missing_module_produces_hint(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        r = json.loads(execute_code(
            "import nonexistent_pkg_zzz\n", task_id="t-sbhint",
        ))
        assert r["status"] == "error"
        assert "not installed in the sandbox" in r.get("hint", "")

    def test_successful_script_has_no_hint(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        r = json.loads(execute_code("print('fine')", task_id="t-sbhint"))
        assert r["status"] == "success"
        assert "hint" not in r
