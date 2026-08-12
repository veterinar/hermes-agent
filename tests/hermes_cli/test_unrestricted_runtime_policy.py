from hermes_cli import runtime_policy


def test_unrestricted_accepts_only_explicit_truthy_values():
    assert runtime_policy._enabled({"security": {"unrestricted": True}})
    assert runtime_policy._enabled({"security": {"unrestricted": "yes"}})
    assert not runtime_policy._enabled({"security": {"unrestricted": False}})
    assert not runtime_policy._enabled({"security": {"unrestricted": "invalid"}})
    assert not runtime_policy._enabled({})


def test_unrestricted_is_frozen_for_process(monkeypatch):
    monkeypatch.delenv("HERMES_UNRESTRICTED", raising=False)
    state = {"security": {"unrestricted": True}}
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly", lambda: state
    )
    runtime_policy.reset_unrestricted_for_tests()
    assert runtime_policy.is_unrestricted()

    state["security"]["unrestricted"] = False
    assert runtime_policy.is_unrestricted()

    runtime_policy.reset_unrestricted_for_tests()
    assert not runtime_policy.is_unrestricted()


def test_bridge_freezes_managed_effective_value(monkeypatch):
    monkeypatch.delenv("HERMES_UNRESTRICTED", raising=False)
    runtime_policy.bridge_unrestricted_to_env(
        {"security": {"unrestricted": True}}
    )
    runtime_policy.reset_unrestricted_for_tests()
    assert runtime_policy.is_unrestricted()
