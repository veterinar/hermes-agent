"""Process-frozen operator policy switches.

``security.unrestricted`` is deliberately resolved once.  A running agent
cannot edit config.yaml and gain capabilities halfway through a session.
Correctness and protocol invariants remain the responsibility of their
callers; this switch only disables Hermes policy restrictions.
"""

from __future__ import annotations

from functools import lru_cache
import os
from typing import Any, Mapping


_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def _enabled(config: Mapping[str, Any]) -> bool:
    security = config.get("security", {})
    if not isinstance(security, Mapping):
        return False
    value = security.get("unrestricted", False)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in _TRUE_VALUES


@lru_cache(maxsize=1)
def is_unrestricted() -> bool:
    """Return the effective unrestricted policy, frozen for this process."""
    frozen = os.getenv("HERMES_UNRESTRICTED")
    if frozen is not None:
        return str(frozen).strip().lower() in _TRUE_VALUES
    try:
        from hermes_cli.config import load_config_readonly

        return _enabled(load_config_readonly())
    except Exception:
        return False


def reset_unrestricted_for_tests() -> None:
    """Clear the process snapshot for isolated tests only."""
    is_unrestricted.cache_clear()


def bridge_unrestricted_to_env(config: Mapping[str, Any]) -> None:
    """Freeze the effective policy for modules that must decide at import."""
    os.environ["HERMES_UNRESTRICTED"] = "1" if _enabled(config) else "0"


__all__ = [
    "bridge_unrestricted_to_env",
    "is_unrestricted",
    "reset_unrestricted_for_tests",
]
