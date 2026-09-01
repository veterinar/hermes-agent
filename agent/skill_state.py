"""Opt-in task-local ``skill_state`` compaction (VetClub author route).

Disabled unless the ``HERMES_SKILL_STATE_PATH`` env var names a canonical,
owner-only, regular JSON file strictly below the active ``HERMES_HOME``.
When active, the agent may publish one explicitly delimited JSON Merge
Patch per assistant tool-call turn; after a valid patch plus complete
tool results, the NEXT provider request is projected to:

  [unchanged system prompt] + [original user task + synthetic canonical
  state message] + [most recent complete assistant tool-call message] +
  [all of its matching tool results]

instead of replaying the whole trajectory.  Projection is request-only:
the durable session transcript, trajectories, and audit output keep the
full unprojected sequence.  Any missing / duplicate / malformed /
oversized / stale / invalid evidence falls back to the existing
full-history route — history is never discarded on doubt.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SKILL_STATE_PATH_ENV = "HERMES_SKILL_STATE_PATH"

SCHEMA_VERSION = 1
MAX_STATE_BYTES = 256 * 1024      # canonical serialized state bound
MAX_PATCH_BYTES = 64 * 1024       # serialized merge-patch bound
MAX_FILE_BYTES = 1024 * 1024     # on-disk document bound

PATCH_OPEN = "<skill-state-patch>"
PATCH_CLOSE = "</skill-state-patch>"

_STATE_OPEN = "<skill-state>"
_STATE_CLOSE = "</skill-state>"


class SkillStateError(Exception):
    """Raised on any invalid / unsafe skill-state evidence."""


# ---------------------------------------------------------------------------
# Activation + path safety
# ---------------------------------------------------------------------------

def resolve_skill_state_path() -> Optional[Path]:
    """Return the validated state path, or ``None`` when the mode is off.

    The env value must name an absolute path that resolves strictly below
    the active ``HERMES_HOME``.  Symlinks anywhere on the final component
    are rejected at open time (O_NOFOLLOW); resolution here checks the
    logical containment so a symlink pointing outside HOME can never
    pass as a "state file".
    """
    raw = os.environ.get(SKILL_STATE_PATH_ENV, "").strip()
    if not raw:
        return None
    p = Path(raw)
    if not p.is_absolute():
        return None
    from hermes_constants import get_hermes_home

    home = Path(get_hermes_home()).resolve()
    try:
        resolved = p.resolve()
    except OSError:
        return None
    if resolved == home or not resolved.is_relative_to(home):
        return None
    return resolved


# ---------------------------------------------------------------------------
# Canonical JSON helpers
# ---------------------------------------------------------------------------

def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _canonical_size_ok(obj: Any, limit: int) -> bool:
    try:
        return len(canonical_json(obj).encode("utf-8")) <= limit
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Document I/O (no-follow, owner-only, atomic replace)
# ---------------------------------------------------------------------------

def _validate_doc(doc: Any) -> Dict[str, Any]:
    if not isinstance(doc, dict):
        raise SkillStateError("state document must be a JSON object")
    if doc.get("version") != SCHEMA_VERSION:
        raise SkillStateError(f"unsupported schema version: {doc.get('version')!r}")
    revision = doc.get("revision")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
        raise SkillStateError("revision must be a non-negative integer")
    state = doc.get("state")
    if not isinstance(state, dict):
        raise SkillStateError("state must be a JSON object")
    if not _canonical_size_ok(state, MAX_STATE_BYTES):
        raise SkillStateError("state exceeds bounded size")
    return doc


def _open_state_read(path: Path) -> Tuple[Any, os.stat_result]:
    fd = os.open(str(path), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise SkillStateError("state path is not a regular file")
        if st.st_size > MAX_FILE_BYTES:
            raise SkillStateError("state file exceeds size bound")
        if st.st_mode & 0o077:
            raise SkillStateError("state file must be owner-only")
        with os.fdopen(fd, "r", encoding="utf-8") as f:
            fd = -1
            return json.load(f), st
    finally:
        if fd >= 0:
            os.close(fd)


def read_state(path: Path) -> Dict[str, Any]:
    """Read + validate the state document.  Raises :class:`SkillStateError`."""
    doc, _ = _open_state_read(path)
    return _validate_doc(doc)


def write_state_atomic(path: Path, doc: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and atomically publish ``doc``; keeps the previous file on failure.

    The staged temp file is written owner-only (0600) in the same directory
    and moved onto the target with :func:`os.replace`.  A symlink sitting
    on the target is rejected before the replace.
    """
    _validate_doc(doc)
    payload = canonical_json(doc)
    if len(payload.encode("utf-8")) > MAX_FILE_BYTES:
        raise SkillStateError("state document exceeds file size bound")
    path = Path(path)
    try:
        if path.is_symlink():
            raise SkillStateError("state path is a symlink")
    except OSError as exc:
        raise SkillStateError(str(exc))
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".skill-state-", suffix=".tmp")
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            fd = -1
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, str(path))
    except BaseException:
        if fd >= 0:
            os.close(fd)
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return doc


# ---------------------------------------------------------------------------
# Delimited merge-patch extraction (RFC 7386)
# ---------------------------------------------------------------------------

def extract_merge_patch(content: Any) -> Optional[Dict[str, Any]]:
    """Extract exactly one delimited JSON Merge Patch from assistant content.

    Returns ``None`` (never raises) when the block is missing, duplicated,
    malformed, oversized, or not a JSON object — callers treat ``None`` as
    "do not compact history".
    """
    if not isinstance(content, str) or PATCH_OPEN not in content:
        return None
    openings = content.count(PATCH_OPEN)
    closings = content.count(PATCH_CLOSE)
    if openings != 1 or closings != 1:
        return None
    start = content.index(PATCH_OPEN) + len(PATCH_OPEN)
    end = content.index(PATCH_CLOSE)
    if end < start:
        return None
    raw = content[start:end].strip()
    if not raw or len(raw.encode("utf-8")) > MAX_PATCH_BYTES:
        return None
    try:
        patch = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(patch, dict):
        return None
    if not _canonical_size_ok(patch, MAX_PATCH_BYTES):
        return None
    return patch


def apply_merge_patch(state: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    """RFC 7386 JSON Merge Patch (null deletes keys)."""

    def merge(target: Any, source: Any) -> Any:
        if not isinstance(source, dict):
            return copy.deepcopy(source)
        if not isinstance(target, dict):
            target = {}
        for key, value in source.items():
            if value is None:
                target.pop(key, None)
            else:
                target[key] = merge(target.get(key), value)
        return target

    return merge(copy.deepcopy(state), patch)


# ---------------------------------------------------------------------------
# System-prompt instruction (stable for the session; initial build only)
# ---------------------------------------------------------------------------

def skill_state_protocol_instruction() -> str:
    return (
        "## Canonical Skill State Protocol\n"
        "A canonical task-state file is active for this session. When your "
        "work changes the task state, include in the SAME assistant message "
        "that contains your tool calls exactly one block delimited by "
        f"{PATCH_OPEN} and {PATCH_CLOSE} containing a single JSON Merge "
        "Patch (RFC 7386) object describing only the changed keys (use null "
        "to delete a key). Omit the block entirely when nothing changed. "
        "Never emit more than one block per message, and never emit the "
        "block in a message without tool calls."
    )


# ---------------------------------------------------------------------------
# Request projection
# ---------------------------------------------------------------------------

def _matching_tool_results(api_messages: List[Dict[str, Any]], assistant_idx: int) -> Optional[List[Dict[str, Any]]]:
    """All tool results after ``assistant_idx`` iff they exactly satisfy its calls."""
    calls = api_messages[assistant_idx].get("tool_calls") or []
    expected = {c.get("id") for c in calls if isinstance(c, dict) and c.get("id")}
    if not expected:
        return None
    results: List[Dict[str, Any]] = []
    seen = set()
    for msg in api_messages[assistant_idx + 1:]:
        if msg.get("role") != "tool":
            return None  # something else interleaved — not a complete observation
        tc_id = msg.get("tool_call_id")
        if tc_id in seen or tc_id not in expected:
            return None
        seen.add(tc_id)
        results.append(msg)
    return results if seen == expected else None


def _patch_signature(patch: Dict[str, Any], assistant_msg: Dict[str, Any]) -> str:
    call_ids = sorted(
        str(c.get("id"))
        for c in (assistant_msg.get("tool_calls") or [])
        if isinstance(c, dict)
    )
    digest = hashlib.sha256(canonical_json(patch).encode("utf-8"))
    digest.update("\x00".join(call_ids))
    return digest.hexdigest()


def _build_state_user_message(task_msg: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    block = f"{_STATE_OPEN}\n{canonical_json(state)}\n{_STATE_CLOSE}"
    projected = copy.deepcopy(task_msg)
    projected.pop("api_content", None)
    if isinstance(projected.get("content"), str):
        projected["content"] = projected["content"] + "\n\n" + block
    elif isinstance(projected.get("content"), list):
        projected["content"] = [*projected["content"], {"type": "text", "text": "\n\n" + block}]
    else:
        projected["content"] = block
    return projected


def project_request(
    agent: Any,
    path: Path,
    api_messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Return the projected request, or the original list on ANY doubt."""
    if not api_messages or api_messages[0].get("role") != "system":
        return api_messages

    # Locate the most recent assistant message carrying tool calls.
    assistant_idx = None
    for idx in range(len(api_messages) - 1, -1, -1):
        if api_messages[idx].get("role") == "assistant" and api_messages[idx].get("tool_calls"):
            assistant_idx = idx
            break
    if assistant_idx is None:
        return api_messages

    tool_results = _matching_tool_results(api_messages, assistant_idx)
    if tool_results is None:
        return api_messages  # incomplete observation — full history

    patch = extract_merge_patch(api_messages[assistant_idx].get("content"))
    if patch is None:
        return api_messages  # missing/duplicate/malformed/oversized — full history

    signature = _patch_signature(patch, api_messages[assistant_idx])
    if getattr(agent, "_skill_state_applied_sig", None) != signature:
        try:
            try:
                doc = read_state(path)
            except (SkillStateError, OSError, ValueError):
                doc = {"version": SCHEMA_VERSION, "revision": 0, "state": {}}
            new_state = apply_merge_patch(doc.get("state") or {}, patch)
            new_doc = {
                "version": SCHEMA_VERSION,
                "revision": int(doc.get("revision", 0)) + 1,
                "state": new_state,
            }
            write_state_atomic(path, new_doc)
        except (SkillStateError, OSError, ValueError):
            return api_messages  # publish failed — never compact on doubt
        agent._skill_state_applied_sig = signature

    try:
        current = read_state(path)
    except (SkillStateError, OSError, ValueError):
        return api_messages  # stale/unreadable canonical state — full history

    # Original task = first user message after the system prompt.
    task_idx = None
    for idx in range(1, len(api_messages)):
        if api_messages[idx].get("role") == "user":
            task_idx = idx
            break
    if task_idx is None:
        return api_messages

    assistant = copy.deepcopy(api_messages[assistant_idx])
    assistant.pop("reasoning", None)
    assistant.pop("reasoning_content", None)
    assistant.pop("reasoning_details", None)
    for key in list(assistant):
        if isinstance(key, str) and key.startswith("_"):
            assistant.pop(key)

    return [
        api_messages[0],
        _build_state_user_message(api_messages[task_idx], current.get("state") or {}),
        assistant,
        *[copy.deepcopy(r) for r in tool_results],
    ]


def maybe_project_skill_state(agent: Any, api_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Loop hook — identity when inactive; fail-open to full history."""
    path = resolve_skill_state_path()
    if path is None:
        return api_messages
    try:
        return project_request(agent, path, api_messages)
    except Exception:
        return api_messages
