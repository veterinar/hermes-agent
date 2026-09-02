"""Opt-in task-local skill-state projection.

Established mode (0.20.5) ported onto Hermes 0.21.0 unchanged in shape:

* Opt-in solely via ``HERMES_SKILL_STATE_PATH``. Unset or invalid → the
  legacy route is byte-for-byte preserved (no projection, no system-prompt
  instruction, no history rewriting).
* The state document is canonical JSON with exactly ``version=1``, a
  nonnegative integer ``revision``, and an object ``state``. All reads and
  writes are bounded and fail closed.
* An accepted patch is a single explicitly delimited JSON Merge Patch
  carried in an assistant tool-call message, evidenced by ALL of the
  matching tool results. Anything missing, duplicate, malformed,
  oversized, stale, or incomplete preserves full history.
* After acceptance, ONLY the next provider request is projected:
  unchanged system prompt, the original user task plus canonical state,
  the latest complete assistant tool-call message, and its matching tool
  results. Durable transcripts remain complete.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import stat
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

ENV_VAR = "HERMES_SKILL_STATE_PATH"

VERSION = 1

# Delimiters around the single JSON Merge Patch in the assistant
# tool-call message content.
BEGIN_MARKER = "<<<HERMES_SKILL_STATE_MERGE_PATCH>>>"
END_MARKER = "<<<END_HERMES_SKILL_STATE_MERGE_PATCH>>>"

# Bounded I/O: refuse anything larger, read or write.
MAX_DOCUMENT_BYTES = 256 * 1024
MAX_PATCH_BYTES = 64 * 1024

# Stable protocol instruction — no state content, no absolute path.
PROTOCOL_INSTRUCTION = (
    "Task-local skill state: durable task state may be kept in a skill-state "
    "document. To update it, emit exactly one delimited JSON Merge Patch in "
    "your next tool-calling assistant message using the markers "
    f"{BEGIN_MARKER} and {END_MARKER} around a canonical JSON object with "
    "the current revision and the merge patch. The patch is applied only "
    "when every matching tool result is present; otherwise the conversation "
    "history is kept complete."
)


def protocol_instruction() -> Optional[str]:
    """The stable protocol instruction, only when the mode is active."""
    if resolve_state_path() is None:
        return None
    return PROTOCOL_INSTRUCTION


# ── Path resolution & validation ────────────────────────────────────


def _hermes_home() -> Path:
    from hermes_constants import get_hermes_home

    return Path(get_hermes_home()).resolve()


def resolve_state_path() -> Optional[Path]:
    """Resolve and validate HERMES_SKILL_STATE_PATH.

    Returns the validated absolute path, or None (inactive → legacy route)
    when unset or invalid. Requirements: absolute, strictly below
    HERMES_HOME, an existing regular non-symlink file, owner-only
    permissions.
    """
    raw = os.environ.get(ENV_VAR)
    if not raw:
        return None
    try:
        path = Path(raw)
        if not path.is_absolute():
            return None
        # Reject symlinks on the ORIGINAL path — resolve() below would
        # follow the link and hide it.
        if stat.S_ISLNK(os.lstat(path).st_mode):
            return None
        resolved = path.resolve()
        home = _hermes_home()
        if resolved == home or home not in resolved.parents:
            return None
        st = os.lstat(resolved)
        if not stat.S_ISREG(st.st_mode):
            return None
        if hasattr(os, "geteuid") and st.st_uid != os.geteuid():
            return None
        if st.st_mode & 0o077:
            return None
        return resolved
    except (OSError, ValueError):
        return None


# ── Canonical document I/O (bounded, fail closed) ───────────────────


def _canonical_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def read_state_document(path: Path) -> Optional[Dict[str, Any]]:
    """Read and validate the state document. None on any failure."""
    try:
        st = os.stat(path)
        if st.st_size > MAX_DOCUMENT_BYTES:
            return None
        with open(path, "rb") as fh:
            data = fh.read(st.st_size)
        if len(data) > MAX_DOCUMENT_BYTES:
            return None
        doc = json.loads(data.decode("utf-8"))
        if not isinstance(doc, dict):
            return None
        if doc.get("version") != VERSION:
            return None
        revision = doc.get("revision")
        if not isinstance(revision, int) or isinstance(revision, bool) or revision < 0:
            return None
        if not isinstance(doc.get("state"), dict):
            return None
        if set(doc.keys()) != {"version", "revision", "state"}:
            return None
        return doc
    except (OSError, ValueError, UnicodeDecodeError):
        return None


def write_state_document(path: Path, doc: Dict[str, Any]) -> bool:
    """Atomically write the canonical document. False on any failure."""
    try:
        payload = _canonical_dumps(doc).encode("utf-8")
        if len(payload) > MAX_DOCUMENT_BYTES:
            return False
        directory = path.parent
        fd, tmp_name = tempfile.mkstemp(dir=str(directory), prefix=".skill-state-")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            os.chmod(tmp_name, 0o600)
            os.replace(tmp_name, str(path))
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
        return True
    except OSError:
        return False


# ── Merge Patch (RFC 7396) ──────────────────────────────────────────


def merge_patch(target: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(target)
    for key, value in patch.items():
        if value is None:
            out.pop(key, None)
        elif isinstance(value, dict):
            out[key] = merge_patch(out.get(key) if isinstance(out.get(key), dict) else {}, value)
        else:
            out[key] = value
    return out


# ── Patch extraction & evidence checks ──────────────────────────────


def extract_patch_block(content: Any) -> Optional[str]:
    """The single delimited patch block, or None when not acceptable.

    None content, zero blocks, or more than one block all refuse.
    """
    if not isinstance(content, str):
        return None
    begin = content.find(BEGIN_MARKER)
    if begin == -1:
        return None
    if content.find(BEGIN_MARKER, begin + 1) != -1:
        return None  # duplicate
    end = content.find(END_MARKER, begin + len(BEGIN_MARKER))
    if end == -1:
        return None
    if content.find(END_MARKER, end + 1) != -1:
        return None  # duplicate
    return content[begin + len(BEGIN_MARKER):end]


def _tool_call_ids(message: Dict[str, Any]) -> List[str]:
    ids: List[str] = []
    for tc in message.get("tool_calls") or []:
        if isinstance(tc, dict) and tc.get("id"):
            ids.append(tc["id"])
    return ids


def check_evidence(
    api_messages: List[Dict[str, Any]], assistant_index: int
) -> Optional[Tuple[str, Dict[str, Any], List[Dict[str, Any]]]]:
    """Validate the patch candidate at ``assistant_index``.

    Returns (block_text, assistant_message_copy, matching_tool_results)
    when acceptable, else None. Requires ALL matching tool results, each
    exactly once; duplicates or missing results refuse.
    """
    assistant = api_messages[assistant_index]
    if assistant.get("role") != "assistant" or not assistant.get("tool_calls"):
        return None
    block = extract_patch_block(assistant.get("content"))
    if block is None or not block.strip():
        return None
    if len(block.encode("utf-8")) > MAX_PATCH_BYTES:
        return None  # oversized
    expected = _tool_call_ids(assistant)
    if not expected:
        return None
    seen: Dict[str, Dict[str, Any]] = {}
    for msg in api_messages[assistant_index + 1:]:
        if msg.get("role") != "tool":
            continue
        tc_id = msg.get("tool_call_id")
        if tc_id in expected:
            if tc_id in seen:
                return None  # duplicate evidence
            if msg.get("content") is None:
                return None  # incomplete evidence
            seen[tc_id] = msg
    if set(seen.keys()) != set(expected):
        return None  # missing evidence
    return block, copy.deepcopy(assistant), [copy.deepcopy(seen[i]) for i in expected]


# ── Accept & apply ──────────────────────────────────────────────────


def accept_patch(block: str, document: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Parse and apply the merge patch against the current document.

    Returns the new document, or None when the patch is malformed or
    stale (revision must equal the current revision exactly).
    """
    try:
        payload = json.loads(block)
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    if set(payload.keys()) != {"revision", "patch"}:
        return None
    revision = payload.get("revision")
    if not isinstance(revision, int) or isinstance(revision, bool):
        return None
    if revision != document["revision"]:
        return None  # stale (or from the future)
    patch = payload.get("patch")
    if not isinstance(patch, dict):
        return None
    new_state = merge_patch(document["state"], patch)
    return {
        "version": VERSION,
        "revision": document["revision"] + 1,
        "state": new_state,
    }


def _first_user_message(api_messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for msg in api_messages:
        if msg.get("role") == "user":
            return msg
    return None


# ── Projection (called from the send path) ──────────────────────────


def apply_skill_state_projection(agent: Any, api_messages: List[Dict[str, Any]]) -> bool:
    """Scan, accept, and project — called once per provider request.

    With the mode inactive this returns False immediately and
    ``api_messages`` is untouched (legacy byte-for-byte route). When a
    delimited patch is accepted, the document is updated atomically with
    a monotonic revision and THIS request (the next one after the
    assistant emitted the patch) is projected in place:

      * original user task plus the canonical state,
      * the latest complete assistant tool-call message,
      * all matching tool results.

    The persisted conversation history (``agent.messages``) is never
    modified. Any failure path simply leaves ``api_messages`` as the full
    history.
    """
    path = resolve_state_path()
    if path is None:
        return False
    document = read_state_document(path)
    if document is None:
        return False

    candidate = None
    for idx in range(len(api_messages) - 1, -1, -1):
        found = check_evidence(api_messages, idx)
        if found is not None:
            candidate = (idx, *found)
            break
    if candidate is None:
        return False
    assistant_index, block, assistant_copy, tool_results = candidate

    new_document = accept_patch(block, document)
    if new_document is None:
        return False
    if not write_state_document(path, new_document):
        return False

    user_msg = _first_user_message(api_messages)
    if user_msg is None:
        return False

    state_line = (
        "Current task-local skill state (canonical JSON):\n"
        + _canonical_dumps(new_document)
    )
    user_content = user_msg.get("content")
    if isinstance(user_content, str):
        projected_user = {**user_msg, "content": user_content + "\n\n" + state_line}
    else:
        projected_user = {**user_msg, "content": state_line}

    api_messages[:] = [projected_user, assistant_copy, *tool_results]
    logger.debug(
        "skill-state patch accepted (revision %d -> %d); next request projected",
        document["revision"],
        new_document["revision"],
    )
    return True
