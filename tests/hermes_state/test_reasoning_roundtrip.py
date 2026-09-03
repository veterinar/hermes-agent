"""Round-trip tests for the structured reasoning columns.

get_messages() returns reasoning_details / codex_reasoning_items /
codex_message_items as the raw TEXT stored in their columns (it only
hydrates content and tool_calls). Callers that feed those rows straight
back into a write — the POST /api/sessions/{id}/fork handler pipes
get_messages() into replace_messages() — must not re-encode that TEXT,
or the forked session replays with reasoning fields decoding to strings
and every isinstance(..., list) consumer silently drops them.
"""
import pytest

from hermes_state import SessionDB


REASONING_DETAILS = [
    {"type": "reasoning.text", "text": "compare both branches first", "format": "unknown"}
]
CODEX_REASONING_ITEMS = [
    {"id": "rs_1", "type": "reasoning", "encrypted_content": "opaque-blob"}
]
CODEX_MESSAGE_ITEMS = [
    {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "done"}],
    }
]


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


def _seed(db, sid="src"):
    """Session with one assistant message carrying all three reasoning fields."""
    db.create_session(sid, source="cli")
    db.append_message(sid, role="user", content="hi")
    db.append_message(
        sid,
        role="assistant",
        content="done",
        reasoning_details=REASONING_DETAILS,
        codex_reasoning_items=CODEX_REASONING_ITEMS,
        codex_message_items=CODEX_MESSAGE_ITEMS,
    )


def _fork(db, src, dst):
    """The fork handler's copy step: raw get_messages rows into replace_messages."""
    db.create_session(dst, source="cli")
    db.replace_messages(dst, db.get_messages(src))


def _assistant(conversation):
    return next(m for m in conversation if m["role"] == "assistant")


class TestDirectWrite:
    """Live-runtime path: structured values in, structured values back."""

    def test_reasoning_fields_hydrate_as_structures(self, db):
        _seed(db)
        msg = _assistant(db.get_messages_as_conversation("src"))
        assert msg["reasoning_details"] == REASONING_DETAILS
        assert msg["codex_reasoning_items"] == CODEX_REASONING_ITEMS
        assert msg["codex_message_items"] == CODEX_MESSAGE_ITEMS


class TestForkRoundTrip:
    """get_messages -> replace_messages must keep the stored TEXT intact."""

    def test_reasoning_details_survive_fork(self, db):
        _seed(db)
        _fork(db, "src", "fork")
        msg = _assistant(db.get_messages_as_conversation("fork"))
        assert msg["reasoning_details"] == REASONING_DETAILS

    def test_codex_reasoning_items_survive_fork(self, db):
        _seed(db)
        _fork(db, "src", "fork")
        msg = _assistant(db.get_messages_as_conversation("fork"))
        assert msg["codex_reasoning_items"] == CODEX_REASONING_ITEMS

    def test_codex_message_items_survive_fork(self, db):
        _seed(db)
        _fork(db, "src", "fork")
        msg = _assistant(db.get_messages_as_conversation("fork"))
        assert msg["codex_message_items"] == CODEX_MESSAGE_ITEMS

    def test_fork_of_fork_stays_stable(self, db):
        # Each extra round-trip used to add another encoding layer.
        _seed(db)
        _fork(db, "src", "fork1")
        _fork(db, "fork1", "fork2")
        msg = _assistant(db.get_messages_as_conversation("fork2"))
        assert msg["reasoning_details"] == REASONING_DETAILS
        assert msg["codex_reasoning_items"] == CODEX_REASONING_ITEMS
        assert msg["codex_message_items"] == CODEX_MESSAGE_ITEMS


class TestAppendMessageRoundTrip:
    """append_message accepts a stored row's already-serialized TEXT too."""

    def test_string_value_not_double_encoded(self, db):
        _seed(db)
        row = next(m for m in db.get_messages("src") if m["role"] == "assistant")
        db.create_session("copy", source="cli")
        db.append_message(
            "copy",
            role="assistant",
            content="done",
            reasoning_details=row["reasoning_details"],
        )
        msg = _assistant(db.get_messages_as_conversation("copy"))
        assert msg["reasoning_details"] == REASONING_DETAILS


ANTHROPIC_CONTENT_BLOCKS = [
    {"type": "text", "text": "part one"},
    {"type": "thinking", "thinking": "hmm", "signature": "sig-1"},
    {"type": "text", "text": "part two"},
]


def _seed_anthropic(db, sid="src"):
    db.create_session(sid, source="cli")
    db.append_message(sid, role="user", content="hi")
    db.append_message(
        sid,
        role="assistant",
        content="done",
        anthropic_content_blocks=ANTHROPIC_CONTENT_BLOCKS,
    )
    return db.get_messages(sid)


class TestAnthropicContentBlocks:
    """Ordered interleaved Anthropic blocks survive every round-trip."""

    def test_blocks_survive_append_and_conversation(self, db):
        _seed_anthropic(db)
        msg = _assistant(db.get_messages_as_conversation("src"))
        assert msg["anthropic_content_blocks"] == ANTHROPIC_CONTENT_BLOCKS

    def test_blocks_survive_replace_roundtrip(self, db):
        rows = _seed_anthropic(db)
        db.create_session("fork", source="cli")
        db.replace_messages("fork", rows)
        msg = _assistant(db.get_messages_as_conversation("fork"))
        assert msg["anthropic_content_blocks"] == ANTHROPIC_CONTENT_BLOCKS

    def test_blocks_survive_close_and_reopen(self, db, tmp_path):
        _seed_anthropic(db)
        db.close()
        db2 = SessionDB(tmp_path / "state.db")
        try:
            msg = _assistant(db2.get_messages_as_conversation("src"))
            assert msg["anthropic_content_blocks"] == ANTHROPIC_CONTENT_BLOCKS
        finally:
            db2.close()

    def test_non_assistant_sidecar_not_stored(self, db):
        db.create_session("u", source="cli")
        db.append_message(
            "u",
            role="user",
            content="hi",
            anthropic_content_blocks=ANTHROPIC_CONTENT_BLOCKS,
        )
        msgs = db.get_messages_as_conversation("u")
        assert "anthropic_content_blocks" not in msgs[0]

    def test_legacy_column_reconciled_and_readable(self, db, tmp_path):
        _seed_anthropic(db)
        db.close()
        import sqlite3

        con = sqlite3.connect(tmp_path / "state.db")
        con.execute("ALTER TABLE messages DROP COLUMN anthropic_content_blocks")
        con.commit()
        con.close()
        db2 = SessionDB(tmp_path / "state.db")
        try:
            msgs = db2.get_messages_as_conversation("src")
            # Column is reconciled back; rows read with no sidecar.
            assert [m["role"] for m in msgs] == ["user", "assistant"]
            assert "anthropic_content_blocks" not in _assistant(msgs)
        finally:
            db2.close()


class TestAnthropicDivergenceClearing:
    """Prefix edits/truncations clear opaque sidecars once, canonically."""

    def test_prefix_edit_clears_sidecars_and_keeps_archive(self, db, tmp_path):
        rows = _seed_anthropic(db)
        # Edit the user turn (first message) and append a new assistant turn.
        edited = [dict(m) for m in rows]
        edited[0]["content"] = "hi (edited)"
        edited.append(
            {
                "role": "assistant",
                "content": "fresh",
                "anthropic_content_blocks": [
                    {"type": "text", "text": "fresh blocks"}
                ],
            }
        )
        db.replace_messages("src", edited, archive_dropped=True)

        all_rows = db.get_messages("src", include_inactive=True)
        # Old canonical rows retained as inactive with old ids.
        inactive = [r for r in all_rows if not r.get("active", 1)]
        active = [r for r in all_rows if r.get("active", 1)]
        assert len(inactive) == 2
        assert [r["id"] for r in inactive] == [r["id"] for r in rows]
        assert len(active) == 3
        # Sidecars cleared on both old (archived) and new dependent rows.
        for r in all_rows:
            assert r.get("anthropic_content_blocks") is None

        # Idempotent reopen: no further clearing, no added rows.
        db.close()
        db2 = SessionDB(tmp_path / "state.db")
        try:
            again = db2.get_messages("src", include_inactive=True)
            assert len(again) == len(all_rows)
            for r in again:
                assert r.get("anthropic_content_blocks") is None
        finally:
            db2.close()

    def test_strict_append_keeps_new_sidecars(self, db):
        rows = _seed_anthropic(db)
        grown = list(rows) + [
            {
                "role": "user",
                "content": "more",
            },
            {
                "role": "assistant",
                "content": "more done",
                "anthropic_content_blocks": [
                    {"type": "text", "text": "appended blocks"}
                ],
            },
        ]
        db.replace_messages("src", grown, archive_dropped=True)
        conv = db.get_messages_as_conversation("src")
        assistants = [m for m in conv if m["role"] == "assistant"]
        assert assistants[0]["anthropic_content_blocks"] == ANTHROPIC_CONTENT_BLOCKS
        assert assistants[1]["anthropic_content_blocks"] == [
            {"type": "text", "text": "appended blocks"}
        ]
