"""Round-trip tests for the structured reasoning columns.

get_messages() returns reasoning_details / codex_reasoning_items /
codex_message_items as the raw TEXT stored in their columns (it only
hydrates content and tool_calls). Callers that feed those rows straight
back into a write — the POST /api/sessions/{id}/fork handler pipes
get_messages() into replace_messages() — must not re-encode that TEXT,
or the forked session replays with reasoning fields decoding to strings
and every isinstance(..., list) consumer silently drops them.
"""
import copy
import sqlite3

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

    def test_strict_append_with_api_content_prefix_keeps_sidecars(self, db):
        # AC-STATE-6: an identical provider-visible prefix (content AND
        # api_content) is a strict append — existing and new sidecars live.
        db.create_session("src", source="cli")
        db.append_message("src", role="user", content="hi")
        db.append_message(
            "src",
            role="assistant",
            content="done",
            api_content="done (provider bytes)",
            anthropic_content_blocks=ANTHROPIC_CONTENT_BLOCKS,
        )
        rows = db.get_messages("src")
        grown = list(rows) + [
            {"role": "user", "content": "more"},
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


class TestApiContentDivergence:
    """AC-STATE-3: clean content match + changed api_content IS divergence."""

    def test_api_content_only_change_invalidates_dependent_sidecars(self, db):
        db.create_session("src", source="cli")
        db.append_message("src", role="user", content="hi")
        db.append_message(
            "src",
            role="assistant",
            content="done",
            api_content="original provider bytes",
            anthropic_content_blocks=ANTHROPIC_CONTENT_BLOCKS,
        )
        rows = db.get_messages("src")
        # Same clean content, DIFFERENT provider-visible api_content.
        edited = [dict(m) for m in rows]
        edited[1]["api_content"] = "rewritten provider bytes"
        db.replace_messages("src", edited, archive_dropped=True)
        conv = db.get_messages_as_conversation("src")
        assistant = _assistant(conv)
        assert "anthropic_content_blocks" not in assistant
        # Caller's dict stripped too.
        assert "anthropic_content_blocks" not in edited[1]
        # The persisted (archived) original row lost its sidecar as well.
        for r in db.get_messages("src", include_inactive=True):
            assert r.get("anthropic_content_blocks") is None

    def test_api_content_absence_normalized_consistently(self, db):
        # Stored row without api_content vs incoming dict without the key:
        # not divergence — sidecar survives an in-place rewrite whose
        # divergence sweep runs against the populated session.
        rows = _seed_anthropic(db)
        edited = [dict(m) for m in rows]
        edited[1].pop("api_content", None)
        db.replace_messages("src", edited)
        assert (
            _assistant(db.get_messages_as_conversation("src"))[
                "anthropic_content_blocks"
            ]
            == ANTHROPIC_CONTENT_BLOCKS
        )


class TestBoundedInvalidationShape:
    """AC-STATE-4: the clearing SQL is constant-size, not per-row placeholders."""

    def test_bounded_update_uses_two_bind_values(self, db, monkeypatch):
        # Capture the clearing UPDATE's bound parameters on the real write
        # connection: it must carry exactly (session_id, first_divergent_id)
        # regardless of transcript length — the bounded form, never an
        # IN list with a placeholder per row.
        real_conn = db._conn
        captured = []

        def _trace(sql, params=()):
            if "anthropic_content_blocks = NULL" in sql:
                captured.append(tuple(params))
            return real_trace(sql, params)

        real_trace = real_conn.execute
        monkeypatch.setattr(real_conn, "execute", _trace)

        db.create_session("src", source="cli")
        # Long enough that a per-row placeholder form would need many binds.
        for i in range(20):
            db.append_message("src", role="user", content=f"q{i}")
            db.append_message(
                "src",
                role="assistant",
                content=f"a{i}",
                anthropic_content_blocks=[{"type": "text", "text": f"blocks {i}"}],
            )
        monkeypatch.undo()
        rows = db.get_messages("src")
        edited = [dict(m) for m in rows]
        edited[0]["content"] = "q0 (edited)"
        monkeypatch.setattr(real_conn, "execute", _trace)
        try:
            db.replace_messages("src", edited)
        finally:
            monkeypatch.undo()
        assert captured == [(("src"), (rows[0]["id"]))]


class TestRollbackMutationSafety:
    """AC-STATE-5: a failed transaction leaves caller dicts deep-equal."""

    def test_failed_write_leaves_inputs_untouched(self, db, monkeypatch):
        rows = _seed_anthropic(db)
        edited = [dict(m) for m in rows]
        edited[0]["content"] = "hi (edited)"
        before = copy.deepcopy(edited)

        def _boom(conn, session_id, messages):
            raise sqlite3.OperationalError("injected write failure")

        monkeypatch.setattr(db, "_insert_message_rows", _boom)
        with pytest.raises(sqlite3.OperationalError):
            db.replace_messages("src", edited, archive_dropped=True)
        assert edited == before
        # Stored rows untouched — the old sidecar is still there.
        assert (
            _assistant(db.get_messages_as_conversation("src"))[
                "anthropic_content_blocks"
            ]
            == ANTHROPIC_CONTENT_BLOCKS
        )


def _clear_at_system(text):
    return {
        "role": "system",
        "content": text,
        "clear_at": "next_user_message",
    }


def _seed_clear_at_pair(db, sid="src"):
    """U1,S1,A1,U2,S2,A2 with two distinct clear_at system rows."""
    db.create_session(sid, source="cli")
    transcript = [
        {"role": "user", "content": "u1"},
        _clear_at_system("system one"),
        {
            "role": "assistant",
            "content": "a1",
            "display_metadata": {"unrelated": {"k": [1, 2]}},
            "anthropic_content_blocks": [{"type": "text", "text": "a1 blocks"}],
        },
        {"role": "user", "content": "u2"},
        _clear_at_system("system two"),
        {
            "role": "assistant",
            "content": "a2",
            "display_metadata": {"other": "meta"},
        },
    ]
    db.replace_messages(sid, transcript)
    return transcript


class TestClearAtSystemRows:
    """AC-STATE-1/2: folded clear_at system rows survive every round trip."""

    def test_replay_after_replace_returns_exact_rows_in_order(self, db):
        transcript = _seed_clear_at_pair(db)
        conv = db.get_messages_as_conversation("src")
        assert [m["role"] for m in conv] == [
            "user",
            "system",
            "assistant",
            "user",
            "system",
            "assistant",
        ]
        # Semantic fields of the system rows only; replay may add
        # persistence metadata (timestamp, _row_id) we don't compare.
        for original, replayed in zip(transcript, conv):
            if original["role"] != "system":
                continue
            assert {k: replayed[k] for k in original} == original
        # No private carrier key leaks into any display metadata.
        for m in conv:
            meta = m.get("display_metadata") or {}
            assert "__hermes_clear_at_system__" not in meta

    def test_survive_close_and_reopen_at_exact_positions(self, db, tmp_path):
        _seed_clear_at_pair(db)
        db.close()
        db2 = SessionDB(tmp_path / "state.db")
        try:
            conv = db2.get_messages_as_conversation("src")
            assert [m["role"] for m in conv] == [
                "user",
                "system",
                "assistant",
                "user",
                "system",
                "assistant",
            ]
            assert conv[1]["content"] == "system one"
            assert conv[4]["content"] == "system two"
            for m in conv:
                meta = m.get("display_metadata") or {}
                assert "__hermes_clear_at_system__" not in meta
        finally:
            db2.close()

    def test_survive_replace_fork_roundtrip_with_metadata(self, db):
        _seed_clear_at_pair(db)
        conv = db.get_messages_as_conversation("src")
        # Fork: replayed conversation straight back through replace.
        db.create_session("fork", source="cli")
        db.replace_messages("fork", [dict(m) for m in conv])
        forked = db.get_messages_as_conversation("fork")
        assert [m["role"] for m in forked] == [m["role"] for m in conv]
        assert forked[1]["content"] == "system one"
        assert forked[4]["content"] == "system two"
        # Unrelated assistant display metadata preserved through the fork.
        assert forked[2]["display_metadata"] == {"unrelated": {"k": [1, 2]}}
        assert forked[5]["display_metadata"] == {"other": "meta"}
        for m in forked:
            meta = m.get("display_metadata") or {}
            assert "__hermes_clear_at_system__" not in meta

    def test_get_messages_returns_physical_system_rows_normalized(self, db):
        _seed_clear_at_pair(db)
        rows = db.get_messages("src")
        assert [r["role"] for r in rows] == [
            "user",
            "system",
            "assistant",
            "user",
            "system",
            "assistant",
        ]
        for row, text in ((rows[1], "system one"), (rows[4], "system two")):
            assert row["role"] == "system"
            assert row["content"] == text
            assert row["clear_at"] == "next_user_message"
        for r in rows:
            meta = r.get("display_metadata") or {}
            assert "__hermes_clear_at_system__" not in meta
