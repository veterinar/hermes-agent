"""Tests for MCP list_* pagination (nextCursor draining).

The MCP spec allows servers to paginate ``tools/list``, ``resources/list``,
and ``prompts/list`` via an opaque ``nextCursor`` token. The Python SDK
fetches one page per call, so hermes must follow the cursor to see items
past page 1. Port of the invariant behind anomalyco/opencode#35439/#35500.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from tools.mcp_tool import _MCP_LIST_MAX_PAGES, _paginate_full_list


def _tool(name):
    t = MagicMock()
    t.name = name
    return t


class TestPaginateFullList:
    def test_single_page_no_cursor(self):
        """A result without nextCursor returns just that page."""
        list_method = AsyncMock(
            return_value=SimpleNamespace(tools=[_tool("a"), _tool("b")])
        )
        items = asyncio.run(_paginate_full_list(list_method, "tools", "srv"))
        assert [t.name for t in items] == ["a", "b"]
        list_method.assert_called_once_with()


    def test_runaway_cursor_capped(self):
        """A server that returns a cursor forever is bounded by the page cap."""
        calls = {"n": 0}

        async def evil_list(cursor=None):
            calls["n"] += 1
            return SimpleNamespace(
                tools=[_tool(f"t{calls['n']}")], nextCursor=f"c{calls['n']}"
            )

        items = asyncio.run(_paginate_full_list(evil_list, "tools", "srv"))
        assert calls["n"] == _MCP_LIST_MAX_PAGES
        assert len(items) == _MCP_LIST_MAX_PAGES

    def test_unrestricted_drains_beyond_default_page_cap(self, monkeypatch):
        """Unrestricted mode has no fixed page-count ceiling."""
        import tools.mcp_tool as mcp_tool

        monkeypatch.setattr(mcp_tool, "is_unrestricted", lambda: True)
        calls = {"n": 0}

        async def many_pages(cursor=None):
            calls["n"] += 1
            next_cursor = f"c{calls['n']}" if calls["n"] < _MCP_LIST_MAX_PAGES + 5 else None
            return SimpleNamespace(
                tools=[_tool(f"t{calls['n']}")], nextCursor=next_cursor
            )

        items = asyncio.run(_paginate_full_list(many_pages, "tools", "srv"))

        assert calls["n"] == _MCP_LIST_MAX_PAGES + 5
        assert len(items) == _MCP_LIST_MAX_PAGES + 5

    def test_unrestricted_still_stops_on_repeated_cursor(self, monkeypatch):
        """Removing the count cap must not permit an infinite cursor cycle."""
        import tools.mcp_tool as mcp_tool

        monkeypatch.setattr(mcp_tool, "is_unrestricted", lambda: True)
        calls = {"n": 0}

        async def repeated_cursor(cursor=None):
            calls["n"] += 1
            return SimpleNamespace(tools=[_tool("item")], nextCursor="same")

        items = asyncio.run(_paginate_full_list(repeated_cursor, "tools", "srv"))

        assert calls["n"] == 2
        assert len(items) == 2


class TestDiscoveryUsesPagination:
    def test_discover_tools_drains_all_pages(self):
        """MCPServerTask._discover_tools registers tools from every page."""
        from tools.mcp_tool import MCPServerTask

        server = MCPServerTask("pag_srv")
        server._config = {"command": "test"}
        pages = {
            None: SimpleNamespace(tools=[_tool("first")], nextCursor="page-2"),
            "page-2": SimpleNamespace(tools=[_tool("second")]),
        }

        async def fake_list(cursor=None):
            return pages[cursor]

        server.session = MagicMock()
        server.session.list_tools = fake_list
        # capability gate: _advertises_tools() returns True when no
        # capability info was captured (legacy fallback), so no override
        # is needed here.

        asyncio.run(server._discover_tools())
        assert [t.name for t in server._tools] == ["first", "second"]
