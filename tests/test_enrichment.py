"""
Unit tests for enrichment/pubmed.py and enrichment/openfda.py.

All aiohttp network calls are mocked — no real HTTP traffic.

Tests:
  - PubMed: correct integer count parsed from esearchresult JSON
  - PubMed: returns 0 on HTTP error
  - PubMed: returns 0 on timeout (asyncio.TimeoutError)
  - PubMed: returns 0 on any unexpected exception
  - PubMed: handles count of 0 correctly
  - OpenFDA: returns "FDA Approved" when results array is non-empty
  - OpenFDA: returns "Not in FDA Database" on 404
  - OpenFDA: returns "Not in FDA Database" when results array is empty
  - OpenFDA: returns "Unknown" on non-404 HTTP error
  - OpenFDA: returns "Unknown" on general exception
"""

from __future__ import annotations

import asyncio
import sys
import os
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from enrichment.pubmed import evidence_count
from enrichment.openfda import fda_status


# ---------------------------------------------------------------------------
# aiohttp mock helpers
# ---------------------------------------------------------------------------

def _make_mock_session(mock_response):
    """Build a MagicMock aiohttp.ClientSession whose .get() is a context manager
    that yields mock_response."""
    session = MagicMock()

    @asynccontextmanager
    async def mock_get(*args, **kwargs):
        yield mock_response

    session.get = mock_get
    return session


def _make_mock_response(status: int, json_data: dict | None = None, raise_on_json=None):
    """Build a fake aiohttp response."""
    response = MagicMock()
    response.status = status

    if status >= 400:
        import aiohttp
        response.raise_for_status = MagicMock(
            side_effect=aiohttp.ClientResponseError(
                request_info=MagicMock(), history=(), status=status
            )
        )
    else:
        response.raise_for_status = MagicMock()

    if raise_on_json:
        response.json = AsyncMock(side_effect=raise_on_json)
    else:
        response.json = AsyncMock(return_value=json_data)

    return response


# ---------------------------------------------------------------------------
# evidence_count() — PubMed
# ---------------------------------------------------------------------------

class TestEvidenceCount:
    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_parses_count_correctly(self):
        json_data = {"esearchresult": {"count": "42"}}
        response = _make_mock_response(200, json_data)
        session = _make_mock_session(response)
        count = self._run(evidence_count("metformin", "Diabetes Mellitus, Type 2", session))
        assert count == 42

    def test_returns_zero_on_count_zero(self):
        json_data = {"esearchresult": {"count": "0"}}
        response = _make_mock_response(200, json_data)
        session = _make_mock_session(response)
        count = self._run(evidence_count("aspirin", "Alzheimer Disease", session))
        assert count == 0

    def test_returns_zero_on_http_error(self):
        import aiohttp
        response = _make_mock_response(500)
        session = _make_mock_session(response)
        count = self._run(evidence_count("ibuprofen", "Hypertension", session))
        assert count == 0

    def test_returns_zero_on_timeout(self):
        response = _make_mock_response(200, raise_on_json=asyncio.TimeoutError())
        session = _make_mock_session(response)
        count = self._run(evidence_count("warfarin", "Atrial Fibrillation", session))
        assert count == 0

    def test_returns_zero_on_malformed_json(self):
        # Missing 'count' key
        json_data = {"esearchresult": {}}
        response = _make_mock_response(200, json_data)
        session = _make_mock_session(response)
        count = self._run(evidence_count("aspirin", "Breast Neoplasms", session))
        assert count == 0

    def test_returns_zero_on_unexpected_exception(self):
        response = _make_mock_response(200, raise_on_json=RuntimeError("unexpected"))
        session = _make_mock_session(response)
        count = self._run(evidence_count("metformin", "Hypertension", session))
        assert count == 0

    def test_returns_integer_type(self):
        json_data = {"esearchresult": {"count": "7"}}
        response = _make_mock_response(200, json_data)
        session = _make_mock_session(response)
        count = self._run(evidence_count("atorvastatin", "Hypertension", session))
        assert isinstance(count, int)

    def test_large_count(self):
        json_data = {"esearchresult": {"count": "12345"}}
        response = _make_mock_response(200, json_data)
        session = _make_mock_session(response)
        count = self._run(evidence_count("aspirin", "Cardiovascular Diseases", session))
        assert count == 12345


# ---------------------------------------------------------------------------
# fda_status() — OpenFDA
# ---------------------------------------------------------------------------

class TestFdaStatus:
    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_fda_approved_when_results_present(self):
        json_data = {"results": [{"openfda": {"brand_name": ["Metformin"]}}]}
        response = _make_mock_response(200, json_data)
        session = _make_mock_session(response)
        status = self._run(fda_status("metformin", session))
        assert status == "FDA Approved"

    def test_not_in_fda_database_on_404(self):
        import aiohttp

        session = MagicMock()

        @asynccontextmanager
        async def mock_get(*args, **kwargs):
            response = MagicMock()
            response.status = 404
            response.raise_for_status = MagicMock()
            response.json = AsyncMock(
                side_effect=aiohttp.ClientResponseError(
                    request_info=MagicMock(), history=(), status=404
                )
            )
            yield response

        session.get = mock_get

        # Override raise_for_status to not raise (404 is handled separately)
        # In the actual code, 404 check is done before raise_for_status
        # We need to simulate the 404 branch properly
        import aiohttp as _aiohttp

        session2 = MagicMock()

        @asynccontextmanager
        async def mock_get_404(*args, **kwargs):
            resp = MagicMock()
            resp.status = 404
            resp.raise_for_status = MagicMock(
                side_effect=_aiohttp.ClientResponseError(
                    request_info=MagicMock(), history=(), status=404
                )
            )
            resp.json = AsyncMock(return_value={})
            yield resp

        session2.get = mock_get_404
        status = self._run(fda_status("totally_unknown_drug_xyz", session2))
        assert status == "Not in FDA Database"

    def test_not_in_fda_database_when_results_empty(self):
        json_data = {"results": []}
        response = _make_mock_response(200, json_data)
        session = _make_mock_session(response)
        status = self._run(fda_status("obscure_drug", session))
        assert status == "Not in FDA Database"

    def test_unknown_on_server_error(self):
        import aiohttp
        session = MagicMock()

        @asynccontextmanager
        async def mock_get(*args, **kwargs):
            resp = MagicMock()
            resp.status = 500
            resp.raise_for_status = MagicMock(
                side_effect=aiohttp.ClientResponseError(
                    request_info=MagicMock(), history=(), status=500
                )
            )
            resp.json = AsyncMock(return_value={})
            yield resp

        session.get = mock_get
        status = self._run(fda_status("some_drug", session))
        assert status == "Unknown"

    def test_unknown_on_connection_error(self):
        session = MagicMock()

        @asynccontextmanager
        async def mock_get(*args, **kwargs):
            raise ConnectionError("network failure")
            yield  # unreachable but required for asynccontextmanager

        session.get = mock_get
        status = self._run(fda_status("some_drug", session))
        assert status == "Unknown"

    def test_unknown_on_json_parse_error(self):
        response = _make_mock_response(200, raise_on_json=ValueError("bad json"))
        session = _make_mock_session(response)
        status = self._run(fda_status("some_drug", session))
        assert status == "Unknown"

    def test_returns_string(self):
        json_data = {"results": [{"openfda": {}}]}
        response = _make_mock_response(200, json_data)
        session = _make_mock_session(response)
        status = self._run(fda_status("aspirin", session))
        assert isinstance(status, str)

    def test_no_results_key_returns_not_in_db(self):
        """If 'results' key is missing from response body."""
        json_data = {}
        response = _make_mock_response(200, json_data)
        session = _make_mock_session(response)
        status = self._run(fda_status("unknown", session))
        assert status == "Not in FDA Database"