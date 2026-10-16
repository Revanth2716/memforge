"""Pagination limits for the REST API."""

from __future__ import annotations

#: Default page size for ``GET /api/v1/memories``.
DEFAULT_PAGE_LIMIT = 50

#: Hard upper bound for a single page.
MAX_PAGE_LIMIT = 100
