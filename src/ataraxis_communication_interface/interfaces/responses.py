"""Provides the paging and projection machinery the Model Context Protocol read tools share."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from dataclasses import dataclass

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

_DEFAULT_ITEM_LIMIT: int = 200
"""The items a semi-detail page carries when the caller names no limit."""

_DEFAULT_DETAILED_LIMIT: int = 50
"""The items a detailed page carries when the caller names no limit."""


@dataclass(frozen=True, slots=True)
class PageWindow:
    """Describes the slice of a matched item set one response carries."""

    start: int
    """The index the page begins at, counted from the first matching item."""
    length: int | None
    """The items the page carries, or None when the page runs to the end of the matches."""
    next_start_row: int | None
    """The ``start_row`` that retrieves the following page, or None when this page ends the matches. A caller walks a
    matched set by following this until it is None, since a page that fills its own limit exactly may still end the
    matches."""

    @property
    def stop(self) -> int | None:
        """Returns the index the page ends before, or None when it runs to the end of the matches."""
        return None if self.length is None else self.start + self.length


def resolve_page(total: int, limit: int, start_row: int) -> PageWindow:
    """Resolves which slice of a matched item set a response carries.

    Args:
        total: The items matching the caller's filters, before any cap.
        limit: The items to carry, or a value at or below zero to carry every match.
        start_row: The index to begin at, counted from the first matching item. A negative value starts at the
            beginning.

    Returns:
        The window describing the page, carrying the start index, its length, and the start row of the next page.
    """
    start = max(0, start_row)
    if start >= total:
        return PageWindow(start=start, length=0, next_start_row=None)
    if limit <= 0:
        return PageWindow(start=start, length=None, next_start_row=None)

    remaining = total - start
    length = min(limit, remaining)
    return PageWindow(start=start, length=length, next_start_row=start + length if length < remaining else None)


def page_fields(window: PageWindow, total: int, listed: int) -> dict[str, Any]:
    """Renders the paging fields a response reports alongside its items.

    Args:
        window: The resolved page window.
        total: The items matching the caller's filters, before any cap.
        listed: The items the response actually carries.

    Returns:
        The listed count, the matched total, the start row, and the next start row.
    """
    return {
        "rows": listed,
        "matched_rows": total,
        "start_row": window.start,
        "next_start_row": window.next_start_row,
    }


def resolve_detail_limit(limit: int | None, *, detailed: bool) -> int:
    """Resolves the page size to use when the caller named none, from the detail the response carries.

    Args:
        limit: The limit the caller named, or None to take the default.
        detailed: Determines whether the response carries full per-item fields.

    Returns:
        The page size to apply.
    """
    if limit is not None:
        return limit
    if not detailed:
        return _DEFAULT_ITEM_LIMIT
    return _DEFAULT_DETAILED_LIMIT


def item_breakdown(items: Sequence[dict[str, Any]], axes: tuple[str, ...]) -> dict[str, dict[str, int]]:
    """Counts how many items carry each value of every filterable axis.

    Args:
        items: The whole matched item set.
        axes: The keys to count, which are the keys a caller may filter by.

    Returns:
        Each present axis mapped to its value counts.
    """
    return {
        axis: _count_values(values=[item.get(axis) for item in items])
        for axis in axes
        if any(axis in item for item in items)
    }


def project_item(item: dict[str, Any], fields: Sequence[str]) -> dict[str, Any]:
    """Narrows one item to the named fields, leaving out the ones carrying nothing.

    Notes:
        An absent key reads as empty, so omitting a field that holds nothing costs a reader no information and keeps
        the common case small.

    Args:
        item: The item to narrow.
        fields: The fields to keep, in the order they should appear.

    Returns:
        The narrowed item.
    """
    narrowed: dict[str, Any] = {}
    for field_name in fields:
        if field_name not in item:
            continue
        value = item[field_name]
        if value is None or (isinstance(value, list | dict | str) and not value):
            continue
        narrowed[field_name] = value
    return narrowed


def reject_unknown(items: Sequence[dict[str, Any]], key: str, values: list[str], subject: str) -> dict[str, Any] | None:
    """Builds the error response for a filter naming a value the scan did not find.

    Notes:
        Reports what is available rather than returning an empty page, because an empty page and a mistyped filter look
        identical to a caller otherwise.

    Args:
        items: The whole matched item set.
        key: The item key being filtered.
        values: The values the caller named.
        subject: The noun naming what one item describes.

    Returns:
        The error response, or None when every named value is present.
    """
    available = sorted({str(item[key]) for item in items if item.get(key) is not None})
    unknown = sorted({value for value in values if value not in available})
    if unknown:
        return {"error": f"No {subject} has '{key}' in {unknown}. Available: {available}."}
    return None


def _count_values(values: Iterable[Any]) -> dict[str, int]:
    """Counts how often each value occurs.

    Notes:
        Values are keyed by their string form, so an enumeration member and its value count as one. A null counts
        under ``none``, since an absent subject is itself a category a caller filters on.

    Args:
        values: The values to count.

    Returns:
        Each value mapped to its count, ordered by value.
    """
    counts: dict[str, int] = {}
    for value in values:
        key = "none" if value is None else str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))
