from __future__ import annotations

import io
import re
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlparse


@dataclass(frozen=True)
class ArchiveFetchPlanEntry:
    url: str
    artifact_name: str
    year: int | None = None


def _path_from_url(url: str) -> str:
    return urlparse(str(url)).path


def _artifact_name_from_url(url: str, fallback: str) -> str:
    path = Path(_path_from_url(url))
    return path.name or fallback


def _infer_year(url: str) -> int | None:
    match = re.search(r"(19|20)\d{2}", _path_from_url(url))
    return int(match.group(0)) if match else None


def discover_archive_links(
    html: str,
    *,
    base_url: str,
    suffixes: Iterable[str],
) -> list[str]:
    allowed = tuple(str(suffix).lower() for suffix in suffixes)
    hrefs = re.findall(r'href=["\']([^"\']+)["\']', html, flags=re.IGNORECASE)
    urls = [
        urljoin(base_url, href)
        for href in hrefs
        if _path_from_url(href).lower().endswith(allowed)
    ]
    return sorted(set(urls))


def filter_urls_for_years(urls: Iterable[str], years: Iterable[int]) -> list[str]:
    url_list = list(urls)
    year_tokens = {str(year) for year in years}
    if not year_tokens:
        return url_list
    filtered = [
        url for url in url_list if any(token in str(url) for token in year_tokens)
    ]
    return filtered or url_list


def iter_yearly_archive_urls(
    *,
    start_year: int,
    end_year: int,
    url_template: str,
    first_year: int,
    yearly_first_year: int | None = None,
    historical_url: str | None = None,
    historical_last_year: int | None = None,
    file_urls: list[str] | None = None,
) -> list[str]:
    if file_urls:
        return list(file_urls)

    urls: list[str] = []
    requested_start = max(start_year, first_year)
    effective_yearly_first = yearly_first_year or first_year

    if (
        historical_url is not None
        and historical_last_year is not None
        and requested_start < effective_yearly_first
        and end_year >= first_year
    ):
        urls.append(historical_url)
        requested_start = max(requested_start, historical_last_year + 1)

    urls.extend(
        url_template.format(year=year)
        for year in range(max(requested_start, effective_yearly_first), end_year + 1)
    )
    return urls


def plan_archive_fetches(
    urls: Iterable[str],
    *,
    years: Iterable[int] | None = None,
    fallback_artifact_prefix: str = "archive",
) -> list[ArchiveFetchPlanEntry]:
    planned_urls = (
        filter_urls_for_years(urls, years or []) if years is not None else list(urls)
    )

    entries: list[ArchiveFetchPlanEntry] = []
    seen: set[str] = set()
    for index, url in enumerate(planned_urls, start=1):
        if url in seen:
            continue
        seen.add(url)
        entries.append(
            ArchiveFetchPlanEntry(
                url=url,
                artifact_name=_artifact_name_from_url(
                    url, f"{fallback_artifact_prefix}_{index}"
                ),
                year=_infer_year(url),
            )
        )
    return entries


def discover_archive_fetches(
    html: str,
    *,
    base_url: str,
    suffixes: Iterable[str],
    years: Iterable[int] | None = None,
    fallback_artifact_prefix: str = "archive",
) -> list[ArchiveFetchPlanEntry]:
    return plan_archive_fetches(
        discover_archive_links(html, base_url=base_url, suffixes=suffixes),
        years=years,
        fallback_artifact_prefix=fallback_artifact_prefix,
    )


def iter_yearly_archive_fetches(
    *,
    start_year: int,
    end_year: int,
    url_template: str,
    first_year: int,
    yearly_first_year: int | None = None,
    historical_url: str | None = None,
    historical_last_year: int | None = None,
    file_urls: list[str] | None = None,
    fallback_artifact_prefix: str = "archive",
) -> list[ArchiveFetchPlanEntry]:
    return plan_archive_fetches(
        iter_yearly_archive_urls(
            start_year=start_year,
            end_year=end_year,
            url_template=url_template,
            first_year=first_year,
            yearly_first_year=yearly_first_year,
            historical_url=historical_url,
            historical_last_year=historical_last_year,
            file_urls=file_urls,
        ),
        years=None,
        fallback_artifact_prefix=fallback_artifact_prefix,
    )


def read_zip_members(
    payload: bytes,
    *,
    suffixes: Iterable[str],
) -> list[tuple[str, bytes]]:
    allowed = tuple(str(suffix).lower() for suffix in suffixes)
    members: list[tuple[str, bytes]] = []
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        for name in zf.namelist():
            if str(name).lower().endswith(allowed):
                members.append((name, zf.read(name)))
    return members


def read_first_zip_member(
    payload: bytes,
    *,
    suffixes: Iterable[str],
) -> tuple[str, bytes] | None:
    members = read_zip_members(payload, suffixes=suffixes)
    return members[0] if members else None
