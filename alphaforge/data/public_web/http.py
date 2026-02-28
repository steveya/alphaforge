from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Mapping, Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class CachedHttpClient:
    def __init__(
        self,
        *,
        cache_dir: str | Path | None = None,
        user_agent: str = "alphaforge/public-web",
        timeout_seconds: int = 30,
        retries: int = 3,
        backoff_seconds: float = 1.0,
        cache_partition: str = "monthly",
    ) -> None:
        self.cache_dir = Path(cache_dir or Path.home() / ".cache/alphaforge/public_web")
        self.user_agent = user_agent
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self.backoff_seconds = backoff_seconds
        self.cache_partition = cache_partition

    def _cache_path(
        self,
        *,
        source: str,
        artifact_name: str,
        url: str,
        params: Optional[Mapping[str, str]] = None,
    ) -> Path:
        query = urlencode(sorted((params or {}).items()))
        cache_key = hashlib.sha256(f"{url}?{query}".encode()).hexdigest()[:12]
        if self.cache_partition == "monthly":
            today = time.gmtime()
            source_dir = (
                self.cache_dir / source / str(today.tm_year) / f"{today.tm_mon:02d}"
            )
        else:
            source_dir = self.cache_dir / source
        source_dir.mkdir(parents=True, exist_ok=True)
        safe_name = artifact_name.replace(os.sep, "_")
        return source_dir / f"{cache_key}_{safe_name}"

    def get_bytes(
        self,
        *,
        url: str,
        source: str,
        artifact_name: str,
        params: Optional[Mapping[str, str]] = None,
        force: bool = False,
    ) -> bytes:
        query = urlencode(params or {})
        full_url = f"{url}?{query}" if query else url
        path = self._cache_path(
            source=source,
            artifact_name=artifact_name,
            url=url,
            params=params,
        )

        if path.exists() and not force:
            return path.read_bytes()

        payload = self._download_with_retry(full_url)
        path.write_bytes(payload)
        metadata = {
            "url": url,
            "full_url": full_url,
            "cached_at_epoch": time.time(),
        }
        path.with_suffix(path.suffix + ".json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        return payload

    def _download_with_retry(self, url: str) -> bytes:
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                req = Request(url, headers={"User-Agent": self.user_agent})
                with urlopen(req, timeout=self.timeout_seconds) as resp:
                    return resp.read()
            except Exception as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(self.backoff_seconds * attempt)
        if last_error is None:
            raise RuntimeError("download failed with unknown error")
        raise RuntimeError(f"download failed for {url}: {last_error}") from last_error
