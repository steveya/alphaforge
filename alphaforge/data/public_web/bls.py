from __future__ import annotations

import json
import os
from urllib.request import Request, urlopen

import pandas as pd

from alphaforge.data.query import Query

from .base import PublicWebSourceBase
from .schema_helpers import single_value_schema
from .utils import make_entity_id


class BLSDataSource(PublicWebSourceBase):
    name = "bls"
    TABLE = "bls_series"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_url: str = "https://api.bls.gov/publicAPI/v2/timeseries/data/",
        chunk_size: int = 25,
        response_provider=None,
    ) -> None:
        super().__init__()
        self._api_key = api_key or os.getenv("BLS_API_KEY")
        self._api_url = api_url
        self._chunk_size = chunk_size
        self._response_provider = response_provider

    def schemas(self):
        return {
            self.TABLE: single_value_schema(
                self.TABLE,
                native_freq="M",
                time_semantics="point",
            )
        }

    def _call(self, series_ids: list[str], start_year: int, end_year: int) -> dict:
        if self._response_provider is not None:
            return self._response_provider(series_ids, start_year, end_year)

        body = {
            "seriesid": series_ids,
            "startyear": str(start_year),
            "endyear": str(end_year),
        }
        if self._api_key:
            body["registrationkey"] = self._api_key

        data = json.dumps(body).encode("utf-8")
        req = Request(
            self._api_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))

    @staticmethod
    def _month_end(year: int, period: str) -> pd.Timestamp | None:
        if not period.startswith("M"):
            return None
        month = int(period[1:])
        return pd.Timestamp(
            year=year, month=month, day=1, tz="UTC"
        ) + pd.offsets.MonthEnd(0)

    def fetch(self, q: Query) -> pd.DataFrame:
        self._require_table(q)
        schema = self._schema()
        entities = self._require_entities(
            q,
            error_message="BLSDataSource requires q.entities with BLS series ids",
        )
        asof_utc = self._asof_utc(q)

        start_year = (
            q.start.year
            if q.start is not None
            else pd.Timestamp.now(tz="UTC").year - 10
        )
        end_year = q.end.year if q.end is not None else pd.Timestamp.now(tz="UTC").year

        rows: list[dict] = []
        for idx in range(0, len(entities), self._chunk_size):
            chunk = entities[idx : idx + self._chunk_size]
            payload = self._call(chunk, start_year, end_year)
            for series in payload.get("Results", {}).get("series", []):
                sid = str(series.get("seriesID", ""))
                for point in series.get("data", []):
                    date = self._month_end(
                        int(point.get("year")), str(point.get("period"))
                    )
                    if date is None:
                        continue
                    rows.append(
                        {
                            "date": date,
                            "entity_id": make_entity_id(sid),
                            "value": pd.to_numeric(point.get("value"), errors="coerce"),
                            "asof_utc": asof_utc,
                        }
                    )

        out = self._frame_from_records(rows, schema=schema)
        return self._finalize(out, q=q, schema=schema)
