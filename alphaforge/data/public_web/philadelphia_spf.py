"""Philadelphia Fed SPF mean-level historical forecast loader."""

from __future__ import annotations

import io
import re
from pathlib import Path

import pandas as pd

from alphaforge.data.query import Query
from alphaforge.data.schema import TableSchema
from alphaforge.data.source import DataSource

from .http import CachedHttpClient
from .utils import (
    apply_query_filters,
    ensure_date_utc,
    make_entity_id,
    project_columns,
    snake_case,
)


class PhiladelphiaSPFMeanLevelSource(DataSource):
    """Historical mean SPF forecasts from the Philadelphia Fed."""

    name: str = "philadelphia_spf"
    TABLE = "philadelphia.spf.mean_level"
    WORKBOOK_URL = (
        "https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/"
        "survey-of-professional-forecasters/historical-data/meanlevel.xlsx"
    )
    RELEASE_URL = (
        "https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/"
        "survey-of-professional-forecasters/spf-release-dates.txt"
    )

    _PERIOD_PATTERN = re.compile(r"(?P<year>\d{4})\s*[:/-]?\s*Q(?P<quarter>[1-4])", re.I)
    _DATE_PATTERN = re.compile(
        r"("
        r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s+\d{4}"
        r"|"
        r"\d{4}-\d{2}-\d{2}"
        r"|"
        r"\d{1,2}/\d{1,2}/\d{4}"
        r")",
        re.I,
    )

    def __init__(
        self,
        *,
        http_client: CachedHttpClient | None = None,
        cache_dir: str | Path | None = None,
        workbook_url: str | None = None,
        release_url: str | None = None,
    ) -> None:
        self._http = http_client or CachedHttpClient(cache_dir=cache_dir)
        self._workbook_url = workbook_url or self.WORKBOOK_URL
        self._release_url = release_url or self.RELEASE_URL

    def schemas(self) -> dict[str, TableSchema]:
        return {
            self.TABLE: TableSchema(
                name=self.TABLE,
                required_columns=["value"],
                canonical_columns=[
                    "value",
                    "sheet_name",
                    "series_name",
                    "survey_period",
                    "release_date",
                ],
                entity_column="entity_id",
                time_column="date",
                native_freq="Q",
                time_semantics="point",
                release_time_column="release_date",
            )
        }

    def _load_workbook(self) -> dict[str, pd.DataFrame]:
        payload = self._http.get_bytes(
            url=self._workbook_url,
            source="philadelphia_spf",
            artifact_name="meanlevel.xlsx",
        )
        return pd.read_excel(io.BytesIO(payload), sheet_name=None)

    def _load_release_calendar(self) -> pd.DataFrame:
        payload = self._http.get_bytes(
            url=self._release_url,
            source="philadelphia_spf",
            artifact_name="spf-release-dates.txt",
        )
        text = payload.decode("utf-8", errors="ignore")
        records: list[dict[str, object]] = []
        for line in text.splitlines():
            period_match = self._PERIOD_PATTERN.search(line)
            if period_match is None:
                continue
            date_matches = self._DATE_PATTERN.findall(line)
            if not date_matches:
                continue
            release_date = pd.to_datetime(date_matches[-1], errors="coerce")
            if pd.isna(release_date):
                continue
            survey_period = (
                f"{int(period_match.group('year'))}Q{int(period_match.group('quarter'))}"
            )
            records.append(
                {
                    "survey_period": survey_period,
                    "release_date": pd.Timestamp(release_date),
                }
            )
        if not records:
            return pd.DataFrame(columns=["survey_period", "release_date"])
        release_df = pd.DataFrame(records).drop_duplicates("survey_period")
        release_df["release_date"] = ensure_date_utc(release_df["release_date"])
        return release_df

    @staticmethod
    def _fallback_date(periods: pd.Series) -> pd.Series:
        labels = periods.astype(str).str.upper().str.replace(" ", "", regex=False)
        out = pd.PeriodIndex(labels, freq="Q").to_timestamp(how="end")
        return ensure_date_utc(pd.Series(out, index=periods.index))

    @staticmethod
    def _clean_sheet(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out.columns = [str(col).strip() for col in out.columns]
        out = out.dropna(how="all")
        out = out.dropna(axis=1, how="all")
        unnamed = [col for col in out.columns if col.lower().startswith("unnamed:")]
        if unnamed:
            out = out.drop(columns=unnamed)
        return out.reset_index(drop=True)

    def _flatten_sheet(self, sheet_name: str, frame: pd.DataFrame) -> pd.DataFrame:
        cleaned = self._clean_sheet(frame)
        if cleaned.empty:
            return pd.DataFrame()

        upper_columns = {col.upper(): col for col in cleaned.columns}
        year_col = upper_columns.get("YEAR")
        quarter_col = upper_columns.get("QUARTER")
        date_col = (
            upper_columns.get("DATE")
            or upper_columns.get("RELEASE_DATE")
            or upper_columns.get("SURVEY_DATE")
        )

        id_columns: list[str] = []
        survey_period = pd.Series(index=cleaned.index, dtype="object")
        if year_col is not None and quarter_col is not None:
            id_columns.extend([year_col, quarter_col])
            year_values = cleaned[year_col].astype("Int64")
            quarter_values = cleaned[quarter_col].astype("Int64")
            survey_period = year_values.astype(str) + "Q" + quarter_values.astype(str)
        elif date_col is not None:
            id_columns.append(date_col)
            dates = pd.to_datetime(cleaned[date_col], errors="coerce")
            survey_period = dates.dt.to_period("Q").astype(str)
        elif year_col is not None:
            id_columns.append(year_col)
            survey_period = cleaned[year_col].astype("Int64").astype(str) + "Q4"
        else:
            return pd.DataFrame()

        value_columns = [col for col in cleaned.columns if col not in id_columns]
        if not value_columns:
            return pd.DataFrame()

        melted = cleaned.assign(survey_period=survey_period).melt(
            id_vars=["survey_period"],
            value_vars=value_columns,
            var_name="series_name",
            value_name="value",
        )
        melted["value"] = pd.to_numeric(melted["value"], errors="coerce")
        melted = melted.dropna(subset=["value", "survey_period"]).copy()
        if melted.empty:
            return pd.DataFrame()

        melted["sheet_name"] = sheet_name
        melted["entity_id"] = [
            make_entity_id("spf", snake_case(sheet_name), snake_case(series_name))
            for series_name in melted["series_name"]
        ]
        return melted.reset_index(drop=True)

    def _to_long(self) -> pd.DataFrame:
        workbook = self._load_workbook()
        release_calendar = self._load_release_calendar()
        frames = [
            self._flatten_sheet(sheet_name, frame)
            for sheet_name, frame in workbook.items()
        ]
        frames = [frame for frame in frames if not frame.empty]
        if not frames:
            return pd.DataFrame(
                columns=[
                    "date",
                    "release_date",
                    "survey_period",
                    "sheet_name",
                    "series_name",
                    "value",
                    "entity_id",
                    "asof_utc",
                ]
            )

        long = pd.concat(frames, ignore_index=True)
        if not release_calendar.empty:
            long = long.merge(release_calendar, on="survey_period", how="left")
        else:
            long["release_date"] = pd.NaT

        missing_release = long["release_date"].isna()
        if missing_release.any():
            long.loc[missing_release, "release_date"] = self._fallback_date(
                long.loc[missing_release, "survey_period"]
            ).values

        long["release_date"] = ensure_date_utc(long["release_date"])
        long["date"] = long["release_date"]
        long["asof_utc"] = long["release_date"]
        return long.sort_values(["date", "sheet_name", "series_name"]).reset_index(drop=True)

    def fetch(self, q: Query) -> pd.DataFrame:
        if q.table != self.TABLE:
            raise ValueError(f"Unknown table: {q.table}")

        long = self._to_long()
        long = apply_query_filters(long, q=q, time_col="date", entity_col="entity_id")
        return project_columns(
            long,
            required_columns=["value"],
            requested_columns=q.columns,
            time_col="date",
            entity_col="entity_id",
        )
