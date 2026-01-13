from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class Universe:
    """Time-varying membership: index=date, columns=entity_id, values=bool."""

    membership: pd.DataFrame

    def entities_on(self, date: pd.Timestamp) -> list[str]:
        # coerce incoming date to naive for membership lookup when needed
        date = pd.Timestamp(date)
        # if tz-aware, compare using naive date to maintain backwards compatibility
        if date.tzinfo is not None:
            date = date.tz_convert(None)
        idx = self.membership.index
        if date not in idx:
            prior = idx[idx <= date]
            if len(prior) == 0:
                return []
            date = prior.max()
        row = self.membership.loc[date]
        return [c for c, v in row.items() if bool(v)]

    def restrict_panel(self, panel: "object") -> "object":
        df = panel.df
        dates = df.index.get_level_values("ts_utc")
        ents = df.index.get_level_values("entity_id")
        keep = []
        cache = {}
        for d, e in zip(dates, ents):
            if d not in cache:
                cache[d] = set(self.entities_on(d))
            keep.append(e in cache[d])
        return type(panel)(df.loc[keep])


@dataclass(frozen=True)
class EntityMetadata:
    """Entity metadata table: index=entity_id, columns like sector/country/etc."""

    df: pd.DataFrame
