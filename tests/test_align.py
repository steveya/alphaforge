import pandas as pd
from alphaforge.data.query import Query
from alphaforge.time.grids import SessionGrid
from alphaforge.time.align import AlignSpec, align_panel, AvailabilityState


def test_align_structural_missingness_monthly(dummy_ctx):
    ctx, dates, entities = dummy_ctx

    # fetch macro table (monthly sparse)
    q = Query(
        table="macro.series",
        columns=["value"],
        start=pd.Timestamp("2020-01-01"),
        end=pd.Timestamp("2020-03-31"),
    )
    panel = ctx.fetch_panel("dummy", q)
    schema = ctx.sources["dummy"].schemas()["macro.series"]

    grid = SessionGrid(
        name="sessions:XNYS", index=pd.DatetimeIndex(dates), calendar="XNYS"
    )
    aligned = align_panel(
        panel, schema, grid, AlignSpec(target_grid="sessions:XNYS", method="ffill")
    )

    av = aligned.availability.df
    # between releases should be NO_UPDATE_EXPECTED (structural)
    # pick a mid-month date that isn't month-end
    mid = pd.Timestamp("2020-02-14", tz="UTC")
    cell = av.loc[(mid, "CPI"), "value"]
    assert cell in (
        AvailabilityState.NO_UPDATE_EXPECTED.value,
        AvailabilityState.AVAILABLE.value,
    )


def test_align_daily_missing_is_abnormal(dummy_ctx):
    ctx, dates, entities = dummy_ctx
    q = Query(
        table="market.ohlcv",
        columns=["close"],
        start=pd.Timestamp("2020-01-01"),
        end=pd.Timestamp("2020-01-31"),
        entities=["AAA"],
    )
    panel = ctx.fetch_panel("dummy", q)
    schema = ctx.sources["dummy"].schemas()["market.ohlcv"]

    # Force a missing close on a specific date (abnormal for daily)
    df = panel.df.copy()
    bad_date = df.index.get_level_values("ts_utc").unique()[5]
    df.loc[(bad_date, "AAA"), "close"] = float("nan")
    # availability grid uses session-labels (midnight UTC), normalize for lookup
    bad_label = bad_date.normalize()
    from alphaforge.data.panel import PanelFrame

    panel2 = PanelFrame(df)

    grid = SessionGrid(
        name="sessions:XNYS",
        index=pd.DatetimeIndex(
            dates[(dates >= "2020-01-01") & (dates <= "2020-01-31")]
        ),
        calendar="XNYS",
    )
    aligned = align_panel(
        panel2, schema, grid, AlignSpec(target_grid="sessions:XNYS", method="none")
    )
    av = aligned.availability.df
    assert (
        av.loc[(bad_label, "AAA"), "close"] == AvailabilityState.MISSING_UNKNOWN.value
    )
