import pandas as pd
from alphaforge.data.panel import PanelFrame


def test_panel_from_long_and_slice(dummy_ctx):
    ctx, dates, entities = dummy_ctx

    df = ctx.sources["dummy"].fetch(
        __import__("alphaforge").data.query.Query(
            table="market.ohlcv", columns=["close"]
        )
    )
    panel = PanelFrame.from_long(df)
    assert isinstance(panel.df.index, pd.MultiIndex)
    assert panel.df.index.names == ["ts_utc", "entity_id"]

    # slice by entities
    sliced = panel.slice(entities=[entities[0]])
    assert sliced.df.index.get_level_values("entity_id").unique().tolist() == [
        entities[0]
    ]
