from __future__ import annotations

import alphaforge
from alphaforge.data.public_web import MOFJGBYieldCurveSource


def test_root_exports_default_public_web_registry() -> None:
    sources = alphaforge.default_public_web_sources()

    assert "mof_jgb_yields" in sources
    assert "philadelphia_spf" in sources
    assert "cftc_cot_disagg" in sources
    assert isinstance(sources["mof_jgb_yields"], MOFJGBYieldCurveSource)
    assert "mof.jgb.yields" in sources["mof_jgb_yields"].schemas()


def test_root_exports_mof_source_constructor() -> None:
    assert alphaforge.MOFJGBYieldCurveSource is MOFJGBYieldCurveSource
