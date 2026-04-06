from __future__ import annotations

from collections.abc import Callable

from alphaforge.data.source import DataSource

from .anp_fuel_prices import ANPFuelPricesDataSource
from .b3_historical_quotes import B3HistoricalQuotesDataSource
from .bcb_sgs import BCBSGSDataSource
from .bea import BEADataSource
from .bls import BLSDataSource
from .cftc_cot import CFTCCoTSource, CFTCDisaggregatedCoTSource
from .cftc_swaps_weekly import CFTCWeeklySwapsSource
from .cme_productslate_reference import CMEProductSlateSource
from .destatis_genesis import DestatisGenesisDataSource
from .dtcc_ppd import DTCCPPDSource
from .ec_weekly_oil_bulletin import ECWeeklyOilBulletinDataSource
from .ecb_sdmx import ECBSDMXDataSource
from .eia import EIADataSource
from .eurex_refdata_contracts import EurexRefdataContractsSource
from .eurex_stats_daily import EurexStatsDailySource
from .eurostat import EurostatDataSource
from .ezoic_adrevenue_daily import EzoicAdRevenueDailySource
from .frb_term_structure import FRBTermStructureBenchmarkSource
from .ibge_sidra import IBGESidraDataSource
from .lch_cdsclear_daily import LCHCDSClearDailySource
from .mof_jgb import MOFJGBYieldCurveSource
from .philadelphia_spf import PhiladelphiaSPFMeanLevelSource

DEFAULT_EUREX_REFDATA_API_URL = "https://www.eurex.com/api/refdata/contracts"


def _default_eurex_refdata_source() -> DataSource:
    return EurexRefdataContractsSource(api_url=DEFAULT_EUREX_REFDATA_API_URL)


DEFAULT_PUBLIC_WEB_SOURCE_FACTORIES: tuple[Callable[[], DataSource], ...] = (
    BLSDataSource,
    BEADataSource,
    EIADataSource,
    EurostatDataSource,
    ECBSDMXDataSource,
    DestatisGenesisDataSource,
    ECWeeklyOilBulletinDataSource,
    IBGESidraDataSource,
    BCBSGSDataSource,
    ANPFuelPricesDataSource,
    B3HistoricalQuotesDataSource,
    DTCCPPDSource,
    CMEProductSlateSource,
    CFTCWeeklySwapsSource,
    CFTCCoTSource,
    CFTCDisaggregatedCoTSource,
    EurexStatsDailySource,
    LCHCDSClearDailySource,
    EzoicAdRevenueDailySource,
    PhiladelphiaSPFMeanLevelSource,
    FRBTermStructureBenchmarkSource,
    _default_eurex_refdata_source,
    MOFJGBYieldCurveSource,
)


def default_public_web_sources() -> dict[str, DataSource]:
    """Construct the default public-web source registry."""

    return {
        source.name: source
        for source in (factory() for factory in DEFAULT_PUBLIC_WEB_SOURCE_FACTORIES)
    }
