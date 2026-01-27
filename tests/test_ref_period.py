import pandas as pd

from alphaforge.time.ref_period import RefPeriod, RefFreq


def test_refperiod_parsing_and_canonicalization():
    q4 = RefPeriod.parse("2024Q4")
    assert q4.freq == RefFreq.Q
    assert q4.year == 2024
    assert q4.period == 4
    assert q4.to_key() == "2024Q4"

    q1 = RefPeriod.parse("2025q1")
    assert q1.to_key() == "2025Q1"

    m1 = RefPeriod.parse("2025-01")
    assert m1.freq == RefFreq.M
    assert m1.to_key() == "2025-01"

    m2 = RefPeriod.parse("2025/01")
    assert m2.to_key() == "2025-01"

    y = RefPeriod.parse("2025")
    assert y.freq == RefFreq.A
    assert y.to_key() == "2025"

    m3 = RefPeriod.parse("2025-01-31")
    assert m3.freq == RefFreq.M
    assert m3.to_key() == "2025-01"


def test_refperiod_end_obs_date():
    assert RefPeriod.parse("2024Q4").end_obs_date() == pd.Timestamp(
        "2024-12-31", tz="UTC"
    )
    assert RefPeriod.parse("2024Q1").end_obs_date() == pd.Timestamp(
        "2024-03-31", tz="UTC"
    )
    assert RefPeriod.parse("2025-02").end_obs_date() == pd.Timestamp(
        "2025-02-28", tz="UTC"
    )
    assert RefPeriod.parse("2024-02").end_obs_date() == pd.Timestamp(
        "2024-02-29", tz="UTC"
    )
    assert RefPeriod.parse("2025").end_obs_date() == pd.Timestamp(
        "2025-12-31", tz="UTC"
    )


def test_refperiod_round_trip_from_obs_end():
    ref = RefPeriod.parse("2024Q4")
    round_trip = RefPeriod.from_obs_date_end(ref.end_obs_date(), freq=RefFreq.Q)
    assert round_trip == ref
