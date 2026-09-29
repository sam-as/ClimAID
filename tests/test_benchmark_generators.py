"""The benchmark generators are reproducible and produce what they document."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import generators as G  # noqa: E402


def test_generators_are_deterministic():
    a, _ = G.seasonal(); b, _ = G.seasonal()
    assert a.equals(b)


def test_seasonal_vs_nonseasonal_strength():
    def month_share(d):
        m = d.groupby(d.time.dt.month).cases.transform("mean")
        return 1 - ((d.cases - m) ** 2).sum() / ((d.cases - d.cases.mean()) ** 2).sum()
    assert month_share(G.seasonal()[0]) > 0.5 > 0.15 > month_share(G.nonseasonal()[0])


def test_realistic_hard_has_documented_problems():
    d, c, truth = G.realistic_hard()
    assert len(d) < 216                                             # missing months
    assert not d.time.isin(["2014-06-01", "2014-07-01", "2014-08-01"]).any()
    assert (truth.set_index("time").reporting_factor.loc["2016-01-01":"2019-12-01"] == 1.6).all()
    assert truth.set_index("time").reporting_factor.loc["2020-06-01"] == 0.4 * 1.6
    assert d.cases.max() > 5 * d.cases.median()                     # outbreak years / entry errors


def test_multi_district_and_cmip_shapes():
    target, others, proj, truth = G.multi_district()
    assert len(others) == 5 and set(proj.model) == {"G1", "G2"} and 0.2 < truth < 0.35
    assert set(G.cmip6().ssp) == {"ssp245", "ssp585"}
