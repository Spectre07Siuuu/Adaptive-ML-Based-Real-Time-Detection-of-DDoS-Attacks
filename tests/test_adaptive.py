import numpy as np
import pandas as pd

from ddos.adaptive.novelty import KnnNovelty
from ddos.adaptive.stream import in_streak

FEATURES = ["a", "b"]


def test_novelty_flags_far_windows_beyond_the_calibration_range():
    rng = np.random.default_rng(0)
    benign = pd.DataFrame({"a": rng.normal(10, 2, 500).clip(0), "b": rng.normal(100, 10, 500)})
    nov = KnnNovelty(FEATURES, quantile=0.99).fit(benign)
    test = pd.DataFrame({"a": [10, 11, 5_000, 50_000], "b": [100, 95, 100, 100]})
    flags = nov.flag(test)
    assert list(flags) == [False, False, True, True]
    s = nov.score(test)
    assert s[3] > s[2] > nov.threshold          # keeps growing further out
    assert nov.flag(benign).mean() < 0.02       # ~1% expected at the 99% quantile


def test_streak_needs_two_adjacent_flagged_windows_of_the_same_peer():
    df = pd.DataFrame({
        "peer_ip":      ["x", "x", "x", "x", "y", "y", "x"],
        "window_start": [1.0, 2.0, 3.0, 7.0, 2.0, 3.0, 8.0],
        "f":            [True, True, False, True, True, False, True],
    })
    assert list(in_streak(df, "f")) == [False, True, False, False, False, False, True]
