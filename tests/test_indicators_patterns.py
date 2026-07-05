import unittest

import pandas as pd

from indicators import detect_candlestick_patterns


class CandlestickPatternTests(unittest.TestCase):
    def test_detects_doji_and_engulfing(self):
        df = pd.DataFrame(
            {
                "Open": [1.00, 1.01, 0.98],
                "High": [1.02, 1.02, 1.05],
                "Low": [0.98, 0.98, 0.97],
                "Close": [1.00, 0.99, 1.04],
            },
            index=pd.date_range("2024-01-01", periods=3, freq="h"),
        )

        result = detect_candlestick_patterns(df)

        self.assertTrue(result.loc[result.index[0], "doji"])
        self.assertTrue(result.loc[result.index[2], "bullish_engulfing"])
        self.assertFalse(result.loc[result.index[2], "bearish_engulfing"])


if __name__ == "__main__":
    unittest.main()
