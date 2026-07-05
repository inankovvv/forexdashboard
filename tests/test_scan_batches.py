import unittest

from scanner import iter_scan_batches


class ScanBatchTests(unittest.TestCase):
    def test_iter_scan_batches_splits_into_single_timeframe_chunks(self):
        batches = list(iter_scan_batches(["15m", "30m", "1h"], batch_size=1))
        self.assertEqual(batches, [["15m"], ["30m"], ["1h"]])


if __name__ == "__main__":
    unittest.main()
