import unittest
from pathlib import Path

import pandas as pd

from synthetic_table_metrics.data_container import Data
from synthetic_table_metrics.synthetic_table_metrics import SyntheticTableMetrics


class TestSyntheticTableMetrics(unittest.TestCase):
    def test_on_census_dataset(self):
        real = pd.read_csv(Path("synthetic_table_metrics", "test_data", "census.csv"))
        synthetic = pd.read_csv(
            Path("synthetic_table_metrics", "test_data", "census.csv")
        )
        synthetic = synthetic.head(200)

        data = Data(real, synthetic)

        metrics = SyntheticTableMetrics()
        metrics = metrics.run(data)

        # Check if result contains the correct keys
        print(metrics)
        ks = list(metrics.keys())
        assert "detectability" in ks
        assert "duplicates" in ks
