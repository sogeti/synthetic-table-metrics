import unittest
from typing import cast
from pathlib import Path

import pandas as pd

from synthetic_table_metrics.data_container import Data
from synthetic_table_metrics.synthetic_table_metrics import SyntheticTableMetrics


class TestSyntheticTableMetrics(unittest.TestCase):
    @staticmethod
    def test_on_census_dataset():
        # The census dataset is larger then the dataset used to develop this package.
        # The size of the dataset caused an error, this is why we explicitely test on
        # this dataset.
        real = pd.read_csv(Path("data", "census.csv"))
        synthetic = cast(
            pd.DataFrame,
            pd.read_csv(Path("data", "census.csv")),
        )
        synthetic = synthetic.head(200)

        data = Data(real, synthetic)

        metrics = SyntheticTableMetrics()
        metrics = metrics.run(data)

        # Check if result contains the correct keys
        ks = list(metrics.keys())
        assert "detectability" in ks
        assert "duplicates" in ks
