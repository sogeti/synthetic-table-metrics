import itertools
import pandas as pd
import numpy as np

from rdt import HyperTransformer

from data_container import Data


def cut_to_similar_length(real, synthetic):
    """
    If the synthetic dataset is longer then the real dataset,
    sample as mych rows from the synthetic as are in the real
    dataset.
    """
    if synthetic.shape[0] > real.shape[0]:
        print("sampling synthetic")
        synthetic = synthetic.sample(real.shape[0])
    return real, synthetic


class DuplicateCounter:
    def __init__(self, max_count=1000):
        # Duplicate counter config
        self.max_count = max_count

    @staticmethod
    def simple_count(data):
        """
        Compare all rows of the real and synthetic dataset. Return
        the number of duplicates.
        """
        c = 0
        for r1, r2 in itertools.product(data.real, data.synthetic):
            if np.allclose(r1, r2, atol=1e-4):
                c += 1
        return c

    @staticmethod
    def count_fraction(data, max_count):
        """
        Select a 1000 rows from each dataset and check if there
        are duplicates between thows
        """

        raise NotImplementedError

    def run(self, data):
        """
        Depending on size of the real data, compare all rows in search
        for duplicates, or check a fraction of the dataset.
        """
        if data.real.shape[0] < self.max_count:
            return self.simple_count(data) / data.real.shape[0]
        else:
            return self.count_fraction(data, self.max_count) / self.max_count


if __name__ == "__main__":
    # Create data object
    real = pd.read_csv("iris_original.csv")
    synthetic = pd.read_csv("iris_bad_data.csv", index_col=False)
    synthetic = synthetic.drop(synthetic.columns[0], axis=1)
    data = Data(real, synthetic)

    # Count duplicates
    dup_counter = DuplicateCounter()
    duplicate_ratio = dup_counter.run(data)
