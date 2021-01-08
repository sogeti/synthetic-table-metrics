import itertools
import pandas as pd
import numpy as np

from rdt.hyper_transformer import HyperTransformer


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
        # DuplicateCounter config
        self.max_count = max_count

    @staticmethod
    def simple_count(data):
        """
        Search all rows of the real and synthetic dataset for duplicates.

        Return the number of duplicates. Return the count as a fraction of
        the length of the real dataset.

        Parameters
        ----------
        data: Data
            data_container

        Returns
        -------
        duplicate_fraction: float
            fraction of duplicates between real and synthetic data
        """
        c = 0
        for r1, r2 in itertools.product(data.real, data.synthetic):
            if np.allclose(r1, r2, atol=1e-4):
                c += 1
        return c / data.real.shape[0]

    @staticmethod
    def simple_count_internal(data):
        """
        Count the duplicate rows within data.synthetic.

        Return the number of duplicate rows as a fraction of the length
        of data.synthetic.

        Parameters
        ----------
        data: Data
            data_container

        Returns
        -------
        duplicate_fraction: float
            fraction of duplicate values in data.synthetic.
        """
        u, c = np.unique(data.synthetic, axis=0, return_counts=True)
        # TODO: are we counting double now, or not?
        return u[c > 1].sum() / data.synthetic.shape[0]

    @staticmethod
    def count_fraction_internal(data, max_count):
        """
        Count the duplicates within a fraction of size max_count within
        the data.synthetic array, return the counts as a fraction of
        the max_count value.
        """
        tmp = (data.synthetic[np.random.randint(data.synthetic.shape[0], size=2), :],)
        u, c = np.unique(tmp, axis=0, return_counts=True)
        # TODO: are we counting double now, or not?
        return u[c > 1].sum() / max_count

    @staticmethod
    def count_fraction(data, max_count):
        """
        Select a 1000 rows from each dataset and check if there
        are duplicates between thows
        """
        c = 0
        for r1, r2 in itertools.product(
            data.real[np.random.randint(data.real.shape[0], size=2), :],
            data.synthetic[np.random.randint(data.synthetic.shape[0], size=2), :],
        ):
            if np.allclose(r1, r2, atol=1e-4):
                c += 1
        return c / max_count

    def run(self, data):
        """
        Depending on size of the real data, compare all rows in search
        for duplicates, or check a fraction of the dataset.
        """
        real_vs_synth = self.simple_count(data)
        synth_duplicates = self.simple_count_internal(data)
        return real_vs_synth, synth_duplicates

        # if data.real.shape[0] < self.max_count:
        # real_vs_synth = self.simple_count(data)
        # synth_duplicates = self.simple_count_internal(data)
        # return real_vs_synth, synth_duplicates
        # else:
        # real_vs_synth = self.count_fraction(data, 1000)
        # synth_duplicates = self.count_fraction_internal(data, 1000)
        # return real_vs_synth, synth_duplicates
