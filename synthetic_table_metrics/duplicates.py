import itertools
import pandas as pd
import numpy as np
from numpy.lib import recfunctions as rfn

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

    @staticmethod
    def count_duplicates(d):
        """
        "Quick" method to count the internal duplicates of a 2D numpy
        array.

        First prepare the array, then count the number of unique rows
        in the array. Finally, return the number of duplicates over
        the number of rows in the array.
        """
        b = np.ascontiguousarray(d).view(
            np.dtype((np.void, d.dtype.itemsize * d.shape[1]))
        )
        n_unique = np.shape(np.unique(b).view(d.dtype).reshape(-1, d.shape[1]))[0]
        return (len(d) - n_unique) / len(d)

    def run(self, data):
        """
        Concatenate real and synth to check for duplicates within
        and between both arrays. Then check the number of duplicates
        within the synthesized dataset.
        """
        real_vs_synth = self.count_duplicates(
            np.concatenate((data.real, data.synthetic), axis=0)
        )
        synth_duplicates = self.count_duplicates(data.synthetic)
        return real_vs_synth, synth_duplicates
