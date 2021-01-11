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
    @staticmethod
    def count_internal_duplicates(d):
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

    @staticmethod
    def count_intersect(d1, d2):

        # Convert to contigious arrays, this method fails for normal
        # arrays when they are very large
        d1 = np.ascontiguousarray(d1)
        d2 = np.ascontiguousarray(d2)

        _, ncols = d1.shape  # Get info on array 1
        # Create a dtype dict
        dtype = {
            "names": [f"{i:f}" for i in range(ncols)],
            "formats": ncols * [d1.dtype],
        }

        # take the intersect of new views of both arrays, this is a
        # trick to make intersect1d work with 2d arrays. Take the
        # length of the number of duplicates returned by interect1d
        duplicates = np.intersect1d(
            d1.view(dtype), d2.view(dtype), assume_unique=True
        ).shape[0]

        # Calculate the fraction of duplicates over the shortest array
        l1, l2 = d1.shape[0], d2.shape[0]
        divider = l1 if l1 < l2 else l2
        return duplicates / divider

    def run(self, data):
        """
        Concatenate real and synth to check for duplicates within
        and between both arrays. Then check the number of duplicates
        within the synthesized dataset.
        """
        real_vs_synth = self.count_intersect(data.real, data.synthetic)
        synth_duplicates = self.count_internal_duplicates(data.synthetic)
        return real_vs_synth, synth_duplicates
