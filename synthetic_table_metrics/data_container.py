"""data_container.py

Simple container class for real and synthetic data.
"""
from rdt.hyper_transformer import HyperTransformer


class Data:
    """
    Container for real and synthetic data as well as their
    transformed versions.
    """

    def __init__(self, real, synthetic):
        self.real_raw = real
        self.synthetic_raw = synthetic
        transformer = HyperTransformer()
        self.real = transformer.fit_transform(real).values
        self.synthetic = transformer.transform(synthetic).values
