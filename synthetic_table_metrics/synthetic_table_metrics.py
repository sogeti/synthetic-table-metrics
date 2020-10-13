import json

import pandas as pd
from .detection import Detection
from .duplicates import DuplicateCounter

# TODO: pass config to classes


class SyntheticTableMetrics:
    def __init__(self):
        # config
        pass

    def run(self, data):
        results = {}
        # Calculate the detection
        detection = Detection()
        results["detectability"] = detection.run(data)

        # Count duplicates
        dup_counter = DuplicateCounter()
        real_vs_synth, synth_duplicates = dup_counter.run(data)
        results["duplicates"] = {
            "real_vs_synth": real_vs_synth,
            "synth_duplicates": synth_duplicates,
        }

        return results
