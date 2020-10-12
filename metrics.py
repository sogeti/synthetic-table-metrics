import json

import pandas as pd
from data_container import Data
from detection import Detection
from duplicates import DuplicateCounter


class Metrics:
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
        results["duplicates"] = dup_counter.run(data)

        return results


if __name__ == "__main__":
    # Load both datasets
    real = pd.read_csv("iris_original.csv")
    synthetic = pd.read_csv("iris_fake_01.csv", index_col=False)
    # synthetic = pd.read_csv("iris_bad_data.csv", index_col=False)
    # synthetic = synthetic.drop(synthetic.columns[0], axis=1)
    data = Data(real, synthetic)

    metrics = Metrics()
    results = metrics.run(data)
    print(json.dumps(results, indent=2))
