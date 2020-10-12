import json
from pathlib import Path

import pandas as pd
from metrics import Metrics
from data_container import Data

# Metrics where synthetic is a clone of the orignal dataset:

# Load both datasets
real = pd.read_csv(Path("data", "iris_original.csv"))
synthetic = pd.read_csv(Path("data", "iris_original.csv"))
data = Data(real, synthetic)
metrics = Metrics()
results = metrics.run(data)
print("COPY:")
print(json.dumps(results, indent=2))


# Metrics on a bad synthetic dataset
synthetic = pd.read_csv(Path("data", "iris_bad_synthetic_data.csv"))
data = Data(real, synthetic)
metrics = Metrics()
results = metrics.run(data)
print("BAD SYNTHETIC DATA:")
print(json.dumps(results, indent=2))
