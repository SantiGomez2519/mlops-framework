import importlib as _importlib

_PKG = __name__

Preprocessing = _importlib.import_module(f"{_PKG}.1_preprocessing").Preprocessing
FeatureEngineering = _importlib.import_module(f"{_PKG}.2_feature_engineering").FeatureEngineering
Training = _importlib.import_module(f"{_PKG}.3_training").Training
