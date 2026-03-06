import json
import joblib
import numpy as np
import torch
from pathlib import Path


class PredictorConfig:

    model_type = None
    weight_mode = None
    feature_mode = None
    w_et = None
    w_gbr = None
    feature_cols = None

    def validate(self):
        if self.model_type is None:
            raise ValueError("model_type is not set")
        if self.feature_mode is None:
            raise ValueError("feature_mode is not set")

        if self.feature_mode not in ("auto", "custom"):
            raise ValueError("Invalid feature_mode")

        if self.feature_mode == "custom":
            if not isinstance(self.feature_cols, list) or len(self.feature_cols) == 0:
                raise ValueError("Invalid feature_cols")

        if self.model_type == "gin_ensemble":
            raise NotImplementedError

        if self.model_type not in ("weighted_ensemble", "extra_trees", "gradient_boost"):
            raise ValueError("Invalid model_type")

        if self.model_type == "weighted_ensemble":
            if self.weight_mode not in ("auto", "fixed", "equal"):
                raise ValueError("Invalid weight_mode")

            if self.weight_mode == "fixed":
                if self.w_et is None or self.w_gbr is None:
                    raise ValueError("Invalid w_et or w_gbr")
                if not isinstance(self.w_et, (int, float)) or not isinstance(self.w_gbr, (int, float)):
                    raise ValueError("Invalid w_et or w_gbr")
                if abs(self.w_et + self.w_gbr - 1.0) > 1e-6:
                    raise ValueError("Invalid w_et or w_gbr")


class LoadPredictor:

    def __init__(self, model_dir, config: PredictorConfig, extractor=None):
        self.model_dir = Path(model_dir)
        self.config = config
        self.extractor = extractor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config.validate()
        self._load_models()

    def _load_models(self):
        cfg_path = self.model_dir / "config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")

        cfg = json.loads(cfg_path.read_text())

        if self.config.feature_mode == "auto":
            self.feature_cols = cfg["feature_cols"]
        else:
            self.feature_cols = self.config.feature_cols

        self.n_features = len(self.feature_cols)

        if self.config.model_type == "weighted_ensemble":
            self.et  = joblib.load(self.model_dir / "et_model.joblib")
            self.gbr = joblib.load(self.model_dir / "gbr_model.joblib")

            if self.config.weight_mode == "auto":
                self.w_et  = cfg["w_et"]
                self.w_gbr = cfg["w_gbr"]
            elif self.config.weight_mode == "fixed":
                self.w_et  = self.config.w_et
                self.w_gbr = self.config.w_gbr
            elif self.config.weight_mode == "equal":
                self.w_et  = 0.5
                self.w_gbr = 0.5

        elif self.config.model_type == "extra_trees":
            self.et    = joblib.load(self.model_dir / "et_model.joblib")
            self.gbr   = None
            self.w_et  = 1.0
            self.w_gbr = 0.0

        elif self.config.model_type == "gradient_boost":
            self.et    = None
            self.gbr   = joblib.load(self.model_dir / "gbr_model.joblib")
            self.w_et  = 0.0
            self.w_gbr = 1.0

    def _run_inference(self, X):
        if self.config.model_type == "weighted_ensemble":
            return float(self.w_et * self.et.predict(X)[0] + self.w_gbr * self.gbr.predict(X)[0])
        elif self.config.model_type == "extra_trees":
            return float(self.et.predict(X)[0])
        elif self.config.model_type == "gradient_boost":
            return float(self.gbr.predict(X)[0])

    def predict(self, json_path, extractor=None):
        _extractor = extractor or self.extractor
        if _extractor is None:
            raise ValueError("Extractor required as parameter or during initialization")

        features = _extractor.extract_from_path(json_path)
        X = np.array([features[col] for col in self.feature_cols]).reshape(1, -1)
        return self._run_inference(X)

    def predict_from_graph(self, G, extractor=None):
        _extractor = extractor or self.extractor
        if _extractor is None:
            raise ValueError("Extractor required as parameter or during initialization")

        features = _extractor.extract_from_graph(G)
        X = np.array([features[col] for col in self.feature_cols]).reshape(1, -1)
        return self._run_inference(X)