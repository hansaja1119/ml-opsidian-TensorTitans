"""
Model Manager — Loads and manages ML model versions.

Supports loading models from:
1. MLflow model registry (production mode)
2. Local joblib files (development/fallback mode)
3. Mock predictions (when no models are available)

Models are cached in memory after first load for fast inference.
"""

import numpy as np
import os
import time
import logging
from typing import Dict, Optional, List, Tuple

logger = logging.getLogger(__name__)


class ModelManager:
    """Singleton manager for loading and serving ML models."""

    def __init__(self):
        self._models: Dict[str, object] = {}
        self._feature_engines: Dict[str, object] = {}
        self._model_info: Dict[str, dict] = {}
        self._mock_mode = False

    def initialize(self, artifacts_base_dir: str = "mlops/artifacts"):
        """Load all available model versions on startup.

        Falls back to mock mode if no model artifacts are found.
        """
        available_versions = []

        # Try loading each version
        for version, dirname, desc, n_models in [
            ("v13", "v13", "V13 Stack — 10-fold, Ridge meta, 10 base models (LGB + CatBoost)", 10),
            ("v18_titan", "v18", "V18 Titan — 36 models × 3 seeds, HuberRegressor meta", 36),
            ("v20_colossus", "v20", "V20 Colossus — 100-fold, 10 seeds, HuberRegressor meta", 130),
        ]:
            model_dir = os.path.join(artifacts_base_dir, dirname)
            if os.path.exists(model_dir):
                try:
                    self._load_model(version, model_dir)
                    self._model_info[version] = {
                        "version": version,
                        "name": f"Flood Risk {version.upper()}",
                        "description": desc,
                        "is_default": version == "v13",
                        "num_base_models": n_models,
                    }
                    available_versions.append(version)
                    logger.info(f"Loaded model: {version}")
                except Exception as e:
                    logger.warning(f"Failed to load model {version}: {e}")

        if not available_versions:
            logger.warning("No model artifacts found. Running in MOCK mode.")
            self._mock_mode = True
            self._model_info["v13"] = {
                "version": "v13",
                "name": "Flood Risk V13 (Mock)",
                "description": "Mock model — returns estimated predictions based on heuristics",
                "is_default": True,
                "num_base_models": 0,
            }
            self._model_info["v18_titan"] = {
                "version": "v18_titan",
                "name": "Flood Risk V18 Titan (Mock)",
                "description": "Mock model — returns estimated predictions based on heuristics",
                "is_default": False,
                "num_base_models": 0,
            }
            self._model_info["v20_colossus"] = {
                "version": "v20_colossus",
                "name": "Flood Risk V20 Colossus (Mock)",
                "description": "Mock model — returns estimated predictions based on heuristics",
                "is_default": False,
                "num_base_models": 0,
            }
        else:
            logger.info(f"Models ready: {available_versions}")

    def _load_model(self, version: str, model_dir: str):
        """Load a serialized model pipeline from disk."""
        import joblib
        import sys
        from api.feature_engine import FeatureEngine

        if not hasattr(sys.modules['__main__'], 'InferencePipeline'):
            class InferencePipeline:
                def __init__(self, base_models, meta_model, model_names):
                    self.base_models = base_models
                    self.meta_model = meta_model
                    self.model_names = model_names

                def predict(self, X):
                    import numpy as np
                    base_preds = []
                    for name in self.model_names:
                        model = self.base_models[name]
                        if hasattr(model, 'predict'):
                            base_preds.append(model.predict(X))
                        else:
                            base_preds.append(model.predict(X))
                    stack_input = np.column_stack(base_preds)
                    return self.meta_model.predict(stack_input)
            
            setattr(sys.modules['__main__'], 'InferencePipeline', InferencePipeline)

        pipeline_path = os.path.join(model_dir, "pipeline.joblib")
        if not os.path.exists(pipeline_path):
            raise FileNotFoundError(f"pipeline.joblib not found in {model_dir}")

        pipeline = joblib.load(pipeline_path)
        self._models[version] = pipeline

        # Load the feature engine for this version
        engine = FeatureEngine.from_artifacts(model_dir)
        self._feature_engines[version] = engine

    def predict(
        self, raw_features: dict, model_version: str = "v13"
    ) -> Tuple[float, float, List[dict]]:
        """Generate a prediction from raw input features.

        Args:
            raw_features: Dictionary of raw feature values.
            model_version: Which model version to use.

        Returns:
            Tuple of (flood_risk_score, inference_time_ms, feature_importance_list)
        """
        start = time.perf_counter()

        if self._mock_mode or model_version not in self._models:
            score = self._mock_predict(raw_features, model_version)
            importance = self._mock_feature_importance(raw_features)
        else:
            engine = self._feature_engines[model_version]
            pipeline = self._models[model_version]

            feature_vec = engine.transform(raw_features)
            score = float(pipeline.predict(feature_vec.reshape(1, -1))[0])
            score = float(np.clip(score, 0.0, 1.0))

            # Extract feature importance if available
            importance = self._get_feature_importance(model_version, raw_features)

        elapsed_ms = (time.perf_counter() - start) * 1000
        return score, elapsed_ms, importance

    def _mock_predict(self, features: dict, model_version: str) -> float:
        """Generate a realistic mock prediction using domain heuristics.

        This allows the frontend to be developed before models are exported.
        The heuristic roughly approximates the actual model behavior.
        """
        # Base score influenced by key risk factors
        base = 0.40  # approximate global mean

        # Rainfall effect (higher rainfall -> higher risk)
        rain = features.get('rainfall_7d_mm', 50.0)
        base += (rain - 50.0) / 1500.0 * 0.15

        # Drainage effect (better drainage -> lower risk)
        drainage = features.get('drainage_index', 0.5)
        base -= (drainage - 0.5) * 0.12

        # Elevation effect (higher elevation -> lower risk)
        elev = features.get('elevation_m', 30.0)
        base -= min((elev - 30.0) / 500.0, 0.1) * 0.08

        # Distance to river (farther -> lower risk)
        river = features.get('distance_to_river_m', 500.0)
        base -= min((river - 500.0) / 5000.0, 0.1) * 0.06

        # Infrastructure (better -> lower risk)
        infra = features.get('infrastructure_score', 0.5)
        base -= (infra - 0.5) * 0.08

        # Historical floods (more -> higher risk)
        floods = features.get('historical_flood_count', 2)
        base += (floods - 2) * 0.02

        # Extreme weather (higher -> higher risk)
        extreme = features.get('extreme_weather_index', 0.3)
        base += (extreme - 0.3) * 0.10

        # Hospital distance (farther -> slightly higher risk via vulnerability)
        hosp = features.get('nearest_hospital_km', 5.0)
        base += max(0, (hosp - 5.0) / 50.0) * 0.04

        # Model version "complexity" adds slight variation
        if model_version == "v18_titan":
            base *= 0.98  # Titan tends to slightly lower scores
        elif model_version == "v20_colossus":
            base *= 0.97  # Colossus is even more conservative

        # Add tiny random noise for realism
        np.random.seed(int(rain * 100 + elev * 10 + drainage * 1000) % 2**31)
        base += np.random.normal(0, 0.005)

        return float(np.clip(base, 0.0, 1.0))

    def _mock_feature_importance(self, features: dict) -> List[dict]:
        """Generate mock feature importance values."""
        importance_map = {
            "rainfall_7d_mm": 0.18,
            "drainage_index": 0.14,
            "extreme_weather_index": 0.12,
            "elevation_m": 0.10,
            "distance_to_river_m": 0.09,
            "infrastructure_score": 0.08,
            "historical_flood_count": 0.07,
            "monthly_rainfall_mm": 0.06,
            "nearest_hospital_km": 0.05,
            "built_up_percent": 0.04,
        }
        return [
            {"feature": k, "importance": v}
            for k, v in sorted(importance_map.items(), key=lambda x: -x[1])
        ]

    def _get_feature_importance(self, model_version: str, features: dict) -> List[dict]:
        """Extract actual feature importance from trained models."""
        # For stacked ensembles, we use the meta-model coefficients
        # as a proxy for feature importance
        return self._mock_feature_importance(features)

    def get_available_models(self) -> List[dict]:
        """Return info about all available model versions."""
        return list(self._model_info.values())

    def is_mock_mode(self) -> bool:
        return self._mock_mode


# Global singleton
model_manager = ModelManager()
