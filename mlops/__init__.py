"""
MLflow Configuration — Central configuration for experiment tracking.
"""

import os

# MLflow tracking server URI
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

# Experiment name
EXPERIMENT_NAME = "flood-risk-simulation"

# Model registry names
MODEL_REGISTRY = {
    "v13": "flood-risk-v13",
    "v18_titan": "flood-risk-v18-titan",
    "v20_colossus": "flood-risk-v20-colossus",
}

# Artifact storage
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
