"""
Urban Planning Simulation Tool — FastAPI Backend

Main application exposing REST endpoints for:
- /api/simulate    — Run flood risk simulation with customized features
- /api/models      — List available model versions
- /api/districts   — Get default feature values per district
- /api/analytics   — Aggregated simulation analytics (admin dashboard)
- /api/health      — Health check
"""

import logging
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from api.schemas import (
    SimulationRequest, SimulationResponse, FeatureImportanceItem,
    ModelInfo, DistrictDefaults, HealthResponse, AnalyticsResponse,
    InterventionPlannerResponse, InterventionItem, SensitivityItem,
    ScenarioComparisonRequest, ScenarioComparisonResponse, ScenarioResult, FeatureDelta,
    DistrictOverviewResponse, DistrictRiskItem,
)
from api.model_manager import model_manager
from api.feature_engine import get_risk_level
from api.database import init_db, get_db
from api.logger import log_simulation, get_analytics

# ─── Logging Setup ──────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


# ─── Lifespan (startup/shutdown) ────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database and load models on startup."""
    logger.info("Initializing database...")
    try:
        init_db()
        logger.info("Database initialized.")
    except Exception as e:
        logger.warning(f"Database init failed (will retry on first request): {e}")

    logger.info("Loading ML models...")
    model_manager.initialize()
    if model_manager.is_mock_mode():
        logger.warning("Running in MOCK mode — no real model artifacts found.")
    else:
        logger.info("Models loaded successfully.")

    yield  # App is running

    logger.info("Shutting down...")


# ─── App Creation ───────────────────────────────────────────────

app = FastAPI(
    title="Flood Risk Simulation API",
    description=(
        "Urban Planning Simulation Tool for predicting flood risk scores "
        "across Sri Lankan districts. Powered by champion ML ensemble models "
        "from the ML Opsidian Genesis competition."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow the Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://frontend:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Default District Values ────────────────────────────────────

# Pre-computed median feature values per district (from training data)
# These serve as sensible defaults when a user selects a district
DISTRICT_DEFAULTS = {
    "Colombo":       {"rainfall_7d_mm": 65.0, "monthly_rainfall_mm": 250.0, "elevation_m": 7.0,
                      "distance_to_river_m": 320.0, "drainage_index": 0.38, "infrastructure_score": 0.72,
                      "nearest_hospital_km": 2.5, "nearest_evac_km": 1.8, "built_up_percent": 68.0,
                      "population_density_per_km2": 3400.0, "ndvi": 0.18, "ndwi": 0.15,
                      "extreme_weather_index": 0.35, "seasonal_index": 0.55, "terrain_roughness_index": 0.12,
                      "socioeconomic_status_index": 0.68, "historical_flood_count": 5,
                      "inundation_area_sqm": 2500.0},
    "Gampaha":       {"rainfall_7d_mm": 55.0, "monthly_rainfall_mm": 220.0, "elevation_m": 12.0,
                      "distance_to_river_m": 450.0, "drainage_index": 0.42, "infrastructure_score": 0.65,
                      "nearest_hospital_km": 3.5, "nearest_evac_km": 2.5, "built_up_percent": 52.0,
                      "population_density_per_km2": 1700.0, "ndvi": 0.25, "ndwi": 0.12,
                      "extreme_weather_index": 0.30, "seasonal_index": 0.50, "terrain_roughness_index": 0.15,
                      "socioeconomic_status_index": 0.62, "historical_flood_count": 4,
                      "inundation_area_sqm": 2000.0},
    "Kalutara":      {"rainfall_7d_mm": 70.0, "monthly_rainfall_mm": 280.0, "elevation_m": 15.0,
                      "distance_to_river_m": 380.0, "drainage_index": 0.40, "infrastructure_score": 0.55,
                      "nearest_hospital_km": 4.0, "nearest_evac_km": 3.0, "built_up_percent": 40.0,
                      "population_density_per_km2": 800.0, "ndvi": 0.32, "ndwi": 0.14,
                      "extreme_weather_index": 0.38, "seasonal_index": 0.52, "terrain_roughness_index": 0.18,
                      "socioeconomic_status_index": 0.55, "historical_flood_count": 4,
                      "inundation_area_sqm": 1800.0},
    "Ratnapura":     {"rainfall_7d_mm": 85.0, "monthly_rainfall_mm": 350.0, "elevation_m": 45.0,
                      "distance_to_river_m": 280.0, "drainage_index": 0.35, "infrastructure_score": 0.42,
                      "nearest_hospital_km": 8.0, "nearest_evac_km": 6.0, "built_up_percent": 22.0,
                      "population_density_per_km2": 450.0, "ndvi": 0.45, "ndwi": 0.18,
                      "extreme_weather_index": 0.45, "seasonal_index": 0.58, "terrain_roughness_index": 0.35,
                      "socioeconomic_status_index": 0.42, "historical_flood_count": 7,
                      "inundation_area_sqm": 3500.0},
    "Kandy":         {"rainfall_7d_mm": 60.0, "monthly_rainfall_mm": 230.0, "elevation_m": 500.0,
                      "distance_to_river_m": 600.0, "drainage_index": 0.48, "infrastructure_score": 0.58,
                      "nearest_hospital_km": 4.5, "nearest_evac_km": 3.5, "built_up_percent": 35.0,
                      "population_density_per_km2": 700.0, "ndvi": 0.38, "ndwi": 0.08,
                      "extreme_weather_index": 0.28, "seasonal_index": 0.48, "terrain_roughness_index": 0.42,
                      "socioeconomic_status_index": 0.55, "historical_flood_count": 3,
                      "inundation_area_sqm": 1200.0},
    "Galle":         {"rainfall_7d_mm": 72.0, "monthly_rainfall_mm": 270.0, "elevation_m": 10.0,
                      "distance_to_river_m": 350.0, "drainage_index": 0.42, "infrastructure_score": 0.52,
                      "nearest_hospital_km": 5.0, "nearest_evac_km": 3.5, "built_up_percent": 38.0,
                      "population_density_per_km2": 650.0, "ndvi": 0.35, "ndwi": 0.16,
                      "extreme_weather_index": 0.36, "seasonal_index": 0.54, "terrain_roughness_index": 0.20,
                      "socioeconomic_status_index": 0.50, "historical_flood_count": 4,
                      "inundation_area_sqm": 2200.0},
    "Matara":        {"rainfall_7d_mm": 68.0, "monthly_rainfall_mm": 260.0, "elevation_m": 8.0,
                      "distance_to_river_m": 400.0, "drainage_index": 0.44, "infrastructure_score": 0.50,
                      "nearest_hospital_km": 5.5, "nearest_evac_km": 4.0, "built_up_percent": 32.0,
                      "population_density_per_km2": 550.0, "ndvi": 0.33, "ndwi": 0.13,
                      "extreme_weather_index": 0.33, "seasonal_index": 0.52, "terrain_roughness_index": 0.22,
                      "socioeconomic_status_index": 0.48, "historical_flood_count": 3,
                      "inundation_area_sqm": 1600.0},
    "Batticaloa":    {"rainfall_7d_mm": 75.0, "monthly_rainfall_mm": 290.0, "elevation_m": 5.0,
                      "distance_to_river_m": 250.0, "drainage_index": 0.32, "infrastructure_score": 0.38,
                      "nearest_hospital_km": 7.0, "nearest_evac_km": 5.5, "built_up_percent": 25.0,
                      "population_density_per_km2": 380.0, "ndvi": 0.28, "ndwi": 0.22,
                      "extreme_weather_index": 0.42, "seasonal_index": 0.60, "terrain_roughness_index": 0.10,
                      "socioeconomic_status_index": 0.38, "historical_flood_count": 6,
                      "inundation_area_sqm": 3000.0},
    "Kurunegala":    {"rainfall_7d_mm": 50.0, "monthly_rainfall_mm": 200.0, "elevation_m": 120.0,
                      "distance_to_river_m": 700.0, "drainage_index": 0.50, "infrastructure_score": 0.52,
                      "nearest_hospital_km": 6.0, "nearest_evac_km": 4.5, "built_up_percent": 28.0,
                      "population_density_per_km2": 520.0, "ndvi": 0.40, "ndwi": 0.08,
                      "extreme_weather_index": 0.25, "seasonal_index": 0.45, "terrain_roughness_index": 0.25,
                      "socioeconomic_status_index": 0.50, "historical_flood_count": 2,
                      "inundation_area_sqm": 800.0},
    "Anuradhapura":  {"rainfall_7d_mm": 45.0, "monthly_rainfall_mm": 180.0, "elevation_m": 90.0,
                      "distance_to_river_m": 900.0, "drainage_index": 0.55, "infrastructure_score": 0.45,
                      "nearest_hospital_km": 10.0, "nearest_evac_km": 7.0, "built_up_percent": 18.0,
                      "population_density_per_km2": 280.0, "ndvi": 0.42, "ndwi": 0.06,
                      "extreme_weather_index": 0.22, "seasonal_index": 0.42, "terrain_roughness_index": 0.15,
                      "socioeconomic_status_index": 0.42, "historical_flood_count": 2,
                      "inundation_area_sqm": 600.0},
}

# Fill remaining districts with generic defaults
GENERIC_DEFAULTS = {
    "rainfall_7d_mm": 50.0, "monthly_rainfall_mm": 200.0, "elevation_m": 30.0,
    "distance_to_river_m": 500.0, "drainage_index": 0.45, "infrastructure_score": 0.50,
    "nearest_hospital_km": 6.0, "nearest_evac_km": 4.0, "built_up_percent": 30.0,
    "population_density_per_km2": 500.0, "ndvi": 0.30, "ndwi": 0.10,
    "extreme_weather_index": 0.30, "seasonal_index": 0.50, "terrain_roughness_index": 0.20,
    "socioeconomic_status_index": 0.50, "historical_flood_count": 2,
    "inundation_area_sqm": 1000.0,
}

ALL_DISTRICTS = [
    "Colombo", "Gampaha", "Kalutara", "Kandy", "Matale", "Nuwara Eliya",
    "Galle", "Matara", "Hambantota", "Jaffna", "Kilinochchi", "Mannar",
    "Mullaitivu", "Vavuniya", "Batticaloa", "Ampara", "Trincomalee",
    "Kurunegala", "Puttalam", "Anuradhapura", "Polonnaruwa", "Badulla",
    "Monaragala", "Ratnapura", "Kegalle"
]


# ─── API Endpoints ──────────────────────────────────────────────

@app.post("/api/simulate", response_model=SimulationResponse)
async def simulate(request: SimulationRequest, db: Session = Depends(get_db)):
    """Run a flood risk simulation with custom feature values.

    Accepts raw feature values, runs the feature engineering pipeline,
    generates a prediction using the selected model version, logs the
    result to PostgreSQL, and returns the flood risk score.
    """
    # Convert request to dict for the model pipeline
    raw_features = request.model_dump()
    model_version = raw_features.pop('model_version', 'v13')
    if hasattr(model_version, 'value'):
        model_version = model_version.value

    # Convert enum values to strings
    for key, value in raw_features.items():
        if hasattr(value, 'value'):
            raw_features[key] = value.value

    # Run inference
    score, latency_ms, importance = model_manager.predict(raw_features, model_version)
    risk_level = get_risk_level(score)

    # Log to database
    try:
        log_simulation(
            db=db,
            request_data=raw_features,
            flood_risk_score=score,
            risk_level=risk_level,
            model_version=model_version,
            inference_time_ms=latency_ms,
        )
    except Exception as e:
        logger.warning(f"Logging failed (non-blocking): {e}")

    return SimulationResponse(
        flood_risk_score=round(score, 4),
        risk_level=risk_level,
        model_version=model_version,
        inference_time_ms=round(latency_ms, 2),
        feature_importance=[
            FeatureImportanceItem(**item) for item in importance
        ],
    )


@app.get("/api/models", response_model=list[ModelInfo])
async def list_models():
    """List all available model versions."""
    return model_manager.get_available_models()


@app.get("/api/districts", response_model=list[DistrictDefaults])
async def list_districts():
    """Return all districts with their default feature values."""
    result = []
    for district in ALL_DISTRICTS:
        defaults = DISTRICT_DEFAULTS.get(district, GENERIC_DEFAULTS)
        result.append(DistrictDefaults(district=district, defaults=defaults))
    return result


@app.get("/api/analytics", response_model=AnalyticsResponse)
async def analytics(db: Session = Depends(get_db)):
    """Return aggregated simulation analytics for the admin dashboard."""
    data = get_analytics(db)
    return AnalyticsResponse(**data)


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    models = [m["version"] for m in model_manager.get_available_models()]
    db_connected = True
    try:
        db = next(get_db())
        db.close()
    except Exception:
        db_connected = False

    return HealthResponse(
        status="healthy" if models else "degraded",
        models_loaded=models,
        database_connected=db_connected,
    )


# ─── Urban Planning Endpoints ──────────────────────────────────

# Actionable features an urban planner can influence
ACTIONABLE_FEATURES = {
    'drainage_index':       {'label': 'Drainage System',     'category': 'Infrastructure', 'min': 0, 'max': 1,     'improve_to': 0.85, 'steps': 10},
    'infrastructure_score': {'label': 'Infrastructure Score', 'category': 'Infrastructure', 'min': 0, 'max': 1,     'improve_to': 0.90, 'steps': 10},
    'nearest_hospital_km':  {'label': 'Hospital Distance',   'category': 'Healthcare',     'min': 0, 'max': 50,    'improve_to': 1.5,  'steps': 10},
    'nearest_evac_km':      {'label': 'Evacuation Center',   'category': 'Emergency',      'min': 0, 'max': 50,    'improve_to': 1.0,  'steps': 10},
    'built_up_percent':     {'label': 'Built-Up Area %',     'category': 'Land Use',       'min': 0, 'max': 100,   'improve_to': None, 'steps': 10},
    'distance_to_river_m':  {'label': 'River Setback',       'category': 'Geography',      'min': 0, 'max': 5000,  'improve_to': 1500, 'steps': 10},
    'elevation_m':          {'label': 'Elevation',           'category': 'Geography',      'min': 0, 'max': 500,   'improve_to': None, 'steps': 10},
}

FEATURE_LABELS = {
    'rainfall_7d_mm': '7-Day Rainfall',
    'monthly_rainfall_mm': 'Monthly Rainfall',
    'drainage_index': 'Drainage Index',
    'infrastructure_score': 'Infrastructure Score',
    'nearest_hospital_km': 'Hospital Distance',
    'nearest_evac_km': 'Evacuation Distance',
    'built_up_percent': 'Built-Up %',
    'distance_to_river_m': 'River Distance',
    'elevation_m': 'Elevation',
    'extreme_weather_index': 'Extreme Weather',
    'seasonal_index': 'Seasonal Index',
    'terrain_roughness_index': 'Terrain Roughness',
    'historical_flood_count': 'Historical Floods',
    'population_density_per_km2': 'Population Density',
    'ndvi': 'NDVI',
    'ndwi': 'NDWI',
    'socioeconomic_status_index': 'Socioeconomic Index',
    'inundation_area_sqm': 'Inundation Area',
}


def _build_raw_features(district: str, overrides: dict) -> dict:
    """Build a complete raw feature dict with district defaults + overrides."""
    defaults = DISTRICT_DEFAULTS.get(district, GENERIC_DEFAULTS)
    raw = {
        'district': district,
        'landcover': 'Urban',
        'soil_type': 'Clay',
        'water_supply': 'Pipe-borne',
        'electricity': 'Yes',
        'road_quality': 'Moderate',
        'urban_rural': 'Urban',
        'water_presence_flag': 'No',
        'flood_occurrence_current_event': 'No',
        'is_good_to_live': 'Yes',
        'reason_not_good_to_live': 'Other',
        'place_name': 'Unknown',
        'generation_date': '2024-06-15',
        'is_synthetic': 0,
        **defaults,
    }
    # Apply overrides (skip None values)
    for k, v in overrides.items():
        if v is not None:
            raw[k] = v
    return raw


@app.get("/api/interventions", response_model=InterventionPlannerResponse)
async def intervention_planner(district: str = "Colombo", model_version: str = "v13"):
    """Analyze which infrastructure interventions reduce flood risk the most.

    For each actionable feature, the endpoint:
    1. Computes the baseline risk score with district defaults
    2. Simulates an "improved" scenario (e.g., drainage from 0.38 to 0.85)
    3. Runs sensitivity analysis across the full feature range
    4. Returns ranked interventions by risk reduction percentage
    """
    import numpy as np

    # Baseline prediction
    baseline_raw = _build_raw_features(district, {})
    baseline_score, _, _ = model_manager.predict(dict(baseline_raw), model_version)
    baseline_level = get_risk_level(baseline_score)

    interventions = []
    sensitivity = []

    for feat_key, feat_info in ACTIONABLE_FEATURES.items():
        current_val = baseline_raw.get(feat_key, 0)

        # --- Sensitivity Analysis (sweep across range) ---
        test_values = np.linspace(feat_info['min'], feat_info['max'], feat_info['steps']).tolist()
        test_scores = []
        for tv in test_values:
            modified_raw = dict(baseline_raw)
            modified_raw[feat_key] = float(tv)
            score, _, _ = model_manager.predict(modified_raw, model_version)
            test_scores.append(round(float(score), 4))

        sensitivity.append(SensitivityItem(
            feature=feat_key,
            label=feat_info['label'],
            values=[round(v, 2) for v in test_values],
            scores=test_scores,
        ))

        # --- Intervention Impact ---
        improved_val = feat_info.get('improve_to')
        if improved_val is not None:
            modified_raw = dict(baseline_raw)
            modified_raw[feat_key] = float(improved_val)
            improved_score, _, _ = model_manager.predict(modified_raw, model_version)

            reduction = 0.0
            if baseline_score > 0:
                reduction = ((baseline_score - improved_score) / baseline_score) * 100

            interventions.append(InterventionItem(
                feature=feat_key,
                feature_label=feat_info['label'],
                current_value=round(float(current_val), 4),
                improved_value=round(float(improved_val), 4),
                baseline_score=round(float(baseline_score), 4),
                improved_score=round(float(improved_score), 4),
                risk_reduction_pct=round(float(reduction), 2),
                category=feat_info['category'],
            ))

    # Sort interventions by impact (highest reduction first)
    interventions.sort(key=lambda x: x.risk_reduction_pct, reverse=True)

    return InterventionPlannerResponse(
        district=district,
        model_version=model_version,
        baseline_score=round(float(baseline_score), 4),
        baseline_risk_level=baseline_level,
        interventions=interventions,
        sensitivity=sensitivity,
    )


@app.post("/api/compare", response_model=ScenarioComparisonResponse)
async def compare_scenarios(request: ScenarioComparisonRequest):
    """Compare two urban planning scenarios side by side.

    Each scenario can specify a different district, model version,
    and feature overrides. Returns the risk delta and per-feature changes.
    """
    # Build and score Scenario A
    a_data = request.scenario_a.model_dump()
    a_label = a_data.pop('label', 'Current State')
    a_district = a_data.pop('district', 'Colombo')
    a_model = a_data.pop('model_version', 'v13')
    a_raw = _build_raw_features(a_district, a_data)
    a_score, _, _ = model_manager.predict(dict(a_raw), a_model)

    # Build and score Scenario B
    b_data = request.scenario_b.model_dump()
    b_label = b_data.pop('label', 'Proposed Plan')
    b_district = b_data.pop('district', 'Colombo')
    b_model = b_data.pop('model_version', 'v13')
    b_raw = _build_raw_features(b_district, b_data)
    b_score, _, _ = model_manager.predict(dict(b_raw), b_model)

    # Compute feature deltas
    numerical_keys = [k for k in a_raw if isinstance(a_raw[k], (int, float)) and k not in ('is_synthetic',)]
    feature_deltas = []
    for key in numerical_keys:
        va = float(a_raw.get(key, 0))
        vb = float(b_raw.get(key, 0))
        delta = vb - va
        if abs(delta) > 0.001:
            direction = "neutral"
            if delta > 0 and key in ('drainage_index', 'infrastructure_score', 'distance_to_river_m', 'elevation_m'):
                direction = "positive"
            elif delta < 0 and key in ('nearest_hospital_km', 'nearest_evac_km', 'rainfall_7d_mm', 'extreme_weather_index'):
                direction = "positive"
            elif delta != 0:
                direction = "negative" if b_score > a_score else "positive"

            feature_deltas.append(FeatureDelta(
                feature=key,
                label=FEATURE_LABELS.get(key, key),
                value_a=round(va, 4),
                value_b=round(vb, 4),
                delta=round(delta, 4),
                impact_direction=direction,
            ))

    # Sort by absolute delta magnitude
    feature_deltas.sort(key=lambda x: abs(x.delta), reverse=True)

    score_delta = round(float(b_score - a_score), 4)
    improvement = "unchanged"
    if score_delta < -0.005:
        improvement = "improved"
    elif score_delta > 0.005:
        improvement = "worsened"

    return ScenarioComparisonResponse(
        scenario_a=ScenarioResult(
            label=a_label, district=a_district,
            flood_risk_score=round(float(a_score), 4),
            risk_level=get_risk_level(a_score),
            features_used={k: v for k, v in a_raw.items() if isinstance(v, (int, float))},
        ),
        scenario_b=ScenarioResult(
            label=b_label, district=b_district,
            flood_risk_score=round(float(b_score), 4),
            risk_level=get_risk_level(b_score),
            features_used={k: v for k, v in b_raw.items() if isinstance(v, (int, float))},
        ),
        score_delta=score_delta,
        risk_improvement=improvement,
        feature_deltas=feature_deltas,
    )


@app.get("/api/district-overview", response_model=DistrictOverviewResponse)
async def district_overview(model_version: str = "v13"):
    """Compute flood risk scores for all 25 districts using default features.

    Returns a ranked overview showing which districts are most at risk,
    allowing urban planners to prioritize resource allocation.
    """
    district_results = []

    for district in ALL_DISTRICTS:
        raw = _build_raw_features(district, {})
        score, _, _ = model_manager.predict(dict(raw), model_version)
        defaults = DISTRICT_DEFAULTS.get(district, GENERIC_DEFAULTS)

        district_results.append(DistrictRiskItem(
            district=district,
            flood_risk_score=round(float(score), 4),
            risk_level=get_risk_level(score),
            key_factors={
                'rainfall_7d_mm': defaults.get('rainfall_7d_mm', 50),
                'drainage_index': defaults.get('drainage_index', 0.45),
                'elevation_m': defaults.get('elevation_m', 30),
                'historical_flood_count': defaults.get('historical_flood_count', 2),
                'infrastructure_score': defaults.get('infrastructure_score', 0.5),
            },
        ))

    # Sort by risk (highest first)
    district_results.sort(key=lambda x: x.flood_risk_score, reverse=True)

    scores = [d.flood_risk_score for d in district_results]
    return DistrictOverviewResponse(
        model_version=model_version,
        districts=district_results,
        highest_risk=district_results[0].district if district_results else "",
        lowest_risk=district_results[-1].district if district_results else "",
        avg_risk_score=round(sum(scores) / len(scores), 4) if scores else 0,
    )
