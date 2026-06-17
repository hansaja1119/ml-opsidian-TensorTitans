"""
Prediction Logger — Logs all simulation requests to PostgreSQL.

Provides both sync logging (for individual requests) and
analytics queries (for the admin dashboard).
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import func, desc

from api.database import SimulationLog, SessionLocal

logger = logging.getLogger(__name__)


def log_simulation(
    db: Session,
    request_data: dict,
    flood_risk_score: float,
    risk_level: str,
    model_version: str,
    inference_time_ms: float,
):
    """Log a single simulation request to the database."""
    try:
        log_entry = SimulationLog(
            timestamp=datetime.utcnow(),
            model_version=model_version,
            district=str(request_data.get('district', 'Unknown')),
            rainfall_7d_mm=request_data.get('rainfall_7d_mm'),
            drainage_index=request_data.get('drainage_index'),
            infrastructure_score=request_data.get('infrastructure_score'),
            nearest_hospital_km=request_data.get('nearest_hospital_km'),
            nearest_evac_km=request_data.get('nearest_evac_km'),
            elevation_m=request_data.get('elevation_m'),
            distance_to_river_m=request_data.get('distance_to_river_m'),
            built_up_percent=request_data.get('built_up_percent'),
            input_features_json=json.dumps(request_data, default=str),
            flood_risk_score=flood_risk_score,
            risk_level=risk_level,
            inference_time_ms=inference_time_ms,
        )
        db.add(log_entry)
        db.commit()
    except Exception as e:
        logger.error(f"Failed to log simulation: {e}")
        db.rollback()


def get_analytics(db: Session) -> dict:
    """Compute aggregated analytics from simulation logs."""
    try:
        total = db.query(func.count(SimulationLog.id)).scalar() or 0

        if total == 0:
            return {
                "total_simulations": 0,
                "avg_latency_ms": 0.0,
                "avg_risk_score": 0.0,
                "most_tweaked_features": {},
                "score_distribution": {"0.0-0.25": 0, "0.25-0.50": 0, "0.50-0.75": 0, "0.75-1.0": 0},
                "recent_simulations": [],
            }

        avg_latency = db.query(func.avg(SimulationLog.inference_time_ms)).scalar() or 0.0
        avg_score = db.query(func.avg(SimulationLog.flood_risk_score)).scalar() or 0.0

        # Score distribution
        low = db.query(func.count(SimulationLog.id)).filter(
            SimulationLog.flood_risk_score < 0.25
        ).scalar() or 0
        med = db.query(func.count(SimulationLog.id)).filter(
            SimulationLog.flood_risk_score >= 0.25,
            SimulationLog.flood_risk_score < 0.50
        ).scalar() or 0
        high = db.query(func.count(SimulationLog.id)).filter(
            SimulationLog.flood_risk_score >= 0.50,
            SimulationLog.flood_risk_score < 0.75
        ).scalar() or 0
        critical = db.query(func.count(SimulationLog.id)).filter(
            SimulationLog.flood_risk_score >= 0.75
        ).scalar() or 0

        # Most tweaked features — count non-null entries for key columns
        feature_counts = {}
        for col_name in ['rainfall_7d_mm', 'drainage_index', 'infrastructure_score',
                         'nearest_hospital_km', 'elevation_m', 'distance_to_river_m',
                         'built_up_percent', 'nearest_evac_km']:
            col = getattr(SimulationLog, col_name)
            count = db.query(func.count(SimulationLog.id)).filter(col.isnot(None)).scalar() or 0
            feature_counts[col_name] = count

        # Recent simulations (last 20)
        recent = db.query(SimulationLog).order_by(
            desc(SimulationLog.timestamp)
        ).limit(20).all()

        recent_list = [{
            "id": r.id,
            "timestamp": r.timestamp.isoformat() if r.timestamp else None,
            "district": r.district,
            "model_version": r.model_version,
            "flood_risk_score": round(r.flood_risk_score, 4),
            "risk_level": r.risk_level,
            "inference_time_ms": round(r.inference_time_ms, 2),
        } for r in recent]

        return {
            "total_simulations": total,
            "avg_latency_ms": round(avg_latency, 2),
            "avg_risk_score": round(avg_score, 4),
            "most_tweaked_features": feature_counts,
            "score_distribution": {
                "0.0-0.25": low,
                "0.25-0.50": med,
                "0.50-0.75": high,
                "0.75-1.0": critical,
            },
            "recent_simulations": recent_list,
        }
    except Exception as e:
        logger.error(f"Failed to compute analytics: {e}")
        return {
            "total_simulations": 0,
            "avg_latency_ms": 0.0,
            "avg_risk_score": 0.0,
            "most_tweaked_features": {},
            "score_distribution": {},
            "recent_simulations": [],
        }
