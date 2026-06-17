"""
Database configuration and models for PostgreSQL.

Uses SQLAlchemy for ORM and connection management.
"""

import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Text
from sqlalchemy.orm import sessionmaker, declarative_base

# ─── Configuration ──────────────────────────────────────────────

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://floodsim:floodsim_password@localhost:5432/floodsim_db"
)

engine = create_engine(DATABASE_URL, echo=False, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ─── Database Models ────────────────────────────────────────────

class SimulationLog(Base):
    """Stores every simulation request for monitoring and analytics."""
    __tablename__ = "simulation_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    model_version = Column(String(50), nullable=False)
    district = Column(String(100), nullable=False)

    # Key input features (stored individually for analytics queries)
    rainfall_7d_mm = Column(Float)
    drainage_index = Column(Float)
    infrastructure_score = Column(Float)
    nearest_hospital_km = Column(Float)
    nearest_evac_km = Column(Float)
    elevation_m = Column(Float)
    distance_to_river_m = Column(Float)
    built_up_percent = Column(Float)

    # Full input stored as JSON string for complete reproducibility
    input_features_json = Column(Text)

    # Output
    flood_risk_score = Column(Float, nullable=False)
    risk_level = Column(String(20), nullable=False)

    # Performance
    inference_time_ms = Column(Float, nullable=False)


def init_db():
    """Create all tables in the database."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Dependency for FastAPI to get a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
