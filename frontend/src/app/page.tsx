'use client';

import { useState, useCallback } from 'react';
import { Waves, MapPin, CloudRain, Mountain, Building2, Zap, AlertTriangle, Target, BarChart3 } from 'lucide-react';
import FeatureSlider from '@/components/FeatureSlider';
import RiskGauge from '@/components/RiskGauge';
import FeatureImportanceChart from '@/components/FeatureImportanceChart';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const ALL_DISTRICTS = [
  'Colombo', 'Gampaha', 'Kalutara', 'Kandy', 'Matale', 'Nuwara Eliya',
  'Galle', 'Matara', 'Hambantota', 'Jaffna', 'Kilinochchi', 'Mannar',
  'Mullaitivu', 'Vavuniya', 'Batticaloa', 'Ampara', 'Trincomalee',
  'Kurunegala', 'Puttalam', 'Anuradhapura', 'Polonnaruwa', 'Badulla',
  'Monaragala', 'Ratnapura', 'Kegalle',
];

const MODEL_VERSIONS = [
  { value: 'v13', label: 'V13 — Ridge Stack (Fast)' },
  { value: 'v18_titan', label: 'V18 Titan — 36 Models (Accurate)' },
  { value: 'v20_colossus', label: 'V20 Colossus — 130 Models (Ultra)' },
];

interface SimulationResult {
  flood_risk_score: number;
  risk_level: string;
  model_version: string;
  inference_time_ms: number;
  feature_importance: { feature: string; importance: number }[];
}

const DEFAULT_FEATURES = {
  rainfall_7d_mm: 50.0,
  monthly_rainfall_mm: 200.0,
  drainage_index: 0.5,
  infrastructure_score: 0.5,
  elevation_m: 30.0,
  distance_to_river_m: 500.0,
  nearest_hospital_km: 5.0,
  nearest_evac_km: 3.0,
  built_up_percent: 30.0,
  extreme_weather_index: 0.3,
  seasonal_index: 0.5,
  terrain_roughness_index: 0.2,
  historical_flood_count: 2,
  population_density_per_km2: 500.0,
  ndvi: 0.3,
  ndwi: 0.1,
  socioeconomic_status_index: 0.5,
  inundation_area_sqm: 1000.0,
};

export default function SimulationPage() {
  const [district, setDistrict] = useState('Colombo');
  const [modelVersion, setModelVersion] = useState('v13');
  const [features, setFeatures] = useState(DEFAULT_FEATURES);
  const [result, setResult] = useState<SimulationResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const updateFeature = useCallback((key: string, value: number) => {
    setFeatures(prev => ({ ...prev, [key]: value }));
  }, []);

  const handleDistrictChange = useCallback(async (newDistrict: string) => {
    setDistrict(newDistrict);
    try {
      const res = await fetch(`${API_URL}/api/districts`);
      if (res.ok) {
        const districts = await res.json();
        const match = districts.find((d: { district: string }) => d.district === newDistrict);
        if (match) {
          setFeatures(prev => ({ ...prev, ...match.defaults }));
        }
      }
    } catch {
      // Use defaults if API is unreachable
    }
  }, []);

  const runSimulation = useCallback(async () => {
    setLoading(true);
    setError(null);

    const payload = {
      ...features,
      district,
      model_version: modelVersion,
      landcover: 'Urban',
      soil_type: 'Clay',
      water_supply: 'Pipe-borne',
      electricity: 'Yes',
      road_quality: 'Moderate',
      urban_rural: 'Urban',
      water_presence_flag: 'No',
      flood_occurrence_current_event: 'No',
      is_good_to_live: 'Yes',
      reason_not_good_to_live: 'Other',
      place_name: 'Unknown',
      generation_date: new Date().toISOString().split('T')[0],
      is_synthetic: 0,
    };

    try {
      const res = await fetch(`${API_URL}/api/simulate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `API Error: ${res.status}`);
      }

      const data: SimulationResult = await res.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to connect to the API');
    } finally {
      setLoading(false);
    }
  }, [features, district, modelVersion]);

  return (
    <main className="page-content">
      {/* Page Header */}
      <div className="page-header animate-in">
        <h1><Waves size={36} style={{ display: 'inline', verticalAlign: 'middle', marginRight: '0.5rem' }} /> Flood Risk Simulator</h1>
        <p>
          Simulate &ldquo;what-if&rdquo; scenarios by adjusting infrastructure and environmental
          parameters. Observe how changes impact the predicted flood risk score.
        </p>
      </div>

      <div className="simulation-grid">
        {/* ─── Left Panel: Controls ─────────────────────── */}
        <div>
          {/* District & Model Selection */}
          <div className="glass-card animate-in" style={{ animationDelay: '100ms' }}>
            <div className="card-header">
              <div className="icon"><MapPin size={16} /></div>
              <h2>Location & Model</h2>
            </div>

            <div className="select-group">
              <label htmlFor="district-select">District</label>
              <div className="select-wrapper">
                <select
                  id="district-select"
                  value={district}
                  onChange={(e) => handleDistrictChange(e.target.value)}
                >
                  {ALL_DISTRICTS.map(d => (
                    <option key={d} value={d}>{d}</option>
                  ))}
                </select>
              </div>
            </div>

            <div className="select-group">
              <label htmlFor="model-select">Model Version</label>
              <div className="select-wrapper">
                <select
                  id="model-select"
                  value={modelVersion}
                  onChange={(e) => setModelVersion(e.target.value)}
                >
                  {MODEL_VERSIONS.map(m => (
                    <option key={m.value} value={m.value}>{m.label}</option>
                  ))}
                </select>
              </div>
            </div>
          </div>

          {/* Weather Parameters */}
          <div className="glass-card animate-in" style={{ marginTop: '1.5rem', animationDelay: '200ms' }}>
            <div className="card-header">
              <div className="icon"><CloudRain size={16} /></div>
              <h2>Weather Scenario</h2>
            </div>

            <FeatureSlider
              id="rainfall_7d_mm" label="7-Day Rainfall" unit=" mm"
              value={features.rainfall_7d_mm} min={0} max={500} step={5}
              onChange={(v) => updateFeature('rainfall_7d_mm', v)}
            />
            <FeatureSlider
              id="monthly_rainfall_mm" label="Monthly Rainfall" unit=" mm"
              value={features.monthly_rainfall_mm} min={0} max={1000} step={10}
              onChange={(v) => updateFeature('monthly_rainfall_mm', v)}
            />
            <FeatureSlider
              id="extreme_weather_index" label="Extreme Weather Index"
              value={features.extreme_weather_index} min={0} max={1} step={0.01}
              onChange={(v) => updateFeature('extreme_weather_index', v)}
            />
            <FeatureSlider
              id="seasonal_index" label="Seasonal Index"
              value={features.seasonal_index} min={0} max={1} step={0.01}
              onChange={(v) => updateFeature('seasonal_index', v)}
            />
          </div>

          {/* Geography */}
          <div className="glass-card animate-in" style={{ marginTop: '1.5rem', animationDelay: '300ms' }}>
            <div className="card-header">
              <div className="icon"><Mountain size={16} /></div>
              <h2>Geography & Terrain</h2>
            </div>

            <FeatureSlider
              id="elevation_m" label="Elevation" unit=" m"
              value={features.elevation_m} min={0} max={500} step={5}
              onChange={(v) => updateFeature('elevation_m', v)}
            />
            <FeatureSlider
              id="distance_to_river_m" label="Distance to River" unit=" m"
              value={features.distance_to_river_m} min={0} max={5000} step={50}
              onChange={(v) => updateFeature('distance_to_river_m', v)}
            />
            <FeatureSlider
              id="drainage_index" label="Drainage Index"
              value={features.drainage_index} min={0} max={1} step={0.01}
              onChange={(v) => updateFeature('drainage_index', v)}
            />
            <FeatureSlider
              id="terrain_roughness_index" label="Terrain Roughness"
              value={features.terrain_roughness_index} min={0} max={1} step={0.01}
              onChange={(v) => updateFeature('terrain_roughness_index', v)}
            />
          </div>

          {/* Infrastructure */}
          <div className="glass-card animate-in" style={{ marginTop: '1.5rem', animationDelay: '400ms' }}>
            <div className="card-header">
              <div className="icon"><Building2 size={16} /></div>
              <h2>Infrastructure & Demographics</h2>
            </div>

            <FeatureSlider
              id="infrastructure_score" label="Infrastructure Score"
              value={features.infrastructure_score} min={0} max={1} step={0.01}
              onChange={(v) => updateFeature('infrastructure_score', v)}
            />
            <FeatureSlider
              id="nearest_hospital_km" label="Nearest Hospital" unit=" km"
              value={features.nearest_hospital_km} min={0} max={50} step={0.5}
              onChange={(v) => updateFeature('nearest_hospital_km', v)}
            />
            <FeatureSlider
              id="nearest_evac_km" label="Nearest Evacuation Center" unit=" km"
              value={features.nearest_evac_km} min={0} max={50} step={0.5}
              onChange={(v) => updateFeature('nearest_evac_km', v)}
            />
            <FeatureSlider
              id="built_up_percent" label="Built-Up Area" unit="%"
              value={features.built_up_percent} min={0} max={100} step={1}
              onChange={(v) => updateFeature('built_up_percent', v)}
            />
            <FeatureSlider
              id="population_density_per_km2" label="Population Density" unit="/km²"
              value={features.population_density_per_km2} min={0} max={10000} step={50}
              onChange={(v) => updateFeature('population_density_per_km2', v)}
            />
            <FeatureSlider
              id="historical_flood_count" label="Historical Flood Count"
              value={features.historical_flood_count} min={0} max={20} step={1}
              onChange={(v) => updateFeature('historical_flood_count', v)}
            />
          </div>

          {/* Simulate Button */}
          <button
            className={`btn-simulate ${loading ? 'loading' : ''}`}
            onClick={runSimulation}
            disabled={loading}
            style={{ marginTop: '1.5rem' }}
          >
            {loading ? 'Simulating...' : 'Run Simulation'}
          </button>

          {error && (
            <div style={{
              marginTop: '1rem',
              padding: '12px 16px',
              background: 'rgba(239, 68, 68, 0.1)',
              border: '1px solid rgba(239, 68, 68, 0.3)',
              borderRadius: '8px',
              color: '#ef4444',
              fontSize: '0.85rem',
            }}>
              <AlertTriangle size={14} style={{ display: 'inline', verticalAlign: 'middle' }} /> {error}
            </div>
          )}
        </div>

        {/* ─── Right Panel: Results ─────────────────────── */}
        <div style={{ position: 'sticky', top: '80px' }}>
          {/* Risk Score Gauge */}
          <div className="glass-card animate-in" style={{ animationDelay: '150ms' }}>
            <div className="card-header">
              <div className="icon"><Target size={16} /></div>
              <h2>Flood Risk Score</h2>
              {result && (
                <span className="badge badge-mock" style={{ marginLeft: 'auto' }}>
                  {result.model_version}
                </span>
              )}
            </div>

            <RiskGauge
              score={result?.flood_risk_score ?? 0}
              riskLevel={result?.risk_level ?? 'Waiting...'}
              modelVersion={result?.model_version ?? 'N/A'}
              inferenceTimeMs={result?.inference_time_ms ?? 0}
            />
          </div>

          {/* Feature Importance */}
          {result && result.feature_importance.length > 0 && (
            <div className="glass-card animate-in" style={{ marginTop: '1.5rem', animationDelay: '250ms' }}>
              <div className="card-header">
                <div className="icon"><BarChart3 size={16} /></div>
                <h2>Feature Importance</h2>
              </div>
              <FeatureImportanceChart data={result.feature_importance} />
            </div>
          )}
        </div>
      </div>
    </main>
  );
}
