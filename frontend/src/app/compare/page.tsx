'use client';

import { useState, useCallback } from 'react';
import { GitCompareArrows, ArrowRight, TrendingUp, TrendingDown, Minus, AlertTriangle } from 'lucide-react';
import RiskGauge from '@/components/RiskGauge';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const ALL_DISTRICTS = [
  'Colombo', 'Gampaha', 'Kalutara', 'Kandy', 'Matale', 'Nuwara Eliya',
  'Galle', 'Matara', 'Hambantota', 'Jaffna', 'Kilinochchi', 'Mannar',
  'Mullaitivu', 'Vavuniya', 'Batticaloa', 'Ampara', 'Trincomalee',
  'Kurunegala', 'Puttalam', 'Anuradhapura', 'Polonnaruwa', 'Badulla',
  'Monaragala', 'Ratnapura', 'Kegalle',
];

interface FeatureConfig {
  key: string;
  label: string;
  min: number;
  max: number;
  step: number;
}

const ADJUSTABLE_FEATURES: FeatureConfig[] = [
  { key: 'drainage_index', label: 'Drainage Index', min: 0, max: 1, step: 0.05 },
  { key: 'infrastructure_score', label: 'Infrastructure Score', min: 0, max: 1, step: 0.05 },
  { key: 'nearest_hospital_km', label: 'Hospital Distance (km)', min: 0, max: 50, step: 0.5 },
  { key: 'nearest_evac_km', label: 'Evacuation Center (km)', min: 0, max: 50, step: 0.5 },
  { key: 'built_up_percent', label: 'Built-Up Area %', min: 0, max: 100, step: 1 },
  { key: 'distance_to_river_m', label: 'River Setback (m)', min: 0, max: 5000, step: 50 },
  { key: 'rainfall_7d_mm', label: '7-Day Rainfall (mm)', min: 0, max: 500, step: 5 },
  { key: 'elevation_m', label: 'Elevation (m)', min: 0, max: 500, step: 5 },
];

interface ScenarioResultData {
  label: string;
  district: string;
  flood_risk_score: number;
  risk_level: string;
}

interface FeatureDelta {
  feature: string;
  label: string;
  value_a: number;
  value_b: number;
  delta: number;
  impact_direction: string;
}

interface ComparisonResult {
  scenario_a: ScenarioResultData;
  scenario_b: ScenarioResultData;
  score_delta: number;
  risk_improvement: string;
  feature_deltas: FeatureDelta[];
}

export default function ComparePage() {
  const [districtA, setDistrictA] = useState('Colombo');
  const [districtB, setDistrictB] = useState('Colombo');
  const [featuresA, setFeaturesA] = useState<Record<string, number | null>>({});
  const [featuresB, setFeaturesB] = useState<Record<string, number | null>>({});
  const [result, setResult] = useState<ComparisonResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const runComparison = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const body = {
        scenario_a: { label: 'Current State', district: districtA, model_version: 'v13', ...featuresA },
        scenario_b: { label: 'Proposed Plan', district: districtB, model_version: 'v13', ...featuresB },
      };
      const res = await fetch(`${API_URL}/api/compare`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setResult(await res.json());
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to compare');
    } finally {
      setLoading(false);
    }
  }, [districtA, districtB, featuresA, featuresB]);

  return (
    <main className="page-content">
      <div className="page-header animate-in">
        <h1><GitCompareArrows size={36} style={{ display: 'inline', verticalAlign: 'middle', marginRight: '0.5rem' }} /> Scenario Comparison</h1>
        <p>
          Compare &ldquo;Current State&rdquo; vs &ldquo;Proposed Plan&rdquo; side by side.
          See exactly how infrastructure changes affect flood risk.
        </p>
      </div>

      {/* Dual-panel Input */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr auto 1fr', gap: '1.5rem', marginBottom: '2rem', alignItems: 'start' }}>
        {/* Scenario A */}
        <div className="glass-card animate-in" style={{ animationDelay: '100ms' }}>
          <div className="card-header">
            <div className="icon" style={{ background: 'rgba(235,0,45,0.15)' }}><span style={{ fontWeight: 800, fontSize: '0.85rem', color: '#eb002d' }}>A</span></div>
            <h2>Current State</h2>
          </div>
          <div className="select-group">
            <label htmlFor="cmp-dist-a">District</label>
            <div className="select-wrapper">
              <select id="cmp-dist-a" value={districtA} onChange={e => setDistrictA(e.target.value)}>
                {ALL_DISTRICTS.map(d => <option key={d} value={d}>{d}</option>)}
              </select>
            </div>
          </div>
          {ADJUSTABLE_FEATURES.map(f => (
            <div key={`a-${f.key}`} className="slider-group">
              <div className="slider-header">
                <span className="slider-label">{f.label}</span>
                <span className="slider-value">{featuresA[f.key] != null ? featuresA[f.key] : 'Default'}</span>
              </div>
              <input type="range" min={f.min} max={f.max} step={f.step}
                value={featuresA[f.key] ?? f.min + (f.max - f.min) / 2}
                onChange={e => setFeaturesA(prev => ({ ...prev, [f.key]: parseFloat(e.target.value) }))}
                style={{ '--progress': `${((((featuresA[f.key] ?? (f.min + (f.max - f.min) / 2)) - f.min) / (f.max - f.min)) * 100)}%` } as React.CSSProperties}
              />
            </div>
          ))}
        </div>

        {/* Center Arrow */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', paddingTop: '8rem' }}>
          <div style={{
            width: '48px', height: '48px', borderRadius: '50%', display: 'flex',
            alignItems: 'center', justifyContent: 'center',
            background: 'linear-gradient(135deg, var(--accent-primary), var(--accent-teal))',
            boxShadow: '0 0 20px rgba(235,0,45,0.3)',
          }}>
            <ArrowRight size={24} color="#fff" />
          </div>
        </div>

        {/* Scenario B */}
        <div className="glass-card animate-in" style={{ animationDelay: '200ms' }}>
          <div className="card-header">
            <div className="icon" style={{ background: 'rgba(20,184,166,0.15)' }}><span style={{ fontWeight: 800, fontSize: '0.85rem', color: '#14b8a6' }}>B</span></div>
            <h2>Proposed Plan</h2>
          </div>
          <div className="select-group">
            <label htmlFor="cmp-dist-b">District</label>
            <div className="select-wrapper">
              <select id="cmp-dist-b" value={districtB} onChange={e => setDistrictB(e.target.value)}>
                {ALL_DISTRICTS.map(d => <option key={d} value={d}>{d}</option>)}
              </select>
            </div>
          </div>
          {ADJUSTABLE_FEATURES.map(f => (
            <div key={`b-${f.key}`} className="slider-group">
              <div className="slider-header">
                <span className="slider-label">{f.label}</span>
                <span className="slider-value">{featuresB[f.key] != null ? featuresB[f.key] : 'Default'}</span>
              </div>
              <input type="range" min={f.min} max={f.max} step={f.step}
                value={featuresB[f.key] ?? f.min + (f.max - f.min) / 2}
                onChange={e => setFeaturesB(prev => ({ ...prev, [f.key]: parseFloat(e.target.value) }))}
                style={{ '--progress': `${((((featuresB[f.key] ?? (f.min + (f.max - f.min) / 2)) - f.min) / (f.max - f.min)) * 100)}%` } as React.CSSProperties}
              />
            </div>
          ))}
        </div>
      </div>

      {/* Compare Button */}
      <button className={`btn-simulate ${loading ? 'loading' : ''}`}
        onClick={runComparison} disabled={loading}
        style={{ maxWidth: '400px', margin: '0 auto 2rem', display: 'block' }}>
        {loading ? 'Comparing...' : 'Compare Scenarios'}
      </button>

      {error && (
        <div style={{ padding: '16px', background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)', borderRadius: '12px', color: '#ef4444', marginBottom: '2rem', textAlign: 'center' }}>
          <AlertTriangle size={14} style={{ display: 'inline', verticalAlign: 'middle' }} /> {error}
        </div>
      )}

      {/* Results */}
      {result && (
        <div className="animate-in">
          {/* Score Comparison */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr auto 1fr', gap: '1.5rem', marginBottom: '2rem', alignItems: 'center' }}>
            <div className="glass-card" style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '0.5rem' }}>Current State</div>
              <RiskGauge score={result.scenario_a.flood_risk_score} riskLevel={result.scenario_a.risk_level} modelVersion="" inferenceTimeMs={0} />
            </div>

            {/* Delta */}
            <div style={{ textAlign: 'center' }}>
              <div style={{
                fontSize: '2rem', fontWeight: 800,
                color: result.risk_improvement === 'improved' ? '#22c55e' : result.risk_improvement === 'worsened' ? '#ef4444' : 'var(--text-muted)',
              }}>
                {result.score_delta > 0 ? '+' : ''}{result.score_delta.toFixed(4)}
              </div>
              <div style={{
                fontSize: '0.8rem', fontWeight: 700, textTransform: 'uppercase',
                color: result.risk_improvement === 'improved' ? '#22c55e' : result.risk_improvement === 'worsened' ? '#ef4444' : 'var(--text-muted)',
                display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '4px',
              }}>
                {result.risk_improvement === 'improved' ? <TrendingDown size={16} /> : result.risk_improvement === 'worsened' ? <TrendingUp size={16} /> : <Minus size={16} />}
                {result.risk_improvement}
              </div>
            </div>

            <div className="glass-card" style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '0.5rem' }}>Proposed Plan</div>
              <RiskGauge score={result.scenario_b.flood_risk_score} riskLevel={result.scenario_b.risk_level} modelVersion="" inferenceTimeMs={0} />
            </div>
          </div>

          {/* Feature Deltas */}
          {result.feature_deltas.length > 0 && (
            <div className="glass-card">
              <div className="card-header">
                <div className="icon"><GitCompareArrows size={16} /></div>
                <h2>Feature Changes</h2>
              </div>
              <table className="logs-table">
                <thead>
                  <tr>
                    <th>Feature</th>
                    <th style={{ textAlign: 'right' }}>Current</th>
                    <th style={{ textAlign: 'center' }}></th>
                    <th style={{ textAlign: 'right' }}>Proposed</th>
                    <th style={{ textAlign: 'right' }}>Delta</th>
                  </tr>
                </thead>
                <tbody>
                  {result.feature_deltas.map(fd => (
                    <tr key={fd.feature}>
                      <td style={{ fontWeight: 600 }}>{fd.label}</td>
                      <td style={{ textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>{fd.value_a.toFixed(2)}</td>
                      <td style={{ textAlign: 'center' }}><ArrowRight size={14} color="var(--text-muted)" /></td>
                      <td style={{ textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>{fd.value_b.toFixed(2)}</td>
                      <td style={{
                        textAlign: 'right', fontWeight: 700, fontVariantNumeric: 'tabular-nums',
                        color: fd.impact_direction === 'positive' ? '#22c55e' : fd.impact_direction === 'negative' ? '#ef4444' : 'var(--text-secondary)',
                      }}>
                        {fd.delta > 0 ? '+' : ''}{fd.delta.toFixed(2)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </main>
  );
}
