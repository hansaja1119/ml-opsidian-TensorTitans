'use client';

import { useState, useEffect, useCallback } from 'react';
import { Lightbulb, ArrowDownRight, TrendingDown, AlertTriangle, RefreshCw } from 'lucide-react';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const ALL_DISTRICTS = [
  'Colombo', 'Gampaha', 'Kalutara', 'Kandy', 'Matale', 'Nuwara Eliya',
  'Galle', 'Matara', 'Hambantota', 'Jaffna', 'Kilinochchi', 'Mannar',
  'Mullaitivu', 'Vavuniya', 'Batticaloa', 'Ampara', 'Trincomalee',
  'Kurunegala', 'Puttalam', 'Anuradhapura', 'Polonnaruwa', 'Badulla',
  'Monaragala', 'Ratnapura', 'Kegalle',
];

interface Intervention {
  feature: string;
  feature_label: string;
  current_value: number;
  improved_value: number;
  baseline_score: number;
  improved_score: number;
  risk_reduction_pct: number;
  category: string;
}

interface SensitivityData {
  feature: string;
  label: string;
  values: number[];
  scores: number[];
}

interface PlannerData {
  district: string;
  model_version: string;
  baseline_score: number;
  baseline_risk_level: string;
  interventions: Intervention[];
  sensitivity: SensitivityData[];
}

function getRiskColor(level: string): string {
  switch (level) {
    case 'Low': return '#22c55e';
    case 'Medium': return '#f59e0b';
    case 'High': return '#f97316';
    case 'Critical': return '#ef4444';
    default: return '#94a3b8';
  }
}

function SensitivityChart({ data }: { data: SensitivityData }) {
  const minScore = Math.min(...data.scores);
  const maxScore = Math.max(...data.scores);
  const range = maxScore - minScore || 0.01;

  return (
    <div style={{ marginTop: '0.75rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: 'var(--text-muted)', marginBottom: '4px' }}>
        <span>{data.values[0]}</span>
        <span>{data.values[data.values.length - 1]}</span>
      </div>
      <div style={{ display: 'flex', gap: '2px', height: '48px', alignItems: 'flex-end' }}>
        {data.scores.map((score, i) => {
          const height = ((score - minScore) / range) * 100;
          const hue = 120 - (score * 120); // green to red
          return (
            <div
              key={i}
              style={{
                flex: 1,
                height: `${Math.max(height, 4)}%`,
                background: `hsl(${Math.max(hue, 0)}, 70%, 50%)`,
                borderRadius: '2px 2px 0 0',
                transition: 'height 0.4s ease',
              }}
              title={`${data.label} = ${data.values[i]} → Score: ${score}`}
            />
          );
        })}
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.65rem', color: 'var(--text-muted)', marginTop: '2px' }}>
        <span>Score: {minScore.toFixed(3)}</span>
        <span>{maxScore.toFixed(3)}</span>
      </div>
    </div>
  );
}

export default function InterventionsPage() {
  const [district, setDistrict] = useState('Colombo');
  const [data, setData] = useState<PlannerData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchInterventions = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_URL}/api/interventions?district=${district}&model_version=v13`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = await res.json();
      setData(json);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch');
    } finally {
      setLoading(false);
    }
  }, [district]);

  useEffect(() => {
    fetchInterventions();
  }, [fetchInterventions]);

  return (
    <main className="page-content">
      <div className="page-header animate-in">
        <h1><Lightbulb size={36} style={{ display: 'inline', verticalAlign: 'middle', marginRight: '0.5rem' }} /> Intervention Planner</h1>
        <p>
          Discover which infrastructure improvements would reduce flood risk the most.
          Select a district to see ranked interventions and sensitivity analysis.
        </p>
      </div>

      {/* District selector */}
      <div className="glass-card animate-in" style={{ maxWidth: '500px', margin: '0 auto 2rem', animationDelay: '100ms' }}>
        <div style={{ display: 'flex', gap: '1rem', alignItems: 'flex-end' }}>
          <div className="select-group" style={{ flex: 1, marginBottom: 0 }}>
            <label htmlFor="int-district">District</label>
            <div className="select-wrapper">
              <select id="int-district" value={district} onChange={e => setDistrict(e.target.value)}>
                {ALL_DISTRICTS.map(d => <option key={d} value={d}>{d}</option>)}
              </select>
            </div>
          </div>
          <button className="btn-simulate" style={{ width: 'auto', padding: '10px 20px', fontSize: '0.85rem' }}
            onClick={fetchInterventions} disabled={loading}>
            <RefreshCw size={14} style={{ marginRight: '6px' }} /> {loading ? 'Analyzing...' : 'Analyze'}
          </button>
        </div>
      </div>

      {error && (
        <div style={{ padding: '16px', background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)', borderRadius: '12px', color: '#ef4444', marginBottom: '2rem', textAlign: 'center' }}>
          <AlertTriangle size={14} style={{ display: 'inline', verticalAlign: 'middle' }} /> {error}
        </div>
      )}

      {data && (
        <>
          {/* Baseline Risk Banner */}
          <div className="glass-card animate-in" style={{ textAlign: 'center', marginBottom: '2rem', animationDelay: '150ms' }}>
            <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '2rem', flexWrap: 'wrap' }}>
              <div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>District</div>
                <div style={{ fontSize: '1.5rem', fontWeight: 800, color: 'var(--text-primary)' }}>{data.district}</div>
              </div>
              <div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Baseline Risk</div>
                <div style={{ fontSize: '2rem', fontWeight: 800, color: getRiskColor(data.baseline_risk_level) }}>
                  {data.baseline_score.toFixed(4)}
                </div>
              </div>
              <div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Risk Level</div>
                <span className={`risk-badge ${data.baseline_risk_level.toLowerCase()}`} style={{ fontSize: '0.85rem', padding: '4px 12px' }}>
                  {data.baseline_risk_level}
                </span>
              </div>
            </div>
          </div>

          {/* Ranked Interventions */}
          <div className="glass-card animate-in" style={{ marginBottom: '2rem', animationDelay: '200ms' }}>
            <div className="card-header">
              <div className="icon"><TrendingDown size={16} /></div>
              <h2>Recommended Interventions</h2>
              <span style={{ marginLeft: 'auto', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                Ranked by risk reduction impact
              </span>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              {data.interventions.map((item, i) => (
                <div key={item.feature} className="animate-in"
                  style={{
                    display: 'flex', alignItems: 'center', gap: '1rem', padding: '1rem',
                    background: 'rgba(255,255,255,0.02)', borderRadius: '12px',
                    border: '1px solid var(--border-glass)', animationDelay: `${i * 80}ms`,
                  }}>
                  {/* Rank */}
                  <div style={{
                    width: '36px', height: '36px', borderRadius: '50%', display: 'flex',
                    alignItems: 'center', justifyContent: 'center', fontWeight: 800, fontSize: '0.9rem',
                    background: i === 0 ? 'rgba(235,0,45,0.15)' : 'rgba(255,255,255,0.05)',
                    color: i === 0 ? '#eb002d' : 'var(--text-secondary)',
                    border: i === 0 ? '2px solid rgba(235,0,45,0.3)' : '1px solid var(--border-glass)',
                    flexShrink: 0,
                  }}>
                    #{i + 1}
                  </div>

                  {/* Info */}
                  <div style={{ flex: 1 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '4px' }}>
                      <span style={{ fontWeight: 700, color: 'var(--text-primary)' }}>{item.feature_label}</span>
                      <span style={{
                        fontSize: '0.65rem', padding: '2px 6px', borderRadius: '4px',
                        background: 'rgba(20,184,166,0.1)', color: '#14b8a6',
                        textTransform: 'uppercase', fontWeight: 600,
                      }}>{item.category}</span>
                    </div>
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                      {item.current_value.toFixed(2)} <ArrowDownRight size={12} style={{ display: 'inline' }} /> {item.improved_value.toFixed(2)}
                    </div>
                  </div>

                  {/* Score change */}
                  <div style={{ textAlign: 'right' }}>
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                      {item.baseline_score.toFixed(4)} → {item.improved_score.toFixed(4)}
                    </div>
                    <div style={{
                      fontSize: '1.1rem', fontWeight: 800,
                      color: item.risk_reduction_pct > 0 ? '#22c55e' : '#ef4444',
                    }}>
                      {item.risk_reduction_pct > 0 ? '−' : '+'}{Math.abs(item.risk_reduction_pct).toFixed(1)}%
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Sensitivity Analysis Grid */}
          <div className="glass-card animate-in" style={{ animationDelay: '300ms' }}>
            <div className="card-header">
              <div className="icon"><Lightbulb size={16} /></div>
              <h2>Sensitivity Analysis</h2>
              <span style={{ marginLeft: 'auto', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                How each feature affects risk score across its full range
              </span>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1.5rem' }}>
              {data.sensitivity.map((s, i) => (
                <div key={s.feature} className="animate-in"
                  style={{
                    padding: '1rem', background: 'rgba(255,255,255,0.02)', borderRadius: '10px',
                    border: '1px solid var(--border-glass)', animationDelay: `${i * 60}ms`,
                  }}>
                  <div style={{ fontWeight: 700, fontSize: '0.85rem', color: 'var(--text-primary)', marginBottom: '2px' }}>
                    {s.label}
                  </div>
                  <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>{s.feature}</div>
                  <SensitivityChart data={s} />
                </div>
              ))}
            </div>
          </div>
        </>
      )}
    </main>
  );
}
