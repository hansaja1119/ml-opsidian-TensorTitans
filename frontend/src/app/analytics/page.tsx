'use client';

import { useState, useEffect } from 'react';
import { BarChart3, AlertTriangle, TrendingUp, Wrench, ClipboardList } from 'lucide-react';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface AnalyticsData {
  total_simulations: number;
  avg_latency_ms: number;
  avg_risk_score: number;
  most_tweaked_features: Record<string, number>;
  score_distribution: Record<string, number>;
  recent_simulations: {
    id: number;
    timestamp: string;
    district: string;
    model_version: string;
    flood_risk_score: number;
    risk_level: string;
    inference_time_ms: number;
  }[];
}

const FEATURE_LABELS: Record<string, string> = {
  rainfall_7d_mm: '7-Day Rainfall',
  drainage_index: 'Drainage Index',
  infrastructure_score: 'Infrastructure',
  nearest_hospital_km: 'Hospital Dist.',
  elevation_m: 'Elevation',
  distance_to_river_m: 'River Distance',
  built_up_percent: 'Built-Up %',
  nearest_evac_km: 'Evacuation Dist.',
};

const DISTRIBUTION_COLORS: Record<string, string> = {
  '0.0-0.25': '#22c55e',
  '0.25-0.50': '#f59e0b',
  '0.50-0.75': '#f97316',
  '0.75-1.0': '#ef4444',
};

export default function AnalyticsPage() {
  const [data, setData] = useState<AnalyticsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchAnalytics();
    const interval = setInterval(fetchAnalytics, 10000); // Auto-refresh every 10s
    return () => clearInterval(interval);
  }, []);

  async function fetchAnalytics() {
    try {
      const res = await fetch(`${API_URL}/api/analytics`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = await res.json();
      setData(json);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch analytics');
    } finally {
      setLoading(false);
    }
  }

  const maxDist = data ? Math.max(...Object.values(data.score_distribution), 1) : 1;
  const maxFeature = data ? Math.max(...Object.values(data.most_tweaked_features), 1) : 1;

  return (
    <main className="page-content">
      <div className="page-header animate-in">
        <h1><BarChart3 size={36} style={{ display: 'inline', verticalAlign: 'middle', marginRight: '0.5rem' }} /> MLOps Analytics Dashboard</h1>
        <p>
          Monitor simulation performance, track model usage, and analyze
          urban planning patterns across all API requests.
        </p>
      </div>

      {error && (
        <div style={{
          padding: '16px',
          background: 'rgba(239, 68, 68, 0.1)',
          border: '1px solid rgba(239, 68, 68, 0.3)',
          borderRadius: '12px',
          color: '#ef4444',
          marginBottom: '2rem',
          textAlign: 'center',
        }}>
          <AlertTriangle size={14} style={{ display: 'inline', verticalAlign: 'middle' }} /> Could not connect to API: {error}. Make sure the backend is running.
        </div>
      )}

      {/* ─── Summary Stats ───────────────────────────── */}
      <div className="analytics-grid">
        <div className="stat-card animate-in" style={{ animationDelay: '100ms' }}>
          <div className="stat-value">{data?.total_simulations ?? '—'}</div>
          <div className="stat-label">Total Simulations</div>
        </div>
        <div className="stat-card animate-in" style={{ animationDelay: '200ms' }}>
          <div className="stat-value">
            {data ? `${data.avg_latency_ms.toFixed(1)}` : '—'}
          </div>
          <div className="stat-label">Avg Latency (ms)</div>
        </div>
        <div className="stat-card animate-in" style={{ animationDelay: '300ms' }}>
          <div className="stat-value">
            {data ? data.avg_risk_score.toFixed(4) : '—'}
          </div>
          <div className="stat-label">Avg Risk Score</div>
        </div>
        <div className="stat-card animate-in" style={{ animationDelay: '400ms' }}>
          <div className="stat-value">
            {data?.recent_simulations.length ?? '—'}
          </div>
          <div className="stat-label">Recent (Last 20)</div>
        </div>
      </div>

      <div className="simulation-grid">
        {/* ─── Left: Charts ──────────────────────────── */}
        <div>
          {/* Score Distribution */}
          <div className="glass-card animate-in" style={{ animationDelay: '200ms' }}>
            <div className="card-header">
              <div className="icon"><TrendingUp size={16} /></div>
              <h2>Risk Score Distribution</h2>
            </div>

            {data && (
              <div className="distribution-chart">
                {Object.entries(data.score_distribution).map(([range, count]) => (
                  <div key={range} className="distribution-bar-wrapper">
                    <div className="distribution-bar-count">{count}</div>
                    <div
                      className="distribution-bar"
                      style={{
                        height: `${Math.max((count / maxDist) * 100, 4)}%`,
                        background: DISTRIBUTION_COLORS[range] || '#eb002d',
                      }}
                    />
                    <div className="distribution-bar-label">{range}</div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Feature Popularity */}
          <div className="glass-card animate-in" style={{ marginTop: '1.5rem', animationDelay: '300ms' }}>
            <div className="card-header">
              <div className="icon"><Wrench size={16} /></div>
              <h2>Most Tweaked Features</h2>
            </div>

            {data && (
              <div className="importance-chart">
                {Object.entries(data.most_tweaked_features)
                  .sort(([, a], [, b]) => b - a)
                  .map(([feature, count], index) => (
                    <div key={feature} className="importance-bar-row animate-in"
                      style={{ animationDelay: `${index * 60}ms` }}>
                      <span className="importance-label">
                        {FEATURE_LABELS[feature] || feature}
                      </span>
                      <div className="importance-bar-container">
                        <div
                          className="importance-bar"
                          style={{ width: `${(count / maxFeature) * 100}%` }}
                        />
                      </div>
                      <span className="importance-value">{count}</span>
                    </div>
                  ))}
              </div>
            )}
          </div>
        </div>

        {/* ─── Right: Recent Logs ────────────────────── */}
        <div>
          <div className="glass-card animate-in" style={{ animationDelay: '250ms' }}>
            <div className="card-header">
              <div className="icon"><ClipboardList size={16} /></div>
              <h2>Recent Simulations</h2>
            </div>

            {data && data.recent_simulations.length > 0 ? (
              <div style={{ overflowX: 'auto' }}>
                <table className="logs-table">
                  <thead>
                    <tr>
                      <th>Time</th>
                      <th>District</th>
                      <th>Score</th>
                      <th>Risk</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.recent_simulations.map((sim) => (
                      <tr key={sim.id}>
                        <td style={{ fontSize: '0.75rem' }}>
                          {sim.timestamp
                            ? new Date(sim.timestamp).toLocaleTimeString()
                            : '—'}
                        </td>
                        <td>{sim.district}</td>
                        <td style={{ fontVariantNumeric: 'tabular-nums', fontWeight: 600 }}>
                          {sim.flood_risk_score.toFixed(4)}
                        </td>
                        <td>
                          <span className={`risk-badge ${sim.risk_level.toLowerCase()}`}>
                            {sim.risk_level}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p style={{ color: 'var(--text-muted)', textAlign: 'center', padding: '2rem 0' }}>
                No simulations recorded yet. Run a simulation to see data here.
              </p>
            )}
          </div>
        </div>
      </div>
    </main>
  );
}
