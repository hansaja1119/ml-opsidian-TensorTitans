'use client';

import { useState, useEffect } from 'react';
import { MapPinned, AlertTriangle, Shield, ShieldAlert, ShieldCheck, ShieldX } from 'lucide-react';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface DistrictRisk {
  district: string;
  flood_risk_score: number;
  risk_level: string;
  key_factors: Record<string, number>;
}

interface OverviewData {
  model_version: string;
  districts: DistrictRisk[];
  highest_risk: string;
  lowest_risk: string;
  avg_risk_score: number;
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

function getRiskBg(level: string): string {
  switch (level) {
    case 'Low': return 'rgba(34,197,94,0.08)';
    case 'Medium': return 'rgba(245,158,11,0.08)';
    case 'High': return 'rgba(249,115,22,0.08)';
    case 'Critical': return 'rgba(239,68,68,0.08)';
    default: return 'rgba(255,255,255,0.03)';
  }
}

function RiskIcon({ level }: { level: string }) {
  const size = 18;
  switch (level) {
    case 'Low': return <ShieldCheck size={size} color="#22c55e" />;
    case 'Medium': return <Shield size={size} color="#f59e0b" />;
    case 'High': return <ShieldAlert size={size} color="#f97316" />;
    case 'Critical': return <ShieldX size={size} color="#ef4444" />;
    default: return <Shield size={size} color="#94a3b8" />;
  }
}

export default function DistrictOverviewPage() {
  const [data, setData] = useState<OverviewData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedDistrict, setSelectedDistrict] = useState<string | null>(null);

  useEffect(() => {
    async function fetchOverview() {
      try {
        const res = await fetch(`${API_URL}/api/district-overview?model_version=v13`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        setData(await res.json());
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch');
      } finally {
        setLoading(false);
      }
    }
    fetchOverview();
  }, []);

  const maxScore = data ? Math.max(...data.districts.map(d => d.flood_risk_score)) : 1;

  // Counts per risk level
  const riskCounts = data ? data.districts.reduce((acc, d) => {
    acc[d.risk_level] = (acc[d.risk_level] || 0) + 1;
    return acc;
  }, {} as Record<string, number>) : {};

  const selected = data?.districts.find(d => d.district === selectedDistrict);

  return (
    <main className="page-content">
      <div className="page-header animate-in">
        <h1><MapPinned size={36} style={{ display: 'inline', verticalAlign: 'middle', marginRight: '0.5rem' }} /> District Risk Overview</h1>
        <p>
          Baseline flood risk scores for all 25 Sri Lankan districts.
          Identify the most vulnerable areas and prioritize resource allocation.
        </p>
      </div>

      {error && (
        <div style={{ padding: '16px', background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)', borderRadius: '12px', color: '#ef4444', marginBottom: '2rem', textAlign: 'center' }}>
          <AlertTriangle size={14} style={{ display: 'inline', verticalAlign: 'middle' }} /> {error}
        </div>
      )}

      {loading && (
        <div style={{ textAlign: 'center', padding: '4rem 0', color: 'var(--text-muted)' }}>
          <div className="animate-pulse" style={{ fontSize: '1.2rem' }}>Computing risk scores for 25 districts...</div>
        </div>
      )}

      {data && (
        <>
          {/* Summary Stats */}
          <div className="analytics-grid" style={{ marginBottom: '2rem' }}>
            <div className="stat-card animate-in" style={{ animationDelay: '100ms' }}>
              <div className="stat-value">{data.districts.length}</div>
              <div className="stat-label">Districts Analyzed</div>
            </div>
            <div className="stat-card animate-in" style={{ animationDelay: '200ms' }}>
              <div className="stat-value" style={{ background: `linear-gradient(135deg, ${getRiskColor('Critical')}, ${getRiskColor('High')})`, WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
                {data.highest_risk}
              </div>
              <div className="stat-label">Highest Risk</div>
            </div>
            <div className="stat-card animate-in" style={{ animationDelay: '300ms' }}>
              <div className="stat-value" style={{ background: `linear-gradient(135deg, ${getRiskColor('Low')}, #14b8a6)`, WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
                {data.lowest_risk}
              </div>
              <div className="stat-label">Lowest Risk</div>
            </div>
            <div className="stat-card animate-in" style={{ animationDelay: '400ms' }}>
              <div className="stat-value">{data.avg_risk_score.toFixed(4)}</div>
              <div className="stat-label">Average Risk</div>
            </div>
          </div>

          {/* Risk Distribution Mini-Bar */}
          <div className="glass-card animate-in" style={{ marginBottom: '2rem', animationDelay: '150ms' }}>
            <div style={{ display: 'flex', gap: '1.5rem', justifyContent: 'center', flexWrap: 'wrap' }}>
              {['Low', 'Medium', 'High', 'Critical'].map(level => (
                <div key={level} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <div style={{ width: '12px', height: '12px', borderRadius: '3px', background: getRiskColor(level) }} />
                  <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                    {level}: <strong style={{ color: 'var(--text-primary)' }}>{riskCounts[level] || 0}</strong>
                  </span>
                </div>
              ))}
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 380px', gap: '1.5rem', alignItems: 'start' }}>
            {/* District Risk Bars */}
            <div className="glass-card animate-in" style={{ animationDelay: '200ms' }}>
              <div className="card-header">
                <div className="icon"><MapPinned size={16} /></div>
                <h2>Risk Ranking</h2>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                {data.districts.map((d, i) => (
                  <div
                    key={d.district}
                    className="animate-in"
                    onClick={() => setSelectedDistrict(d.district)}
                    style={{
                      display: 'flex', alignItems: 'center', gap: '0.75rem',
                      padding: '8px 12px', borderRadius: '8px', cursor: 'pointer',
                      background: selectedDistrict === d.district ? getRiskBg(d.risk_level) : 'transparent',
                      border: selectedDistrict === d.district ? `1px solid ${getRiskColor(d.risk_level)}40` : '1px solid transparent',
                      transition: 'all 0.2s ease',
                      animationDelay: `${i * 30}ms`,
                    }}
                  >
                    {/* Rank */}
                    <span style={{ width: '24px', fontSize: '0.75rem', fontWeight: 700, color: 'var(--text-muted)', textAlign: 'right' }}>
                      {i + 1}
                    </span>
                    {/* Icon */}
                    <RiskIcon level={d.risk_level} />
                    {/* Name */}
                    <span style={{ width: '120px', fontWeight: 600, fontSize: '0.85rem', color: 'var(--text-primary)' }}>
                      {d.district}
                    </span>
                    {/* Bar */}
                    <div style={{ flex: 1, height: '8px', background: 'var(--bg-secondary)', borderRadius: '4px', overflow: 'hidden' }}>
                      <div style={{
                        height: '100%', borderRadius: '4px',
                        width: `${(d.flood_risk_score / maxScore) * 100}%`,
                        background: getRiskColor(d.risk_level),
                        transition: 'width 0.6s cubic-bezier(0.4, 0, 0.2, 1)',
                      }} />
                    </div>
                    {/* Score */}
                    <span style={{
                      fontWeight: 700, fontSize: '0.8rem', fontVariantNumeric: 'tabular-nums',
                      color: getRiskColor(d.risk_level), minWidth: '55px', textAlign: 'right',
                    }}>
                      {d.flood_risk_score.toFixed(4)}
                    </span>
                    {/* Badge */}
                    <span className={`risk-badge ${d.risk_level.toLowerCase()}`} style={{ minWidth: '55px', textAlign: 'center' }}>
                      {d.risk_level}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            {/* District Detail Panel */}
            <div style={{ position: 'sticky', top: '80px' }}>
              <div className="glass-card animate-in" style={{ animationDelay: '250ms' }}>
                <div className="card-header">
                  <div className="icon"><Shield size={16} /></div>
                  <h2>District Details</h2>
                </div>

                {selected ? (
                  <div>
                    <div style={{ textAlign: 'center', marginBottom: '1.5rem' }}>
                      <div style={{ fontSize: '1.4rem', fontWeight: 800, color: 'var(--text-primary)' }}>{selected.district}</div>
                      <div style={{ fontSize: '2.5rem', fontWeight: 800, color: getRiskColor(selected.risk_level), lineHeight: 1.2, margin: '0.5rem 0' }}>
                        {selected.flood_risk_score.toFixed(4)}
                      </div>
                      <span className={`risk-badge ${selected.risk_level.toLowerCase()}`} style={{ fontSize: '0.85rem', padding: '4px 12px' }}>
                        {selected.risk_level}
                      </span>
                    </div>

                    <div className="section-divider" />

                    <div style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--accent-teal)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: '0.75rem' }}>
                      Key Risk Factors
                    </div>
                    {Object.entries(selected.key_factors).map(([key, val]) => {
                      const labels: Record<string, string> = {
                        rainfall_7d_mm: 'Rainfall (7d)', drainage_index: 'Drainage',
                        elevation_m: 'Elevation', historical_flood_count: 'Flood History',
                        infrastructure_score: 'Infrastructure',
                      };
                      return (
                        <div key={key} style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 0', borderBottom: '1px solid rgba(255,255,255,0.03)' }}>
                          <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>{labels[key] || key}</span>
                          <span style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--text-primary)', fontVariantNumeric: 'tabular-nums' }}>
                            {typeof val === 'number' && val < 1 ? val.toFixed(2) : val}
                          </span>
                        </div>
                      );
                    })}

                    <div style={{ marginTop: '1.5rem' }}>
                      <a href={`/interventions?district=${selected.district}`}
                        style={{
                          display: 'block', textAlign: 'center', padding: '10px', borderRadius: '8px',
                          background: 'linear-gradient(135deg, var(--accent-primary), var(--accent-secondary))',
                          color: 'white', fontWeight: 700, fontSize: '0.85rem', textDecoration: 'none',
                          textTransform: 'uppercase', letterSpacing: '0.05em',
                        }}>
                        View Interventions →
                      </a>
                    </div>
                  </div>
                ) : (
                  <div style={{ textAlign: 'center', padding: '3rem 1rem', color: 'var(--text-muted)' }}>
                    <MapPinned size={48} style={{ opacity: 0.3, marginBottom: '1rem' }} />
                    <p>Click on a district to view details</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </>
      )}
    </main>
  );
}
