'use client';

import { useCallback, useState } from 'react';
import {
  AlertTriangle,
  ArrowRight,
  Bot,
  CheckCircle2,
  ChevronDown,
  CircleDollarSign,
  ListChecks,
  Route,
  Search,
  ShieldCheck,
  Sparkles,
  Target,
  XCircle,
} from 'lucide-react';
import RiskGauge from '@/components/RiskGauge';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const ALL_DISTRICTS = [
  'Colombo', 'Gampaha', 'Kalutara', 'Kandy', 'Matale', 'Nuwara Eliya',
  'Galle', 'Matara', 'Hambantota', 'Jaffna', 'Kilinochchi', 'Mannar',
  'Mullaitivu', 'Vavuniya', 'Batticaloa', 'Ampara', 'Trincomalee',
  'Kurunegala', 'Puttalam', 'Anuradhapura', 'Polonnaruwa', 'Badulla',
  'Monaragala', 'Ratnapura', 'Kegalle',
];

const TARGET_LEVELS = ['Low', 'Medium', 'High'] as const;

const BUDGET_PROFILES = [
  { value: 'low_cost', label: 'Low Cost', description: 'Favor practical, lower-cost actions' },
  { value: 'balanced', label: 'Balanced', description: 'Balance impact and implementation cost' },
  { value: 'aggressive', label: 'Aggressive', description: 'Prioritize maximum risk reduction' },
] as const;

const ACTIONABLE_FEATURES = [
  { key: 'drainage_index', label: 'Drainage System', category: 'Infrastructure' },
  { key: 'infrastructure_score', label: 'Infrastructure Score', category: 'Infrastructure' },
  { key: 'nearest_evac_km', label: 'Evacuation Center Distance', category: 'Emergency' },
  { key: 'nearest_hospital_km', label: 'Hospital Distance', category: 'Healthcare' },
  { key: 'distance_to_river_m', label: 'River Setback', category: 'Geography' },
  { key: 'built_up_percent', label: 'Built-Up Area', category: 'Land Use' },
];

interface PlanStep {
  feature: string;
  feature_label: string;
  category: string;
  from_value: number;
  to_value: number;
  score_before: number;
  score_after: number;
  absolute_reduction: number;
  relative_reduction_pct: number;
  cost_level: string;
  rationale: string;
}

interface SearchTraceItem {
  step: number;
  tried_feature: string;
  tried_value: number;
  resulting_score: number;
  resulting_risk_level: string;
  accepted: boolean;
}

interface AlternativePlan {
  name: string;
  optimized_score: number;
  optimized_risk_level: string;
  target_reached: boolean;
  risk_reduction_pct: number;
  steps: PlanStep[];
}

interface OptimizePlanResponse {
  district: string;
  model_version: string;
  baseline_score: number;
  baseline_risk_level: string;
  target_risk_level: string;
  target_score_threshold: number;
  target_reached: boolean;
  optimized_score: number;
  optimized_risk_level: string;
  risk_reduction_pct: number;
  recommended_plan: PlanStep[];
  search_trace: SearchTraceItem[];
  alternatives: AlternativePlan[];
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

function formatFeature(feature: string): string {
  return ACTIONABLE_FEATURES.find(item => item.key === feature)?.label
    ?? feature.replaceAll('_', ' ');
}

function PlanStepCard({ step, index }: { step: PlanStep; index: number }) {
  return (
    <div
      className="animate-in"
      style={{
        display: 'grid',
        gridTemplateColumns: '48px minmax(0, 1fr) auto',
        gap: '1rem',
        alignItems: 'center',
        padding: '1rem',
        borderRadius: '12px',
        background: 'rgba(255,255,255,0.025)',
        border: '1px solid var(--border-glass)',
        animationDelay: `${index * 80}ms`,
      }}
    >
      <div style={{
        width: '42px',
        height: '42px',
        borderRadius: '50%',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'rgba(20,184,166,0.12)',
        border: '1px solid rgba(20,184,166,0.3)',
        color: '#14b8a6',
        fontWeight: 800,
      }}>
        {index + 1}
      </div>

      <div style={{ minWidth: 0 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
          <strong style={{ color: 'var(--text-primary)' }}>{step.feature_label}</strong>
          <span style={{
            padding: '2px 7px',
            borderRadius: '5px',
            background: 'rgba(255,255,255,0.05)',
            color: 'var(--text-muted)',
            fontSize: '0.65rem',
            fontWeight: 700,
            textTransform: 'uppercase',
          }}>
            {step.category}
          </span>
          <span style={{
            padding: '2px 7px',
            borderRadius: '5px',
            background: 'rgba(245,158,11,0.1)',
            color: '#f59e0b',
            fontSize: '0.65rem',
            fontWeight: 700,
            textTransform: 'uppercase',
          }}>
            {step.cost_level} cost
          </span>
        </div>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem',
          margin: '0.4rem 0',
          color: 'var(--text-secondary)',
          fontVariantNumeric: 'tabular-nums',
        }}>
          <span>{step.from_value.toFixed(2)}</span>
          <ArrowRight size={14} />
          <span style={{ color: 'var(--text-primary)', fontWeight: 700 }}>{step.to_value.toFixed(2)}</span>
        </div>
        <p style={{ margin: 0, color: 'var(--text-muted)', fontSize: '0.78rem', lineHeight: 1.5 }}>
          {step.rationale}
        </p>
      </div>

      <div style={{ textAlign: 'right', minWidth: '110px' }}>
        <div style={{ color: '#22c55e', fontWeight: 800, fontSize: '1.1rem' }}>
          −{step.relative_reduction_pct.toFixed(2)}%
        </div>
        <div style={{ color: 'var(--text-muted)', fontSize: '0.72rem' }}>
          {step.score_before.toFixed(4)} → {step.score_after.toFixed(4)}
        </div>
      </div>
    </div>
  );
}

export default function OptimizePage() {
  const [district, setDistrict] = useState('Colombo');
  const [targetRiskLevel, setTargetRiskLevel] = useState<(typeof TARGET_LEVELS)[number]>('Medium');
  const [budgetProfile, setBudgetProfile] = useState('balanced');
  const [maxSteps, setMaxSteps] = useState(3);
  const [allowedFeatures, setAllowedFeatures] = useState<string[]>(
    ACTIONABLE_FEATURES.map(feature => feature.key),
  );
  const [result, setResult] = useState<OptimizePlanResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const toggleFeature = useCallback((feature: string) => {
    setAllowedFeatures(current => (
      current.includes(feature)
        ? current.filter(item => item !== feature)
        : [...current, feature]
    ));
  }, []);

  const generatePlan = useCallback(async () => {
    if (allowedFeatures.length === 0) {
      setError('Select at least one allowed intervention feature.');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch(`${API_URL}/api/optimize-plan`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          district,
          model_version: 'v13',
          target_risk_level: targetRiskLevel,
          max_steps: maxSteps,
          budget_profile: budgetProfile,
          allowed_features: allowedFeatures,
        }),
      });

      if (!response.ok) {
        const body = await response.json().catch(() => null);
        const detail = typeof body?.detail === 'string'
          ? body.detail
          : `Planner request failed with HTTP ${response.status}`;
        throw new Error(detail);
      }

      setResult(await response.json());
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : 'Unable to generate a mitigation plan.',
      );
    } finally {
      setLoading(false);
    }
  }, [allowedFeatures, budgetProfile, district, maxSteps, targetRiskLevel]);

  return (
    <main className="page-content">
      <div className="page-header animate-in">
        <h1>
          <Bot size={36} style={{ display: 'inline', verticalAlign: 'middle', marginRight: '0.5rem' }} />
          AI Mitigation Planner
        </h1>
        <p>
          Generate a model-guided counterfactual plan that balances flood-risk reduction,
          intervention cost, and the number of practical actions.
        </p>
      </div>

      <div className="simulation-grid">
        <div>
          <div className="glass-card animate-in" style={{ animationDelay: '100ms' }}>
            <div className="card-header">
              <div className="icon"><Target size={16} /></div>
              <h2>Planning Goal</h2>
            </div>

            <div className="select-group">
              <label htmlFor="optimize-district">District</label>
              <div className="select-wrapper">
                <select
                  id="optimize-district"
                  value={district}
                  onChange={event => setDistrict(event.target.value)}
                >
                  {ALL_DISTRICTS.map(item => <option key={item} value={item}>{item}</option>)}
                </select>
              </div>
            </div>

            <div className="select-group">
              <label htmlFor="target-risk">Target Risk Level</label>
              <div className="select-wrapper">
                <select
                  id="target-risk"
                  value={targetRiskLevel}
                  onChange={event => setTargetRiskLevel(event.target.value as (typeof TARGET_LEVELS)[number])}
                >
                  {TARGET_LEVELS.map(level => <option key={level} value={level}>{level}</option>)}
                </select>
              </div>
            </div>

            <div className="select-group">
              <label htmlFor="budget-profile">Budget Profile</label>
              <div className="select-wrapper">
                <select
                  id="budget-profile"
                  value={budgetProfile}
                  onChange={event => setBudgetProfile(event.target.value)}
                >
                  {BUDGET_PROFILES.map(profile => (
                    <option key={profile.value} value={profile.value}>{profile.label}</option>
                  ))}
                </select>
              </div>
              <div style={{ marginTop: '0.4rem', color: 'var(--text-muted)', fontSize: '0.72rem' }}>
                {BUDGET_PROFILES.find(profile => profile.value === budgetProfile)?.description}
              </div>
            </div>

            <div className="slider-group">
              <div className="slider-header">
                <label className="slider-label" htmlFor="max-steps">Maximum Plan Steps</label>
                <span className="slider-value">{maxSteps}</span>
              </div>
              <input
                id="max-steps"
                type="range"
                min={1}
                max={5}
                step={1}
                value={maxSteps}
                onChange={event => setMaxSteps(Number(event.target.value))}
                style={{ '--progress': `${((maxSteps - 1) / 4) * 100}%` } as React.CSSProperties}
              />
              <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-muted)', fontSize: '0.68rem' }}>
                <span>1 focused action</span>
                <span>5 actions</span>
              </div>
            </div>
          </div>

          <div className="glass-card animate-in" style={{ marginTop: '1.5rem', animationDelay: '180ms' }}>
            <div className="card-header">
              <div className="icon"><ListChecks size={16} /></div>
              <h2>Allowed Interventions</h2>
              <span style={{ marginLeft: 'auto', color: 'var(--text-muted)', fontSize: '0.72rem' }}>
                {allowedFeatures.length}/{ACTIONABLE_FEATURES.length} selected
              </span>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(210px, 1fr))', gap: '0.75rem' }}>
              {ACTIONABLE_FEATURES.map(feature => {
                const checked = allowedFeatures.includes(feature.key);
                return (
                  <label
                    key={feature.key}
                    htmlFor={`feature-${feature.key}`}
                    style={{
                      display: 'flex',
                      gap: '0.75rem',
                      alignItems: 'center',
                      padding: '0.8rem',
                      borderRadius: '10px',
                      cursor: 'pointer',
                      border: checked ? '1px solid rgba(20,184,166,0.4)' : '1px solid var(--border-glass)',
                      background: checked ? 'rgba(20,184,166,0.08)' : 'rgba(255,255,255,0.02)',
                      transition: 'all 0.2s ease',
                    }}
                  >
                    <input
                      id={`feature-${feature.key}`}
                      type="checkbox"
                      checked={checked}
                      onChange={() => toggleFeature(feature.key)}
                      style={{ width: '17px', height: '17px', accentColor: '#14b8a6' }}
                    />
                    <span>
                      <span style={{ display: 'block', color: 'var(--text-primary)', fontWeight: 650, fontSize: '0.84rem' }}>
                        {feature.label}
                      </span>
                      <span style={{ color: 'var(--text-muted)', fontSize: '0.68rem' }}>{feature.category}</span>
                    </span>
                  </label>
                );
              })}
            </div>
          </div>

          <button
            className={`btn-simulate ${loading ? 'loading' : ''}`}
            onClick={generatePlan}
            disabled={loading}
            style={{ marginTop: '1.5rem' }}
          >
            <Sparkles size={17} style={{ marginRight: '0.5rem' }} />
            {loading ? 'Generating Plan...' : 'Generate Mitigation Plan'}
          </button>

          {error && (
            <div style={{
              marginTop: '1rem',
              padding: '14px 16px',
              borderRadius: '10px',
              background: 'rgba(239,68,68,0.1)',
              border: '1px solid rgba(239,68,68,0.3)',
              color: '#ef4444',
            }}>
              <AlertTriangle size={15} style={{ display: 'inline', verticalAlign: 'middle', marginRight: '0.4rem' }} />
              {error}
            </div>
          )}
        </div>

        <div style={{ position: 'sticky', top: '80px' }}>
          <div className="glass-card animate-in" style={{ animationDelay: '140ms', textAlign: 'center' }}>
            <div className="card-header">
              <div className="icon"><ShieldCheck size={16} /></div>
              <h2>Planner Status</h2>
            </div>
            {result ? (
              <>
                {result.target_reached
                  ? <CheckCircle2 size={50} color="#22c55e" />
                  : <XCircle size={50} color="#f59e0b" />}
                <h3 style={{ color: result.target_reached ? '#22c55e' : '#f59e0b', margin: '0.75rem 0 0.25rem' }}>
                  {result.target_reached ? 'Target Reached' : 'Best Available Plan'}
                </h3>
                <p style={{ color: 'var(--text-muted)', margin: 0, fontSize: '0.8rem' }}>
                  Target: {result.target_risk_level} (&lt; {result.target_score_threshold.toFixed(2)})
                </p>
                <div className="section-divider" />
                <div style={{ display: 'flex', justifyContent: 'space-around', gap: '1rem' }}>
                  <div>
                    <div style={{ color: '#22c55e', fontWeight: 800, fontSize: '1.5rem' }}>
                      {result.risk_reduction_pct.toFixed(2)}%
                    </div>
                    <div style={{ color: 'var(--text-muted)', fontSize: '0.68rem', textTransform: 'uppercase' }}>
                      Risk Reduction
                    </div>
                  </div>
                  <div>
                    <div style={{ color: 'var(--text-primary)', fontWeight: 800, fontSize: '1.5rem' }}>
                      {result.recommended_plan.length}
                    </div>
                    <div style={{ color: 'var(--text-muted)', fontSize: '0.68rem', textTransform: 'uppercase' }}>
                      Plan Steps
                    </div>
                  </div>
                </div>
              </>
            ) : (
              <div style={{ padding: '2.5rem 1rem', color: 'var(--text-muted)' }}>
                <Bot size={48} style={{ opacity: 0.3, marginBottom: '0.75rem' }} />
                <p style={{ margin: 0 }}>Configure the planning goal and generate a plan.</p>
              </div>
            )}
          </div>
        </div>
      </div>

      {result && (
        <div className="animate-in" style={{ marginTop: '2rem' }}>
          <div className="simulation-grid" style={{ marginBottom: '2rem' }}>
            <div className="glass-card">
              <div className="card-header">
                <div className="icon"><AlertTriangle size={16} /></div>
                <h2>Baseline Risk</h2>
              </div>
              <RiskGauge
                score={result.baseline_score}
                riskLevel={result.baseline_risk_level}
                modelVersion={result.model_version}
                inferenceTimeMs={0}
              />
            </div>
            <div className="glass-card">
              <div className="card-header">
                <div className="icon"><ShieldCheck size={16} /></div>
                <h2>Optimized Risk</h2>
              </div>
              <RiskGauge
                score={result.optimized_score}
                riskLevel={result.optimized_risk_level}
                modelVersion={result.model_version}
                inferenceTimeMs={0}
              />
            </div>
          </div>

          <div className="glass-card" style={{ marginBottom: '2rem' }}>
            <div className="card-header">
              <div className="icon"><Route size={16} /></div>
              <h2>Recommended Mitigation Plan</h2>
              <span style={{ marginLeft: 'auto', color: getRiskColor(result.optimized_risk_level), fontWeight: 700, fontSize: '0.78rem' }}>
                {result.optimized_risk_level} risk
              </span>
            </div>

            {result.recommended_plan.length > 0 ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
                {result.recommended_plan.map((step, index) => (
                  <PlanStepCard key={`${step.feature}-${index}`} step={step} index={index} />
                ))}
              </div>
            ) : (
              <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)' }}>
                <CheckCircle2 size={38} color="#22c55e" style={{ marginBottom: '0.5rem' }} />
                <p style={{ margin: 0 }}>
                  No mitigation steps are required for this target, or no allowed action improved the score.
                </p>
              </div>
            )}
          </div>

          <details className="glass-card" style={{ marginBottom: '2rem' }}>
            <summary style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.75rem',
              cursor: 'pointer',
              listStyle: 'none',
              color: 'var(--text-primary)',
              fontWeight: 700,
            }}>
              <div className="icon"><Search size={16} /></div>
              Search Trace
              <span style={{ color: 'var(--text-muted)', fontSize: '0.75rem', fontWeight: 500 }}>
                {result.search_trace.length} candidates evaluated
              </span>
              <ChevronDown size={17} style={{ marginLeft: 'auto' }} />
            </summary>
            <div style={{ overflowX: 'auto', marginTop: '1.25rem' }}>
              {result.search_trace.length > 0 ? (
                <table className="logs-table">
                  <thead>
                    <tr>
                      <th>Step</th>
                      <th>Feature</th>
                      <th style={{ textAlign: 'right' }}>Tried Value</th>
                      <th style={{ textAlign: 'right' }}>Score</th>
                      <th>Risk</th>
                      <th>Decision</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.search_trace.map((item, index) => (
                      <tr key={`${item.step}-${item.tried_feature}-${item.tried_value}-${index}`}>
                        <td>{item.step}</td>
                        <td>{formatFeature(item.tried_feature)}</td>
                        <td style={{ textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>{item.tried_value.toFixed(3)}</td>
                        <td style={{ textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>{item.resulting_score.toFixed(4)}</td>
                        <td>
                          <span className={`risk-badge ${item.resulting_risk_level.toLowerCase()}`}>
                            {item.resulting_risk_level}
                          </span>
                        </td>
                        <td style={{ color: item.accepted ? '#22c55e' : 'var(--text-muted)', fontWeight: 700 }}>
                          {item.accepted ? 'Accepted' : 'Explored'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <p style={{ color: 'var(--text-muted)', textAlign: 'center' }}>
                  The baseline already met the target, so no candidates were evaluated.
                </p>
              )}
            </div>
          </details>

          {result.alternatives.length > 0 && (
            <div className="glass-card">
              <div className="card-header">
                <div className="icon"><CircleDollarSign size={16} /></div>
                <h2>Alternative Plans</h2>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '1rem' }}>
                {result.alternatives.map((alternative, index) => (
                  <details
                    key={`${alternative.name}-${index}`}
                    style={{
                      padding: '1rem',
                      borderRadius: '12px',
                      background: 'rgba(255,255,255,0.025)',
                      border: '1px solid var(--border-glass)',
                    }}
                  >
                    <summary style={{ cursor: 'pointer', listStyle: 'none' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '1rem', alignItems: 'start' }}>
                        <div>
                          <div style={{ color: 'var(--text-primary)', fontWeight: 700 }}>{alternative.name}</div>
                          <div style={{ color: 'var(--text-muted)', fontSize: '0.72rem', marginTop: '0.25rem' }}>
                            {alternative.steps.length} steps · {alternative.risk_reduction_pct.toFixed(2)}% reduction
                          </div>
                        </div>
                        <span className={`risk-badge ${alternative.optimized_risk_level.toLowerCase()}`}>
                          {alternative.optimized_score.toFixed(4)}
                        </span>
                      </div>
                    </summary>
                    <div style={{ marginTop: '1rem', borderTop: '1px solid var(--border-glass)', paddingTop: '0.75rem' }}>
                      {alternative.steps.length > 0 ? alternative.steps.map((step, stepIndex) => (
                        <div key={`${step.feature}-${stepIndex}`} style={{ padding: '0.45rem 0', color: 'var(--text-secondary)', fontSize: '0.78rem' }}>
                          <strong style={{ color: 'var(--text-primary)' }}>{stepIndex + 1}. {step.feature_label}</strong>
                          {' '}{step.from_value.toFixed(2)} → {step.to_value.toFixed(2)}
                        </div>
                      )) : (
                        <p style={{ color: 'var(--text-muted)', fontSize: '0.78rem', margin: 0 }}>
                          No alternative action sequence was available.
                        </p>
                      )}
                    </div>
                  </details>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </main>
  );
}
