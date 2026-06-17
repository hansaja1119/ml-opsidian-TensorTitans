'use client';

import { useMemo } from 'react';

interface RiskGaugeProps {
  score: number;
  riskLevel: string;
  modelVersion: string;
  inferenceTimeMs: number;
}

export default function RiskGauge({
  score,
  riskLevel,
  modelVersion,
  inferenceTimeMs,
}: RiskGaugeProps) {
  const { color, glowColor } = useMemo(() => {
    if (score < 0.25) return { color: '#22c55e', glowColor: 'rgba(34, 197, 94, 0.3)' };
    if (score < 0.50) return { color: '#f59e0b', glowColor: 'rgba(245, 158, 11, 0.3)' };
    if (score < 0.75) return { color: '#f97316', glowColor: 'rgba(249, 115, 22, 0.3)' };
    return { color: '#ef4444', glowColor: 'rgba(239, 68, 68, 0.3)' };
  }, [score]);

  // SVG arc calculations for a semi-circle gauge
  const radius = 90;
  const circumference = Math.PI * radius; // half circle
  const offset = circumference - (score * circumference);

  return (
    <div className="risk-gauge-container">
      <svg
        className="risk-gauge-svg"
        width="240"
        height="150"
        viewBox="0 0 240 150"
        style={{ '--gauge-color': glowColor } as React.CSSProperties}
      >
        {/* Background arc */}
        <path
          className="gauge-bg"
          d="M 30 130 A 90 90 0 0 1 210 130"
          fill="none"
          strokeWidth="12"
          strokeLinecap="round"
        />
        {/* Filled arc */}
        <path
          className="gauge-fill"
          d="M 30 130 A 90 90 0 0 1 210 130"
          fill="none"
          stroke={color}
          strokeWidth="12"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
        />
        {/* Score text */}
        <text
          className="gauge-score-text"
          x="120"
          y="110"
        >
          {score.toFixed(4)}
        </text>
        {/* Risk level label */}
        <text
          className="gauge-label-text"
          x="120"
          y="135"
          fill={color}
        >
          {riskLevel}
        </text>
      </svg>

      <div className="risk-meta">
        <div className="risk-meta-item">
          <div className="label">Model</div>
          <div className="value">{modelVersion.toUpperCase()}</div>
        </div>
        <div className="risk-meta-item">
          <div className="label">Latency</div>
          <div className="value">{inferenceTimeMs.toFixed(1)}ms</div>
        </div>
      </div>
    </div>
  );
}
