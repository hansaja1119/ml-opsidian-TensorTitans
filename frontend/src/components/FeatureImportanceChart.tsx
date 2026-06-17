'use client';

interface FeatureImportanceItem {
  feature: string;
  importance: number;
}

interface FeatureImportanceChartProps {
  data: FeatureImportanceItem[];
}

const FEATURE_DISPLAY_NAMES: Record<string, string> = {
  rainfall_7d_mm: '7-Day Rainfall',
  drainage_index: 'Drainage Index',
  extreme_weather_index: 'Extreme Weather',
  elevation_m: 'Elevation',
  distance_to_river_m: 'River Distance',
  infrastructure_score: 'Infrastructure',
  historical_flood_count: 'Flood History',
  monthly_rainfall_mm: 'Monthly Rainfall',
  nearest_hospital_km: 'Hospital Dist.',
  built_up_percent: 'Built-Up %',
};

export default function FeatureImportanceChart({ data }: FeatureImportanceChartProps) {
  const maxImportance = Math.max(...data.map(d => d.importance), 0.01);

  return (
    <div className="importance-chart">
      {data.map((item, index) => (
        <div
          key={item.feature}
          className="importance-bar-row animate-in"
          style={{ animationDelay: `${index * 60}ms` }}
        >
          <span className="importance-label">
            {FEATURE_DISPLAY_NAMES[item.feature] || item.feature}
          </span>
          <div className="importance-bar-container">
            <div
              className="importance-bar"
              style={{ width: `${(item.importance / maxImportance) * 100}%` }}
            />
          </div>
          <span className="importance-value">
            {(item.importance * 100).toFixed(1)}%
          </span>
        </div>
      ))}
    </div>
  );
}
