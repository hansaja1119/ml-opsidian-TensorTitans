'use client';

interface FeatureSliderProps {
  id: string;
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  unit?: string;
  onChange: (value: number) => void;
}

export default function FeatureSlider({
  id,
  label,
  value,
  min,
  max,
  step,
  unit = '',
  onChange,
}: FeatureSliderProps) {
  const progress = ((value - min) / (max - min)) * 100;

  return (
    <div className="slider-group">
      <div className="slider-header">
        <label htmlFor={id} className="slider-label">{label}</label>
        <span className="slider-value">
          {typeof value === 'number' ? value.toFixed(step < 1 ? 2 : 0) : value}{unit}
        </span>
      </div>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        style={{ '--progress': `${progress}%` } as React.CSSProperties}
      />
      <div className="slider-bounds">
        <span>{min}{unit}</span>
        <span>{max}{unit}</span>
      </div>
    </div>
  );
}
