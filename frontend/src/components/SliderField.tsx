interface SliderFieldProps {
  label: string
  value: number
  min: number
  max: number
  step: number
  onChange: (v: number) => void
}

export function SliderField({ label, value, min, max, step, onChange }: SliderFieldProps) {
  return (
    <div className="field">
      <label>
        {label}: <span className="tabular">{value}</span>
      </label>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} />
    </div>
  )
}
