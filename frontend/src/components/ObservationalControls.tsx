import type { SimulateObservationalParams } from '../types'
import { SliderField } from './SliderField'

interface Props {
  dataMode: 'simulate' | 'upload'
  onDataModeChange: (mode: 'simulate' | 'upload') => void
  params: SimulateObservationalParams
  onParamsChange: (params: SimulateObservationalParams) => void
  onFileSelected: (file: File) => void
}

export function ObservationalControls({ dataMode, onDataModeChange, params, onParamsChange, onFileSelected }: Props) {
  function set<K extends keyof SimulateObservationalParams>(key: K, value: SimulateObservationalParams[K]) {
    onParamsChange({ ...params, [key]: value })
  }

  return (
    <div className="controls">
      <label className="radio-row">
        <input type="radio" checked={dataMode === 'simulate'} onChange={() => onDataModeChange('simulate')} />
        Simulate
      </label>
      <label className="radio-row">
        <input type="radio" checked={dataMode === 'upload'} onChange={() => onDataModeChange('upload')} />
        Upload CSV
      </label>

      {dataMode === 'upload' ? (
        <div className="field">
          <label htmlFor="observational-file">CSV columns: treatment, outcome, covariate_1, covariate_2, ...</label>
          <input
            id="observational-file"
            type="file"
            accept=".csv"
            onChange={(e) => e.target.files?.[0] && onFileSelected(e.target.files[0])}
          />
        </div>
      ) : (
        <>
          <SliderField label="Sample size" value={params.n} min={500} max={20000} step={500} onChange={(v) => set('n', v)} />
          <SliderField label="True effect" value={params.true_effect} min={-10} max={10} step={0.5} onChange={(v) => set('true_effect', v)} />
          <SliderField
            label="Confounding strength"
            value={params.confounding_strength}
            min={0}
            max={5}
            step={0.1}
            onChange={(v) => set('confounding_strength', v)}
          />
          <div className="field">
            <label htmlFor="obs-seed">Random seed</label>
            <input id="obs-seed" type="number" value={params.seed} onChange={(e) => set('seed', Number(e.target.value))} />
          </div>
        </>
      )}

      <SliderField label="Matching caliper" value={params.caliper} min={0.01} max={0.5} step={0.01} onChange={(v) => set('caliper', v)} />
    </div>
  )
}
