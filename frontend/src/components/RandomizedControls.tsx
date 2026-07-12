import type { SimulateRandomizedParams } from '../types'
import { SliderField } from './SliderField'

interface Props {
  dataMode: 'simulate' | 'upload'
  onDataModeChange: (mode: 'simulate' | 'upload') => void
  params: SimulateRandomizedParams
  onParamsChange: (params: SimulateRandomizedParams) => void
  onFileSelected: (file: File) => void
  disabled: boolean
}

export function RandomizedControls({ dataMode, onDataModeChange, params, onParamsChange, onFileSelected, disabled }: Props) {
  function set<K extends keyof SimulateRandomizedParams>(key: K, value: SimulateRandomizedParams[K]) {
    onParamsChange({ ...params, [key]: value })
  }

  return (
    <fieldset disabled={disabled} className="controls">
      <label className="radio-row">
        <input type="radio" checked={dataMode === 'simulate'} onChange={() => onDataModeChange('simulate')} />
        Simulate
      </label>
      <label className="radio-row">
        <input type="radio" checked={dataMode === 'upload'} onChange={() => onDataModeChange('upload')} />
        Upload CSV
      </label>

      {dataMode === 'upload' && (
        <div className="field">
          <label htmlFor="randomized-file">CSV columns: group, outcome, pre_covariate</label>
          <input
            id="randomized-file"
            type="file"
            accept=".csv"
            onChange={(e) => e.target.files?.[0] && onFileSelected(e.target.files[0])}
          />
        </div>
      )}

      {dataMode === 'simulate' && (
        <>
          <SliderField label="Sample size per arm" value={params.n_per_arm} min={100} max={20000} step={100} onChange={(v) => set('n_per_arm', v)} />
          <SliderField label="True effect" value={params.true_effect} min={-10} max={10} step={0.5} onChange={(v) => set('true_effect', v)} />
          <SliderField label="Baseline std dev" value={params.baseline_std} min={1} max={50} step={1} onChange={(v) => set('baseline_std', v)} />
          <SliderField
            label="Extra treatment-unrelated noise (std dev)"
            value={params.extra_noise_std}
            min={0}
            max={50}
            step={1}
            onChange={(v) => set('extra_noise_std', v)}
          />
          <SliderField
            label="Covariate correlation with extra noise"
            value={params.extra_noise_correlation}
            min={0}
            max={1}
            step={0.05}
            onChange={(v) => set('extra_noise_correlation', v)}
          />
          <div className="field">
            <label htmlFor="seed">Random seed</label>
            <input id="seed" type="number" value={params.seed} onChange={(e) => set('seed', Number(e.target.value))} />
          </div>
        </>
      )}

      <label className="checkbox-row">
        <input type="checkbox" checked={params.include_peeking} onChange={(e) => set('include_peeking', e.target.checked)} />
        Include peeking scenario
      </label>
      {params.include_peeking && (
        <SliderField label="Peeking checkpoint size" value={params.checkpoint_size} min={10} max={500} step={10} onChange={(v) => set('checkpoint_size', v)} />
      )}
    </fieldset>
  )
}
