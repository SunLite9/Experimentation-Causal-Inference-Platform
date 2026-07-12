import { useEffect, useRef, useState } from 'react'
import './App.css'
import {
  fetchFlagship,
  simulateObservational,
  simulateRandomized,
  uploadObservational,
  uploadRandomized,
} from './api'
import { CausalResults } from './components/CausalResults'
import { FlagshipBanner } from './components/FlagshipBanner'
import { ObservationalControls } from './components/ObservationalControls'
import { RandomizedControls } from './components/RandomizedControls'
import { RandomizedResults } from './components/RandomizedResults'
import type { CausalAnalysis, RandomizedAnalysis, SimulateObservationalParams, SimulateRandomizedParams } from './types'

type StudyType = 'randomized' | 'observational'
type DataMode = 'simulate' | 'upload'

const DEFAULT_RANDOMIZED: SimulateRandomizedParams = {
  n_per_arm: 5000,
  true_effect: 2.0,
  baseline_mean: 100.0,
  baseline_std: 20.0,
  extra_noise_std: 0,
  extra_noise_correlation: 0,
  covariate_correlation: 0.7,
  seed: 42,
  include_peeking: false,
  checkpoint_size: 100,
}

const DEFAULT_OBSERVATIONAL: SimulateObservationalParams = {
  n: 10000,
  true_effect: 5.0,
  confounding_strength: 2.0,
  caliper: 0.05,
  seed: 42,
}

function App() {
  const [flagship, setFlagship] = useState(false)
  const [studyType, setStudyType] = useState<StudyType>('randomized')
  const [dataMode, setDataMode] = useState<DataMode>('simulate')

  const [randomizedParams, setRandomizedParams] = useState(DEFAULT_RANDOMIZED)
  const [observationalParams, setObservationalParams] = useState(DEFAULT_OBSERVATIONAL)

  const [randomizedResult, setRandomizedResult] = useState<RandomizedAnalysis | null>(null)
  const [causalResult, setCausalResult] = useState<CausalAnalysis | null>(null)
  const [flagshipResult, setFlagshipResult] = useState<RandomizedAnalysis | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    if (!flagship) return
    setLoading(true)
    setError(null)
    fetchFlagship()
      .then(setFlagshipResult)
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false))
  }, [flagship])

  useEffect(() => {
    if (flagship || dataMode !== 'simulate') return
    if (debounceRef.current) clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(() => {
      setLoading(true)
      setError(null)
      const request =
        studyType === 'randomized' ? simulateRandomized(randomizedParams) : simulateObservational(observationalParams)
      request
        .then((result) => {
          if (studyType === 'randomized') setRandomizedResult(result as RandomizedAnalysis)
          else setCausalResult(result as CausalAnalysis)
        })
        .catch((e: Error) => setError(e.message))
        .finally(() => setLoading(false))
    }, 250)
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [flagship, studyType, dataMode, randomizedParams, observationalParams])

  function handleFileSelected(file: File) {
    setLoading(true)
    setError(null)
    const request =
      studyType === 'randomized'
        ? uploadRandomized(file, randomizedParams.include_peeking ? randomizedParams.checkpoint_size : null)
        : uploadObservational(file, observationalParams.caliper)
    request
      .then((result) => {
        if (studyType === 'randomized') setRandomizedResult(result as RandomizedAnalysis)
        else setCausalResult(result as CausalAnalysis)
      })
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false))
  }

  function selectStudyType(next: StudyType) {
    setStudyType(next)
    setDataMode('simulate')
  }

  return (
    <div className="app-shell">
      <header className="app-header">
        <h1>Experimentation Causal Inference Platform</h1>
        <p>
          Simulate or upload experiment data and compare the naive analysis against the corrected analysis —
          CUPED, sequential testing, or propensity matching — side by side.
        </p>
      </header>

      <div className="app-body">
        <aside className="sidebar">
          <div className="sidebar-section">
            <h2>Flagship demo</h2>
            {!flagship ? (
              <button type="button" className="btn btn--primary" onClick={() => setFlagship(true)}>
                Load flagship demo
              </button>
            ) : (
              <button type="button" className="btn" onClick={() => setFlagship(false)}>
                Clear flagship demo
              </button>
            )}
          </div>

          <div className="sidebar-section">
            <h2>Study type</h2>
            <label className="radio-row">
              <input
                type="radio"
                checked={studyType === 'randomized'}
                disabled={flagship}
                onChange={() => selectStudyType('randomized')}
              />
              Randomized experiment
            </label>
            <label className="radio-row">
              <input
                type="radio"
                checked={studyType === 'observational'}
                disabled={flagship}
                onChange={() => selectStudyType('observational')}
              />
              Observational (non-randomized)
            </label>
          </div>

          <div className="sidebar-section">
            <h2>Data</h2>
            {studyType === 'randomized' ? (
              <RandomizedControls
                dataMode={dataMode}
                onDataModeChange={setDataMode}
                params={randomizedParams}
                onParamsChange={setRandomizedParams}
                onFileSelected={handleFileSelected}
                disabled={flagship}
              />
            ) : (
              <ObservationalControls
                dataMode={dataMode}
                onDataModeChange={setDataMode}
                params={observationalParams}
                onParamsChange={setObservationalParams}
                onFileSelected={handleFileSelected}
              />
            )}
          </div>
        </aside>

        <main className="main-content">
          {loading && <p className="status-line">Loading…</p>}
          {error && <p className="status-line status-line--error">{error}</p>}

          {flagship && flagshipResult && (
            <>
              <FlagshipBanner />
              <RandomizedResults analysis={flagshipResult} />
            </>
          )}

          {!flagship && studyType === 'randomized' && randomizedResult && <RandomizedResults analysis={randomizedResult} />}
          {!flagship && studyType === 'observational' && causalResult && <CausalResults analysis={causalResult} />}

          {!flagship && dataMode === 'upload' && studyType === 'randomized' && !randomizedResult && (
            <p className="status-line">Upload a CSV with columns: group (control/treatment), outcome, pre_covariate.</p>
          )}
          {!flagship && dataMode === 'upload' && studyType === 'observational' && !causalResult && (
            <p className="status-line">Upload a CSV with columns: treatment (0/1), outcome, covariate_1, covariate_2, ...</p>
          )}
        </main>
      </div>
    </div>
  )
}

export default App
