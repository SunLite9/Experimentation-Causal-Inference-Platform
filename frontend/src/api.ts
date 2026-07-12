import type {
  CausalAnalysis,
  RandomizedAnalysis,
  SimulateObservationalParams,
  SimulateRandomizedParams,
} from './types'

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(body.detail ?? `Request failed (${res.status})`)
  }
  return res.json() as Promise<T>
}

export function simulateRandomized(params: SimulateRandomizedParams): Promise<RandomizedAnalysis> {
  return fetch('/api/randomized/simulate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  }).then((res) => handle<RandomizedAnalysis>(res))
}

export function uploadRandomized(file: File, checkpointSize: number | null): Promise<RandomizedAnalysis> {
  const form = new FormData()
  form.append('file', file)
  const query = checkpointSize ? `?checkpoint_size=${checkpointSize}` : ''
  return fetch(`/api/randomized/upload${query}`, { method: 'POST', body: form }).then((res) =>
    handle<RandomizedAnalysis>(res),
  )
}

export function simulateObservational(params: SimulateObservationalParams): Promise<CausalAnalysis> {
  return fetch('/api/observational/simulate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  }).then((res) => handle<CausalAnalysis>(res))
}

export function uploadObservational(file: File, caliper: number): Promise<CausalAnalysis> {
  const form = new FormData()
  form.append('file', file)
  return fetch(`/api/observational/upload?caliper=${caliper}`, { method: 'POST', body: form }).then((res) =>
    handle<CausalAnalysis>(res),
  )
}

export function fetchFlagship(): Promise<RandomizedAnalysis> {
  return fetch('/api/flagship').then((res) => handle<RandomizedAnalysis>(res))
}
