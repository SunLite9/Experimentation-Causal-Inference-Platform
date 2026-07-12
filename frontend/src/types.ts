export interface TTestResult {
  effect: number
  ci_lower: number
  ci_upper: number
  p_value: number
  significant: boolean
}

export interface PeekingCheckpoint {
  n_per_arm: number
  naive_p_value: number
  sequential_p_value: number
}

export interface PeekingResult {
  checkpoints: PeekingCheckpoint[]
  naive_first_flag_n: number | null
  sequential_first_flag_n: number | null
}

export interface SRMCheck {
  n_control: number
  n_treatment: number
  expected_ratio: number
  p_value: number
  srm_detected: boolean
}

export interface RandomizedAnalysis {
  true_effect: number | null
  srm: SRMCheck
  naive: TTestResult
  cuped: TTestResult
  variance_reduction_pct: number
  peeking: PeekingResult | null
}

export interface CausalAnalysis {
  true_effect: number | null
  naive_effect: number
  matched_effect: number
  matched_ci_lower: number
  matched_ci_upper: number
  n_matched: number
  n_treated: number
}

export interface SimulateRandomizedParams {
  n_per_arm: number
  true_effect: number
  baseline_mean: number
  baseline_std: number
  extra_noise_std: number
  extra_noise_correlation: number
  covariate_correlation: number
  seed: number
  include_peeking: boolean
  checkpoint_size: number
}

export interface SimulateObservationalParams {
  n: number
  true_effect: number
  confounding_strength: number
  caliper: number
  seed: number
}
