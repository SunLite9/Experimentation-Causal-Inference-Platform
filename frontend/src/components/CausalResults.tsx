import type { CausalAnalysis } from '../types'
import { VerdictBadge } from './VerdictBadge'

interface Props {
  analysis: CausalAnalysis
}

export function CausalResults({ analysis }: Props) {
  const { naive_effect, matched_effect, matched_ci_lower, matched_ci_upper, n_matched, n_treated, true_effect } = analysis
  const matchedSignificant = matched_ci_lower > 0 || matched_ci_upper < 0

  return (
    <div>
      {true_effect !== null && (
        <p className="true-effect-caption">
          True effect (ground truth): <span className="tabular">{true_effect.toFixed(4)}</span>
        </p>
      )}
      <div className="results-grid">
        <div className="card">
          <h3>Naive treated-vs-control comparison</h3>
          <p>Ignores confounding — biased if treatment wasn't randomized.</p>
          <p className="metric tabular">{naive_effect.toFixed(4)}</p>
          <p className="stat-list-label">(no valid CI without adjusting for confounding — shown for comparison only)</p>
        </div>

        <div className="card">
          <h3>Propensity score matching</h3>
          <p className="metric tabular">{matched_effect.toFixed(4)}</p>
          <dl className="stat-list">
            <div>
              <dt>95% CI</dt>
              <dd className="tabular">
                [{matched_ci_lower.toFixed(4)}, {matched_ci_upper.toFixed(4)}]
              </dd>
            </div>
            <div>
              <dt>Matched pairs</dt>
              <dd className="tabular">
                {n_matched} / {n_treated} treated units
              </dd>
            </div>
          </dl>
          <VerdictBadge significant={matchedSignificant} effect={matched_effect} />
        </div>
      </div>
    </div>
  )
}
