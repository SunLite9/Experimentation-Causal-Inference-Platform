import type { RandomizedAnalysis } from '../types'
import { PeekingChart } from './PeekingChart'
import { VerdictBadge } from './VerdictBadge'

interface Props {
  analysis: RandomizedAnalysis
}

export function RandomizedResults({ analysis }: Props) {
  const { srm, naive, cuped, variance_reduction_pct, true_effect, peeking } = analysis

  return (
    <div>
      {srm.srm_detected && (
        <div className="banner banner--critical">
          <strong>Sample ratio mismatch detected.</strong> Observed split: {srm.n_control} control /{' '}
          {srm.n_treatment} treatment (expected {(srm.expected_ratio * 100).toFixed(0)}% treatment,
          chi-square p = {srm.p_value < 0.0001 ? srm.p_value.toExponential(2) : srm.p_value.toFixed(4)}).
          The actual split doesn't match the intended allocation — this usually means randomization,
          logging, or filtering is broken somewhere upstream. The results below are not trustworthy
          until this is investigated, regardless of what their own p-values say.
        </div>
      )}

      {true_effect !== null && (
        <p className="true-effect-caption">
          True effect (ground truth): <span className="tabular">{true_effect.toFixed(4)}</span>
        </p>
      )}

      <div className={srm.srm_detected ? 'results-suppressed' : undefined}>
        <div className="results-grid">
          <div className="card">
            <h3>Naive t-test</h3>
            <p className="metric tabular">{naive.effect.toFixed(4)}</p>
            <dl className="stat-list">
              <div>
                <dt>95% CI</dt>
                <dd className="tabular">
                  [{naive.ci_lower.toFixed(4)}, {naive.ci_upper.toFixed(4)}]
                </dd>
              </div>
              <div>
                <dt>p-value</dt>
                <dd className="tabular">{naive.p_value.toFixed(4)}</dd>
              </div>
            </dl>
            <VerdictBadge significant={naive.significant} effect={naive.effect} withheld={srm.srm_detected} />
          </div>

          <div className="card">
            <h3>CUPED-adjusted t-test</h3>
            <p className="metric tabular">{cuped.effect.toFixed(4)}</p>
            <dl className="stat-list">
              <div>
                <dt>95% CI</dt>
                <dd className="tabular">
                  [{cuped.ci_lower.toFixed(4)}, {cuped.ci_upper.toFixed(4)}]
                </dd>
              </div>
              <div>
                <dt>p-value</dt>
                <dd className="tabular">{cuped.p_value.toFixed(4)}</dd>
              </div>
              <div>
                <dt>Variance reduction</dt>
                <dd className="tabular">{variance_reduction_pct.toFixed(1)}%</dd>
              </div>
            </dl>
            <VerdictBadge significant={cuped.significant} effect={cuped.effect} withheld={srm.srm_detected} />
          </div>
        </div>

        {peeking && (
          <div className="card card--wide">
            <h3>Peeking check</h3>
            <p>p-value at each look as data accumulates.</p>
            <PeekingChart checkpoints={peeking.checkpoints} alpha={0.05} />
            <div className="results-grid" style={{ marginTop: 16 }}>
              <div>
                <p className="stat-list-label">If you peeked and stopped at the first significant naive check:</p>
                {srm.srm_detected ? (
                  <span className="badge badge--neutral">Verdict withheld — SRM detected</span>
                ) : peeking.naive_first_flag_n !== null ? (
                  <span className="badge badge--critical">
                    Ship — crossed p &lt; 0.05 at n={peeking.naive_first_flag_n}/arm
                  </span>
                ) : (
                  <span className="badge badge--neutral">Never crossed significance — don't ship</span>
                )}
              </div>
              <div>
                <p className="stat-list-label">Sequential (mSPRT) test, checked at the same points:</p>
                {srm.srm_detected ? (
                  <span className="badge badge--neutral">Verdict withheld — SRM detected</span>
                ) : peeking.sequential_first_flag_n !== null ? (
                  <span className="badge badge--critical">
                    Ship — crossed p &lt; 0.05 at n={peeking.sequential_first_flag_n}/arm
                  </span>
                ) : (
                  <span className="badge badge--good">Never crossed significance — correctly don't ship</span>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
