export function FlagshipBanner() {
  return (
    <div className="banner">
      <strong>Flagship demo loaded.</strong> This scenario simulates a genuine +1.5
      treatment effect that's being masked by noise unrelated to the treatment (a
      pre-experiment covariate captures 92% of that noise). The naive t-test on the
      raw outcome fails to reach significance and would say <strong>don't ship</strong> —
      killing a real, working feature. CUPED strips out the noise the covariate
      already explained and reveals the same effect as statistically significant:{' '}
      <strong>ship</strong>. Nothing about the treatment effect itself changed — only
      the noise around it was removed. Full write-up in the README's "Flagship demo" section.
    </div>
  )
}
