interface VerdictBadgeProps {
  significant: boolean
  effect: number
  withheld?: boolean
}

export function VerdictBadge({ significant, effect, withheld }: VerdictBadgeProps) {
  if (withheld) {
    return (
      <span className="badge badge--neutral">
        <span aria-hidden="true">?</span> Verdict withheld — SRM detected
      </span>
    )
  }
  if (significant && effect > 0) {
    return (
      <span className="badge badge--good">
        <span aria-hidden="true">✓</span> Ship
      </span>
    )
  }
  if (significant && effect < 0) {
    return (
      <span className="badge badge--critical">
        <span aria-hidden="true">✕</span> Don't ship — negative effect
      </span>
    )
  }
  return (
    <span className="badge badge--neutral">
      <span aria-hidden="true">–</span> Don't ship — not significant
    </span>
  )
}
