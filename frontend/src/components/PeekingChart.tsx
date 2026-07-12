import { useState } from 'react'
import type { PeekingCheckpoint } from '../types'

interface PeekingChartProps {
  checkpoints: PeekingCheckpoint[]
  alpha: number
}

const WIDTH = 680
const HEIGHT = 300
const MARGIN = { top: 16, right: 56, bottom: 32, left: 40 }
const PLOT_W = WIDTH - MARGIN.left - MARGIN.right
const PLOT_H = HEIGHT - MARGIN.top - MARGIN.bottom

export function PeekingChart({ checkpoints, alpha }: PeekingChartProps) {
  const [hoverIndex, setHoverIndex] = useState<number | null>(null)

  if (checkpoints.length === 0) return null

  const minN = checkpoints[0].n_per_arm
  const maxN = checkpoints[checkpoints.length - 1].n_per_arm
  const xScale = (n: number) =>
    MARGIN.left + (maxN === minN ? 0 : ((n - minN) / (maxN - minN)) * PLOT_W)
  const yScale = (p: number) => MARGIN.top + (1 - Math.min(p, 1)) * PLOT_H

  const naivePath = checkpoints
    .map((c, i) => `${i === 0 ? 'M' : 'L'} ${xScale(c.n_per_arm)} ${yScale(c.naive_p_value)}`)
    .join(' ')
  const sequentialPath = checkpoints
    .map((c, i) => `${i === 0 ? 'M' : 'L'} ${xScale(c.n_per_arm)} ${yScale(c.sequential_p_value)}`)
    .join(' ')

  const yTicks = [0, 0.25, 0.5, 0.75, 1]
  const last = checkpoints[checkpoints.length - 1]
  const hovered = hoverIndex !== null ? checkpoints[hoverIndex] : null

  function handleMove(e: React.MouseEvent<SVGRectElement>) {
    const rect = e.currentTarget.getBoundingClientRect()
    const relX = e.clientX - rect.left
    const frac = Math.min(1, Math.max(0, relX / PLOT_W))
    const targetN = minN + frac * (maxN - minN)
    let closest = 0
    let closestDist = Infinity
    checkpoints.forEach((c, i) => {
      const dist = Math.abs(c.n_per_arm - targetN)
      if (dist < closestDist) {
        closestDist = dist
        closest = i
      }
    })
    setHoverIndex(closest)
  }

  return (
    <div className="peeking-chart">
      <div className="chart-legend">
        <span className="legend-item">
          <span className="legend-swatch" style={{ background: 'var(--series-naive)' }} />
          Naive t-test p-value
        </span>
        <span className="legend-item">
          <span className="legend-swatch" style={{ background: 'var(--series-sequential)' }} />
          Sequential (mSPRT) p-value
        </span>
      </div>
      <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img" aria-label="p-value at each look as data accumulates">
        {yTicks.map((t) => (
          <g key={t}>
            <line
              x1={MARGIN.left}
              x2={WIDTH - MARGIN.right}
              y1={yScale(t)}
              y2={yScale(t)}
              stroke="var(--gridline)"
              strokeWidth={1}
            />
            <text x={MARGIN.left - 8} y={yScale(t) + 4} textAnchor="end" className="axis-label">
              {t}
            </text>
          </g>
        ))}

        <line
          x1={MARGIN.left}
          x2={WIDTH - MARGIN.right}
          y1={yScale(alpha)}
          y2={yScale(alpha)}
          stroke="var(--text-muted)"
          strokeWidth={1}
          strokeDasharray="4 4"
        />
        <text x={WIDTH - MARGIN.right + 6} y={yScale(alpha) + 4} className="axis-label">
          α={alpha}
        </text>

        <line
          x1={MARGIN.left}
          x2={MARGIN.left}
          y1={MARGIN.top}
          y2={HEIGHT - MARGIN.bottom}
          stroke="var(--baseline)"
          strokeWidth={1}
        />
        <line
          x1={MARGIN.left}
          x2={WIDTH - MARGIN.right}
          y1={HEIGHT - MARGIN.bottom}
          y2={HEIGHT - MARGIN.bottom}
          stroke="var(--baseline)"
          strokeWidth={1}
        />
        <text x={MARGIN.left} y={HEIGHT - 6} className="axis-label">
          {minN}
        </text>
        <text x={WIDTH - MARGIN.right} y={HEIGHT - 6} textAnchor="end" className="axis-label">
          {maxN} samples/arm
        </text>

        <path d={naivePath} fill="none" stroke="var(--series-naive)" strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
        <path
          d={sequentialPath}
          fill="none"
          stroke="var(--series-sequential)"
          strokeWidth={2}
          strokeLinejoin="round"
          strokeLinecap="round"
        />

        <circle cx={xScale(last.n_per_arm)} cy={yScale(last.naive_p_value)} r={5} fill="var(--series-naive)" stroke="var(--surface-1)" strokeWidth={2} />
        <circle
          cx={xScale(last.n_per_arm)}
          cy={yScale(last.sequential_p_value)}
          r={5}
          fill="var(--series-sequential)"
          stroke="var(--surface-1)"
          strokeWidth={2}
        />

        {hovered && (
          <g>
            <line
              x1={xScale(hovered.n_per_arm)}
              x2={xScale(hovered.n_per_arm)}
              y1={MARGIN.top}
              y2={HEIGHT - MARGIN.bottom}
              stroke="var(--text-muted)"
              strokeWidth={1}
            />
            <circle cx={xScale(hovered.n_per_arm)} cy={yScale(hovered.naive_p_value)} r={5} fill="var(--series-naive)" stroke="var(--surface-1)" strokeWidth={2} />
            <circle
              cx={xScale(hovered.n_per_arm)}
              cy={yScale(hovered.sequential_p_value)}
              r={5}
              fill="var(--series-sequential)"
              stroke="var(--surface-1)"
              strokeWidth={2}
            />
          </g>
        )}

        <rect
          x={MARGIN.left}
          y={MARGIN.top}
          width={PLOT_W}
          height={PLOT_H}
          fill="transparent"
          onMouseMove={handleMove}
          onMouseLeave={() => setHoverIndex(null)}
        />
      </svg>
      {hovered && (
        <div className="chart-tooltip">
          <strong>{hovered.n_per_arm} samples/arm</strong>
          <span>
            <span className="legend-swatch" style={{ background: 'var(--series-naive)' }} /> naive p = {hovered.naive_p_value.toFixed(4)}
          </span>
          <span>
            <span className="legend-swatch" style={{ background: 'var(--series-sequential)' }} /> sequential p ={' '}
            {hovered.sequential_p_value.toFixed(4)}
          </span>
        </div>
      )}
    </div>
  )
}
