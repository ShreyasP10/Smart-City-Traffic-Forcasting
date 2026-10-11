import { useEffect, useRef, useState } from 'react'

export function CountUp({ value, duration = 600, decimals = 0, suffix = '', prefix = '' }) {
  const [shown, setShown] = useState(value)
  const from = useRef(value)
  useEffect(() => {
    const start = from.current
    if (start === value) return
    let raf
    const t0 = performance.now()
    const tick = (t) => {
      const p = Math.min(1, (t - t0) / duration)
      const e = 1 - Math.pow(1 - p, 3)
      setShown(start + (value - start) * e)
      if (p < 1) raf = requestAnimationFrame(tick)
      else from.current = value
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [value, duration])
  const txt = decimals ? shown.toFixed(decimals) : Math.round(shown).toLocaleString()
  return <span>{prefix}{txt}{suffix}</span>
}

export function LevelBadge({ level, colors }) {
  const hex = (colors && colors[level]) || '#6b7280'
  return (
    <span className="badge" style={{ background: hex }}>
      {level} {level === 'Severe' ? '🔴' : level === 'High' ? '🟠' : level === 'Moderate' ? '🟡' : '🟢'}
    </span>
  )
}

export function GaugeNeedle({ ratio }) {
  const c = Math.max(0, Math.min(1, 1 - (ratio ?? 0.7)))
  const ang = -90 + c * 180
  const col = c < 0.4 ? '#ef4444' : c < 0.65 ? '#eab308' : '#22c55e'
  const tick = (deg) => {
    const a = (deg * Math.PI) / 180
    const x1 = 80 + 50 * Math.cos(a), y1 = 84 - 50 * Math.sin(a)
    const x2 = 80 + 57 * Math.cos(a), y2 = 84 - 57 * Math.sin(a)
    return { x1, y1, x2, y2 }
  }
  const ticks = [180, 135, 90, 45, 0].map(tick)
  return (
    <svg viewBox="0 0 160 100" width="160" className="gauge">
      <path d="M18 84 A62 62 0 0 1 142 84" className="gauge-track" strokeWidth="11" fill="none" strokeLinecap="round" />
      <path d="M18 84 A62 62 0 0 1 35.4 40.9" stroke="#ef4444" strokeWidth="10" fill="none" strokeLinecap="round" />
      <path d="M38.5 37.9 A62 62 0 0 1 73.5 22.3" stroke="#f97316" strokeWidth="10" fill="none" strokeLinecap="round" />
      <path d="M77.8 22.1 A62 62 0 0 1 114.7 32.6" stroke="#eab308" strokeWidth="10" fill="none" strokeLinecap="round" />
      <path d="M118.2 35.1 A62 62 0 0 1 142 84" stroke="#22c55e" strokeWidth="10" fill="none" strokeLinecap="round" />
      {ticks.map((t, i) => (
        <line key={i} {...t} className="gauge-tickmark" strokeWidth="2" strokeLinecap="round" />
      ))}
      <g transform={`translate(80,84) rotate(${ang})`} style={{ transition: 'transform .8s cubic-bezier(.34,1.4,.64,1)' }}>
        <line x1="0" y1="12" x2="0" y2="-56" className="gauge-needle" strokeWidth="3.5" strokeLinecap="round" />
      </g>
      <circle className="gauge-hub" cx="80" cy="84" r="7" strokeWidth="2.5" />
      <circle cx="80" cy="84" r="2.5" fill={col} />
      <text x="14" y="98" className="gauge-tick" fontSize="9">0</text>
      <text x="76" y="14" className="gauge-tick" fontSize="9">50</text>
      <text x="134" y="98" className="gauge-tick" fontSize="9">100</text>
    </svg>
  )
}

export function FusionCard({ flow, pred }) {
  const ratio = flow?.congestion_ratio ?? 0.7
  const livePct = Math.round((1 - ratio) * 100)
  return (
    <div className="card">
      <h4>🧩 How live data shapes the forecast</h4>
      <p className="muted">Fused = model history corrected by live speed (ratio {Number(ratio).toFixed(2)})</p>
      <div className="fusionbar">
        <div className="fusionlive" style={{ width: `${livePct}%` }} />
      </div>
      <p className="muted">{livePct}% live signal · base {pred?.base_vol?.toLocaleString()} → adapted {pred?.adapted_vol?.toLocaleString()} → final {pred?.final_vol?.toLocaleString()} veh/hr</p>
    </div>
  )
}
