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
  return (
    <svg viewBox="0 0 160 96" width="150">
      <path d="M18 84 A62 62 0 0 1 142 84" stroke="rgba(255,255,255,.12)" strokeWidth="11" fill="none" strokeLinecap="round" />
      <path d="M18 84 A62 62 0 0 1 142 84" stroke={col} strokeWidth="9" fill="none" strokeLinecap="round" opacity="0.85" />
      <g transform={`translate(80,84) rotate(${ang})`} style={{ transition: 'transform .8s' }}>
        <polygon points="0,-64 3,-6 0,0 -3,-6" fill="#f8fafc" />
      </g>
      <circle cx="80" cy="84" r="6" fill="#0f172a" stroke="#f8fafc" strokeWidth="2.5" />
      <text x="18" y="94" fill="#94a3b8" fontSize="9">0%</text>
      <text x="74" y="30" fill="#94a3b8" fontSize="9">flow</text>
      <text x="132" y="94" fill="#94a3b8" fontSize="9">100%</text>
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
