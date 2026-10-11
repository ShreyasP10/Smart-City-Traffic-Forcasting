import { useEffect, useState } from 'react'
import { Bar, BarChart, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { api, LEVEL_EMOJI } from '../api'

const SCENARIOS = [
  { name: 'Evening peak + monsoon', s: { hour: 18, dow: 2, rain: 8.5, temp_c: 26, clouds: 95 } },
  { name: 'Sunny Sunday', s: { hour: 14, dow: 6, weekend: 1, rain: 0, temp_c: 31, clouds: 10 } },
  { name: 'Morning rush', s: { hour: 8, dow: 3, rain: 0, temp_c: 24, clouds: 30 } },
  { name: 'Late night', s: { hour: 2, dow: 4, rain: 0, temp_c: 22, clouds: 5 } },
  { name: 'Holiday midday', s: { hour: 12, dow: 1, holiday: 1, rain: 0, temp_c: 28, clouds: 25 } },
]
const DAYS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']

export default function Simulator() {
  const [p, setP] = useState({ hour: 18, dow: 2, month: 10, weekend: 0, holiday: 0, temp_c: 26, rain: 2, snow: 0, clouds: 60, weather: 'Clouds' })
  const [out, setOut] = useState(null)
  const set = (k, v) => setP((s) => ({ ...s, [k]: v }))
  useEffect(() => {
    const t = setTimeout(() => api.whatif({ ...p, dow: (p.dow + 6) % 7 }).then(setOut).catch(() => {}), 250)
    return () => clearTimeout(t)
  }, [p])
  return (
    <div>
      <h3>🧪 What-if simulator — drive the model yourself</h3>
      <div className="pills">
        {SCENARIOS.map((s) => (
          <button key={s.name} onClick={() => setP((cur) => ({ ...cur, ...s.s }))}>{s.name}</button>
        ))}
      </div>
      <div className="sliders">
        <label>Hour {p.hour}:00 <input type="range" min="0" max="23" value={p.hour} onChange={(e) => set('hour', +e.target.value)} /></label>
        <div className="pills">{DAYS.map((d, i) => <button key={d} className={p.dow === i ? 'active' : ''} onClick={() => { set('dow', i); set('weekend', i === 0 || i === 6 ? 1 : 0) }}>{d}</button>)}</div>
        <label>Rain {p.rain} mm/h <input type="range" min="0" max="15" step="0.5" value={p.rain} onChange={(e) => set('rain', +e.target.value)} /></label>
        <label>Temp {p.temp_c}°C <input type="range" min="-10" max="45" value={p.temp_c} onChange={(e) => set('temp_c', +e.target.value)} /></label>
        <label>Clouds {p.clouds}% <input type="range" min="0" max="100" step="5" value={p.clouds} onChange={(e) => set('clouds', +e.target.value)} /></label>
        <label><input type="checkbox" checked={!!p.weekend} onChange={(e) => set('weekend', e.target.checked ? 1 : 0)} /> Weekend</label>
        <label><input type="checkbox" checked={!!p.holiday} onChange={(e) => set('holiday', e.target.checked ? 1 : 0)} /> Holiday</label>
      </div>
      {!out ? <p>Loading…</p> : (
        <>
          <div className="cards">
            <div className="card"><h4>Corridor volume</h4><b>{out.volume.toLocaleString()} veh/hr</b></div>
            <div className="card"><h4>Congestion index</h4><b>{LEVEL_EMOJI[out.level]} {out.level}</b></div>
            <div className="card"><h4>Advice</h4><span>{out.prescription}</span></div>
          </div>
          <h4>GBM ensemble inference — marginal attribution waterfall</h4>
          <p className="muted">Base US prediction {out.volume.toLocaleString()} → calibration {out.calibration_delta >= 0 ? '+' : ''}{(out.calibration_delta || 0).toLocaleString()} → adapted {(out.adapted_volume ?? out.volume).toLocaleString()} ({out.adapted_level || out.level})</p>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={[
              { n: 'Base (US)', lo: 0, w: out.volume },
              { n: 'Calibration Δ', lo: Math.min(out.volume, out.adapted_volume ?? out.volume), w: Math.abs(out.calibration_delta || 0) },
              { n: 'Adapted (India)', lo: 0, w: out.adapted_volume ?? out.volume },
            ]}>
              <XAxis dataKey="n" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="lo" stackId="s" fill="transparent" />
              <Bar dataKey="w" stackId="s" fill="#7c3aed" name="veh/hr" />
            </BarChart>
          </ResponsiveContainer>
          <h4>What the model relies on (global importance)</h4>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={out.importances.slice(0, 10)} layout="vertical" margin={{ left: 90 }}>
              <XAxis type="number" />
              <YAxis type="category" dataKey="feature" width={80} />
              <Tooltip />
              <ReferenceLine x={0} stroke="#64748b" />
              <Bar dataKey="importance" fill="#06b6d4" />
            </BarChart>
          </ResponsiveContainer>
        </>
      )}
    </div>
  )
}
