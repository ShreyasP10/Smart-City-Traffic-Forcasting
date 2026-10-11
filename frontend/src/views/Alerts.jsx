import { useEffect, useMemo, useState } from 'react'

export default function Alerts({ forecast, city }) {
  const [threshold, setThreshold] = useState('High')
  const [read, setRead] = useState(false)
  const [cleared, setCleared] = useState(false)
  useEffect(() => { setRead(false); setCleared(false) }, [city, forecast?.at])

  const items = useMemo(() => {
    if (!forecast?.rows) return []
    const out = []
    const hit = (lvl) => (threshold === 'Severe' ? lvl === 'Severe' : lvl === 'High' || lvl === 'Severe')
    forecast.rows.forEach((r) => {
      if (hit(r.final_level)) {
        out.push({
          id: `${r.in_min}`, level: r.final_level, horizon: r.in_min === 0 ? 'Now' : `+${r.in_min} min`,
          pct: Math.round((r.final_vol / 6200) * 100),
          title: `${r.final_level} congestion ${r.in_min === 0 ? 'detected' : 'forecast'} — ${city}`,
          desc: r.in_min === 0
            ? `Live conditions crossed the ${threshold} threshold at ${r.final_vol.toLocaleString()} veh/hr.`
            : `Model projects ${r.final_level} traffic (${r.final_vol.toLocaleString()} veh/hr) in ${r.in_min} min. Consider alternates.`,
        })
      }
    })
    return out
  }, [forecast, threshold, city])

  const visible = cleared ? [] : items
  const unread = !read ? visible.length : 0
  return (
    <div>
      <h3>🚨 Alerts — threshold rules</h3>
      <p className="muted">Evaluated on live conditions + 15/30/60-min forecasts every refresh cycle.</p>
      <div className="pills">
        <button className={threshold === 'High' ? 'active' : ''} onClick={() => setThreshold('High')}>High or worse (≥65%)</button>
        <button className={threshold === 'Severe' ? 'active' : ''} onClick={() => setThreshold('Severe')}>Severe only (≥85%)</button>
      </div>
      <div className="row">
        <b>Warning feed ({visible.length})</b>
        {unread > 0 && <span className="badge" style={{ background: '#ef4444' }}>{unread} new</span>}
        <button onClick={() => setRead(true)}>Mark read</button>
        <button onClick={() => setCleared(true)}>Clear</button>
      </div>
      {visible.length === 0 ? (
        <div className="card"><h4>✅ Corridors operating within limits</h4><span>No active {threshold === 'Severe' ? 'Severe-only' : 'High-or-worse'} alerts.</span></div>
      ) : visible.map((a) => (
        <div className="card alertcard" key={a.id}>
          <b>{a.level} — {a.title}</b>
          <span>{a.horizon} · {a.pct}% index</span>
          <p>{a.desc}</p>
        </div>
      ))}
    </div>
  )
}
