import { useEffect, useMemo, useState } from 'react'
import { Area, CartesianGrid, ComposedChart, Legend, Line, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { api } from '../api'

const CORRIDORS = ['City Center Hub', 'North Radial Ring', 'North-East Expressway', 'West Central Avenue', 'East Business Corridor', 'South Coastal Link', 'South-East Bypass', 'North-West Arterial', 'South-West Terminal']
const LS_KEY = 'traffic_custom_events_v1'
const DOW = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']

const dkey = (d) => `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`
const addDays = (d, n) => { const x = new Date(d); x.setDate(x.getDate() + n); return x }

function preloaded() {
  const t = new Date()
  const E = (off, title, sh, s, e, cat, att, surge, venue) => ({
    id: `ev-${off}-${sh}`, title, short: title.split(' ').slice(0, 2).join(' '),
    date: dkey(addDays(t, off)), start: s, end: e, category: cat, attendees: att, surge, venue,
    corridors: CORRIDORS.slice(0, 3), custom: false,
  })
  return [
    E(1, 'Grand Garba & Dandiya Night', 'Garba Night', '19:00', '02:00', 'cultural', 52000, 1.88, 'Community Grounds'),
    E(2, 'Festive Dandiya Evening', 'Dandiya', '19:30', '01:30', 'cultural', 38000, 1.65, 'City Grounds'),
    E(4, 'Dussehra Mela Procession', 'Dussehra Mela', '16:00', '22:30', 'procession', 44000, 1.74, 'Main Avenue'),
    E(7, 'T20 Cricket Match', 'T20 Match', '18:00', '23:30', 'sports', 58000, 1.92, 'Stadium'),
    E(10, 'Diwali Bazaar Night Market', 'Diwali Bazaar', '14:00', '23:00', 'cultural', 65000, 1.82, 'Market District'),
    E(14, 'City Marathon', 'Marathon', '05:00', '11:00', 'sports', 22000, 1.55, 'Ring Road'),
    E(18, 'Tech Summit', 'Tech Summit', '08:30', '18:00', 'civic', 30000, 1.45, 'Convention Center'),
    E(21, 'Music Concert', 'Concert', '19:00', '23:00', 'concert', 35000, 1.6, 'Arena'),
  ]
}

// Simplified surge curve port: arrival / active / dispersal
function surgeAt(ev, h) {
  const s = Number(ev.start.slice(0, 2)), e0 = Number(ev.end.slice(0, 2))
  const end = e0 <= s ? e0 + 24 : e0
  const H = h < s ? h + 24 : h
  const m = ev.surge || 1.5
  if (H === s - 1 || (s === 0 && h === 23)) return 1 + (m - 1) * 0.35
  if (H >= s && H <= end) {
    const p = end === s ? 0.5 : (H - s) / (end - s)
    return 1 + (m - 1) * (0.6 + 0.4 * Math.sin(p * Math.PI))
  }
  if (H === end + 1) return 1 + (m - 1) * 0.45
  return 1
}

function diurnal(h, weekend) {
  if (!weekend) {
    if (h >= 7 && h <= 9) return 0.82
    if (h >= 17 && h <= 20) return 0.88
    if (h >= 10 && h <= 16) return 0.58
    if (h >= 21 || h <= 4) return 0.2
    return 0.42
  }
  if (h >= 12 && h <= 16) return 0.66
  if (h >= 17 && h <= 21) return 0.6
  if (h >= 22 || h <= 5) return 0.16
  return 0.4
}

const lvlOf = (v) => (v < 0.4 ? 'Low' : v < 0.65 ? 'Moderate' : v < 0.85 ? 'High' : 'Severe')

export default function Calendar() {
  const now = useMemo(() => new Date(), [])
  const [ym, setYm] = useState({ y: now.getFullYear(), m: now.getMonth() })
  const [sel, setSel] = useState(dkey(addDays(now, 1)))
  const [mode, setMode] = useState('grid')
  const [custom, setCustom] = useState(() => { try { return JSON.parse(localStorage.getItem(LS_KEY) || '[]') } catch { return [] } })
  const [modal, setModal] = useState(null)
  const [sim, setSim] = useState({ att: 45000, rain: 0, police: false })
  const [hours, setHours] = useState(null)

  const events = useMemo(() => [...preloaded(), ...custom], [custom])
  const byDate = useMemo(() => {
    const m = {}
    events.forEach((e) => { (m[e.date] = m[e.date] || []).push(e) })
    return m
  }, [events])

  const days = useMemo(() => {
    const out = []
    const n = new Date(ym.y, ym.m + 1, 0).getDate()
    for (let d = 1; d <= n; d++) {
      const dt = new Date(ym.y, ym.m, d)
      const k = dkey(dt)
      const we = dt.getDay() === 0 || dt.getDay() === 6
      const evs = byDate[k] || []
      const peak = Math.max(...Array.from({ length: 24 }, (_, h) => {
        const mult = Math.max(1, ...evs.map((e) => surgeAt(e, h)))
        return Math.min(0.99, diurnal(h, we) * (evs.length ? mult : 1))
      }))
      out.push({ d, k, dow: dt.getDay(), peak, level: lvlOf(peak), evs, today: k === dkey(now) })
    }
    return out
  }, [ym, byDate, now])

  const selEvs = byDate[sel] || []
  const selDow = new Date(sel + 'T12:00:00').getDay()

  useEffect(() => {
    setHours(null)
    const run = async () => {
      const rows = await Promise.all(Array.from({ length: 24 }, (_, h) =>
        api.whatif({ hour: h, dow: (selDow + 6) % 7, month: ym.m + 1, weekend: selDow === 0 || selDow === 6 ? 1 : 0, temp_c: 29, rain: sim.rain, clouds: sim.rain > 0 ? 70 : 25 }).catch(() => null)
      ))
      const policeF = sim.police ? 0.78 : 1
      const attF = 0.7 + (sim.att / 90000) * 0.6
      setHours(rows.map((r, h) => {
        const base = r ? r.volume : 2500
        const mult = Math.max(1, ...selEvs.map((e) => surgeAt({ ...e, surge: 1 + (e.surge - 1) * attF * policeF }, h)))
        const ev = Math.round(base * mult * (sim.rain > 3 ? 1.12 : sim.rain > 0 ? 1.05 : 1))
        return { h, label: `${String(h).padStart(2, '0')}:00`, base, ev, level: lvlOf(Math.min(0.99, ev / 6200)) }
      }))
    }
    if (mode === 'day') run()
  }, [mode, sel, sim.att, sim.rain, sim.police])

  const saveCustom = (e) => {
    const next = [...custom, { ...e, id: `custom-${Date.now()}`, custom: true }]
    setCustom(next)
    localStorage.setItem(LS_KEY, JSON.stringify(next))
    setModal(null)
    setSel(e.date)
    setMode('day')
  }

  const peak = hours?.reduce((a, b) => (a.ev > b.ev ? a : b))
  const corridors = CORRIDORS.slice(0, 6).map((c, i) => {
    const affected = selEvs.some((e) => (e.corridors || []).includes(c)) || i < 3
    const drop = peak ? Math.min(60, Math.round((peak.ev / Math.max(peak.base, 1) - 1) * (affected ? 55 : 25))) : 0
    const spd = Math.max(11, Math.round(46 * (1 - drop / 100)))
    return { c, base: 46, spd, drop, level: spd > 30 ? 'Low' : spd > 23 ? 'Moderate' : spd > 14 ? 'High' : 'Severe', detour: drop > 20 && spd >= 16 }
  })

  const firstDow = new Date(ym.y, ym.m, 1).getDay()
  const monthName = new Date(ym.y, ym.m, 1).toLocaleString('en', { month: 'long' })

  return (
    <div>
      <h3>🏮 Event calendar & future traffic predictions</h3>
      <div className="pills">
        <button className={mode === 'grid' ? 'active' : ''} onClick={() => setMode('grid')}>Month calendar ({days.length} days)</button>
        <button className={mode === 'schedule' ? 'active' : ''} onClick={() => setMode('schedule')}>Event schedule ({events.length})</button>
        <button className={mode === 'day' ? 'active' : ''} onClick={() => setMode('day')}>Day inspector ({sel})</button>
        <button onClick={() => setModal({ date: sel })}>+ Add event</button>
      </div>

      {mode === 'grid' && (
        <>
          <div className="row">
            <button onClick={() => setYm((s) => (s.m === 0 ? { y: s.y - 1, m: 11 } : { ...s, m: s.m - 1 }))}>‹ Prev</button>
            <b>{monthName} {ym.y}</b>
            <button onClick={() => setYm((s) => (s.m === 11 ? { y: s.y + 1, m: 0 } : { ...s, m: s.m + 1 }))}>Next ›</button>
          </div>
          <div className="calgrid">
            {['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'].map((d) => <div key={d} className="calhead">{d}</div>)}
            {Array.from({ length: firstDow }).map((_, i) => <div key={`b${i}`} className="calblank" />)}
            {days.map((d) => (
              <div key={d.k} className={`calcell ${d.level.toLowerCase()} ${d.k === sel ? 'sel' : ''}`} onClick={() => { setSel(d.k); setMode('day') }}>
                <b>{d.d}{d.today ? ' •' : ''}</b>
                <span className="calpct">{Math.round(d.peak * 100)}%</span>
                <span className="calev">{d.evs.length ? `🏮 ${d.evs[0].short}` : 'regular'}</span>
              </div>
            ))}
          </div>
          <p className="muted">Cells: event-adjusted estimates (diurnal curve × surge). Click a day for model-backed hourly detail.</p>
        </>
      )}

      {mode === 'schedule' && events.map((e) => (
        <div className="card" key={e.id}>
          <b>🏮 {e.title}</b> <span className="muted">{e.date} · {e.start}–{e.end} · {e.venue} · {e.attendees.toLocaleString()} people</span>
          <p>Surge ×{e.surge} on {(e.corridors || []).join(', ')}</p>
          <button onClick={() => { setSel(e.date); setMode('day') }}>Hour-by-hour timeline →</button>
        </div>
      ))}

      {mode === 'day' && (
        <>
          <h4>24-hour inspector — {sel} {selEvs.map((e) => e.short).join(', ')}</h4>
          {!hours ? <p>Querying model for 24 hours…</p> : (
            <>
              <ResponsiveContainer width="100%" height={260}>
                <ComposedChart data={hours}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="label" interval={2} />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line dataKey="base" stroke="#06b6d4" strokeDasharray="4 4" dot={false} name="Baseline" />
                  <Area dataKey="ev" stroke="#f59e0b" fill="#f59e0b" fillOpacity={0.25} name="Event surge" />
                  <ReferenceLine y={4400} stroke="#ef4444" strokeDasharray="4 4" label="Severe" />
                  <ReferenceLine y={3400} stroke="#eab308" strokeDasharray="4 4" label="High" />
                </ComposedChart>
              </ResponsiveContainer>
              <div className="sliders">
                <label>Crowd {sim.att.toLocaleString()} <input type="range" min="5000" max="90000" step="5000" value={sim.att} onChange={(e) => setSim((s) => ({ ...s, att: +e.target.value }))} /></label>
                <label>Rain {sim.rain} mm <input type="range" min="0" max="15" value={sim.rain} onChange={(e) => setSim((s) => ({ ...s, rain: +e.target.value }))} /></label>
                <label><input type="checkbox" checked={sim.police} onChange={(e) => setSim((s) => ({ ...s, police: e.target.checked }))} /> Police green corridor (−22%)</label>
              </div>
              <table className="tbl">
                <thead><tr><th>Corridor</th><th>Base</th><th>Predicted</th><th>Drop</th><th>Status</th></tr></thead>
                <tbody>{corridors.map((c) => (
                  <tr key={c.c}><td>{c.c}</td><td>{c.base} km/h</td><td>{c.spd} km/h</td><td>−{c.drop}%</td><td>{c.level}{c.detour ? ' · suggested detour' : ''}</td></tr>
                ))}</tbody>
              </table>
            </>
          )}
        </>
      )}

      {modal && <EventModal initial={modal.date} onClose={() => setModal(null)} onSave={saveCustom} />}
    </div>
  )
}

function EventModal({ initial, onClose, onSave }) {
  const [f, setF] = useState({ title: 'Local Garba Night', short: 'Garba', date: initial, start: '19:00', end: '01:00', category: 'cultural', attendees: 35000, venue: 'Community Grounds', corridors: CORRIDORS.slice(0, 2) })
  const set = (k, v) => setF((s) => ({ ...s, [k]: v }))
  const surge = f.attendees > 50000 ? 1.85 : f.attendees > 25000 ? 1.65 : f.attendees > 10000 ? 1.48 : 1.35
  const toggle = (c) => set('corridors', f.corridors.includes(c) ? f.corridors.filter((x) => x !== c) : [...f.corridors, c])
  return (
    <div className="modalbg" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <h3>+ Add event & predict traffic</h3>
        <label>Title <input value={f.title} onChange={(e) => set('title', e.target.value)} /></label>
        <div className="row">
          <label>Date <input type="date" value={f.date} onChange={(e) => set('date', e.target.value)} /></label>
          <label>Start <input type="time" value={f.start} onChange={(e) => set('start', e.target.value)} /></label>
          <label>End <input type="time" value={f.end} onChange={(e) => set('end', e.target.value)} /></label>
        </div>
        <div className="row">
          <label>Category <select value={f.category} onChange={(e) => set('category', e.target.value)}>
            {['cultural', 'sports', 'concert', 'civic', 'procession', 'custom'].map((c) => <option key={c} value={c}>{c}</option>)}
          </select></label>
          <label>Crowd <input type="number" min="500" max="150000" step="2500" value={f.attendees} onChange={(e) => set('attendees', +e.target.value)} /></label>
        </div>
        <label>Venue <input value={f.venue} onChange={(e) => set('venue', e.target.value)} /></label>
        <div className="pills">{CORRIDORS.map((c) => <button key={c} type="button" className={f.corridors.includes(c) ? 'active' : ''} onClick={() => toggle(c)}>{c}</button>)}</div>
        <p className="muted">Estimated surge ×{surge}</p>
        <div className="row">
          <button onClick={onClose}>Cancel</button>
          <button className="primary" disabled={!f.title || !f.date || f.corridors.length === 0}
            onClick={() => onSave({ ...f, surge, short: f.short || f.title.split(' ').slice(0, 2).join(' ') })}>Add & predict →</button>
        </div>
      </div>
    </div>
  )
}
