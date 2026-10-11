import { useEffect, useMemo, useState } from 'react'
import { Area, CartesianGrid, ComposedChart, Legend, Line, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { api } from '../api'
import { CORRIDORS, LS_KEY, addDays, diurnal, dkey, lvlOf, preloaded, surgeAt } from '../data/events'
const DOW = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']

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
