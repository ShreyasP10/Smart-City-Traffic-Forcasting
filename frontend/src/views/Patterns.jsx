import { useEffect, useState } from 'react'
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { api } from '../api'

const DAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

export default function Patterns({ data }) {
  const [rainC, setRainC] = useState(null)
  const [holC, setHolC] = useState(null)
  const [wxC, setWxC] = useState(null)

  useEffect(() => {
    let on = true
    const run = async () => {
      try {
        // 1. rainfall (mm) vs congestion at evening peak Wednesday
        const rains = [0, 1, 2, 3, 5, 8, 12, 15]
        const rr = await Promise.all(rains.map((r) =>
          api.whatif({ hour: 18, dow: 2, month: 10, temp_c: 27, rain: r, clouds: r > 0 ? 80 : 25 }).catch(() => null)))
        if (on) setRainC(rains.map((r, i) => ({ rain: `${r}mm`, vol: rr[i]?.volume ?? 0, level: rr[i]?.level ?? '–' })))
        // 2. public holiday vs standard workday across dayparts (Monday)
        const parts = [{ n: 'Morning 8h', h: 8 }, { n: 'Midday 13h', h: 13 }, { n: 'Evening 18h', h: 18 }, { n: 'Night 23h', h: 23 }]
        const hh = await Promise.all(parts.flatMap((p) => [
          api.whatif({ hour: p.h, dow: 0, holiday: 0 }).catch(() => null),
          api.whatif({ hour: p.h, dow: 0, holiday: 1 }).catch(() => null),
        ]))
        if (on) setHolC(parts.map((p, i) => ({
          part: p.n, workday: hh[i * 2]?.volume ?? 0, holiday: hh[i * 2 + 1]?.volume ?? 0,
        })))
        // 3. clear vs harsh weather types at peak
        const wxs = ['Clear', 'Clouds', 'Fog', 'Rain', 'Thunderstorm']
        const ww = await Promise.all(wxs.map((w) =>
          api.whatif({ hour: 18, dow: 2, weather: w, rain: w === 'Rain' ? 4 : w === 'Thunderstorm' ? 9 : 0, clouds: w === 'Clear' ? 10 : 85 }).catch(() => null)))
        if (on) setWxC(wxs.map((w, i) => ({ wx: w, vol: ww[i]?.volume ?? 0, level: ww[i]?.level ?? '–' })))
      } catch { /* offline: sections stay empty */ }
    }
    run()
    return () => { on = false }
  }, [])

  if (!data) return <p>Loading…</p>
  const merged = data.hourly.map((h) => ({
    hour: h.hour,
    all: h.vol,
    weekday: data.hourly_weekday?.find((x) => x.hour === h.hour)?.vol,
    weekend: data.hourly_weekend?.find((x) => x.hour === h.hour)?.vol,
  }))
  return (
    <div>
      <h3>📊 Historical patterns (what the model learned)</h3>
      <h4>Diurnal hourly profile — weekday vs weekend</h4>
      <ResponsiveContainer width="100%" height={240}>
        <LineChart data={merged}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="hour" />
          <YAxis />
          <Tooltip />
          <Legend />
          <Line type="monotone" dataKey="weekday" stroke="#0284c7" strokeWidth={2} dot={false} name="Weekday" />
          <Line type="monotone" dataKey="weekend" stroke="#22c55e" strokeWidth={2} dot={false} name="Weekend" />
        </LineChart>
      </ResponsiveContainer>
      <h4>Weekday load</h4>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={data.weekday.map((d) => ({ day: DAYS[d.dow], vol: d.vol }))}>
          <XAxis dataKey="day" />
          <YAxis />
          <Tooltip />
          <Bar dataKey="vol" fill="#22c55e" />
        </BarChart>
      </ResponsiveContainer>
      <h4>7×24 weekly congestion heatmap</h4>
      <div className="heatgrid" style={{ '--cols': 24 }}>
        {data.heatmap.values.map((row, di) =>
          row.map((v, hi) => {
            const t = Math.min(1, v / 6000)
            const bg = `rgba(239,68,68,${0.08 + t * 0.85})`
            return <div key={`${di}-${hi}`} className="heatcell" title={`${DAYS[di]} ${hi}:00 — ${v}`} style={{ background: bg }} />
          })
        )}
      </div>
      <h4>🌧 Rainfall vs congestion (model sweep, Wed 18:00)</h4>
      {!rainC ? <p className="muted">Sweeping model…</p> : (
        <ResponsiveContainer width="100%" height={230}>
          <BarChart data={rainC}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="rain" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="vol" fill="#0284c7" name="predicted vol" />
          </BarChart>
        </ResponsiveContainer>
      )}
      <h4>🎉 Public holiday vs standard workday load (Monday)</h4>
      {!holC ? <p className="muted">Sweeping model…</p> : (
        <ResponsiveContainer width="100%" height={230}>
          <BarChart data={holC}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="part" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="workday" fill="#0284c7" name="Workday" />
            <Bar dataKey="holiday" fill="#7c3aed" name="Holiday" />
          </BarChart>
        </ResponsiveContainer>
      )}
      <h4>☀️ Clear vs harsh weather types (peak hour)</h4>
      {!wxC ? <p className="muted">Sweeping model…</p> : (
        <ResponsiveContainer width="100%" height={230}>
          <BarChart data={wxC}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="wx" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="vol" fill="#f97316" name="predicted vol" />
          </BarChart>
        </ResponsiveContainer>
      )}
      <div className="cards">
        <div className="card"><h4>AM peak 7–10h</h4><b>{data.peaks.am.toLocaleString()}</b></div>
        <div className="card"><h4>PM peak 17–21h</h4><b>{data.peaks.pm.toLocaleString()}</b></div>
        <div className="card"><h4>Weekday avg</h4><b>{data.peaks.weekday.toLocaleString()}</b></div>
        <div className="card"><h4>Weekend avg</h4><b>{data.peaks.weekend.toLocaleString()}</b></div>
      </div>
    </div>
  )
}
