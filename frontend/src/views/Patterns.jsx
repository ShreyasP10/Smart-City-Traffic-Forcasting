import { Bar, BarChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

const DAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

export default function Patterns({ data }) {
  if (!data) return <p>Loading…</p>
  return (
    <div>
      <h3>📊 Historical patterns (what the model learned)</h3>
      <h4>Average volume by hour (peak-hour structure)</h4>
      <ResponsiveContainer width="100%" height={240}>
        <LineChart data={data.hourly}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="hour" />
          <YAxis />
          <Tooltip />
          <Line type="monotone" dataKey="vol" stroke="#3b82f6" strokeWidth={2} dot={false} />
        </LineChart>
      </ResponsiveContainer>
      <h4>Weekday vs weekend</h4>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={data.weekday.map((d) => ({ day: DAYS[d.dow], vol: d.vol }))}>
          <XAxis dataKey="day" />
          <YAxis />
          <Tooltip />
          <Bar dataKey="vol" fill="#22c55e" />
        </BarChart>
      </ResponsiveContainer>
      <h4>Hour × day heatmap</h4>
      <div className="heatgrid" style={{ '--cols': 24 }}>
        {data.heatmap.values.map((row, di) =>
          row.map((v, hi) => {
            const t = Math.min(1, v / 6000)
            const bg = `rgba(239,68,68,${0.08 + t * 0.85})`
            return <div key={`${di}-${hi}`} className="heatcell" title={`${DAYS[di]} ${hi}:00 — ${v}`} style={{ background: bg }} />
          })
        )}
      </div>
      <div className="cards">
        <div className="card"><h4>AM peak 7–10h</h4><b>{data.peaks.am.toLocaleString()}</b></div>
        <div className="card"><h4>PM peak 17–21h</h4><b>{data.peaks.pm.toLocaleString()}</b></div>
        <div className="card"><h4>Weekday avg</h4><b>{data.peaks.weekday.toLocaleString()}</b></div>
        <div className="card"><h4>Weekend avg</h4><b>{data.peaks.weekend.toLocaleString()}</b></div>
      </div>
    </div>
  )
}
