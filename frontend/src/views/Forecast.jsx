import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

export default function Forecast({ data }) {
  if (!data || !data.rows) return <p>Loading forecast data…</p>
  const rows = data.rows.map((r) => ({
    h: `+${r.in_min}m (${r.time})`,
    final: r.final_vol,
    adapted: r.adapted_vol,
    base: r.base_vol,
    level: r.final_level,
  }))
  return (
    <div>
      <h3>📈 Short-term forecast — {data.route}</h3>
      <table className="tbl">
        <thead><tr><th>Horizon</th><th>At</th><th>Volume</th><th>Level</th></tr></thead>
        <tbody>
          {data.rows.map((r) => (
            <tr key={r.in_min}><td>+{r.in_min} min</td><td>{r.time}</td><td>{r.final_vol.toLocaleString()}</td><td>{r.final_level}</td></tr>
          ))}
        </tbody>
      </table>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={rows}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="h" />
          <YAxis />
          <Tooltip />
          <Legend />
          <Line type="monotone" dataKey="final" stroke="#f97316" strokeWidth={3} dot />
          <Line type="monotone" dataKey="adapted" stroke="#3b82f6" dot={false} />
          <Line type="monotone" dataKey="base" stroke="#9ca3af" dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
