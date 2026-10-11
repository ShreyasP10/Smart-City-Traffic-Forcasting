import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

export default function Adaptation({ data }) {
  if (!data || !data.metrics || !data.stages) return <p>Loading adaptation data…</p>
  const m = data.metrics || {}
  const drop = m.calib_mae_unadapted ? ((1 - (m.calib_mae_adapted || 0) / m.calib_mae_unadapted) * 100).toFixed(1) : '–'
  return (
    <div>
      <h3>🔁 Cross-context adaptation: US base → Indian city</h3>
      <p>Base (XGBoost, US history): MAE <b>{m.base_mae?.toFixed(1)}</b>, RMSE <b>{m.base_rmse?.toFixed(1)}</b></p>
      <p>Without adaptation: <b>{m.calib_mae_unadapted?.toFixed(1)}</b> → with Ridge calibration: <b>{m.calib_mae_adapted?.toFixed(1)}</b> ({drop}% drop)</p>
      <table className="tbl">
        <thead><tr><th>Stage</th><th>Volume</th><th>Level</th></tr></thead>
        <tbody>
          {data.stages.map((s) => (
            <tr key={s.stage}><td>{s.stage}</td><td>{s.volume.toLocaleString()}</td><td>{s.level}</td></tr>
          ))}
        </tbody>
      </table>
      <p>🔧 Calibration adjusted the US prediction by <b>{data.delta >= 0 ? '+' : ''}{data.delta.toLocaleString()} veh/hr</b>.</p>
      <h3>🇮🇳 Indian context calibration (live collector)</h3>
      <p>Logged <b>{data.collector.rows}</b> live observations across <b>{data.collector.locations}</b> locations.</p>
      {data.collector.points.length > 0 && (
        <ResponsiveContainer width="100%" height={240}>
          <LineChart data={data.collector.points}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="t" tick={false} />
            <YAxis domain={[0, 1.1]} />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="ratio" stroke="#f97316" dot={false} />
          </LineChart>
        </ResponsiveContainer>
      )}
    </div>
  )
}
