import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis, Cell } from 'recharts'

export default function Compare({ data, routes, a, b, setA, setB }) {
  return (
    <div>
      <h3>⚖ Route comparison</h3>
      <div className="row">
        <label>Route A <select value={a} onChange={(e) => setA(e.target.value)}>
          {routes.map((r) => <option key={r} value={r}>{r}</option>)}
        </select></label>
        <label>Route B <select value={b} onChange={(e) => setB(e.target.value)}>
          {routes.map((r) => <option key={r} value={r}>{r}</option>)}
        </select></label>
      </div>
      {!data ? <p>Loading…</p> : (
        <>
          <table className="tbl">
            <thead><tr><th>Route</th><th>Level</th><th>Vol</th><th>Speed</th><th>Trip</th></tr></thead>
            <tbody>
              {data.results.map((r) => (
                <tr key={r.route} className={r.route === data.best ? 'best' : ''}>
                  <td>{r.route}</td><td>{r.level}</td><td>{r.vol.toLocaleString()}</td><td>{r.speed} km/h</td><td>{r.trip_min} min</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="ok">✅ Recommendation: <b>{data.best}</b></p>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={data.results}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="route" tick={false} />
              <YAxis />
              <Tooltip />
              <Bar dataKey="trip_min">
                {data.results.map((r, i) => (
                  <Cell key={i} fill={(data.colors && data.colors[r.level]) || '#3b82f6'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </>
      )}
    </div>
  )
}
