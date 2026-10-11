import { useEffect, useState } from 'react'
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { api } from '../api'
import CorridorField from '../components/CorridorField'

// Free corridor-vs-corridor comparison: type any two places, we route both.
export default function Compare({ city }) {
  const [a, setA] = useState({ name: 'Andheri', lat: 19.1136, lon: 72.8697 })
  const [b, setB] = useState({ name: 'BKC', lat: 19.0596, lon: 72.8656 })
  const [res, setRes] = useState(null)
  const [loading, setLoading] = useState(false)

  const run = async () => {
    if (!a?.lat || !b?.lat) return
    setLoading(true)
    try {
      const [ra, rb] = await Promise.all([
        api.corridor(a.lat, a.lon, b.lat, b.lon),
        api.corridor(b.lat, b.lon, a.lat, a.lon),
      ])
      const rows = [
        { route: `${a.name} → ${b.name}`, level: ra.pred.final_level, vol: ra.pred.final_vol, trip_min: ra.route.live_min },
        { route: `${b.name} → ${a.name}`, level: rb.pred.final_level, vol: rb.pred.final_vol, trip_min: rb.route.live_min },
      ]
      const best = rows[0].trip_min <= rows[1].trip_min ? rows[0] : rows[1]
      setRes({ results: rows, best: best.route, colors: ra.colors })
    } catch { setRes(null) }
    setLoading(false)
  }

  useEffect(() => { run() }, [])
  const norm = (v) => (typeof v === 'object' ? v : { name: v, lat: null, lon: null })
  const setA2 = (v) => setA(typeof v === 'object' ? v : { name: v, lat: null, lon: null })
  const setB2 = (v) => setB(typeof v === 'object' ? v : { name: v, lat: null, lon: null })

  return (
    <div>
      <h3>⚖ Route comparison — any two places</h3>
      <div className="row">
        <CorridorField label="From" value={a.name} routes={[]} onPick={(v) => setA2(norm(v))} />
        <CorridorField label="To" value={b.name} routes={[]} onPick={(v) => setB2(norm(v))} />
        <button className="primary" onClick={run} disabled={loading || !a?.lat || !b?.lat}>
          {loading ? 'Comparing…' : 'Compare →'}
        </button>
      </div>
      {(!a?.lat || !b?.lat) && <p className="muted">Pick both ends from suggestions or search results (needs map coordinates).</p>}
      {!res ? <p>{loading ? 'Routing both directions…' : 'Compare to see results.'}</p> : (
        <>
          <table className="tbl">
            <thead><tr><th>Direction</th><th>Level</th><th>Vol</th><th>Trip</th></tr></thead>
            <tbody>
              {res.results.map((r) => (
                <tr key={r.route} className={r.route === res.best ? 'best' : ''}>
                  <td>{r.route}</td><td>{r.level}</td><td>{r.vol.toLocaleString()}</td><td>{r.trip_min} min</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="ok">✅ Recommendation: <b>{res.best}</b></p>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={res.results}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="route" tick={false} />
              <YAxis />
              <Tooltip />
              <Bar dataKey="trip_min">
                {res.results.map((r, i) => (
                  <Cell key={i} fill={(res.colors && res.colors[r.level]) || '#3b82f6'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </>
      )}
    </div>
  )
}
