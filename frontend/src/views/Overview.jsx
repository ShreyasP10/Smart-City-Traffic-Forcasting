import { useEffect, useState } from 'react'
import { Bar, BarChart, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { api, LEVEL_EMOJI } from '../api'
import { CountUp, FusionCard, GaugeNeedle } from '../components/common'

function verdictOf(rows, liveLevel) {
  if (!rows?.length) return 'Analyzing corridor throughput…'
  const worst = rows.reduce((a, b) => (a.final_vol > b.final_vol ? a : b))
  const order = { Low: 0, Moderate: 1, High: 2, Severe: 3 }
  if (worst.final_level === 'Severe' && liveLevel !== 'Severe')
    return `Congestion is building rapidly: expect Severe within ${worst.in_min} min.`
  if (order[worst.final_level] > order[liveLevel])
    return `Traffic pressure is rising toward ${worst.final_level} (+${worst.in_min} min).`
  if (rows[rows.length - 1].final_vol < rows[0].final_vol * 0.93)
    return 'Traffic is steadily easing — good window to travel.'
  if (liveLevel === 'Severe') return 'Severe bottlenecking persists — use alternates.'
  return `Traffic remains ${liveLevel} and steady.`
}

export default function Overview({ data, forecast, city, route, rain, holiday }) {
  const [insight, setInsight] = useState(null)
  useEffect(() => {
    if (city && route) api.insight(city, route).then(setInsight).catch(() => {})
  }, [city, route])
  if (!data) return <p>Loading…</p>
  const { flow, weather, pred, colors, shap } = data
  const lvl = pred.final_level
  const verdict = verdictOf(forecast?.rows, lvl)
  return (
    <div>
      {lvl === 'Severe' && (
        <div className="alert">🚨 SEVERE congestion — consider an alternate route.</div>
      )}
      <div className="cards">
        <div className="card"><h4>Live Now · congestion</h4><b><CountUp value={Math.round((1 - flow.congestion_ratio) * 100)} suffix="%" /></b><GaugeNeedle ratio={flow.congestion_ratio} /></div>
        <div className="card"><h4>Live speed</h4><b><CountUp value={flow.current_speed} /> km/h</b><span>free {flow.free_flow_speed} · {flow.congestion_ratio < 0.5 ? 'jammed' : 'flowing'}</span></div>
        <div className="card"><h4>Predicted volume</h4><b><CountUp value={pred.final_vol} /> veh/hr</b><span>{lvl} · delay +{flow.delay_min} min</span></div>
        <div className="card"><h4>Weather</h4><b>{weather.weather_main} {(weather.temp - 273.15).toFixed(0)}°C</b><span>rain {weather.rain_1h} mm/h · clouds {weather.clouds_all}%</span></div>
      </div>
      <h3>
        <span className="badge" style={{ background: colors[lvl] }}>{LEVEL_EMOJI[lvl]} {lvl}</span>
        {(rain || holiday) && <span className="tag">demo override active</span>}
      </h3>
      <p className="verdict">📣 City transit verdict: <b>{verdict}</b></p>
      <p className="muted">Sources: {flow.source} / {weather.source}</p>
      <FusionCard flow={flow} pred={pred} />
      <h3>🔍 Why this prediction?</h3>
      {shap.headline && <p><b>{shap.headline}</b></p>}
      {shap.summary && <p className="muted">{shap.summary}</p>}
      {shap.drivers?.length > 0 && (
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={shap.drivers} layout="vertical" margin={{ left: 120 }}>
            <XAxis type="number" />
            <YAxis type="category" dataKey="name" width={110} />
            <Tooltip />
            <Bar dataKey="impact">
              {shap.drivers.map((d, i) => (
                <Cell key={i} fill={d.impact > 0 ? '#ef4444' : '#22c55e'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      )}
      <div className="card">
        <h4>✨ Traffic insight & advisory {insight?.source === 'gemini' ? '(AI)' : '(advisory)'}</h4>
        <p>{insight ? insight.text : 'Generating commute analysis…'}</p>
      </div>
    </div>
  )
}
