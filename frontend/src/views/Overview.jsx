import { useEffect, useState } from 'react'
import { Bar, BarChart, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { api, LEVEL_EMOJI } from '../api'
import { CountUp, FusionCard, GaugeNeedle } from '../components/common'
import { nextEvent } from '../data/events'

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

export default function Overview({ data, forecast, city, route, rain, holiday, goCalendar }) {
  const [insight, setInsight] = useState(null)
  const [ev] = useState(() => nextEvent())
  useEffect(() => {
    if (city && route) api.insight(city, route).then(setInsight).catch(() => {})
  }, [city, route])
  if (!data || !data.pred || !data.flow) return <p>Loading overview data…</p>
  const { flow, weather, pred, colors, shap } = data
  const lvl = pred.final_level
  const verdict = verdictOf(forecast?.rows, lvl)
  return (
    <div>
      {lvl === 'Severe' && (
        <div className="alert">🚨 SEVERE congestion — consider an alternate route.</div>
      )}
      <div className="cards">
        <div className="card"><h4>Live Now · congestion</h4><div className="bignum"><CountUp value={Math.round((1 - flow.congestion_ratio) * 100)} /><span className="unit">% congested</span></div><GaugeNeedle ratio={flow.congestion_ratio} /><span className="sub">fluidity {Math.round(flow.congestion_ratio * 100)}% of free-flow speed</span></div>
        <div className="card"><h4>Live speed</h4><div className="bignum"><CountUp value={flow.current_speed} /><span className="unit">km/h</span></div><span className="sub">free {flow.free_flow_speed} · {flow.congestion_ratio < 0.5 ? 'jammed' : 'flowing'}</span></div>
        <div className="card"><h4>Predicted volume</h4><div className="bignum"><CountUp value={pred.final_vol} /><span className="unit">veh/hr</span></div><span className="sub">{lvl} · delay +{flow.delay_min} min</span></div>
        <div className="card"><h4>Weather</h4><div className="bignum">{(weather.temp - 273.15).toFixed(0)}<span className="unit">°C · {weather.weather_main}</span></div><span className="sub">rain {weather.rain_1h} mm/h · clouds {weather.clouds_all}%</span></div>
      </div>
      <h3>
        <span className="badge" style={{ background: colors[lvl] }}>{LEVEL_EMOJI[lvl]} {lvl}</span>
        {(rain || holiday) && <span className="tag">demo override active</span>}
      </h3>
      <p className="verdict">📣 City transit verdict: <b>{verdict}</b></p>
      <p className="muted">Sources: {flow.source} / {weather.source}</p>
      <FusionCard flow={flow} pred={pred} />
      {data.causes?.causes?.length > 0 && (
        <div className="card">
          <h4>🧐 Why is there traffic here? (ranked causes)</h4>
          {data.causes.causes.map((c, i) => (
            <div key={i} className="row" style={{ alignItems: 'flex-start' }}>
              <span>{c.kind === 'incident' ? '🚧' : c.kind === 'weather' ? '🌧' : c.kind === 'demand' ? '👥' : c.kind === 'bottleneck' ? '🚗' : '✅'}</span>
              <span><b>{i + 1}. {c.title}</b><br /><span className="muted">{c.detail}</span></span>
            </div>
          ))}
          <p className="muted">Incidents: {data.causes.incidents_source || 'key needed'} · demand & weather from live model inputs.</p>
        </div>
      )}
      <div className="cards">
        <div className="card"><h4>Legacy (history only)</h4><div className="bignum">{pred.base_vol}<span className="unit">veh/hr</span></div><span className="sub">static US baseline · {pred.base_level}</span></div>
        <div className="card"><h4>Upgraded (live fused)</h4><div className="bignum">{pred.final_vol}<span className="unit">veh/hr</span></div><span className="sub">{pred.base_vol ? `${pred.final_vol >= pred.base_vol ? '+' : ''}${Math.round((pred.final_vol - pred.base_vol) / pred.base_vol * 100)}% real-world correction` : ''} · {lvl}</span></div>
      </div>
      {ev && (
        <div className="card">
          <h4>🏮 Upcoming: {ev.title} — {ev.date}</h4>
          <span>{ev.start}–{ev.end} · {(ev.attendees || 0).toLocaleString()}+ expected · surge ×{ev.surge}</span>
          <div><button onClick={() => goCalendar && goCalendar()}>Open event calendar →</button></div>
        </div>
      )}
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
