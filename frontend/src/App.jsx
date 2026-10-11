import { useEffect, useState } from 'react'
import { api } from './api'
import Overview from './views/Overview'
import Forecast from './views/Forecast'
import Heatmap from './views/Heatmap'
import Patterns from './views/Patterns'
import Adaptation from './views/Adaptation'
import Compare from './views/Compare'
import Simulator from './views/Simulator'
import Diagnostics from './views/Diagnostics'
import Alerts from './views/Alerts'
import Calendar from './views/Calendar'
import SettingsModal from './components/SettingsModal'

const TABS = [
  ['overview', '📍 Overview'],
  ['forecast', '🔮 Forecast'],
  ['heatmap', '🗺 Heatmap'],
  ['patterns', '📊 Patterns'],
  ['sim', '🧪 Simulator'],
  ['diag', '🔬 Diagnostics'],
  ['adapt', '🔁 Adaptation'],
  ['calendar', '🏮 Calendar'],
  ['alerts', '🚨 Alerts'],
  ['compare', '⚖ Compare'],
]

export default function App() {
  const [cities, setCities] = useState([])
  const [city, setCity] = useState('Mumbai')
  const [route, setRoute] = useState('Andheri → BKC')
  const [hour, setHour] = useState('')
  const [rain, setRain] = useState(false)
  const [holiday, setHoliday] = useState(false)
  const [tab, setTab] = useState('overview')
  const [data, setData] = useState({})
  const [loading, setLoading] = useState(false)
  const [cmpA, setCmpA] = useState('')
  const [cmpB, setCmpB] = useState('')
  const [custom, setCustom] = useState(null)
  const [query, setQuery] = useState('')
  const [geo, setGeo] = useState([])
  const [settings, setSettings] = useState(false)
  const [live, setLive] = useState(null)
  const [countdown, setCountdown] = useState(60)
  const [tick, setTick] = useState(0)

  useEffect(() => {
    api.cities().then((d) => {
      setCities(d.cities)
      const m = d.cities.find((c) => c.name === 'Mumbai')
      if (m) {
        setRoute(m.routes[0].name)
        setCmpA(m.routes[0].name)
        setCmpB(m.routes[1]?.name ?? m.routes[0].name)
      }
    }).catch(() => {})
    api.health().then(setLive).catch(() => {})
  }, [])

  const routes = cities.find((c) => c.name === city)?.routes ?? []
  const ll = custom && city === 'Custom' ? { lat: custom.lat, lon: custom.lon } : {}
  const ov = { rain, holiday, ...ll }

  useEffect(() => {
    if (!city || (!route && city !== 'Custom')) return
    setLoading(true)
    const h = hour === '' ? null : Number(hour)
    const jobs = {
      overview: async () => {
        const [o, f] = await Promise.all([
          api.overview(city === 'Custom' ? 'Mumbai' : city, route, { ...ov, ...(h != null ? { hour: h } : {}) }),
          api.forecast(city === 'Custom' ? 'Mumbai' : city, route, ov).catch(() => null),
        ])
        if (f) setData((s) => ({ ...s, forecast: f }))
        return o
      },
      forecast: () => api.forecast(city === 'Custom' ? 'Mumbai' : city, route, ov),
      heatmap: async () => {
        const [m, f] = await Promise.all([
          api.heatmap(city === 'Custom' ? 'Mumbai' : city),
          api.forecast(city === 'Custom' ? 'Mumbai' : city, route, ov).catch(() => null),
        ])
        if (f) setData((s) => ({ ...s, forecast: f }))
        return m
      },
      patterns: () => api.patterns(),
      sim: () => Promise.resolve(true),
      diag: () => Promise.resolve(true),
      adapt: () => api.adaptation(city, route),
      calendar: () => Promise.resolve(true),
      alerts: () => api.forecast(city === 'Custom' ? 'Mumbai' : city, route, ov),
      compare: () => (cmpA && cmpB ? api.compare(city, cmpA, cmpB) : Promise.resolve(null)),
    }
    jobs[tab]().then((d) => setData((s) => ({ ...s, [tab]: d }))).catch(() => {}).finally(() => setLoading(false))
  }, [tab, city, route, hour, rain, holiday, cmpA, cmpB, custom, tick])

  useEffect(() => {
    const id = setInterval(() => setCountdown((c) => {
      if (c <= 1) { setTick((t) => t + 1); return 60 }
      return c - 1
    }), 1000)
    return () => clearInterval(id)
  }, [])

  const search = async (q) => {
    setQuery(q)
    if (q.length < 3) { setGeo([]); return }
    try { setGeo((await api.geocode(q)).results || []) } catch { setGeo([]) }
  }

  return (
    <div className="app">
      <header>
        <h1>🚦 Smart City Traffic Forecasting</h1>
        <p className="muted">US-trained XGBoost → live TomTom → calibrated for Indian cities
          {live && (live.tomtom || live.openweather ? ' · 🟢 live' : ' · 🟡 simulated')}</p>
      </header>
      <div className="controls">
        <label>City
          <select value={city} onChange={(e) => {
            const c = e.target.value
            setCity(c)
            setCustom(c === 'Custom' ? custom : null)
            const rs = cities.find((x) => x.name === c)?.routes ?? []
            if (rs.length) {
              setRoute(rs[0].name)
              setCmpA(rs[0].name)
              setCmpB(rs[1]?.name ?? rs[0].name)
            }
          }}>
            {cities.map((c) => <option key={c.name} value={c.name}>{c.name}</option>)}
            {custom && <option value="Custom">{custom.name} (custom)</option>}
          </select>
        </label>
        {city !== 'Custom' && (
          <label>Route
            <select value={route} onChange={(e) => setRoute(e.target.value)}>
              {routes.map((r) => <option key={r.name} value={r.name}>{r.name}</option>)}
            </select>
          </label>
        )}
        <label>Hour override
          <select value={hour} onChange={(e) => setHour(e.target.value)}>
            <option value="">now</option>
            {Array.from({ length: 24 }, (_, i) => <option key={i} value={i}>{String(i).padStart(2, '0')}:00</option>)}
          </select>
        </label>
        <label className="check"><input type="checkbox" checked={rain} onChange={(e) => setRain(e.target.checked)} /> 🌧 Simulate rain</label>
        <label className="check"><input type="checkbox" checked={holiday} onChange={(e) => setHoliday(e.target.checked)} /> 🎉 Holiday</label>
        <label>Any city (geocode)
          <input placeholder="Search global city…" value={query} onChange={(e) => search(e.target.value)} />
        </label>
        <button onClick={() => { setTick((t) => t + 1); setCountdown(60) }}>⟳ {countdown}s</button>
        <button onClick={() => setSettings(true)}>⚙ Settings</button>
      </div>
      {geo.length > 0 && (
        <div className="card">
          {geo.map((g, i) => (
            <div key={i} className="row">
              <span>{g.name} ({g.country})</span>
              <button onClick={() => { setCustom(g); setCity('Custom'); setGeo([]); setQuery('') }}>Use this location →</button>
            </div>
          ))}
        </div>
      )}
      {custom && city === 'Custom' && <p className="muted">📍 Custom point {custom.name} ({custom.lat.toFixed(4)}, {custom.lon.toFixed(4)}) — Overview + Forecast use live data here; other tabs show {city === 'Custom' ? 'Mumbai' : city} reference.</p>}
      <nav className="tabs">
        {TABS.map(([k, label]) => (
          <button key={k} className={tab === k ? 'active' : ''} onClick={() => setTab(k)}>{label}</button>
        ))}
      </nav>
      <main>
        {loading && <p className="muted">Fetching live data…</p>}
        {tab === 'overview' && <Overview data={data.overview} forecast={data.forecast} city={city} route={city === 'Custom' ? custom?.name : route} rain={rain} holiday={holiday} />}
        {tab === 'forecast' && <Forecast data={data.forecast} />}
        {tab === 'heatmap' && <Heatmap data={data.heatmap} city={city} forecast={data.forecast} />}
        {tab === 'patterns' && <Patterns data={data.patterns} />}
        {tab === 'sim' && <Simulator />}
        {tab === 'diag' && <Diagnostics />}
        {tab === 'adapt' && <Adaptation data={data.adapt} />}
        {tab === 'calendar' && <Calendar />}
        {tab === 'alerts' && <Alerts forecast={data.alerts} city={city} />}
        {tab === 'compare' && (
          <Compare data={data.compare} routes={routes.map((r) => r.name)} a={cmpA} b={cmpB} setA={setCmpA} setB={setCmpB} />
        )}
      </main>
      <SettingsModal open={settings} onClose={() => setSettings(false)} onSaved={() => { api.health().then(setLive).catch(() => {}); setTick((t) => t + 1) }} />
    </div>
  )
}
