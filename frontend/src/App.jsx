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

import CorridorPlanner from './views/CorridorPlanner'
import CorridorField from './components/CorridorField'
import CityField from './components/CityField'
import ErrorBoundary from './components/ErrorBoundary'

const TABS = [
  ['overview', '📍 Overview'],
  ['route', '🛣 Route'],
  ['forecast', '🔮 Forecast'],
  ['heatmap', '🗺 Map'],
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
  const [dow, setDow] = useState('')
  const [rainMm, setRainMm] = useState(0)
  const [holiday, setHoliday] = useState(false)
  const [tab, setTab] = useState('overview')
  const [data, setData] = useState({})
  const [loading, setLoading] = useState(false)
  const [custom, setCustom] = useState(null)
  const [settings, setSettings] = useState(false)
  const [live, setLive] = useState(null)
  const [countdown, setCountdown] = useState(60)
  const [tick, setTick] = useState(0)
  const [toast, setToast] = useState(null)
  const [model, setModel] = useState(null)
  const [backendDown, setBackendDown] = useState(false)
  const [theme, setTheme] = useState(() => {
    try { return localStorage.getItem('traffic-theme') || 'light' } catch { return 'light' }
  })

  useEffect(() => {
    document.documentElement.dataset.theme = theme === 'dark' ? 'dark' : 'light'
    try { localStorage.setItem('traffic-theme', theme) } catch { /* private mode */ }
  }, [theme])

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
    api.health().then((h) => { setLive(h); setBackendDown(false) }).catch(() => setBackendDown(true))
    api.model().then(setModel).catch(() => {})
    api.model().then(setModel).catch(() => {})
  }, [])

  const routes = cities.find((c) => c.name === city)?.routes ?? []
  const routeIsPoint = route && typeof route === 'object'
  const routeName = routeIsPoint ? route.name : route
  const routeLL = routeIsPoint ? { lat: route.lat, lon: route.lon }
    : custom && city === 'Custom' ? { lat: custom.lat, lon: custom.lon } : {}
  const apiCity = city === 'Custom' ? 'Mumbai' : city
  const ov = { rain_mm: rainMm, holiday, ...routeLL, ...(dow === '' ? {} : { dow: Number(dow) }) }

  const setRoutePick = (v) => {
    if (typeof v === 'object') setRoute(v)
    else setRoute(v)
  }

  useEffect(() => {
    if (!city || (!route && city !== 'Custom')) return
    setLoading(true)
    const h = hour === '' ? null : Number(hour)
    const base = { ...ov, ...(h != null ? { hour: h } : {}) }
    const rname = routeIsPoint ? 'Andheri → BKC' : route
    const jobs = {
      overview: async () => {
        const [o, f] = await Promise.all([
          api.overview(apiCity, rname, { ...base, ...routeLL }),
          api.forecast(apiCity, rname, { ...base, ...routeLL }).catch(() => null),
        ])
        if (f) setData((s) => ({ ...s, forecast: f }))
        return o
      },
      route: () => Promise.resolve(true),
      forecast: () => api.forecast(apiCity, rname, { ...base, ...routeLL }),
      heatmap: async () => {
        const [m, f] = await Promise.all([
          api.heatmap(apiCity),
          api.forecast(apiCity, rname, { ...base, ...routeLL }).catch(() => null),
        ])
        if (f) setData((s) => ({ ...s, forecast: f }))
        return m
      },
      patterns: () => api.patterns(),
      sim: () => Promise.resolve(true),
      diag: () => Promise.resolve(true),
      adapt: () => api.adaptation(apiCity, rname, routeLL),
      calendar: () => Promise.resolve(true),
      alerts: () => api.forecast(apiCity, rname, { ...base, ...routeLL }),
      compare: () => Promise.resolve(true),
    }
    const runner = jobs[tab] || (() => Promise.resolve(true))
    runner().then((d) => setData((s) => ({ ...s, [tab]: d }))).catch(() => {}).finally(() => setLoading(false))
  }, [tab, city, route, hour, dow, rainMm, holiday, custom, tick])

  useEffect(() => {
    const id = setInterval(() => setCountdown((c) => {
      if (c <= 1) { setTick((t) => t + 1); return 60 }
      return c - 1
    }), 1000)
    return () => clearInterval(id)
  }, [])

  const pickCity = (c) => {
    if (typeof c === 'object') {
      setCustom(c)
      setCity('Custom')
      return
    }
    setCity(c)
    setCustom(null)
    const rs = cities.find((x) => x.name === c)?.routes ?? []
    if (rs.length) setRoute(rs[0].name)
  }

  useEffect(() => {
    const p = data.overview?.pred
    if (p && p.final_level === 'Severe') {
      const lbl = routeIsPoint ? route.name : route
      const id = `${city}-${lbl}-${data.overview.at}`
      setToast((t) => (t && t.id === id ? t : { id, title: `Severe congestion — ${city}`, desc: `${lbl}: ${p.final_vol.toLocaleString()} veh/hr right now.` }))
    }
  }, [data.overview])

  useEffect(() => {
    if (!toast) return
    const t = setTimeout(() => setToast(null), 6000)
    return () => clearTimeout(t)
  }, [toast])

  return (
    <div className="app">
      <header>
        <h1>🚦 Smart City Traffic Forecasting</h1>
        <p className="muted">US-trained XGBoost → live TomTom → calibrated for Indian cities
          {live && (live.tomtom || live.openweather ? ' · 🟢 live' : ' · 🟡 simulated')}</p>
      </header>
      <div className="controls">
        <CityField cities={cities} value={city} custom={custom} onPick={pickCity} />
        <CorridorField label="Corridor — type anything" value={routeName} routes={routes.map((r) => r.name)} onPick={setRoutePick} />
        <div className="dowpills">
          <span className="dowlabel">Day</span>
          <div className="pills">
            <button className={dow === '' ? 'active' : ''} onClick={() => setDow('')}>Today</button>
            {['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'].map((d, i) => (
              <button key={d} className={dow === i ? 'active' : ''} onClick={() => setDow(dow === i ? '' : i)}>{d}</button>
            ))}
          </div>
        </div>
        <label>Hour override
          <select value={hour} onChange={(e) => setHour(e.target.value)}>
            <option value="">now</option>
            {Array.from({ length: 24 }, (_, i) => <option key={i} value={i}>{String(i).padStart(2, '0')}:00</option>)}
          </select>
        </label>
        <div className="raincard">
          <span className="dowlabel">Rain</span>
          <div className="pills">
            {[{ v: 0, l: '☀️ Dry' }, { v: 1.5, l: '🌦 Drizzle' }, { v: 8, l: '🌧 Heavy' }].map((o) => (
              <button key={o.l} className={rainMm === o.v ? 'active' : ''} onClick={() => setRainMm(o.v)} title={o.v ? `${o.v} mm/h in model` : 'no rain'}>{o.l}</button>
            ))}
          </div>
        </div>
        <button className={holiday ? 'activebtn' : ''} onClick={() => setHoliday(!holiday)} title="Public-holiday traffic pattern (lighter commute, event peaks)">
          🎉 Holiday{holiday ? ' ON' : ''}
        </button>
        <button onClick={() => { setTick((t) => t + 1); setCountdown(60) }}>⟳ {countdown}s</button>
        <button onClick={() => setTheme((t) => (t === 'light' ? 'dark' : 'light'))} title="Toggle dark / light theme">
          {theme === 'light' ? '🌙 Dark' : '☀️ Light'}
        </button>
        <button onClick={() => setSettings(true)}>⚙ Settings</button>
      </div>
      {custom && city === 'Custom' && <p className="muted">📍 Custom point {custom.name} ({custom.lat.toFixed(4)}, {custom.lon.toFixed(4)}) — Overview + Forecast use live data here.</p>}
      {routeIsPoint && <p className="muted">🛣 Free corridor point: {route.name} ({route.lat.toFixed(4)}, {route.lon.toFixed(4)}) — geocoded, not predefined.</p>}
      <nav className="tabs">
        {TABS.map(([k, label]) => (
          <button key={k} className={tab === k ? 'active' : ''} onClick={() => setTab(k)}>{label}</button>
        ))}
      </nav>
      <main>
        {backendDown && (
          <div className="alert">
            <b>🔴 Backend offline — this page can't load data.</b>
            <p>Open a second terminal in the project folder and run:</p>
            <p><code>python -m uvicorn api:app --port 8000</code></p>
            <button onClick={() => {
              api.health()
                .then((h) => { setLive(h); setBackendDown(false); setTick((t) => t + 1) })
                .catch(() => setBackendDown(true))
            }}>Retry connection →</button>
          </div>
        )}
        {loading && <p className="muted">Fetching live data…</p>}
        {!loading && backendDown && <p className="muted">City suggestions still work (world search runs in your browser). Predictions need the backend.</p>}
        <ErrorBoundary key={tab}>
        {tab === 'overview' && <Overview data={data.overview} forecast={data.forecast} city={city} route={routeName} rain={rainMm > 0} holiday={holiday} goCalendar={() => setTab('calendar')} />}
        {tab === 'route' && <CorridorPlanner />}
        {tab === 'forecast' && <Forecast data={data.forecast} />}
        {tab === 'heatmap' && <Heatmap data={data.heatmap} city={city} forecast={data.forecast} dark={theme === 'dark'} />}
        {tab === 'patterns' && <Patterns data={data.patterns} />}
        {tab === 'sim' && <Simulator />}
        {tab === 'diag' && <Diagnostics />}
        {tab === 'adapt' && <Adaptation data={data.adapt} />}
        {tab === 'calendar' && <Calendar />}
        {tab === 'alerts' && <Alerts forecast={data.alerts} city={city} />}
        {tab === 'compare' && <Compare />}
        </ErrorBoundary>
      </main>
      <SettingsModal open={settings} onClose={() => setSettings(false)} onSaved={() => { api.health().then((h) => { setLive(h); setBackendDown(false) }).catch(() => {}); setTick((t) => t + 1) }} />
      {toast && (
        <div className="toast" onClick={() => setToast(null)}>
          <b>🔔 {toast.title}</b>
          <span>{toast.desc}</span>
        </div>
      )}
      <footer className="foot">
        <span>TrafficSense · live congestion forecasting</span>
        {model && <span className="muted">{model.backend} · MAE {model.metrics?.base_mae?.toFixed(0)}{model.custom ? ' · custom-trained' : ''} · {live && (live.tomtom ? 'TomTom live' : 'simulated')}</span>}
        <button onClick={() => setSettings(true)}>Model settings →</button>
      </footer>
    </div>
  )
}
