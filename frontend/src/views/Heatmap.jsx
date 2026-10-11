import { useEffect, useRef, useState } from 'react'
import html2canvas from 'html2canvas'
import 'leaflet/dist/leaflet.css'
import { CircleMarker, MapContainer, Polyline, Popup, TileLayer, Tooltip, useMap } from 'react-leaflet'
import { api } from '../api'

function FlyTo({ center }) {
  const map = useMap()
  const key = `${center[0].toFixed(3)},${center[1].toFixed(3)}`
  const last = useRef(key)
  useEffect(() => {
    if (last.current !== key) {
      last.current = key
      map.flyTo(center, 11, { duration: 1.0 })
    }
  }, [key])
  return null
}

const HORIZONS = [0, 15, 30, 60]

export default function Heatmap({ data, city, forecast, dark }) {
  const [hz, setHz] = useState(0)
  const [mode, setMode] = useState('corridors')
  const [grid, setGrid] = useState(null)
  const [routes, setRoutes] = useState(null)
  const [raster, setRaster] = useState(false)
  const [tileUrl, setTileUrl] = useState(null)
  const [exporting, setExporting] = useState(false)
  const wrapRef = useRef(null)
  if (!data || !data.points) return <p>Loading heatmap data…</p>
  const pts = data.points
  const center = pts.length ? [pts[0].lat, pts[0].lon] : [19.076, 72.8777]
  const baseTiles = dark
    ? 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png'
    : 'https://tile.openstreetmap.org/{z}/{x}/{y}.png'

  useEffect(() => {
    api.tiles().then((t) => t.live && setTileUrl(t.url)).catch(() => {})
  }, [])

  useEffect(() => {
    if (mode === 'grid' && city) {
      api.grid(city, hz).then(setGrid).catch(() => setGrid(null))
    }
    if (mode === 'routes' && city) {
      api.routes(city, hz).then(setRoutes).catch(() => setRoutes(null))
    }
  }, [mode, city, hz])

  // Horizon: scale each point's volume by forecast trend ratio
  let ratio = 1
  if (hz > 0 && forecast?.rows?.length) {
    const now = forecast.rows[0]?.final_vol || 1
    const at = (forecast.rows.find((r) => r.in_min === hz) || {}).final_vol || now
    ratio = at / now
  }
  const adj = pts.map((p) => ({ ...p, dvol: Math.round(p.vol * ratio) }))
  const shown = mode === 'grid' && grid?.points
    ? grid.points.map((p) => ({ route: p.name, lat: p.lat, lon: p.lon, level: p.level, vol: p.vol, dvol: p.vol, speed: p.speed }))
    : adj
  const routeLines = mode === 'routes' && routes?.routes ? routes.routes : null

  const exportPng = async () => {
    if (!wrapRef.current) return
    setExporting(true)
    try {
      const canvas = await html2canvas(wrapRef.current, { useCORS: true, backgroundColor: '#0f172a', scale: 2 })
      const a = document.createElement('a')
      a.download = `traffic-${city}-T${hz}-${Date.now()}.png`
      a.href = canvas.toDataURL('image/png')
      a.click()
    } catch { /* tiles may taint canvas offline */ }
    setExporting(false)
  }

  return (
    <div>
      <h3>🗺 Congestion map — {data.city}</h3>
      <div className="maptools">
        <div className="pills">
          <button className={mode === 'corridors' ? 'active' : ''} onClick={() => setMode('corridors')}>Corridors</button>
          <button className={mode === 'routes' ? 'active' : ''} onClick={() => setMode('routes')}>Routes</button>
          <button className={mode === 'grid' ? 'active' : ''} onClick={() => setMode('grid')}>9-point grid</button>
        </div>
        <div className="pills">
          {HORIZONS.map((h) => (
            <button key={h} className={hz === h ? 'active' : ''} onClick={() => setHz(h)}>
              {h === 0 ? 'Live' : `+${h}m`}
            </button>
          ))}
        </div>
        <div className="pills">
          {tileUrl && (
            <button className={raster ? 'active' : ''} onClick={() => setRaster(!raster)}>
              🛰 TomTom flow overlay
            </button>
          )}
          <button onClick={exportPng} disabled={exporting}>{exporting ? '…' : '📷 Export PNG'}</button>
        </div>
      </div>
      <div ref={wrapRef}>
        <MapContainer center={center} zoom={11} preferCanvas style={{ height: 480, borderRadius: 12 }}>
          <TileLayer url={baseTiles} />
          {raster && tileUrl && <TileLayer url={tileUrl} opacity={0.65} zIndex={10} />}
          <FlyTo center={center} />
          {routeLines
            ? routeLines.map((r) => (
              <Polyline key={r.route} positions={r.geometry && r.geometry.length ? r.geometry : []}
                pathOptions={{ color: data.colors[r.level], weight: 6, opacity: 0.9 }}>
                <Popup>
                  <b>{r.route}</b><br />
                  Level: <b>{r.level}</b> · {r.vol.toLocaleString()} veh/hr<br />
                  Speed: {r.speed} km/h (free {r.free})<br />
                  {r.dist_km} km · {r.trip_min} min · delay +{r.delay_min} min
                </Popup>
              </Polyline>
            ))
            : shown.map((p) => (
            <CircleMarker
              key={p.route}
              center={[p.lat, p.lon]}
              radius={8 + Math.min(22, p.dvol / 250)}
              pathOptions={{ color: '#fff', weight: 2, fillColor: data.colors[p.level], fillOpacity: 0.85 }}
            >
              <Tooltip>{p.route}: {p.level} ({p.dvol.toLocaleString()})</Tooltip>
              <Popup>
                <b>{p.route}</b><br />
                Horizon: {hz === 0 ? 'Live now' : `+${hz} min forecast`}<br />
                Level: <b>{p.level}</b><br />
                Volume: {p.dvol.toLocaleString()} veh/hr<br />
                Speed: {p.speed} km/h
              </Popup>
            </CircleMarker>
          ))}
        </MapContainer>
      </div>
      <div className="legend">
        {['Low', 'Moderate', 'High', 'Severe'].map((l) => (
          <span key={l}><i style={{ background: data.colors[l] }} />{l}</span>
        ))}
        <span className="muted">bubble size ∝ volume · {hz === 0 ? 'live' : `forecast +${hz}m`}{data.points[0]?.source?.startsWith('sim') ? ' · simulated' : ''}</span>
      </div>
      {routeLines && (
        <>
          <h4>Every route — live stats</h4>
          <table className="tbl">
            <thead><tr><th>Route</th><th>Level</th><th>Volume</th><th>Speed</th><th>Trip</th><th>Delay</th></tr></thead>
            <tbody>
              {routeLines.map((r) => (
                <tr key={r.route}><td>{r.route}</td><td>{r.level}</td><td>{r.vol.toLocaleString()}</td><td>{r.speed} km/h</td><td>{r.trip_min} min</td><td>+{r.delay_min}</td></tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </div>
  )
}
