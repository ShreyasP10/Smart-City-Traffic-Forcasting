import { useEffect, useState } from 'react'
import 'leaflet/dist/leaflet.css'
import { CircleMarker, MapContainer, Polyline, Popup, TileLayer, Tooltip, useMapEvents } from 'react-leaflet'
import { api, LEVEL_HEX } from '../api'

function ClickSetter({ target, onPick }) {
  useMapEvents({
    click(e) {
      if (target) onPick({ lat: +e.latlng.lat.toFixed(5), lon: +e.latlng.lng.toFixed(5) })
    },
  })
  return null
}

import Autocomplete from '../components/Autocomplete'

const POPULAR = [
  { name: 'Andheri West', lat: 19.1136, lon: 72.8697 },
  { name: 'BKC', lat: 19.0596, lon: 72.8656 },
  { name: 'Dadar', lat: 19.0176, lon: 72.8428 },
  { name: 'Colaba', lat: 18.922, lon: 72.8347 },
  { name: 'Connaught Place, Delhi', lat: 28.6315, lon: 77.2167 },
]

function SearchBox({ label, value, onSelect }) {
  const suggest = async (q, popular) => {
    if (popular) {
      return POPULAR.map((p) => ({ key: p.name, icon: '⭐', primary: p.name, secondary: 'popular', data: p }))
    }
    let geo = []
    try {
      const d = await api.geocode(q)
      geo = (d.results || []).slice(0, 5).map((g, i) => ({
        key: `g${i}-${g.lat}`, icon: '📍', primary: g.name, secondary: g.country, data: { name: g.name, lat: g.lat, lon: g.lon },
      }))
    } catch { /* offline */ }
    return geo
  }
  return (
    <label>{label}
      <Autocomplete
        placeholder="Type any place…"
        initial={value?.name || ''}
        getSuggestions={suggest}
        onPick={(it) => onSelect(it.data)}
      />
    </label>
  )
}

export default function CorridorPlanner() {
  const [src, setSrc] = useState({ name: 'Andheri', lat: 19.1136, lon: 72.8697 })
  const [dst, setDst] = useState({ name: 'BKC', lat: 19.0596, lon: 72.8656 })
  const [pick, setPick] = useState(null)
  const [out, setOut] = useState(null)
  const [loading, setLoading] = useState(false)

  const go = async (a = src, b = dst) => {
    if (!a || !b) return
    setLoading(true)
    try { setOut(await api.corridor(a.lat, a.lon, b.lat, b.lon)) } catch { setOut(null) }
    setLoading(false)
  }
  useEffect(() => { go() }, [])

  const center = [(src.lat + dst.lat) / 2, (src.lon + dst.lon) / 2]
  const lvl = out?.pred.final_level
  return (
    <div>
      <h3>🛣 Route planner — any source, any destination</h3>
      <div className="row">
        <SearchBox label="Source" value={src} onSelect={setSrc} />
        <button title="Swap" onClick={() => { setSrc(dst); setDst(src) }}>⇄</button>
        <SearchBox label="Destination" value={dst} onSelect={setDst} />
        <button className="primary" onClick={() => go()} disabled={loading}>{loading ? 'Routing…' : 'Plan route →'}</button>
      </div>
      <div className="pills">
        <button className={pick === 'src' ? 'active' : ''} onClick={() => setPick(pick === 'src' ? null : 'src')}>📍 Click map = source</button>
        <button className={pick === 'dst' ? 'active' : ''} onClick={() => setPick(pick === 'dst' ? null : 'dst')}>📍 Click map = destination</button>
        {pick && <span className="muted">Click anywhere on the map…</span>}
      </div>
      <div className="mapwrap">
      <MapContainer center={center} zoom={12} style={{ height: 380, minHeight: 380, borderRadius: 12 }} key={`${center[0]}-${center[1]}`}>
        <TileLayer url="https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png" />
        <ClickSetter target={pick} onPick={(p) => {
          const named = { ...p, name: `Pinned (${p.lat.toFixed(3)}, ${p.lon.toFixed(3)})` }
          if (pick === 'src') setSrc(named); else setDst(named)
          setPick(null)
        }} />
        <Polyline positions={[[src.lat, src.lon], [dst.lat, dst.lon]]} pathOptions={{ color: lvl ? LEVEL_HEX[lvl] : '#0284c7', weight: 5, opacity: 0.85, dashArray: out?.live_routing ? null : '8 8' }} />
        {[{ ...src, tag: 'A · source' }, { ...dst, tag: 'B · destination' }].map((p, i) => (
          <CircleMarker key={i} center={[p.lat, p.lon]} radius={10}
            pathOptions={{ color: '#fff', weight: 2, fillColor: i === 0 ? '#0284c7' : '#7c3aed', fillOpacity: 1 }}>
            <Tooltip permanent>{p.tag}</Tooltip>
            <Popup><b>{p.name}</b><br />{p.lat.toFixed(4)}, {p.lon.toFixed(4)}</Popup>
          </CircleMarker>
        ))}
      </MapContainer>
      </div>
      {!out ? <p>{loading ? 'Routing…' : 'Plan a route to see prediction.'}</p> : (
        <>
          <div className="cards">
            <div className="card"><h4>Distance</h4><div className="bignum">{out.route.dist_km}<span className="unit">km</span></div><span className="sub">{out.live_routing ? 'live TomTom routing' : 'haversine estimate'}</span></div>
            <div className="card"><h4>Travel time</h4><div className="bignum">{out.route.live_min}<span className="unit">min</span></div><span className="sub">free-flow {out.route.free_min} · delay +{out.route.delay_min}</span></div>
            <div className="card"><h4>Mid-corridor</h4><div className="bignum">{out.pred.final_vol}<span className="unit">veh/hr</span></div><span className="sub">{lvl} congestion</span></div>
          </div>
          <p className="verdict">💡 {out.advice}</p>
        </>
      )}
    </div>
  )
}
