import { useEffect, useState } from 'react'
import 'leaflet/dist/leaflet.css'
import { CircleMarker, MapContainer, Polyline, Popup, TileLayer, Tooltip, useMap, useMapEvents } from 'react-leaflet'
import { api, LEVEL_HEX } from '../api'
import Autocomplete from '../components/Autocomplete'

function ClickSetter({ target, onPick }) {
  useMapEvents({
    click(e) {
      if (target) onPick({ lat: +e.latlng.lat.toFixed(5), lon: +e.latlng.lng.toFixed(5) })
    },
  })
  return null
}

function FlyTo({ center }) {
  const map = useMap()
  useEffect(() => {
    if (center && center[0] && center[1]) {
      map.flyTo(center, 12, { duration: 0.8 })
    }
  }, [center])
  return null
}

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
  const [opt, setOpt] = useState(0)
  const [loading, setLoading] = useState(false)

  const go = async (a = src, b = dst) => {
    if (!a || !b) return
    setLoading(true)
    try {
      const r = await api.corridor(a.lat, a.lon, b.lat, b.lon, 2)
      setOut(r)
      setOpt(0)
    } catch { setOut(null) }
    setLoading(false)
  }
  useEffect(() => { go() }, [])

  const center = [
    (Number(src?.lat ?? 19.1136) + Number(dst?.lat ?? 19.0596)) / 2,
    (Number(src?.lon ?? 72.8697) + Number(dst?.lon ?? 72.8656)) / 2,
  ]
  const lvl = out?.pred?.final_level
  const selOpt = out ? ((out.options || [out.route]).find((o) => (o.id ?? 0) === opt) || out.route) : null
  const line = selOpt?.geometry?.length
    ? selOpt.geometry
    : (src?.lat != null && dst?.lat != null ? [[src.lat, src.lon], [dst.lat, dst.lon]] : [])
  const roadAligned = !!(selOpt?.geometry?.length)
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
      <MapContainer center={center} zoom={12} style={{ height: 380, minHeight: 380, borderRadius: 12 }}>
        <TileLayer url="https://tile.openstreetmap.org/{z}/{x}/{y}.png" />
        <FlyTo center={center} />
        <ClickSetter target={pick} onPick={(p) => {
          const named = { ...p, name: `Pinned (${p.lat.toFixed(3)}, ${p.lon.toFixed(3)})` }
          if (pick === 'src') setSrc(named); else setDst(named)
          setPick(null)
        }} />
        {src?.lat != null && dst?.lat != null && (
          <Polyline positions={line} pathOptions={{ color: lvl ? LEVEL_HEX[lvl] : '#0284c7', weight: 5, opacity: 0.9, dashArray: roadAligned ? null : '8 8' }} />
        )}
        {[{ ...src, tag: 'A · source' }, { ...dst, tag: 'B · destination' }].filter((p) => p && p.lat != null && p.lon != null).map((p, i) => (
          <CircleMarker key={i} center={[p.lat, p.lon]} radius={10}
            pathOptions={{ color: '#fff', weight: 2, fillColor: i === 0 ? '#0284c7' : '#7c3aed', fillOpacity: 1 }}>
            <Tooltip permanent>{p.tag}</Tooltip>
            <Popup><b>{p.name || 'Point'}</b><br />{Number(p.lat).toFixed(4)}, {Number(p.lon).toFixed(4)}</Popup>
          </CircleMarker>
        ))}
      </MapContainer>
      </div>
      {!out ? <p>{loading ? 'Routing…' : 'Plan a route to see prediction.'}</p> : (
        <>
          <h4>Select path ({(out.options || [out.route]).length} options{out.live_routing ? ' · live TomTom' : ' · estimated'}{roadAligned ? ' · road-aligned 🛣' : ''})</h4>
          <div className="pills">
            {(out.options || [out.route]).map((o, i) => (
              <button key={o.id ?? i} className={opt === (o.id ?? i) ? 'active' : ''}
                onClick={() => setOpt(o.id ?? i)}>
                {o.label || `Path ${i + 1}`} · {o.live_min} min · {o.dist_km} km
              </button>
            ))}
          </div>
          {(() => {
            const sel = (out.options || [out.route]).find((o) => (o.id ?? 0) === opt) || out.route || {}
            return (
              <>
                <div className="cards">
                  <div className="card"><h4>Distance ({sel.label || 'Path'})</h4><div className="bignum">{sel.dist_km ?? '—'}<span className="unit">km</span></div><span className="sub">{out.live_routing ? 'live TomTom routing' : 'haversine estimate'}</span></div>
                  <div className="card"><h4>Travel time</h4><div className="bignum">{sel.live_min ?? '—'}<span className="unit">min</span></div><span className="sub">free-flow {sel.free_min ?? '—'} · delay +{sel.delay_min ?? '—'}</span></div>
                  <div className="card"><h4>Mid-corridor</h4><div className="bignum">{out.pred?.final_vol ?? '—'}<span className="unit">veh/hr</span></div><span className="sub">{lvl || 'Normal'} congestion</span></div>
                </div>
                <p className="verdict">💡 {out.advice || 'Corridor flowing.'}</p>
              </>
            )
          })()}
        </>
      )}
    </div>
  )
}
