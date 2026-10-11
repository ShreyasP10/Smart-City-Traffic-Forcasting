import { useEffect, useState } from 'react'
import { api } from '../api'

export default function SettingsModal({ open, onClose, onSaved }) {
  const [tt, setTt] = useState('')
  const [ow, setOw] = useState('')
  const [health, setHealth] = useState(null)
  const [msg, setMsg] = useState('')
  useEffect(() => {
    if (open) api.health().then(setHealth).catch(() => {})
  }, [open ])
  if (!open) return null
  const save = async () => {
    const r = await api.setKeys(tt.trim(), ow.trim()).catch(() => null)
    if (r?.ok) {
      setMsg('Keys saved — backend is now live where keys validate.')
      setHealth({ ok: true, tomtom: r.tomtom, openweather: r.openweather })
      onSaved && onSaved()
    } else setMsg('Save failed — is the backend running?')
    setTimeout(() => setMsg(''), 3000)
  }
  return (
    <div className="modalbg" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <h3>⚙ Telemetry & model settings</h3>
        <p>
          Backend: {health ? (health.tomtom || health.openweather ? '🟢 live sources connected' : '🟡 simulated mode') : '…'}
          {health && ` (TomTom ${health.tomtom ? '✅' : '—'} · Weather ${health.openweather ? '✅' : '—'})`}
        </p>
        <label>TomTom API key <input type="password" placeholder="paste key value, not key ID" value={tt} onChange={(e) => setTt(e.target.value)} /></label>
        <label>OpenWeather key <input type="password" placeholder="optional — Open-Meteo fallback covers" value={ow} onChange={(e) => setOw(e.target.value)} /></label>
        <div className="row">
          <button className="primary" onClick={save}>Save keys</button>
          <button onClick={onClose}>Close</button>
        </div>
        {msg && <p className="ok">{msg}</p>}
        <p className="muted">Keys live in backend memory only (never written to disk by this dialog).</p>
      </div>
    </div>
  )
}
