import { useEffect, useState } from 'react'
import { api } from '../api'

export default function SettingsModal({ open, onClose, onSaved }) {
  const [tt, setTt] = useState('')
  const [ow, setOw] = useState('')
  const [health, setHealth] = useState(null)
  const [model, setModel] = useState(null)
  const [msg, setMsg] = useState('')
  const [file, setFile] = useState(null)
  const [hp, setHp] = useState({ num_trees: 60, max_depth: 6, lr: 0.05 })
  const [training, setTraining] = useState(false)
  const [trainMsg, setTrainMsg] = useState('')
  useEffect(() => {
    if (open) {
      api.health().then(setHealth).catch(() => {})
      api.model().then(setModel).catch(() => {})
    }
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
        <h3>🧠 Model training (UCI CSV format)</h3>
        {model && <p className="muted">Active: {model.backend} · MAE {model.metrics?.base_mae?.toFixed(1)}{model.custom ? ` · custom (${model.source})` : ' · stock bundle'}</p>}
        <label>Training CSV <input type="file" accept=".csv" onChange={(e) => setFile(e.target.files[0] || null)} /></label>
        <p className="muted">Columns: date_time, traffic_volume (+ holiday, temp, rain_1h, snow_1h, clouds_all, weather_main). ≥500 rows.</p>
        <div className="row">
          <label>Trees <input type="number" min="10" max="150" value={hp.num_trees} onChange={(e) => setHp((s) => ({ ...s, num_trees: +e.target.value }))} /></label>
          <label>Depth <input type="number" min="2" max="8" value={hp.max_depth} onChange={(e) => setHp((s) => ({ ...s, max_depth: +e.target.value }))} /></label>
          <label>LR <input type="number" min="0.01" max="0.3" step="0.01" value={hp.lr} onChange={(e) => setHp((s) => ({ ...s, lr: +e.target.value }))} /></label>
        </div>
        <div className="row">
          <button className="primary" disabled={!file || training} onClick={async () => {
            setTraining(true)
            setTrainMsg('Training in browser… ~1 min for 48k rows. Do not close.')
            try {
              const r = await api.retrain(file, hp)
              setTrainMsg(`✅ Trained ${r.backend}: MAE ${r.metrics.base_mae.toFixed(1)} — live bundle swapped.`)
              api.model().then(setModel).catch(() => {})
              onSaved && onSaved()
            } catch (e) {
              setTrainMsg(`❌ Train failed: ${(e?.response?.data?.detail) || e.message}`)
            }
            setTraining(false)
          }}>{training ? 'Training…' : 'Retrain in-app →'}</button>
        </div>
        {trainMsg && <p className={trainMsg.startsWith('✅') ? 'ok' : ''}>{trainMsg}</p>}
      </div>
    </div>
  )
}
