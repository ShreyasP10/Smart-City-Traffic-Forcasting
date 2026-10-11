import Autocomplete from './Autocomplete'
import { api } from '../api'

// Free-text corridor field: type anything — matches predefined corridors,
// falls back to geocoding the text into a map point.
export default function CorridorField({ label, value, routes, onPick }) {
  const display = typeof value === 'object' ? value?.name || '' : value || ''
  const suggest = async (q, popular) => {
    const list = routes || []
    if (popular) {
      return list.slice(0, 5).map((r) => ({ key: r, icon: '🛣', primary: r, secondary: 'corridor', data: r }))
    }
    const matches = list.filter((r) => r.toLowerCase().includes(q.toLowerCase())).slice(0, 5)
      .map((r) => ({ key: r, icon: '🛣', primary: r, secondary: 'corridor', data: r }))
    let geo = []
    try {
      const d = await api.geocode(q)
      geo = (d.results || []).slice(0, 4).map((g, i) => ({
        key: `g${i}-${g.lat}`, icon: '📍', primary: g.name, secondary: g.country,
        data: { name: g.name, lat: g.lat, lon: g.lon },
      }))
    } catch { /* offline */ }
    return [...matches, ...geo]
  }
  return (
    <label>{label}
      <Autocomplete
        placeholder="Type corridor or any place…"
        initial={display}
        getSuggestions={suggest}
        onPick={(it) => onPick(it.data)}
      />
    </label>
  )
}
