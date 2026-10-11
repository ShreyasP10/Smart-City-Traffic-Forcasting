import Autocomplete from './Autocomplete'
import { api } from '../api'

// Free-text city field: known cities first, then any geocoded world city.
// FALLBACK_CITIES keeps suggestions alive even when the backend is down.
const FALLBACK_CITIES = ['Mumbai', 'Bengaluru', 'Delhi NCR', 'Pune']

export default function CityField({ cities, value, custom, onPick }) {
  const display = value === 'Custom' ? custom?.name || '' : value || ''
  const names = cities && cities.length ? cities.map((c) => c.name) : FALLBACK_CITIES
  const suggest = async (q, popular) => {
    if (popular) {
      return names.slice(0, 5).map((n) => ({ key: n, icon: '🏙', primary: n, secondary: 'saved city', data: n }))
    }
    const known = names.filter((n) => n.toLowerCase().includes(q.toLowerCase()))
      .map((n) => ({ key: n, icon: '🏙', primary: n, secondary: 'saved city', data: n }))
    let geo = []
    try {
      const d = await api.geocode(q)
      geo = (d.results || []).slice(0, 4).map((g, i) => ({
        key: `g${i}-${g.lat}`, icon: '📍', primary: g.name, secondary: g.country, data: g,
      }))
    } catch { /* fully offline */ }
    return [...known, ...geo]
  }
  return (
    <label>City — type anywhere
      <Autocomplete
        placeholder="Mumbai, Delhi, Tokyo…"
        initial={display}
        getSuggestions={suggest}
        onPick={(it) => onPick(it.data)}
      />
    </label>
  )
}
