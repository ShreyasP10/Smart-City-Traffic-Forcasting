import Autocomplete from './Autocomplete'
import { api } from '../api'

// Free-text city field: known cities first, then any geocoded world city.
export default function CityField({ cities, value, custom, onPick }) {
  const display = value === 'Custom' ? custom?.name || '' : value || ''
  const suggest = async (q, popular) => {
    if (popular) {
      return (cities || []).filter((c) => !c.name.startsWith('Minneapolis'))
        .slice(0, 5).map((c) => ({ key: c.name, icon: '🏙', primary: c.name, secondary: 'saved city', data: c.name }))
    }
    const known = (cities || []).filter((c) => c.name.toLowerCase().includes(q.toLowerCase()))
      .map((c) => ({ key: c.name, icon: '🏙', primary: c.name, secondary: 'saved city', data: c.name }))
    let geo = []
    try {
      const d = await api.geocode(q)
      geo = (d.results || []).slice(0, 4).map((g, i) => ({
        key: `g${i}-${g.lat}`, icon: '📍', primary: g.name, secondary: g.country, data: g,
      }))
    } catch { /* offline */ }
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
