import axios from 'axios'

const get = async (path, params) => (await axios.get(path, { params })).data
const post = async (path, body) => (await axios.post(path, body)).data

export const api = {
  health: () => get('/api/health'),
  cities: () => get('/api/cities'),
  thresholds: () => get('/api/thresholds'),
  overview: (city, route, opts = {}) => get('/api/overview', { city, route, ...opts }),
  forecast: (city, route, opts = {}) => get('/api/forecast', { city, route, ...opts }),
  heatmap: (city) => get('/api/heatmap', { city }),
  grid: (city, horizon) => get('/api/grid', { city, horizon }),
  routes: (city, horizon) => get('/api/routes', { city, horizon }),
  patterns: () => get('/api/patterns'),
  adaptation: (city, route, opts = {}) => get('/api/adaptation', { city, route, ...opts }),
  compare: (city, a, b) => get('/api/compare', { city, a, b }),
  whatif: (p) => get('/api/whatif', p),
  diagnostics: () => get('/api/diagnostics'),
  model: () => get('/api/model'),
  retrain: (file, params) => {
    const fd = new FormData()
    fd.append('file', file)
    Object.entries(params || {}).forEach(([k, v]) => fd.append(k, v))
    return axios.post('/api/retrain', fd, { timeout: 300000 }).then((r) => r.data)
  },
  corridor: (olat, olon, dlat, dlon, alts = 2) => get('/api/corridor', { olat, olon, dlat, dlon, alts }),
  grid: (city, horizon) => get('/api/grid', { city, horizon }),
  insight: (city, route) => get('/api/insight', { city, route }),
  geocode: async (q) => {
    // Primary: backend proxy. Fallback: Nominatim direct from browser
    // (same as reference app's client-side geocodeApi) so suggestions
    // keep working even when the backend is down.
    try {
      return await get('/api/geocode', { q })
    } catch {
      const r = await fetch(`https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(q)}&limit=5&addressdetails=1`, {
        headers: { Accept: 'application/json' },
      })
      if (!r.ok) return { results: [] }
      const j = await r.json()
      return {
        results: j.map((g) => ({
          name: String(g.display_name || '').split(',').slice(0, 2).join(','),
          lat: parseFloat(g.lat),
          lon: parseFloat(g.lon),
          country: (g.address || {}).country || '',
        })),
      }
    }
  },
  tiles: () => get('/api/tiles'),
  setKeys: (tomtom, openweather) => post('/api/keys', { tomtom, openweather }),
};

export const LEVEL_EMOJI = { Low: '🟢', Moderate: '🟡', High: '🟠', Severe: '🔴' };
export const LEVEL_HEX = { Low: '#22c55e', Moderate: '#eab308', High: '#f97316', Severe: '#ef4444' };
