// Shared event data + surge math (Calendar view + Overview banner).
export const CORRIDORS = ['City Center Hub', 'North Radial Ring', 'North-East Expressway', 'West Central Avenue', 'East Business Corridor', 'South Coastal Link', 'South-East Bypass', 'North-West Arterial', 'South-West Terminal']
export const LS_KEY = 'traffic_custom_events_v1'

export const dkey = (d) => `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`
export const addDays = (d, n) => { const x = new Date(d); x.setDate(x.getDate() + n); return x }
export const lvlOf = (v) => (v < 0.4 ? 'Low' : v < 0.65 ? 'Moderate' : v < 0.85 ? 'High' : 'Severe')

export function preloaded() {
  const t = new Date()
  const E = (off, title, sh, s, e, cat, att, surge, venue) => ({
    id: `ev-${off}-${sh}`, title, short: title.split(' ').slice(0, 2).join(' '),
    date: dkey(addDays(t, off)), start: s, end: e, category: cat, attendees: att, surge, venue,
    corridors: CORRIDORS.slice(0, 3), custom: false,
  })
  return [
    E(1, 'Grand Garba & Dandiya Night', 'Garba Night', '19:00', '02:00', 'cultural', 52000, 1.88, 'Community Grounds'),
    E(2, 'Festive Dandiya Evening', 'Dandiya', '19:30', '01:30', 'cultural', 38000, 1.65, 'City Grounds'),
    E(4, 'Dussehra Mela Procession', 'Dussehra Mela', '16:00', '22:30', 'procession', 44000, 1.74, 'Main Avenue'),
    E(7, 'T20 Cricket Match', 'T20 Match', '18:00', '23:30', 'sports', 58000, 1.92, 'Stadium'),
    E(10, 'Diwali Bazaar Night Market', 'Diwali Bazaar', '14:00', '23:00', 'cultural', 65000, 1.82, 'Market District'),
    E(14, 'City Marathon', 'Marathon', '05:00', '11:00', 'sports', 22000, 1.55, 'Ring Road'),
    E(18, 'Tech Summit', 'Tech Summit', '08:30', '18:00', 'civic', 30000, 1.45, 'Convention Center'),
    E(21, 'Music Concert', 'Concert', '19:00', '23:00', 'concert', 35000, 1.6, 'Arena'),
  ]
}

export function loadCustom() {
  try { return JSON.parse(localStorage.getItem(LS_KEY) || '[]') } catch { return [] }
}

// Simplified surge curve: arrival / active / dispersal
export function surgeAt(ev, h) {
  const s = Number(ev.start.slice(0, 2)), e0 = Number(ev.end.slice(0, 2))
  const end = e0 <= s ? e0 + 24 : e0
  const H = h < s ? h + 24 : h
  const m = ev.surge || 1.5
  if (H === s - 1 || (s === 0 && h === 23)) return 1 + (m - 1) * 0.35
  if (H >= s && H <= end) {
    const p = end === s ? 0.5 : (H - s) / (end - s)
    return 1 + (m - 1) * (0.6 + 0.4 * Math.sin(p * Math.PI))
  }
  if (H === end + 1) return 1 + (m - 1) * 0.45
  return 1
}

export function diurnal(h, weekend) {
  if (!weekend) {
    if (h >= 7 && h <= 9) return 0.82
    if (h >= 17 && h <= 20) return 0.88
    if (h >= 10 && h <= 16) return 0.58
    if (h >= 21 || h <= 4) return 0.2
    return 0.42
  }
  if (h >= 12 && h <= 16) return 0.66
  if (h >= 17 && h <= 21) return 0.6
  if (h >= 22 || h <= 5) return 0.16
  return 0.4
}

export function nextEvent() {
  const t = dkey(new Date())
  const all = [...preloaded(), ...loadCustom()].filter((e) => e.date >= t)
  all.sort((a, b) => (a.date < b.date ? -1 : 1))
  return all[0] || null
}
