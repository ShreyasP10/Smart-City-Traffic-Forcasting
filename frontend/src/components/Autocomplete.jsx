import { useEffect, useRef, useState } from 'react'

// Google-style autocomplete: type-ahead list, match highlight,
// ArrowUp/Down + Enter + Escape, mouse pick, popular defaults on focus.
export default function Autocomplete({ placeholder, initial = '', minChars = 2, getSuggestions, onPick }) {
  const [q, setQ] = useState(initial)
  const [open, setOpen] = useState(false)
  const [items, setItems] = useState([])
  const [hi, setHi] = useState(0)
  const box = useRef(null)

  useEffect(() => { setQ(initial) }, [initial])

  useEffect(() => {
    if (q.trim().length < minChars) {
      // popular defaults on focus / short query
      Promise.resolve(getSuggestions('', true)).then((d) => setItems(d || [])).catch(() => setItems([]))
      return
    }
    const t = setTimeout(() => {
      getSuggestions(q.trim(), false).then((d) => { setItems(d || []); setHi(0) }).catch(() => setItems([]))
    }, 300)
    return () => clearTimeout(t)
  }, [q])

  useEffect(() => {
    const close = (e) => { if (box.current && !box.current.contains(e.target)) setOpen(false) }
    document.addEventListener('click', close)
    return () => document.removeEventListener('click', close)
  }, [])

  const choose = (it) => {
    if (!it) return
    setQ(it.primary)
    setOpen(false)
    setItems([])
    onPick && onPick(it)
  }

  const hl = (text) => {
    const needle = q.trim()
    if (!needle || minChars > 0 && needle.length < minChars) return text
    const i = text.toLowerCase().indexOf(needle.toLowerCase())
    if (i < 0) return text
    return <>{text.slice(0, i)}<mark>{text.slice(i, i + needle.length)}</mark>{text.slice(i + needle.length)}</>
  }

  return (
    <div ref={box} className="combo">
      <input
        placeholder={placeholder}
        value={q}
        onChange={(e) => { setQ(e.target.value); setOpen(true) }}
        onFocus={() => setOpen(true)}
        onKeyDown={(e) => {
          if (e.key === 'ArrowDown') { e.preventDefault(); setHi((h) => Math.min(items.length - 1, h + 1)) }
          else if (e.key === 'ArrowUp') { e.preventDefault(); setHi((h) => Math.max(0, h - 1)) }
          else if (e.key === 'Enter') choose(items[hi])
          else if (e.key === 'Escape') setOpen(false)
        }}
      />
      {open && items.length > 0 && (
        <div className="combo-list">
          {items.map((it, i) => (
            <button type="button" key={it.key} className={`combo-item${i === hi ? ' hi' : ''}`}
              onMouseEnter={() => setHi(i)} onClick={() => choose(it)}>
              <span className="combo-ic">{it.icon}</span>
              <span className="combo-tx">{hl(it.primary)}{it.secondary && <span className="muted"> · {it.secondary}</span>}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
