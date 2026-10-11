import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'

// Last-resort surface: any uncaught error (outside React boundaries,
// HMR breakage, Leaflet init races) becomes a visible banner instead
// of a mysterious blank page. Copy its text when reporting issues.
window.addEventListener('error', (e) => {
  if (document.getElementById('fatal-banner')) return
  const d = document.createElement('div')
  d.id = 'fatal-banner'
  d.style.cssText = 'position:fixed;top:0;left:0;right:0;z-index:9999;background:#7f1d1d;color:#fff;padding:10px 16px;font:14px system-ui;white-space:pre-wrap;'
  d.textContent = '⚠️ App error: ' + (e.message || e.error) + ' — hard-refresh (Ctrl+Shift+R); if it persists, paste this text.'
  document.body.prepend(d)
})

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
