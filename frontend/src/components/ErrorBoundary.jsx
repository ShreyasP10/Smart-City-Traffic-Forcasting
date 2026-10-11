import React from 'react'

export default class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props)
    this.state = { error: null }
  }
  static getDerivedStateFromError(error) {
    return { error }
  }
  componentDidCatch(error) {
    if (this.props.onError) this.props.onError(error)
  }
  render() {
    if (this.state.error) {
      return (
        <div className="card">
          <h4>⚠️ This view crashed — nothing else is affected</h4>
          <p className="muted">{String(this.state.error?.message || this.state.error)}</p>
          <button onClick={() => this.setState({ error: null })}>Retry view →</button>
        </div>
      )
    }
    return this.props.children
  }
}
