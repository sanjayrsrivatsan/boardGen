import React, { useState, useEffect } from 'react'
import BoardView from './BoardView.jsx'
import Controls from './Controls.jsx'
import ClimbCard from './ClimbCard.jsx'

const API_URL = 'http://localhost:8000'

export default function App() {
  const [boards, setBoards] = useState([])
  const [selectedBoard, setSelectedBoard] = useState(null)
  const [boardConfig, setBoardConfig] = useState(null)
  const [climbs, setClimbs] = useState([])
  const [selectedClimb, setSelectedClimb] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  // Fetch available boards on mount
  useEffect(() => {
    fetch(`${API_URL}/boards`)
      .then(res => res.json())
      .then(data => {
        setBoards(data)
        if (data.length > 0) setSelectedBoard(data[0].name)
      })
      .catch(err => setError('Failed to connect to API'))
  }, [])

  // Fetch board config when selection changes
  useEffect(() => {
    if (!selectedBoard) return
    fetch(`${API_URL}/board-config/${selectedBoard}`)
      .then(res => res.json())
      .then(setBoardConfig)
      .catch(err => setError(`Failed to load ${selectedBoard} config`))
  }, [selectedBoard])

  const handleGenerate = async (params) => {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch(`${API_URL}/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ board: selectedBoard, ...params })
      })
      const data = await res.json()
      setClimbs(data.climbs)
      if (data.climbs.length > 0) setSelectedClimb(data.climbs[0])
    } catch (err) {
      setError('Generation failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
      {/* Header */}
      <header style={{ padding: '1rem 2rem', background: '#fff', borderBottom: '1px solid #ddd', boxShadow: '0 1px 3px rgba(0,0,0,0.1)' }}>
        <h1 style={{ fontSize: '1.5rem', fontWeight: 600, color: '#333' }}>BoardDiffusion</h1>
        <p style={{ color: '#666', fontSize: '0.875rem' }}>AI-powered climbing problem generator</p>
      </header>

      {/* Main content */}
      <main style={{ display: 'flex', flex: 1, padding: '1rem', gap: '1rem' }}>
        {/* Left panel: Board selector and controls */}
        <aside style={{ width: 300, display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          {/* Board selector */}
          <div style={{ background: '#fff', padding: '1rem', borderRadius: 8, border: '1px solid #ddd' }}>
            <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 500, color: '#333' }}>Board</label>
            <select
              value={selectedBoard || ''}
              onChange={e => setSelectedBoard(e.target.value)}
              style={{ width: '100%', padding: '0.5rem', borderRadius: 4, border: '1px solid #ccc', background: '#fff', color: '#333' }}
            >
              {boards.map(b => (
                <option key={b.name} value={b.name}>{b.name}</option>
              ))}
            </select>
          </div>

          {/* Controls */}
          {boardConfig && (
            <Controls
              config={boardConfig}
              onGenerate={handleGenerate}
              loading={loading}
            />
          )}
        </aside>

        {/* Center: Board visualization */}
        <div style={{ flex: 1, display: 'flex', justifyContent: 'center', alignItems: 'flex-start', paddingTop: '1rem' }}>
          {boardConfig && (
            <BoardView
              config={boardConfig}
              climb={selectedClimb}
            />
          )}
        </div>

        {/* Right panel: Generated climbs */}
        <aside style={{ width: 300, display: 'flex', flexDirection: 'column', gap: '0.5rem', overflowY: 'auto', maxHeight: 'calc(100vh - 100px)' }}>
          <h3 style={{ padding: '0.5rem 1rem', background: '#fff', borderRadius: 8, border: '1px solid #ddd', color: '#333' }}>
            Generated Climbs ({climbs.length})
          </h3>
          {climbs.map((climb, i) => (
            <ClimbCard
              key={i}
              climb={climb}
              selected={selectedClimb?.id === climb.id}
              onClick={() => setSelectedClimb(climb)}
            />
          ))}
        </aside>
      </main>

      {/* Error display */}
      {error && (
        <div style={{ padding: '1rem', background: '#dc3545', color: '#fff', textAlign: 'center' }}>
          {error}
        </div>
      )}
    </div>
  )
}
