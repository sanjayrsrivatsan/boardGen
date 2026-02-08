import React, { useState } from 'react'

/**
 * Generation controls panel.
 * Adapts to board configuration (e.g., hides angle for MoonBoard).
 */
export default function Controls({ config, onGenerate, loading }) {
  const [difficulty, setDifficulty] = useState(14)
  const [angle, setAngle] = useState(config.default_angle || 40)
  const [isClassic, setIsClassic] = useState(false)
  const [quality, setQuality] = useState(3.0)
  const [guidance, setGuidance] = useState(3.0)
  const [temperature, setTemperature] = useState(0.8)
  const [nSamples, setNSamples] = useState(10)
  const [nHolds, setNHolds] = useState('')

  const handleSubmit = (e) => {
    e.preventDefault()
    onGenerate({
      n_samples: nSamples,
      difficulty,
      angle: config.angle_conditioning ? angle : undefined,
      is_classic: isClassic,
      quality,
      guidance_scale: guidance,
      temperature,
      n_holds: nHolds ? parseInt(nHolds) : undefined,
    })
  }

  const inputStyle = {
    width: '100%',
    padding: '0.5rem',
    borderRadius: 4,
    border: 'none',
    background: '#0f3460',
    color: '#fff',
  }

  const labelStyle = {
    display: 'block',
    marginBottom: '0.25rem',
    fontSize: '0.875rem',
    color: '#aaa',
  }

  return (
    <form onSubmit={handleSubmit} style={{ background: '#16213e', padding: '1rem', borderRadius: 8 }}>
      <h3 style={{ marginBottom: '1rem' }}>Generation Settings</h3>

      {/* Difficulty */}
      <div style={{ marginBottom: '1rem' }}>
        <label style={labelStyle}>Difficulty (V-grade ~{Math.round(difficulty/2 - 3)})</label>
        <input
          type="range"
          min={6}
          max={32}
          value={difficulty}
          onChange={e => setDifficulty(Number(e.target.value))}
          style={{ width: '100%' }}
        />
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: '#666' }}>
          <span>V0</span>
          <span>V{Math.round(difficulty/2 - 3)}</span>
          <span>V14+</span>
        </div>
      </div>

      {/* Angle (only if board supports it) */}
      {config.angle_conditioning && (
        <div style={{ marginBottom: '1rem' }}>
          <label style={labelStyle}>Angle: {angle}°</label>
          <input
            type="range"
            min={config.angle_range?.[0] || 0}
            max={config.angle_range?.[1] || 70}
            step={5}
            value={angle}
            onChange={e => setAngle(Number(e.target.value))}
            style={{ width: '100%' }}
          />
        </div>
      )}

      {/* Classic toggle */}
      <div style={{ marginBottom: '1rem' }}>
        <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
          <input
            type="checkbox"
            checked={isClassic}
            onChange={e => setIsClassic(e.target.checked)}
          />
          Target classic quality
        </label>
      </div>

      {/* Quality */}
      <div style={{ marginBottom: '1rem' }}>
        <label style={labelStyle}>Quality: {quality.toFixed(1)} stars</label>
        <input
          type="range"
          min={1}
          max={5}
          step={0.5}
          value={quality}
          onChange={e => setQuality(Number(e.target.value))}
          style={{ width: '100%' }}
        />
      </div>

      {/* Advanced settings */}
      <details style={{ marginBottom: '1rem' }}>
        <summary style={{ cursor: 'pointer', color: '#888' }}>Advanced</summary>
        <div style={{ marginTop: '0.5rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
          <div>
            <label style={labelStyle}>Guidance scale: {guidance}</label>
            <input
              type="range"
              min={0}
              max={10}
              step={0.5}
              value={guidance}
              onChange={e => setGuidance(Number(e.target.value))}
              style={{ width: '100%' }}
            />
          </div>
          <div>
            <label style={labelStyle}>Temperature: {temperature}</label>
            <input
              type="range"
              min={0.1}
              max={2}
              step={0.1}
              value={temperature}
              onChange={e => setTemperature(Number(e.target.value))}
              style={{ width: '100%' }}
            />
          </div>
          <div>
            <label style={labelStyle}>Hold count (empty = auto)</label>
            <input
              type="number"
              min={4}
              max={20}
              value={nHolds}
              onChange={e => setNHolds(e.target.value)}
              placeholder="Auto"
              style={inputStyle}
            />
          </div>
        </div>
      </details>

      {/* Number of samples */}
      <div style={{ marginBottom: '1rem' }}>
        <label style={labelStyle}>Number of climbs</label>
        <input
          type="number"
          min={1}
          max={50}
          value={nSamples}
          onChange={e => setNSamples(Number(e.target.value))}
          style={inputStyle}
        />
      </div>

      {/* Generate button */}
      <button
        type="submit"
        disabled={loading}
        style={{
          width: '100%',
          padding: '0.75rem',
          borderRadius: 4,
          border: 'none',
          background: loading ? '#555' : '#e94560',
          color: '#fff',
          fontSize: '1rem',
          fontWeight: 600,
          cursor: loading ? 'wait' : 'pointer',
        }}
      >
        {loading ? 'Generating...' : 'Generate Climbs'}
      </button>
    </form>
  )
}
