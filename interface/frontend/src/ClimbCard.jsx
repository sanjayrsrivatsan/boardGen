import React from 'react'

/**
 * Card displaying a generated climb's summary.
 */
export default function ClimbCard({ climb, selected, onClick }) {
  const { n_holds, valid, metadata, frames_string } = climb

  return (
    <div
      onClick={onClick}
      style={{
        padding: '0.75rem',
        background: selected ? '#e3f2fd' : '#fff',
        borderRadius: 8,
        cursor: 'pointer',
        border: selected ? '2px solid #007bff' : '1px solid #ddd',
        transition: 'all 0.15s ease',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span style={{ fontWeight: 600, color: '#333' }}>
          {n_holds} holds
        </span>
        <span
          style={{
            padding: '0.25rem 0.5rem',
            borderRadius: 4,
            fontSize: '0.75rem',
            background: valid ? '#28a745' : '#dc3545',
            color: '#fff',
          }}
        >
          {valid ? 'Valid' : 'Invalid'}
        </span>
      </div>

      {metadata?.difficulty && (
        <div style={{ fontSize: '0.875rem', color: '#666', marginTop: '0.25rem' }}>
          Target: ~V{Math.round(metadata.difficulty/2 - 3)}
          {metadata.angle && ` @ ${metadata.angle}°`}
        </div>
      )}

      <div
        style={{
          marginTop: '0.5rem',
          fontSize: '0.75rem',
          color: '#999',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap',
        }}
      >
        {frames_string?.slice(0, 40)}...
      </div>
    </div>
  )
}
