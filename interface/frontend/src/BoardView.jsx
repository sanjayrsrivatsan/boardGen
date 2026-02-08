import React from 'react'

/**
 * Visualization of the climbing board with actual board image.
 * Overlays glowing LED circles for active holds.
 */
export default function BoardView({ config, climb }) {
  if (!config) return null

  const { placement_coords, placement_ids, role_colors, role_names, board } = config

  // Board image dimensions and scaling
  const imgSize = 4320  // Square image
  const displayScale = 0.14  // Scale down for display
  const width = imgSize * displayScale
  const height = imgSize * displayScale

  // Coordinate mapping - adjusted for image alignment
  // The board image has padding at edges, these values tuned for the 12x12 image
  const boardPadX = 25
  const boardPadY = 45
  const boardWidth = width - 50
  const boardHeight = height - 70

  // Get active holds from climb
  const activeHolds = new Map()
  if (climb?.hold_list) {
    for (const hold of climb.hold_list) {
      activeHolds.set(hold.placement_id, hold)
    }
  }

  // Map role names to colors
  const getRoleColor = (roleName) => {
    const name = roleName?.toLowerCase() || ''
    if (name.includes('start')) return role_colors?.start || '#00FF00'
    if (name.includes('finish') || name.includes('end') || name.includes('top'))
      return role_colors?.finish || '#FF0000'
    if (name.includes('foot')) return role_colors?.foot_only || '#FF00FF'
    return role_colors?.middle || '#0000FF'
  }

  // Board image path - using the full 12x12 board image
  const boardImage = `/app/images/tension/board-12x12.png`

  return (
    <div style={{ position: 'relative', width, height }}>
      {/* Board background - light with subtle shadow */}
      <div
        style={{
          position: 'absolute',
          width: '100%',
          height: '100%',
          background: '#fff',
          borderRadius: 8,
          border: '1px solid #ddd',
          boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
        }}
      />

      {/* Board image with holds */}
      <img
        src={boardImage}
        alt="Tension Board"
        style={{
          position: 'absolute',
          width: '100%',
          height: '100%',
          objectFit: 'contain',
          opacity: 0.9,
        }}
      />

      {/* SVG overlay for LEDs */}
      <svg
        width={width}
        height={height}
        style={{ position: 'absolute', top: 0, left: 0 }}
      >
        {/* LED glow filter */}
        <defs>
          <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="4" result="coloredBlur"/>
            <feMerge>
              <feMergeNode in="coloredBlur"/>
              <feMergeNode in="SourceGraphic"/>
            </feMerge>
          </filter>
          <filter id="strongGlow" x="-100%" y="-100%" width="300%" height="300%">
            <feGaussianBlur stdDeviation="8" result="coloredBlur"/>
            <feMerge>
              <feMergeNode in="coloredBlur"/>
              <feMergeNode in="coloredBlur"/>
              <feMergeNode in="SourceGraphic"/>
            </feMerge>
          </filter>
        </defs>

        {/* Active hold LEDs */}
        {placement_coords.map((coord, i) => {
          const placementId = placement_ids[i]
          const activeHold = activeHolds.get(placementId)

          if (!activeHold) return null

          // Map normalized coords to image pixels
          const x = boardPadX + coord[0] * boardWidth
          const y = height - boardPadY - coord[1] * boardHeight  // Flip Y

          const color = getRoleColor(activeHold.role)

          return (
            <g key={placementId}>
              {/* Outer glow */}
              <circle
                cx={x}
                cy={y}
                r={12}
                fill={color}
                opacity={0.3}
                filter="url(#strongGlow)"
              />
              {/* Inner bright LED */}
              <circle
                cx={x}
                cy={y}
                r={6}
                fill={color}
                filter="url(#glow)"
              />
              {/* Center bright spot */}
              <circle
                cx={x}
                cy={y}
                r={3}
                fill="#fff"
                opacity={0.8}
              />
            </g>
          )
        })}
      </svg>

      {/* Legend */}
      <div style={{
        position: 'absolute',
        bottom: 10,
        left: 10,
        background: 'rgba(255,255,255,0.95)',
        padding: '8px 12px',
        borderRadius: 6,
        fontSize: 11,
        border: '1px solid #ddd',
      }}>
        {role_colors && Object.entries(role_colors).map(([role, color]) => (
          <div key={role} style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
            <div style={{
              width: 12,
              height: 12,
              borderRadius: '50%',
              background: color,
              boxShadow: `0 0 4px ${color}`,
            }} />
            <span style={{ color: '#333' }}>{role.replace('_', ' ')}</span>
          </div>
        ))}
      </div>

      {/* Climb info */}
      {climb && (
        <div style={{
          position: 'absolute',
          top: 10,
          right: 10,
          background: 'rgba(255,255,255,0.95)',
          padding: '8px 12px',
          borderRadius: 6,
          fontSize: 12,
          border: '1px solid #ddd',
        }}>
          <div style={{ color: '#666' }}>Holds: {climb.n_holds}</div>
          <div style={{ color: climb.valid ? '#28a745' : '#dc3545' }}>
            {climb.valid ? '✓ Valid' : '✗ Invalid'}
          </div>
        </div>
      )}
    </div>
  )
}
