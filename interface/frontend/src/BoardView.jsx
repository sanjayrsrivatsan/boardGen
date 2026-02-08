import React from 'react'

/**
 * SVG visualization of the climbing board.
 * Renders all placements as circles, highlighting active holds with role colors.
 */
export default function BoardView({ config, climb }) {
  if (!config) return null

  const { placement_coords, placement_ids, role_colors, role_names } = config
  const width = 400
  const height = 600
  const padding = 20
  const holdRadius = 8

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
    if (name.includes('start')) return role_colors.start || '#00FF00'
    if (name.includes('finish') || name.includes('end') || name.includes('top'))
      return role_colors.finish || '#FF0000'
    if (name.includes('foot')) return role_colors.foot_only || '#FF00FF'
    return role_colors.middle || '#0000FF'
  }

  return (
    <svg
      width={width}
      height={height}
      style={{ background: '#222', borderRadius: 8 }}
    >
      {/* Board outline */}
      <rect
        x={padding}
        y={padding}
        width={width - 2*padding}
        height={height - 2*padding}
        fill="none"
        stroke="#444"
        strokeWidth={2}
        rx={4}
      />

      {/* Placements */}
      {placement_coords.map((coord, i) => {
        const placementId = placement_ids[i]
        const x = padding + coord[0] * (width - 2*padding)
        const y = height - padding - coord[1] * (height - 2*padding) // Flip Y for climbing

        const activeHold = activeHolds.get(placementId)
        const isActive = !!activeHold

        return (
          <circle
            key={placementId}
            cx={x}
            cy={y}
            r={isActive ? holdRadius + 2 : holdRadius - 2}
            fill={isActive ? getRoleColor(activeHold.role) : '#333'}
            stroke={isActive ? '#fff' : '#444'}
            strokeWidth={isActive ? 2 : 1}
            opacity={isActive ? 1 : 0.4}
          />
        )
      })}

      {/* Legend */}
      <g transform={`translate(${padding + 10}, ${height - 80})`}>
        {Object.entries(role_colors).map(([role, color], i) => (
          <g key={role} transform={`translate(0, ${i * 18})`}>
            <circle cx={6} cy={6} r={5} fill={color} />
            <text x={16} y={10} fill="#888" fontSize={11}>
              {role.replace('_', ' ')}
            </text>
          </g>
        ))}
      </g>
    </svg>
  )
}
