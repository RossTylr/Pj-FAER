/**
 * Mini Schematic Component
 *
 * Compact crucifix schematic for side-by-side scenario comparison.
 * Shows key capacities and OPEL status at a glance.
 */

import React from "react";

// === TYPE DEFINITIONS ===

interface MiniSchematicProps {
  label: string;
  n_triage: number;
  n_ed_bays: number;
  n_theatre: number;
  n_itu: number;
  n_ward: number;
  scaling_enabled: boolean;
  opel_enabled: boolean;
  opel_3_threshold: number;
  surge_beds: number;
  util_ed?: number;
  util_itu?: number;
  util_ward?: number;
}

// === CONFIGURATION ===

const VIEWBOX_WIDTH = 400;
const VIEWBOX_HEIGHT = 200;

const NODE_W = 55;
const NODE_H = 32;

// Node positions (centered crucifix layout)
const POSITIONS: Record<string, { x: number; y: number }> = {
  arrivals: { x: 30, y: 80 },
  triage: { x: 100, y: 80 },
  ed_bays: { x: 175, y: 80 },
  theatre: { x: 250, y: 80 },
  discharge: { x: 325, y: 80 },
  itu: { x: 175, y: 30 },
  ward: { x: 175, y: 130 },
};

const STATUS_COLORS = {
  normal: { fill: "#f0fff4", stroke: "#28a745" },
  warning: { fill: "#fffbf0", stroke: "#ffc107" },
  critical: { fill: "#fff5f5", stroke: "#dc3545" },
  entry: { fill: "#e3f2fd", stroke: "#1976d2" },
  exit: { fill: "#f3e5f5", stroke: "#7b1fa2" },
  process: { fill: "#f0fff4", stroke: "#28a745" },
};

// === MAIN COMPONENT ===

const MiniSchematic: React.FC<MiniSchematicProps> = ({
  label,
  n_triage,
  n_ed_bays,
  n_theatre,
  n_itu,
  n_ward,
  scaling_enabled,
  opel_enabled,
  opel_3_threshold,
  surge_beds,
  util_ed,
  util_itu,
  util_ward,
}) => {
  // Determine node color based on type and utilisation
  const getNodeColor = (nodeId: string, nodeType: string) => {
    if (nodeType === "entry") return STATUS_COLORS.entry;
    if (nodeType === "exit") return STATUS_COLORS.exit;
    if (nodeType === "process") return STATUS_COLORS.process;

    // For resource nodes, check utilisation
    let util: number | undefined;
    if (nodeId === "ed_bays") util = util_ed;
    else if (nodeId === "itu") util = util_itu;
    else if (nodeId === "ward") util = util_ward;

    if (util !== undefined) {
      if (util >= 0.9) return STATUS_COLORS.critical;
      if (util >= 0.7) return STATUS_COLORS.warning;
    }
    return STATUS_COLORS.normal;
  };

  // Render a single node
  const renderNode = (
    nodeId: string,
    nodeLabel: string,
    value: string,
    nodeType: string
  ) => {
    const pos = POSITIONS[nodeId];
    const colors = getNodeColor(nodeId, nodeType);

    return (
      <g
        key={nodeId}
        transform={`translate(${pos.x - NODE_W / 2}, ${pos.y - NODE_H / 2})`}
      >
        <rect
          width={NODE_W}
          height={NODE_H}
          rx={6}
          fill={colors.fill}
          stroke={colors.stroke}
          strokeWidth={1.5}
        />
        <text
          x={NODE_W / 2}
          y={13}
          textAnchor="middle"
          fontSize={10}
          fontWeight="bold"
          fill="#333"
        >
          {nodeLabel}
        </text>
        <text
          x={NODE_W / 2}
          y={26}
          textAnchor="middle"
          fontSize={12}
          fill={colors.stroke}
        >
          {value}
        </text>
      </g>
    );
  };

  // Render arrow between nodes
  const renderArrow = (from: string, to: string) => {
    const p1 = POSITIONS[from];
    const p2 = POSITIONS[to];

    // Adjust start/end points based on direction
    let x1 = p1.x;
    let y1 = p1.y;
    let x2 = p2.x;
    let y2 = p2.y;

    if (Math.abs(x2 - x1) > Math.abs(y2 - y1)) {
      // Horizontal
      x1 = x1 + NODE_W / 2;
      x2 = x2 - NODE_W / 2;
    } else {
      // Vertical
      y1 = y1 + (y2 > y1 ? NODE_H / 2 : -NODE_H / 2);
      y2 = y2 + (y2 > y1 ? -NODE_H / 2 : NODE_H / 2);
    }

    return (
      <line
        key={`${from}-${to}`}
        x1={x1}
        y1={y1}
        x2={x2}
        y2={y2}
        stroke="#adb5bd"
        strokeWidth={1.5}
        markerEnd="url(#mini-arrow)"
      />
    );
  };

  return (
    <svg
      viewBox={`0 0 ${VIEWBOX_WIDTH} ${VIEWBOX_HEIGHT}`}
      width="100%"
      height="100%"
      preserveAspectRatio="xMidYMid meet"
      style={{
        fontFamily: "system-ui, -apple-system, sans-serif",
        background: "#fafafa",
        borderRadius: "8px",
      }}
    >
      <defs>
        <marker
          id="mini-arrow"
          markerWidth={8}
          markerHeight={8}
          refX={7}
          refY={3}
          orient="auto"
        >
          <path d="M0,0 L0,6 L8,3 z" fill="#adb5bd" />
        </marker>
      </defs>

      {/* Title */}
      <text x={10} y={18} fontSize={11} fontWeight="bold" fill="#333">
        {label}
      </text>

      {/* OPEL Status Badge */}
      {scaling_enabled ? (
        <g transform={`translate(${VIEWBOX_WIDTH - 90}, 6)`}>
          <rect
            width={80}
            height={16}
            rx={8}
            fill={opel_enabled ? "#28a745" : "#6c757d"}
            fillOpacity={0.15}
            stroke={opel_enabled ? "#28a745" : "#6c757d"}
            strokeWidth={0.5}
          />
          <text
            x={40}
            y={12}
            textAnchor="middle"
            fontSize={9}
            fill={opel_enabled ? "#28a745" : "#6c757d"}
          >
            {opel_enabled
              ? `OPEL @ ${Math.round(opel_3_threshold * 100)}%`
              : "Custom Rules"}
          </text>
        </g>
      ) : (
        <g transform={`translate(${VIEWBOX_WIDTH - 90}, 6)`}>
          <rect
            width={80}
            height={16}
            rx={8}
            fill="#6c757d"
            fillOpacity={0.1}
          />
          <text x={40} y={12} textAnchor="middle" fontSize={9} fill="#6c757d">
            No Scaling
          </text>
        </g>
      )}

      {/* Arrows */}
      {renderArrow("arrivals", "triage")}
      {renderArrow("triage", "ed_bays")}
      {renderArrow("ed_bays", "theatre")}
      {renderArrow("theatre", "discharge")}
      {renderArrow("ed_bays", "itu")}
      {renderArrow("ed_bays", "ward")}

      {/* Nodes */}
      {renderNode("arrivals", "Arr", "→", "entry")}
      {renderNode("triage", "Tri", String(n_triage), "process")}
      {renderNode("ed_bays", "ED", String(n_ed_bays), "resource")}
      {renderNode("theatre", "Thtr", String(n_theatre), "resource")}
      {renderNode("discharge", "Out", "→", "exit")}
      {renderNode("itu", "ITU", String(n_itu), "resource")}
      {renderNode("ward", "Ward", String(n_ward), "resource")}

      {/* Surge capacity indicator */}
      {scaling_enabled && surge_beds > 0 && (
        <text x={10} y={VIEWBOX_HEIGHT - 10} fontSize={9} fill="#666">
          Surge: +{surge_beds} beds @ OPEL 3
        </text>
      )}
    </svg>
  );
};

export default MiniSchematic;
