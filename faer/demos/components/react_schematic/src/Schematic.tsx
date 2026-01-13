/**
 * Main Schematic Component
 *
 * Renders hospital flow schematic as SVG with:
 * - Node boxes showing capacity/utilisation
 * - Flow edges with volume indicators
 * - Status colours and blocking indicators
 * - Click interaction for node details
 * - Hover states for enhanced UX
 */

import React, { useState, useEffect } from "react";
import { Streamlit, ComponentProps } from "streamlit-component-lib";
import "./styles/schematic.css";

// === TYPE DEFINITIONS ===

interface NodeData {
  id: string;
  label: string;
  node_type: string;
  capacity: number | null;
  occupied: number;
  throughput_per_hour: number;
  mean_wait_mins: number;
  utilisation: number;
  status: string;
}

interface EdgeData {
  source: string;
  target: string;
  volume_per_hour: number;
  is_blocked: boolean;
}

interface SchematicData {
  timestamp: string;
  nodes: Record<string, NodeData>;
  edges: EdgeData[];
  total_in_system: number;
  total_throughput_24h: number;
  overall_status: string;
}

interface SchematicProps extends ComponentProps {
  args: {
    data: SchematicData;
    width: number;
    height: number;
  };
}

// === CONFIGURATION ===

// ViewBox dimensions for responsive scaling
// Canvas: 1400x1000, schematic scaled 20% larger
const VIEWBOX_WIDTH = 1400;
const VIEWBOX_HEIGHT = 1000;

// Node positions - Left-to-right crucifix layout with Theatre as center
// Using 270px spacing (225 × 1.2 = 20% larger)
//
//                              [ITU]
//                                ↑
// [Arrivals] → [Triage] → [ED Bays] → [Theatre] → [Discharge]
//                                ↓
//                             [Ward]
//
// Horizontal: 270px between node centers on main lane
// Vertical: 270px from Theatre to ITU/Ward
// Canvas: 1400x1000, main lane at y=500 (centered)
//
const SPACING = 270;
const MAIN_Y = 500;

const NODE_POSITIONS: Record<string, { x: number; y: number; col: number; lane: string }> = {
  // Entry column (left, stacked vertically) - x=120
  ambulance: { x: 120, y: MAIN_Y - 120, col: 0, lane: "entry" },
  walkin: { x: 120, y: MAIN_Y, col: 0, lane: "entry" },
  hems: { x: 120, y: MAIN_Y + 120, col: 0, lane: "entry" },
  // Triage (assessment) - x=120 + 270 = 390
  triage: { x: 120 + SPACING, y: MAIN_Y, col: 1, lane: "main" },
  // ED Bays - x=390 + 270 = 660
  ed_bays: { x: 120 + SPACING * 2, y: MAIN_Y, col: 2, lane: "main" },
  // Theatre (center hub of crucifix) - x=660 + 270 = 930
  theatre: { x: 120 + SPACING * 3, y: MAIN_Y, col: 3, lane: "main" },
  // ITU (top arm of crucifix - above Theatre)
  itu: { x: 120 + SPACING * 3, y: MAIN_Y - SPACING, col: 3, lane: "top" },
  // Ward (bottom arm of crucifix - below Theatre)
  ward: { x: 120 + SPACING * 3, y: MAIN_Y + SPACING, col: 3, lane: "bottom" },
  // Discharge (exit, far right) - x=930 + 270 = 1200
  discharge: { x: 120 + SPACING * 4, y: MAIN_Y, col: 4, lane: "main" },
};

const STATUS_COLORS: Record<string, { fill: string; stroke: string; text: string }> = {
  normal: { fill: "#f0fff4", stroke: "#28a745", text: "#166534" },
  warning: { fill: "#fffbf0", stroke: "#ffc107", text: "#92400e" },
  critical: { fill: "#fff5f5", stroke: "#dc3545", text: "#991b1b" },
};

const NODE_TYPE_COLORS: Record<string, { fill: string; stroke: string; text: string }> = {
  entry: { fill: "#e3f2fd", stroke: "#1976d2", text: "#1565c0" },
  exit: { fill: "#f3e5f5", stroke: "#7b1fa2", text: "#6a1b9a" },
};

// Node dimensions (20% larger: 130×90 → 156×108)
const NODE_WIDTH = 156;
const NODE_HEIGHT = 108;

// === MAIN COMPONENT ===

const Schematic: React.FC<SchematicProps> = ({ args }) => {
  const { data, height } = args;
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);

  useEffect(() => {
    Streamlit.setFrameHeight(height);
  }, [height]);

  const handleNodeClick = (nodeId: string) => {
    setSelectedNode(nodeId);
    Streamlit.setComponentValue(nodeId);
  };

  // === RENDER EDGE ===
  const renderEdge = (edge: EdgeData, index: number) => {
    const source = NODE_POSITIONS[edge.source];
    const target = NODE_POSITIONS[edge.target];
    if (!source || !target) return null;

    const strokeColor = edge.is_blocked ? "#dc3545" : "#adb5bd";
    const baseStrokeWidth = Math.max(1.5, Math.min(3, edge.volume_per_hour / 5));
    const strokeWidth = edge.is_blocked ? baseStrokeWidth + 0.5 : baseStrokeWidth;
    const strokeDash = edge.is_blocked ? "6,3" : undefined;

    let path: string;
    let labelX: number;
    let labelY: number;

    // Determine edge path based on layout
    const isHorizontalMain = source.lane === "main" && target.lane === "main";
    const isVerticalUp = source.lane === "main" && target.lane === "top";
    const isVerticalDown = source.lane === "main" && target.lane === "bottom";
    const isEntryToMain = source.lane === "entry" && target.lane === "main";

    if (isHorizontalMain) {
      // Horizontal flow on main lane
      const x1 = source.x + NODE_WIDTH / 2;
      const y1 = source.y;
      const x2 = target.x - NODE_WIDTH / 2;
      const y2 = target.y;
      path = `M ${x1} ${y1} L ${x2} ${y2}`;
      labelX = (x1 + x2) / 2;
      labelY = y1 - 8;
    } else if (isVerticalUp) {
      // ED to ITU (up)
      const x1 = source.x;
      const y1 = source.y - NODE_HEIGHT / 2;
      const x2 = target.x;
      const y2 = target.y + NODE_HEIGHT / 2;
      path = `M ${x1} ${y1} L ${x2} ${y2}`;
      labelX = x1 + 12;
      labelY = (y1 + y2) / 2;
    } else if (isVerticalDown) {
      // ED to Ward (down)
      const x1 = source.x;
      const y1 = source.y + NODE_HEIGHT / 2;
      const x2 = target.x;
      const y2 = target.y - NODE_HEIGHT / 2;
      path = `M ${x1} ${y1} L ${x2} ${y2}`;
      labelX = x1 + 12;
      labelY = (y1 + y2) / 2;
    } else if (isEntryToMain) {
      // Entry nodes to triage (curved)
      const x1 = source.x + NODE_WIDTH / 2;
      const y1 = source.y;
      const x2 = target.x - NODE_WIDTH / 2;
      const y2 = target.y;
      const midX = (x1 + x2) / 2;
      path = `M ${x1} ${y1} C ${midX} ${y1}, ${midX} ${y2}, ${x2} ${y2}`;
      labelX = midX;
      labelY = (y1 + y2) / 2;
    } else if (target.lane === "top" && source.lane !== "main") {
      // Theatre to ITU (curved)
      const x1 = source.x;
      const y1 = source.y - NODE_HEIGHT / 2;
      const x2 = target.x + NODE_WIDTH / 2;
      const y2 = target.y;
      path = `M ${x1} ${y1} C ${x1} ${y2 - 30}, ${x2 + 30} ${y2}, ${x2} ${y2}`;
      labelX = (x1 + x2) / 2;
      labelY = y2 - 20;
    } else if (source.lane === "top" || source.lane === "bottom") {
      // ITU/Ward to discharge (curved to the right)
      const x1 = source.x + NODE_WIDTH / 2;
      const y1 = source.y;
      const x2 = target.x - NODE_WIDTH / 2;
      const y2 = target.y;
      const midX = (x1 + x2) / 2 + 50;
      path = `M ${x1} ${y1} C ${midX} ${y1}, ${midX} ${y2}, ${x2} ${y2}`;
      labelX = midX - 20;
      labelY = (y1 + y2) / 2;
    } else if (target.lane === "bottom" && source.lane === "top") {
      // ITU to Ward (curved down on left side)
      const x1 = source.x - NODE_WIDTH / 2;
      const y1 = source.y;
      const x2 = target.x - NODE_WIDTH / 2;
      const y2 = target.y;
      path = `M ${x1} ${y1} C ${x1 - 40} ${y1}, ${x2 - 40} ${y2}, ${x2} ${y2}`;
      labelX = x1 - 30;
      labelY = (y1 + y2) / 2;
    } else {
      // Default curved connection
      const x1 = source.x + NODE_WIDTH / 2;
      const y1 = source.y;
      const x2 = target.x - NODE_WIDTH / 2;
      const y2 = target.y;
      const midX = (x1 + x2) / 2;
      path = `M ${x1} ${y1} C ${midX} ${y1}, ${midX} ${y2}, ${x2} ${y2}`;
      labelX = midX;
      labelY = (y1 + y2) / 2;
    }

    return (
      <g key={`edge-${index}`} className="edge-group">
        {/* Shadow/glow for blocked edges */}
        {edge.is_blocked && (
          <path
            d={path}
            fill="none"
            stroke="#dc3545"
            strokeWidth={strokeWidth + 3}
            strokeOpacity={0.25}
            strokeDasharray={strokeDash}
          />
        )}
        {/* Main edge path */}
        <path
          d={path}
          fill="none"
          stroke={strokeColor}
          strokeWidth={strokeWidth}
          strokeDasharray={strokeDash}
          markerEnd={edge.is_blocked ? "url(#blocked-arrow)" : "url(#arrow)"}
          className="edge-path"
        />
        {/* Volume label for significant flows */}
        {edge.volume_per_hour >= 2.0 && (
          <text
            x={labelX}
            y={labelY}
            fontSize={9}
            fill="#666"
            textAnchor="middle"
            className="edge-label"
          >
            {edge.volume_per_hour.toFixed(1)}/hr
          </text>
        )}
      </g>
    );
  };

  // === RENDER NODE ===
  const renderNode = (node: NodeData) => {
    const pos = NODE_POSITIONS[node.id];
    if (!pos) return null;

    // Determine colors based on node type and status
    let colors = STATUS_COLORS[node.status] || STATUS_COLORS.normal;
    if (node.node_type === "entry" || node.node_type === "exit") {
      const typeColors = NODE_TYPE_COLORS[node.node_type];
      colors = {
        fill: typeColors.fill,
        stroke: typeColors.stroke,
        text: typeColors.text,
      };
    }

    const isHovered = hoveredNode === node.id;
    const isSelected = selectedNode === node.id;
    const scale = isHovered ? 1.02 : 1;
    const shadowOpacity = isHovered ? 0.2 : 0.1;

    // Calculate utilisation bar width (matches bar padding of 12px on each side)
    const utilBarWidth = (NODE_WIDTH - 24) * node.utilisation;

    return (
      <g
        key={node.id}
        transform={`translate(${pos.x - NODE_WIDTH / 2}, ${pos.y - NODE_HEIGHT / 2}) scale(${scale})`}
        onClick={() => handleNodeClick(node.id)}
        onMouseEnter={() => setHoveredNode(node.id)}
        onMouseLeave={() => setHoveredNode(null)}
        className="node-group"
        style={{ cursor: "pointer" }}
      >
        {/* Drop shadow */}
        <rect
          x={3}
          y={3}
          width={NODE_WIDTH}
          height={NODE_HEIGHT}
          rx={10}
          fill="black"
          fillOpacity={shadowOpacity}
        />

        {/* Node background */}
        <rect
          width={NODE_WIDTH}
          height={NODE_HEIGHT}
          rx={10}
          fill={colors.fill}
          stroke={colors.stroke}
          strokeWidth={isSelected ? 3 : 2}
          className="node-rect"
        />

        {/* Selection indicator */}
        {isSelected && (
          <rect
            x={-3}
            y={-3}
            width={NODE_WIDTH + 6}
            height={NODE_HEIGHT + 6}
            rx={12}
            fill="none"
            stroke={colors.stroke}
            strokeWidth={1}
            strokeDasharray="5,3"
          />
        )}

        {/* Status indicator dot */}
        <circle
          cx={NODE_WIDTH - 18}
          cy={18}
          r={7}
          fill={STATUS_COLORS[node.status]?.stroke || "#666"}
        />

        {/* Label */}
        <text
          x={NODE_WIDTH / 2}
          y={26}
          textAnchor="middle"
          fontSize={16}
          fontWeight="bold"
          fill={colors.text}
          className="node-label"
        >
          {node.label}
        </text>

        {/* Utilisation bar (only for nodes with capacity) */}
        {node.capacity && (
          <>
            <rect
              x={12}
              y={42}
              width={NODE_WIDTH - 24}
              height={12}
              rx={6}
              fill="#e9ecef"
            />
            <rect
              x={12}
              y={42}
              width={utilBarWidth}
              height={12}
              rx={6}
              fill={STATUS_COLORS[node.status]?.stroke || "#28a745"}
              className="util-bar"
            />
          </>
        )}

        {/* Capacity/throughput text */}
        <text
          x={NODE_WIDTH / 2}
          y={72}
          textAnchor="middle"
          fontSize={14}
          fill="#333"
        >
          {node.capacity
            ? `${node.occupied}/${node.capacity} (${(node.utilisation * 100).toFixed(0)}%)`
            : `${node.throughput_per_hour.toFixed(1)}/hr`}
        </text>

        {/* Wait time */}
        <text
          x={NODE_WIDTH / 2}
          y={94}
          textAnchor="middle"
          fontSize={12}
          fill="#666"
        >
          {node.mean_wait_mins > 0 ? `Wait: ${node.mean_wait_mins.toFixed(0)}m` : ""}
        </text>
      </g>
    );
  };

  // === RENDER LEGEND ===
  const renderLegend = () => (
    <g transform={`translate(${VIEWBOX_WIDTH - 145}, 60)`}>
      <rect
        width={126}
        height={180}
        fill="white"
        stroke="#dee2e6"
        rx={8}
        fillOpacity={0.95}
      />
      <text x={12} y={22} fontSize={12} fontWeight="bold" fill="#333">
        Legend
      </text>
      <circle cx={22} cy={44} r={5} fill="#28a745" />
      <text x={34} y={48} fontSize={11} fill="#333">Normal (&lt;70%)</text>
      <circle cx={22} cy={66} r={5} fill="#ffc107" />
      <text x={34} y={70} fontSize={11} fill="#333">Warning (70-90%)</text>
      <circle cx={22} cy={88} r={5} fill="#dc3545" />
      <text x={34} y={92} fontSize={11} fill="#333">Critical (&gt;90%)</text>
      <line x1={14} y1={112} x2={38} y2={112} stroke="#dc3545" strokeWidth={2} strokeDasharray="5,3" />
      <text x={46} y={116} fontSize={11} fill="#333">Blocked</text>
      <circle cx={22} cy={136} r={5} fill="#1976d2" />
      <text x={34} y={140} fontSize={11} fill="#333">Entry</text>
      <circle cx={22} cy={158} r={5} fill="#7b1fa2" />
      <text x={34} y={162} fontSize={11} fill="#333">Exit</text>
    </g>
  );

  // === RENDER HEADER ===
  const renderHeader = () => {
    const statusColor = STATUS_COLORS[data.overall_status]?.stroke || "#666";
    return (
      <g transform="translate(24, 22)">
        <text fontSize={14} fill="#666">
          <tspan fontWeight="bold" fill="#333">{data.timestamp}</tspan>
          <tspan dx={18}>|</tspan>
          <tspan dx={18}>{data.total_in_system} in system</tspan>
          <tspan dx={18}>|</tspan>
          <tspan dx={18}>{data.total_throughput_24h}/24hr</tspan>
          <tspan dx={18}>|</tspan>
          <tspan dx={18}>Status: </tspan>
          <tspan fontWeight="bold" fill={statusColor}>{data.overall_status.toUpperCase()}</tspan>
        </text>
      </g>
    );
  };

  // === MAIN RENDER ===
  return (
    <svg
      viewBox={`0 0 ${VIEWBOX_WIDTH} ${VIEWBOX_HEIGHT}`}
      width="100%"
      height="100%"
      preserveAspectRatio="xMidYMid meet"
      style={{ fontFamily: "system-ui, -apple-system, sans-serif", background: "#fafafa" }}
      className="schematic-svg"
    >
      {/* Definitions */}
      <defs>
        {/* Arrow marker */}
        <marker
          id="arrow"
          markerWidth="10"
          markerHeight="10"
          refX="9"
          refY="3"
          orient="auto"
          markerUnits="strokeWidth"
        >
          <path d="M0,0 L0,6 L9,3 z" fill="#adb5bd" />
        </marker>
        {/* Blocked arrow marker */}
        <marker
          id="blocked-arrow"
          markerWidth="10"
          markerHeight="10"
          refX="9"
          refY="3"
          orient="auto"
          markerUnits="strokeWidth"
        >
          <path d="M0,0 L0,6 L9,3 z" fill="#dc3545" />
        </marker>
        {/* Gradient for utilisation bars */}
        <linearGradient id="utilGradient" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" style={{ stopColor: "#28a745", stopOpacity: 1 }} />
          <stop offset="70%" style={{ stopColor: "#ffc107", stopOpacity: 1 }} />
          <stop offset="100%" style={{ stopColor: "#dc3545", stopOpacity: 1 }} />
        </linearGradient>
      </defs>

      {/* Background */}
      <rect width={VIEWBOX_WIDTH} height={VIEWBOX_HEIGHT} fill="#fafafa" />

      {/* Header */}
      {renderHeader()}

      {/* Edges (rendered behind nodes) */}
      <g className="edges-layer">
        {data.edges.map((edge, i) => renderEdge(edge, i))}
      </g>

      {/* Nodes */}
      <g className="nodes-layer">
        {Object.values(data.nodes).map((node) => renderNode(node))}
      </g>

      {/* Legend */}
      {renderLegend()}
    </svg>
  );
};

export default Schematic;
