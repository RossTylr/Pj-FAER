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

// Node positions (hardcoded layout for demo - hospital flow layout)
// Centered layout that fits within 850px width
const NODE_POSITIONS: Record<string, { x: number; y: number; row: number }> = {
  // Entry row (y=80) - spread across top
  ambulance: { x: 140, y: 80, row: 0 },
  walkin: { x: 350, y: 80, row: 0 },
  hems: { x: 560, y: 80, row: 0 },
  // Triage row (y=200) - centered
  triage: { x: 350, y: 200, row: 1 },
  // ED row (y=320) - centered
  ed_bays: { x: 350, y: 320, row: 2 },
  // Downstream row (y=460) - spread across bottom
  theatre: { x: 100, y: 460, row: 3 },
  itu: { x: 270, y: 460, row: 3 },
  ward: { x: 440, y: 460, row: 3 },
  discharge: { x: 610, y: 460, row: 3 },
};

const STATUS_COLORS: Record<string, { fill: string; stroke: string; text: string }> = {
  normal: { fill: "#f0fff4", stroke: "#28a745", text: "#166534" },
  warning: { fill: "#fffbf0", stroke: "#ffc107", text: "#92400e" },
  critical: { fill: "#fff5f5", stroke: "#dc3545", text: "#991b1b" },
};

const NODE_TYPE_COLORS: Record<string, { fill: string; stroke: string }> = {
  entry: { fill: "#e3f2fd", stroke: "#1976d2" },
  exit: { fill: "#fce4ec", stroke: "#c2185b" },
};

const NODE_WIDTH = 130;
const NODE_HEIGHT = 90;

// === MAIN COMPONENT ===

const Schematic: React.FC<SchematicProps> = ({ args }) => {
  const { data, width, height } = args;
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
    const strokeWidth = edge.is_blocked ? 3 : Math.max(1, Math.min(4, edge.volume_per_hour / 5));
    const strokeDash = edge.is_blocked ? "8,4" : undefined;

    // Calculate edge start/end points based on node positions
    let x1 = source.x;
    let y1 = source.y + NODE_HEIGHT / 2;
    let x2 = target.x;
    let y2 = target.y - NODE_HEIGHT / 2;

    // If same row (horizontal connection), adjust points
    if (source.row === target.row) {
      y1 = source.y;
      y2 = target.y;
      x1 = source.x + NODE_WIDTH / 2;
      x2 = target.x - NODE_WIDTH / 2;
    }

    // Calculate control points for curved lines
    const midY = (y1 + y2) / 2;
    const path = source.row === target.row
      ? `M ${x1} ${y1} L ${x2} ${y2}`
      : `M ${x1} ${y1} C ${x1} ${midY}, ${x2} ${midY}, ${x2} ${y2}`;

    return (
      <g key={`edge-${index}`} className="edge-group">
        {/* Shadow/glow for blocked edges */}
        {edge.is_blocked && (
          <path
            d={path}
            fill="none"
            stroke="#dc3545"
            strokeWidth={strokeWidth + 4}
            strokeOpacity={0.3}
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
        {edge.volume_per_hour >= 1.0 && (
          <text
            x={(x1 + x2) / 2 + (source.row === target.row ? 0 : 15)}
            y={(y1 + y2) / 2}
            fontSize={10}
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
      colors = {
        ...colors,
        fill: NODE_TYPE_COLORS[node.node_type].fill,
        stroke: NODE_TYPE_COLORS[node.node_type].stroke,
      };
    }

    const isHovered = hoveredNode === node.id;
    const isSelected = selectedNode === node.id;
    const scale = isHovered ? 1.02 : 1;
    const shadowOpacity = isHovered ? 0.2 : 0.1;

    // Calculate utilisation bar width
    const utilBarWidth = (NODE_WIDTH - 20) * node.utilisation;

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
          x={2}
          y={2}
          width={NODE_WIDTH}
          height={NODE_HEIGHT}
          rx={8}
          fill="black"
          fillOpacity={shadowOpacity}
        />

        {/* Node background */}
        <rect
          width={NODE_WIDTH}
          height={NODE_HEIGHT}
          rx={8}
          fill={colors.fill}
          stroke={colors.stroke}
          strokeWidth={isSelected ? 3 : 2}
          className="node-rect"
        />

        {/* Selection indicator */}
        {isSelected && (
          <rect
            x={-2}
            y={-2}
            width={NODE_WIDTH + 4}
            height={NODE_HEIGHT + 4}
            rx={10}
            fill="none"
            stroke={colors.stroke}
            strokeWidth={1}
            strokeDasharray="4,2"
          />
        )}

        {/* Status indicator dot */}
        <circle
          cx={NODE_WIDTH - 15}
          cy={15}
          r={6}
          fill={STATUS_COLORS[node.status]?.stroke || "#666"}
        />

        {/* Label */}
        <text
          x={NODE_WIDTH / 2}
          y={22}
          textAnchor="middle"
          fontSize={13}
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
              x={10}
              y={35}
              width={NODE_WIDTH - 20}
              height={10}
              rx={5}
              fill="#e9ecef"
            />
            <rect
              x={10}
              y={35}
              width={utilBarWidth}
              height={10}
              rx={5}
              fill={STATUS_COLORS[node.status]?.stroke || "#28a745"}
              className="util-bar"
            />
          </>
        )}

        {/* Capacity/throughput text */}
        <text
          x={NODE_WIDTH / 2}
          y={60}
          textAnchor="middle"
          fontSize={12}
          fill="#333"
        >
          {node.capacity
            ? `${node.occupied}/${node.capacity} (${(node.utilisation * 100).toFixed(0)}%)`
            : `${node.throughput_per_hour.toFixed(1)}/hr`}
        </text>

        {/* Wait time */}
        <text
          x={NODE_WIDTH / 2}
          y={78}
          textAnchor="middle"
          fontSize={10}
          fill="#666"
        >
          {node.mean_wait_mins > 0 ? `Wait: ${node.mean_wait_mins.toFixed(0)}m` : ""}
        </text>
      </g>
    );
  };

  // === RENDER LEGEND ===
  const renderLegend = () => (
    <g transform="translate(20, 490)">
      <rect
        width={680}
        height={50}
        fill="white"
        stroke="#dee2e6"
        rx={6}
        fillOpacity={0.95}
      />
      <text x={15} y={20} fontSize={11} fontWeight="bold" fill="#333">
        Legend:
      </text>
      <circle cx={80} cy={16} r={5} fill="#28a745" />
      <text x={92} y={20} fontSize={10} fill="#333">Normal (&lt;70%)</text>
      <circle cx={200} cy={16} r={5} fill="#ffc107" />
      <text x={212} y={20} fontSize={10} fill="#333">Warning (70-90%)</text>
      <circle cx={340} cy={16} r={5} fill="#dc3545" />
      <text x={352} y={20} fontSize={10} fill="#333">Critical (&gt;90%)</text>
      <line x1={460} y1={16} x2={490} y2={16} stroke="#dc3545" strokeWidth={2} strokeDasharray="4,2" />
      <text x={498} y={20} fontSize={10} fill="#333">Blocked Flow</text>
      <circle cx={600} cy={16} r={5} fill="#1976d2" />
      <text x={612} y={20} fontSize={10} fill="#333">Entry</text>
      <circle cx={80} cy={38} r={5} fill="#c2185b" />
      <text x={92} y={42} fontSize={10} fill="#333">Exit</text>
    </g>
  );

  // === RENDER HEADER ===
  const renderHeader = () => {
    const statusColor = STATUS_COLORS[data.overall_status]?.stroke || "#666";
    return (
      <g transform="translate(20, 10)">
        <text fontSize={18} fontWeight="bold" fill="#333">
          🏥 System Schematic — {data.timestamp}
        </text>
        <g transform="translate(0, 30)">
          <text fontSize={12} fill="#666">
            In System: <tspan fontWeight="bold" fill="#333">{data.total_in_system}</tspan>
          </text>
          <text x={130} fontSize={12} fill="#666">
            24h Throughput: <tspan fontWeight="bold" fill="#333">{data.total_throughput_24h}</tspan>
          </text>
          <text x={300} fontSize={12} fill="#666">
            Status: <tspan fontWeight="bold" fill={statusColor}>{data.overall_status.toUpperCase()}</tspan>
          </text>
        </g>
      </g>
    );
  };

  // === MAIN RENDER ===
  return (
    <svg
      width={width}
      height={height}
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
      <rect width={width} height={height} fill="#fafafa" />

      {/* Header */}
      {renderHeader()}

      {/* Section labels - left aligned */}
      <text x={25} y={55} fontSize={11} fill="#888" fontWeight="500">📥 Arrivals</text>
      <text x={25} y={175} fontSize={11} fill="#888" fontWeight="500">🏷️ Assessment</text>
      <text x={25} y={295} fontSize={11} fill="#888" fontWeight="500">🚨 Emergency Dept</text>
      <text x={25} y={435} fontSize={11} fill="#888" fontWeight="500">🏨 Downstream</text>

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
