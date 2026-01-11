# Schematic Demos

This directory contains isolated demos comparing **Pure Streamlit** vs **React Component** approaches for rendering hospital flow schematics.

## Purpose

These demos exist to evaluate two different approaches for visualizing hospital patient flow:

1. **Pure Streamlit** (`schematic_streamlit.py`) - Uses native Streamlit components + custom CSS/HTML
2. **React Component** (`schematic_react.py`) - Uses a custom React/SVG component

Both demos render **identical sample data** to allow fair visual comparison.

## Isolation

These demos are **COMPLETELY ISOLATED** from the main application:

- No imports from main application code
- No exports to main application code
- Self-contained sample data in `sample_data.py`
- Can be deleted without affecting the main app

## Running the Demos

### Demo 1: Pure Streamlit

```bash
# From project root
streamlit run faer/demos/schematic_streamlit.py
```

No build step required - runs immediately.

### Demo 2: React Component

The React component requires a build step first:

```bash
# Build the React component
cd faer/demos/components/react_schematic
npm install
npm run build
cd ../../../..

# Run the demo
streamlit run faer/demos/schematic_react.py
```

**Note:** If the React component isn't built, the demo will show a static SVG fallback.

## File Structure

```
faer/demos/
├── __init__.py                 # Package marker
├── README.md                   # This file
├── sample_data.py              # Shared data structures (NodeState, FlowEdge, SchematicData)
├── schematic_streamlit.py      # Demo 1: Pure Streamlit implementation
├── schematic_react.py          # Demo 2: React wrapper + fallback
└── components/
    └── react_schematic/
        ├── package.json        # React dependencies
        ├── tsconfig.json       # TypeScript configuration
        ├── webpack.config.js   # Webpack build configuration
        ├── src/
        │   ├── index.tsx       # Streamlit component entry point
        │   ├── Schematic.tsx   # Main React/SVG component
        │   └── styles/
        │       └── schematic.css
        └── build/
            ├── index.html      # Component HTML template
            └── bundle.js       # Compiled output (after npm run build)
```

## Sample Data

Both demos use identical data from `sample_data.py`:

- **9 nodes**: 3 entry (Ambulance, Walk-in, HEMS), 1 process (Triage), 4 resources (ED, Theatre, ITU, Ward), 1 exit (Discharge)
- **13 edges**: Patient flow connections between nodes
- **Blocking state**: ITU at 100% capacity with blocked inflows
- **Status levels**: Normal (<70%), Warning (70-90%), Critical (>90%)

## Evaluation Criteria

| Criterion | Weight | Description |
|-----------|--------|-------------|
| **Visual Quality** | 25% | Professional appearance, polish |
| **Flow Lines** | 15% | Ability to show arrows/connections between nodes |
| **Interactivity** | 20% | Hover effects, click handling, tooltips |
| **Responsiveness** | 10% | How well it handles resize |
| **Dev Time** | 15% | Time to implement |
| **Maintenance** | 10% | Ease of updates and modifications |
| **Dependencies** | 5% | External tooling required |

### Scoring (1-5)

- **5** = Excellent
- **4** = Good
- **3** = Adequate
- **2** = Poor
- **1** = Unacceptable

## Quick Comparison

| Aspect | Pure Streamlit | React Component |
|--------|----------------|-----------------|
| Build step | None | `npm install && npm run build` |
| Flow lines | Limited (vertical arrows only) | Full SVG paths with curves |
| Interactivity | CSS hover, Streamlit widgets | Click, hover, selection, tooltips |
| Visual quality | Good | Excellent |
| Dev complexity | Low | Medium-High |
| Dependencies | None | React, webpack, TypeScript |

## Development

### Modifying Streamlit Demo

Edit `schematic_streamlit.py` directly. Changes are reflected immediately on page refresh.

### Modifying React Demo

1. Edit files in `components/react_schematic/src/`
2. Run `npm run build` (or `npm run dev` for watch mode)
3. Refresh the Streamlit page

### Adding New Sample Data

Edit `sample_data.py` to modify the test scenario. Both demos will automatically use the updated data.
