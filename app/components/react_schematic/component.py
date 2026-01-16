"""React Schematic Component wrapper for Streamlit.

Renders the interactive hospital flow schematic using a custom React/SVG component.
Includes graceful fallback when the React bundle is not built.
"""

from pathlib import Path
from typing import Optional

import streamlit.components.v1 as components

# Path to compiled React component
COMPONENT_PATH = Path(__file__).parent / "build"

# Declare the component (lazy initialization)
_react_schematic = None


def is_built() -> bool:
    """Check if the React component has been built.

    Returns:
        True if bundle.js exists in the build directory
    """
    bundle_path = COMPONENT_PATH / "bundle.js"
    return bundle_path.exists()


def get_component():
    """Get the declared component, initializing on first call.

    Returns:
        The Streamlit component, or None if not built
    """
    global _react_schematic
    if _react_schematic is None and is_built():
        _react_schematic = components.declare_component(
            "react_schematic",
            path=str(COMPONENT_PATH),
        )
    return _react_schematic


def react_schematic(
    data: dict,
    width: int = 1400,
    height: int = 750,
    key: str = None,
) -> Optional[str]:
    """Render React-based schematic component.

    Args:
        data: SchematicData as JSON-serializable dict (use to_dict())
        width: Component width in pixels
        height: Component height in pixels
        key: Streamlit component key for state management

    Returns:
        Clicked node ID if any, else None
    """
    component = get_component()
    if component is None:
        return None
    return component(
        data=data,
        width=width,
        height=height,
        key=key,
        default=None,
    )


def mini_schematic_svg(data) -> str:
    """Render mini schematic as inline SVG.

    This is a simpler fallback that doesn't require the React component
    to be built. Always available.

    Args:
        data: MiniSchematicData object

    Returns:
        SVG string for inline rendering with st.markdown(svg, unsafe_allow_html=True)
    """
    from .data import render_mini_schematic_svg
    return render_mini_schematic_svg(data)
