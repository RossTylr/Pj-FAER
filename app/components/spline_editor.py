"""Spline-based arrival profile editor component (Phase 8b).

Provides an interactive UI for editing arrival profiles using draggable
control points with PCHIP spline interpolation.

Usage:
    from app.components.spline_editor import render_spline_editor

    spline_config = render_spline_editor(
        session_key='spline_arrivals',
        baseline_rates=BASE_HOURLY_RATES,
    )
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
from typing import List, Optional

from faer.core.scenario import (
    SplineKnot,
    SplineArrivalConfig,
    BASE_HOURLY_RATES,
)
from faer.core.arrivals import build_spline_profile


def render_spline_editor(
    session_key: str = 'spline_arrivals',
    baseline_rates: Optional[List[float]] = None,
) -> SplineArrivalConfig:
    """Render interactive spline arrival profile editor.

    This component provides:
    - A Plotly chart showing baseline rates, interpolated curve, and knots
    - Click-to-select knot functionality
    - Sliders to adjust selected knot's hour and multiplier
    - Constraint controls (preserve volume, max/min multiplier)
    - Add/delete knot buttons

    Args:
        session_key: Session state key for storing spline config.
        baseline_rates: 24-hour baseline pattern. Defaults to BASE_HOURLY_RATES.

    Returns:
        SplineArrivalConfig with user's current configuration.
    """
    if baseline_rates is None:
        baseline_rates = BASE_HOURLY_RATES

    # Initialize from session state or create defaults
    if session_key not in st.session_state:
        st.session_state[session_key] = SplineArrivalConfig()

    config = st.session_state[session_key]

    # Initialize selected knot index
    selected_key = f'{session_key}_selected_knot'
    if selected_key not in st.session_state:
        st.session_state[selected_key] = None

    # Build and display the chart
    _render_spline_chart(config, baseline_rates, session_key)

    # Knot editor panel
    _render_knot_editor(config, session_key)

    # Constraint controls
    _render_constraint_controls(config, session_key)

    # Update session state
    st.session_state[session_key] = config

    return config


def _render_spline_chart(
    config: SplineArrivalConfig,
    baseline_rates: List[float],
    session_key: str,
) -> None:
    """Render the Plotly chart with baseline, spline curve, and knots."""

    # Build the interpolated profile
    try:
        profile = build_spline_profile(
            knots=config.get_sorted_knots(),
            baseline_rates=baseline_rates,
            preserve_volume=config.preserve_volume,
            min_mult=config.min_multiplier,
            max_mult=config.max_multiplier,
            resolution_mins=config.resolution_mins,
        )
    except Exception as e:
        st.error(f"Error building spline profile: {e}")
        return

    fig = go.Figure()

    # 1. Baseline line (dashed gray)
    hours = list(range(24))
    fig.add_trace(go.Scatter(
        x=hours,
        y=baseline_rates,
        mode='lines',
        name='Baseline',
        line=dict(dash='dash', color='gray', width=1),
        hoverinfo='text',
        hovertext=[f"Hour {h}: {baseline_rates[h]:.1f}/hr (baseline)" for h in hours],
    ))

    # 2. Interpolated spline curve (filled area)
    spline_hours = np.linspace(0, 24, 96)
    spline_rates = [profile.get_rate(h * 60) for h in spline_hours]

    fig.add_trace(go.Scatter(
        x=spline_hours,
        y=spline_rates,
        mode='lines',
        name='Spline Profile',
        line=dict(color='#2E86AB', width=2),
        fill='tozeroy',
        fillcolor='rgba(46, 134, 171, 0.2)',
        hoverinfo='text',
        hovertext=[f"Hour {h:.1f}: {spline_rates[i]:.2f}/hr" for i, h in enumerate(spline_hours)],
    ))

    # 3. Knot markers
    sorted_knots = config.get_sorted_knots()
    knot_hours = [k.hour for k in sorted_knots]
    knot_rates = [baseline_rates[int(k.hour) % 24] * k.multiplier for k in sorted_knots]

    # Determine selected knot for highlighting
    selected_idx = st.session_state.get(f'{session_key}_selected_knot', None)

    # Create marker colors (highlight selected)
    marker_colors = []
    marker_sizes = []
    for i in range(len(sorted_knots)):
        if selected_idx is not None and i == selected_idx:
            marker_colors.append('#E94F37')  # Red for selected
            marker_sizes.append(18)
        else:
            marker_colors.append('#F39237')  # Orange for unselected
            marker_sizes.append(14)

    fig.add_trace(go.Scatter(
        x=knot_hours,
        y=knot_rates,
        mode='markers+text',
        name='Control Knots',
        marker=dict(
            size=marker_sizes,
            color=marker_colors,
            symbol='circle',
            line=dict(width=2, color='white'),
        ),
        text=[f'{k.multiplier:.1f}x' for k in sorted_knots],
        textposition='top center',
        textfont=dict(size=10, color='#333'),
        hoverinfo='text',
        hovertext=[
            f"Knot {i+1}: Hour {k.hour:.1f}, {k.multiplier:.1f}x"
            for i, k in enumerate(sorted_knots)
        ],
    ))

    # Chart layout
    max_rate = max(max(baseline_rates) * config.max_multiplier, max(spline_rates) * 1.1)

    fig.update_layout(
        title=dict(
            text='Spline Arrival Profile Editor',
            font=dict(size=16),
        ),
        xaxis=dict(
            title='Hour of Day',
            range=[-0.5, 24.5],
            dtick=2,
            gridcolor='rgba(128, 128, 128, 0.2)',
        ),
        yaxis=dict(
            title='Arrivals per Hour',
            range=[0, max_rate],
            gridcolor='rgba(128, 128, 128, 0.2)',
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1,
        ),
        hovermode='closest',
        plot_bgcolor='white',
        margin=dict(l=50, r=20, t=60, b=50),
    )

    # Render with click events using streamlit-plotly-events
    try:
        from streamlit_plotly_events import plotly_events

        selected_points = plotly_events(
            fig,
            click_event=True,
            select_event=False,
            hover_event=False,
            key=f'{session_key}_chart',
        )

        # Handle click on knot marker
        if selected_points:
            point = selected_points[0]
            # Check if clicked on knots trace (index 2)
            if point.get('curveNumber') == 2:
                knot_idx = point.get('pointIndex', 0)
                st.session_state[f'{session_key}_selected_knot'] = knot_idx
                st.rerun()

    except ImportError:
        # Fallback if streamlit-plotly-events not installed
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Install `streamlit-plotly-events` for click-to-select functionality")


def _render_knot_editor(config: SplineArrivalConfig, session_key: str) -> None:
    """Render the knot selection and editing panel."""

    st.markdown("---")
    st.subheader("Edit Control Knots")

    sorted_knots = config.get_sorted_knots()
    selected_idx = st.session_state.get(f'{session_key}_selected_knot', None)

    # Knot selector (dropdown as alternative to clicking)
    col1, col2 = st.columns([3, 1])

    with col1:
        knot_options = [f"Knot {i+1}: Hour {k.hour:.1f}, {k.multiplier:.1f}x"
                        for i, k in enumerate(sorted_knots)]

        # Default to first knot if none selected
        default_idx = selected_idx if selected_idx is not None else 0

        selected_label = st.selectbox(
            "Select knot to edit (or click on chart)",
            options=knot_options,
            index=min(default_idx, len(knot_options) - 1),
            key=f'{session_key}_knot_selector',
        )

        # Update selected index from dropdown
        new_selected_idx = knot_options.index(selected_label) if selected_label in knot_options else 0
        if new_selected_idx != selected_idx:
            st.session_state[f'{session_key}_selected_knot'] = new_selected_idx
            selected_idx = new_selected_idx

    with col2:
        st.caption(f"Total: {len(sorted_knots)} knots")

    # Edit selected knot
    if selected_idx is not None and 0 <= selected_idx < len(config.knots):
        # Find the actual knot in the original (unsorted) list
        # The sorted list may have different indices
        original_knots = config.knots
        sorted_knot = sorted_knots[selected_idx]

        # Find this knot in the original list
        original_idx = None
        for i, k in enumerate(original_knots):
            if k.hour == sorted_knot.hour and k.multiplier == sorted_knot.multiplier:
                original_idx = i
                break

        if original_idx is not None:
            knot = original_knots[original_idx]

            col1, col2, col3 = st.columns([2, 2, 1])

            with col1:
                new_hour = st.slider(
                    "Hour",
                    min_value=0.0,
                    max_value=24.0,
                    value=float(knot.hour),
                    step=0.5,
                    key=f'{session_key}_edit_hour',
                    help="Time of day (0-24 hours)",
                )

            with col2:
                new_mult = st.slider(
                    "Multiplier",
                    min_value=float(config.min_multiplier),
                    max_value=float(config.max_multiplier),
                    value=float(knot.multiplier),
                    step=0.1,
                    key=f'{session_key}_edit_mult',
                    format="%.1fx",
                    help="1.0x = baseline, 2.0x = double arrivals",
                )

            with col3:
                st.write("")  # Spacer
                st.write("")  # Spacer
                if st.button("Delete", key=f'{session_key}_delete_knot', type='secondary'):
                    if len(config.knots) > 3:
                        config.knots.pop(original_idx)
                        st.session_state[f'{session_key}_selected_knot'] = None
                        st.rerun()
                    else:
                        st.warning("Minimum 3 knots required")

            # Update knot if values changed
            if new_hour != knot.hour or new_mult != knot.multiplier:
                config.knots[original_idx] = SplineKnot(new_hour, new_mult)
                st.session_state[session_key] = config

    # Add knot button
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if st.button("Add Knot", key=f'{session_key}_add_knot', type='primary'):
            if len(config.knots) < 12:
                # Add at midpoint hour that doesn't already have a knot
                existing_hours = {k.hour for k in config.knots}
                for candidate in [12.0, 6.0, 18.0, 3.0, 9.0, 15.0, 21.0]:
                    if candidate not in existing_hours:
                        config.add_knot(candidate, 1.0)
                        st.session_state[f'{session_key}_selected_knot'] = len(config.knots) - 1
                        st.rerun()
                        break
            else:
                st.warning("Maximum 12 knots reached")

    with col2:
        if st.button("Reset to Baseline", key=f'{session_key}_reset'):
            st.session_state[session_key] = SplineArrivalConfig()
            st.session_state[f'{session_key}_selected_knot'] = None
            st.rerun()


def _render_constraint_controls(config: SplineArrivalConfig, session_key: str) -> None:
    """Render constraint toggles and controls."""

    st.markdown("---")

    with st.expander("Constraints & Settings", expanded=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            preserve_vol = st.checkbox(
                "Preserve Total Volume",
                value=config.preserve_volume,
                key=f'{session_key}_preserve_volume',
                help="Scale curve so total daily arrivals match baseline",
            )
            config.preserve_volume = preserve_vol

        with col2:
            max_mult = st.slider(
                "Max Multiplier",
                min_value=1.5,
                max_value=4.0,
                value=float(config.max_multiplier),
                step=0.1,
                key=f'{session_key}_max_mult',
                format="%.1fx",
            )
            config.max_multiplier = max_mult

        with col3:
            min_mult = st.slider(
                "Min Multiplier",
                min_value=0.0,
                max_value=0.5,
                value=float(config.min_multiplier),
                step=0.05,
                key=f'{session_key}_min_mult',
                format="%.2fx",
            )
            config.min_multiplier = min_mult

        st.session_state[session_key] = config


def get_spline_summary(config: SplineArrivalConfig, baseline_rates: List[float] = None) -> dict:
    """Get summary statistics for a spline configuration.

    Args:
        config: SplineArrivalConfig to summarize.
        baseline_rates: Baseline rates (default BASE_HOURLY_RATES).

    Returns:
        Dict with summary stats: n_knots, est_daily_arrivals, max_rate, min_rate.
    """
    if baseline_rates is None:
        baseline_rates = BASE_HOURLY_RATES

    try:
        profile = build_spline_profile(
            knots=config.get_sorted_knots(),
            baseline_rates=baseline_rates,
            preserve_volume=config.preserve_volume,
            min_mult=config.min_multiplier,
            max_mult=config.max_multiplier,
            resolution_mins=config.resolution_mins,
        )

        # Calculate stats
        rates = [rate for _, rate in profile.schedule]
        total = sum(rates) * (config.resolution_mins / 60)  # Approx daily total

        return {
            'n_knots': len(config.knots),
            'est_daily_arrivals': total,
            'max_rate': max(rates),
            'min_rate': min(rates),
            'avg_rate': np.mean(rates),
        }
    except Exception:
        return {
            'n_knots': len(config.knots),
            'est_daily_arrivals': 0,
            'max_rate': 0,
            'min_rate': 0,
            'avg_rate': 0,
        }
