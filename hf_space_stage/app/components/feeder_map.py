"""Feeder-map data assembly + pydeck rendering for the dashboard.

Two public entry points:

* :func:`build_map_data` — pure-data helper returning
  ``(nodes_df, edges_df)``. Kept side-effect-free so the smoke test in
  ``tests/test_dashboard.py`` can call it without a Streamlit context.
* :func:`build_deck` — composes a :class:`pydeck.Deck` with a scatter
  layer (nodes) and a line layer (edges). The Streamlit caller passes
  this into ``st.pydeck_chart``.

Coordinate handling
-------------------
The IEEE 123 BusCoords.dat uses a local engineering grid (x, y in feet),
not latitude/longitude. For pydeck we normalise to a compact lon/lat
window centred on Phoenix Sky Harbor (KPHX) so the map renders at a
recognisable Arizona location and hover tooltips still show the underlying
bus data. The transform is affine — pan/zoom is fine but distances on
the map are not to scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Map view centre — KPHX so the feeder renders over Phoenix.
MAP_CENTER_LON: float = -112.0740
MAP_CENTER_LAT: float = 33.4484

#: Size of the normalised footprint on the map, in degrees lon/lat.
MAP_SPAN_DEG: float = 0.05

#: ANSI C84.1 voltage colour thresholds (pu) for the scatter layer.
VOLTAGE_GREEN_MIN: float = 0.97
VOLTAGE_YELLOW_MIN: float = 0.95

#: RGB colour triples (pydeck expects uint8 lists).
COLOR_GREEN: tuple[int, int, int] = (46, 204, 113)
COLOR_YELLOW: tuple[int, int, int] = (241, 196, 15)
COLOR_RED: tuple[int, int, int] = (231, 76, 60)
COLOR_GREY: tuple[int, int, int] = (127, 140, 141)

#: Edge colour by element type.
EDGE_COLORS: dict[str, tuple[int, int, int]] = {
    "line": (70, 90, 120),
    "switch": (155, 89, 182),
    "transformer": (230, 126, 34),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Bounds:
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def scale(self) -> float:
        return max(self.x_max - self.x_min, self.y_max - self.y_min, 1.0)


def _compute_bounds(graph: nx.Graph) -> _Bounds:
    xs = [float(d.get("x", 0.0)) for _, d in graph.nodes(data=True)]
    ys = [float(d.get("y", 0.0)) for _, d in graph.nodes(data=True)]
    # Ignore the (0, 0) sentinels from buses without published coords.
    xs_valid = [v for v in xs if v != 0.0] or [0.0, 1.0]
    ys_valid = [v for v in ys if v != 0.0] or [0.0, 1.0]
    return _Bounds(min(xs_valid), max(xs_valid), min(ys_valid), max(ys_valid))


def _project(x: float, y: float, bounds: _Bounds) -> tuple[float, float]:
    """Normalise (x, y) engineering coords → (lon, lat) around KPHX."""
    scale = bounds.scale()
    nx_ = (x - bounds.x_min) / scale
    ny_ = (y - bounds.y_min) / scale
    lon = MAP_CENTER_LON + (nx_ - 0.5) * MAP_SPAN_DEG
    lat = MAP_CENTER_LAT + (ny_ - 0.5) * MAP_SPAN_DEG
    return lon, lat


def _voltage_color(v: float | None) -> tuple[int, int, int]:
    if v is None:
        return COLOR_GREY
    if v >= VOLTAGE_GREEN_MIN:
        return COLOR_GREEN
    if v >= VOLTAGE_YELLOW_MIN:
        return COLOR_YELLOW
    return COLOR_RED


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_map_data(
    graph: nx.Graph,
    voltages: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assemble node + edge dataframes for the feeder map.

    Args:
        graph: Output of :func:`gridsense.topology.load_ieee123`.
        voltages: Optional ``bus_name -> per-unit voltage`` mapping from a
            :class:`gridsense.power_flow.SnapshotResult`. Unknown buses get
            voltage ``1.0`` (assumed nominal) but neutral-grey colour so the
            map still renders before a snapshot is run.

    Returns:
        ``(nodes_df, edges_df)`` where ``nodes_df`` has columns
        ``[bus, x, y, lon, lat, voltage_pu, kw, kvar, color, radius, tooltip]``
        and ``edges_df`` has
        ``[name, kind, src, dst, src_lon, src_lat, dst_lon, dst_lat,
           path, color]``.
    """
    bounds = _compute_bounds(graph)
    voltages = voltages or {}

    # -- Nodes -----------------------------------------------------------
    node_rows: list[dict] = []
    for bus, attrs in graph.nodes(data=True):
        x = float(attrs.get("x", 0.0))
        y = float(attrs.get("y", 0.0))
        lon, lat = _project(x, y, bounds)
        kw = float(attrs.get("kw_load", 0.0))
        kvar = float(attrs.get("kvar_load", 0.0))
        v = voltages.get(bus)
        v_val = float(v) if v is not None else 1.0
        color = _voltage_color(v)
        # Radius (map units): scales with kW but clamped so the smallest
        # buses are still clickable.
        radius = 20.0 + (kw ** 0.5) * 4.0
        tooltip = (
            f"<b>Bus {bus}</b><br/>"
            f"V = {v_val:.3f} pu<br/>"
            f"kW = {kw:.1f}<br/>"
            f"kvar = {kvar:.1f}"
        )
        node_rows.append(
            {
                "bus": bus,
                "x": x,
                "y": y,
                "lon": lon,
                "lat": lat,
                "voltage_pu": v_val,
                "kw": kw,
                "kvar": kvar,
                "color": [int(c) for c in color],
                "radius": float(radius),
                "tooltip": tooltip,
            }
        )
    nodes_df = pd.DataFrame(node_rows)

    # -- Edges -----------------------------------------------------------
    edge_rows: list[dict] = []
    node_lookup = {
        row["bus"]: (row["lon"], row["lat"]) for row in node_rows
    }
    for u, v, attrs in graph.edges(data=True):
        src_lon, src_lat = node_lookup.get(u, (MAP_CENTER_LON, MAP_CENTER_LAT))
        dst_lon, dst_lat = node_lookup.get(v, (MAP_CENTER_LON, MAP_CENTER_LAT))
        kind = str(attrs.get("element_type", "line"))
        color = EDGE_COLORS.get(kind, EDGE_COLORS["line"])
        edge_rows.append(
            {
                "name": str(attrs.get("name", "")),
                "kind": kind,
                "src": u,
                "dst": v,
                "src_lon": src_lon,
                "src_lat": src_lat,
                "dst_lon": dst_lon,
                "dst_lat": dst_lat,
                "path": [[src_lon, src_lat], [dst_lon, dst_lat]],
                "color": [int(c) for c in color],
            }
        )
    edges_df = pd.DataFrame(edge_rows)

    return nodes_df, edges_df


def build_deck(nodes_df: pd.DataFrame, edges_df: pd.DataFrame):
    """Compose a :class:`pydeck.Deck` for the map tab.

    Import of pydeck is deferred so importing this module has no heavy
    side-effects (keeps the pytest smoke test fast).
    """
    import pydeck as pdk

    line_layer = pdk.Layer(
        "PathLayer",
        data=edges_df,
        get_path="path",
        get_color="color",
        width_scale=1,
        width_min_pixels=2,
        get_width=3,
        pickable=False,
    )
    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=nodes_df,
        get_position="[lon, lat]",
        get_fill_color="color",
        get_radius="radius",
        radius_units="meters",
        radius_min_pixels=3,
        radius_max_pixels=14,
        pickable=True,
        auto_highlight=True,
    )
    view_state = pdk.ViewState(
        latitude=MAP_CENTER_LAT,
        longitude=MAP_CENTER_LON,
        zoom=13,
        pitch=0,
        bearing=0,
    )
    deck = pdk.Deck(
        layers=[line_layer, scatter_layer],
        initial_view_state=view_state,
        tooltip={"html": "{tooltip}", "style": {"color": "white"}},
        map_style=None,  # light default; no Mapbox token required.
    )
    return deck


def render() -> None:
    """Backwards-compat placeholder — the main app renders the tab directly."""
    return None
