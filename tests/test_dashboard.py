"""Smoke tests for the Streamlit dashboard.

The actual UI rendering is covered by the (optional) Playwright flow in
T+9; these tests only verify the module imports cleanly without triggering
the Streamlit runtime and that the map-data helper returns a
well-formed DataFrame.
"""

from __future__ import annotations

import importlib
import sys


def test_app_imports() -> None:
    """Importing ``app.streamlit_app`` must not execute the full UI.

    The module is guarded by ``_running_as_streamlit()`` so that pytest
    (which has no Streamlit runtime) can import it without calls to
    ``st.title`` / ``st.tabs`` raising ``NoSessionContextError``.
    """
    # Drop a possibly-stale import so the guard is actually exercised.
    sys.modules.pop("app.streamlit_app", None)
    module = importlib.import_module("app.streamlit_app")

    # Public handles the Streamlit runtime will call:
    assert hasattr(module, "main"), "main() entry point missing"
    assert callable(module.main)
    # Guard helper exists and reports False outside `streamlit run`.
    assert hasattr(module, "_running_as_streamlit")
    assert module._running_as_streamlit() is False


def test_build_map_data() -> None:
    """``build_map_data`` returns node + edge DataFrames with expected columns."""
    import pandas as pd

    from app.components.feeder_map import build_map_data
    from gridsense.topology import load_ieee123

    graph = load_ieee123()
    # Fake a few voltages so the colour-assignment code path is exercised.
    fake_voltages = {"1": 0.94, "7": 0.96, "51": 1.01}
    nodes_df, edges_df = build_map_data(graph, fake_voltages)

    assert isinstance(nodes_df, pd.DataFrame)
    assert isinstance(edges_df, pd.DataFrame)

    expected_node_cols = {"bus", "x", "y", "voltage_pu", "kw"}
    assert expected_node_cols.issubset(set(nodes_df.columns))

    # Shape sanity: one row per topology node / edge (no duplication).
    assert len(nodes_df) == graph.number_of_nodes()
    assert len(edges_df) == graph.number_of_edges()

    # Fake voltages land on their buses.
    bus1 = nodes_df[nodes_df["bus"] == "1"].iloc[0]
    assert abs(bus1["voltage_pu"] - 0.94) < 1e-9
    # A red-band bus picks up the red colour.
    assert list(bus1["color"]) == [231, 76, 60]

    # Buses without an explicit voltage default to nominal 1.0 pu and
    # neutral grey.
    bus_no_v = nodes_df[~nodes_df["bus"].isin(fake_voltages.keys())].iloc[0]
    assert abs(bus_no_v["voltage_pu"] - 1.0) < 1e-9
    assert list(bus_no_v["color"]) == [127, 140, 141]
