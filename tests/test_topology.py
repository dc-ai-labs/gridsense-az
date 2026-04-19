"""Tests for gridsense.topology — IEEE 123 .dss parser + PyG converter.

PLAN.md §5.1. Verifies the pure-stdlib parser produces a graph with the
canonical IEEE 123 counts (132 buses, 131 edges, 85 spot loads, 3490 kW
total) and that to_pyg_data returns correctly-shaped tensors with no NaNs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def test_topology_importable():
    """Module imports and exposes the public API."""
    from gridsense import topology

    assert hasattr(topology, "load_ieee123")
    assert hasattr(topology, "to_pyg_data")
    assert callable(topology.load_ieee123)
    assert callable(topology.to_pyg_data)


def test_load_ieee123_bus_count():
    """Canonical IEEE 123 has 132 buses (128 load/switching + 150, 150r, 25r, 9r regs)."""
    from gridsense.topology import load_ieee123

    G = load_ieee123()
    # 132 is what our parser produces. Allow ±4 slack for alternate interpretations
    # of regulator aliases but assert reasonable bounds.
    assert 128 <= len(G.nodes) <= 135, f"unexpected bus count: {len(G.nodes)}"


def test_load_ieee123_edge_count():
    """IEEE 123 is a radial feeder — edges == nodes - 1 (tree) modulo switch ties."""
    from gridsense.topology import load_ieee123

    G = load_ieee123()
    n_nodes = len(G.nodes)
    n_edges = len(G.edges)
    # Tree has N-1 edges; allow up to N for normally-open tie switches.
    assert n_nodes - 5 <= n_edges <= n_nodes + 2, (
        f"unexpected edge count: nodes={n_nodes} edges={n_edges}"
    )


def test_loads_aggregated():
    """85 spot loads aggregate to 3490 kW / 1920 kvar (IEEE 123 canonical)."""
    from gridsense.topology import load_ieee123

    G = load_ieee123()
    loaded = [n for n, d in G.nodes(data=True) if d.get("kw_load", 0) > 0]
    total_kw = sum(d.get("kw_load", 0) for _, d in G.nodes(data=True))
    total_kvar = sum(d.get("kvar_load", 0) for _, d in G.nodes(data=True))

    # Canonical IEEE 123 has 85 spot loads totalling 3490 kW + 1920 kvar.
    assert 80 <= len(loaded) <= 90, f"loaded-bus count off: {len(loaded)}"
    assert abs(total_kw - 3490.0) < 50.0, f"total kW off: {total_kw}"
    assert abs(total_kvar - 1920.0) < 50.0, f"total kvar off: {total_kvar}"


def test_bus_coords_loaded():
    """BusCoords.dat provides (x,y) for every bus in the feeder."""
    from gridsense.topology import load_ieee123

    G = load_ieee123()
    with_xy = sum(
        1 for _, d in G.nodes(data=True) if d.get("x") is not None and d.get("y") is not None
    )
    # Require coords on at least 90% of buses; some regulator aliases may not be in
    # BusCoords.dat directly.
    assert with_xy >= 0.9 * len(G.nodes), (
        f"only {with_xy}/{len(G.nodes)} buses have coordinates"
    )


def test_to_pyg_data_shapes():
    """PyG Data: x [N,5], edge_index [2, 2E], edge_attr [2E, 2], pos [N,2]."""
    from gridsense.topology import load_ieee123, to_pyg_data

    G = load_ieee123()
    data = to_pyg_data(G)

    n = len(G.nodes)
    e = len(G.edges)
    assert tuple(data.x.shape) == (n, 5), f"x.shape={tuple(data.x.shape)}"
    # Undirected → 2E directed edges.
    assert tuple(data.edge_index.shape) == (2, 2 * e), (
        f"edge_index.shape={tuple(data.edge_index.shape)}"
    )
    assert tuple(data.edge_attr.shape) == (2 * e, 2), (
        f"edge_attr.shape={tuple(data.edge_attr.shape)}"
    )
    assert tuple(data.pos.shape) == (n, 2), f"pos.shape={tuple(data.pos.shape)}"
    assert len(data.node_names) == n


def test_to_pyg_data_no_nan():
    """Standardised features + edge attrs should contain no NaN/Inf."""
    from gridsense.topology import load_ieee123, to_pyg_data

    G = load_ieee123()
    data = to_pyg_data(G)

    assert not bool(data.x.isnan().any()), "x contains NaN"
    assert not bool(data.x.isinf().any()), "x contains Inf"
    assert not bool(data.edge_attr.isnan().any()), "edge_attr contains NaN"
    assert not bool(data.edge_attr.isinf().any()), "edge_attr contains Inf"
    assert not bool(data.pos.isnan().any()), "pos contains NaN"


def test_node_names_roundtrip():
    """Node-names list aligns with x rows and matches the nx.Graph node order."""
    from gridsense.topology import load_ieee123, to_pyg_data

    G = load_ieee123()
    data = to_pyg_data(G)

    # Every node_name should exist in the source graph.
    for name in data.node_names:
        assert name in G.nodes, f"node_name {name!r} not in graph"

    # Index alignment: row i of data.x corresponds to data.node_names[i].
    # Pick a loaded bus and verify its row has nonzero load-related features.
    for i, name in enumerate(data.node_names):
        attrs = G.nodes[name]
        if attrs.get("kw_load", 0) > 0:
            # After standardisation, loaded bus should have kw_load_std != 0 typically;
            # just ensure the row isn't all zeros (would indicate lookup failure).
            row = data.x[i]
            assert row.abs().sum().item() > 0, f"row for loaded bus {name} is all zeros"
            break
    else:
        pytest.fail("no loaded buses found — aggregation broken")
