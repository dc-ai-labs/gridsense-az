"""IEEE 123-bus feeder topology parser → NetworkX graph + PyG Data.

Pure-stdlib parser of OpenDSS (.dss) circuit descriptions. Does NOT import
`opendssdirect` — so this module can be used for adjacency construction and
dashboard rendering without a physics solver installed.

Entry points
------------
* :func:`load_ieee123` — parse ``data/ieee123/IEEE123Master.dss`` (and its
  ``Redirect``-ed includes) into a :class:`networkx.Graph`.
* :func:`to_pyg_data` — convert the graph into a
  :class:`torch_geometric.data.Data` with standardised node features,
  undirected ``edge_index``, edge attributes, raw coordinates, and the
  ordered list of bus names for round-trip mapping.

The parser recognises ``New Line``, ``New Load``, ``New Transformer``, and
``Redirect`` directives.  It is tolerant of the conventions used throughout
the Kersting/EPRI OpenDSS corpus: case-insensitive keywords, ``~``
continuation lines, ``!`` comments (inline and whole-line), bus-phase
suffixes (``150.1.2.3``), and bracketed ``buses=[...]`` transformer syntax.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import networkx as nx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

DEFAULT_ROOT: Path = Path(__file__).resolve().parents[2] / "data" / "ieee123"
"""Default directory containing ``IEEE123Master.dss`` and its includes."""

MASTER_FILENAME: str = "IEEE123Master.dss"
COORDS_FILENAME: str = "BusCoords.dat"

DEFAULT_BASE_KV: float = 4.16
"""Primary distribution voltage of the IEEE 123-bus feeder, in kV."""

DEFAULT_PHASES: int = 3
KFT_TO_FT: float = 1_000.0
"""``units=kft`` → multiply Length by this to get feet."""

# Regexes are compiled once at import and reused.
_COMMENT_RE = re.compile(r"\s*!.*$")
_KEY_VALUE_RE = re.compile(
    r"(?P<key>[A-Za-z_%][A-Za-z0-9_%]*)\s*=\s*"
    r"(?P<val>\"[^\"]*\"|'[^']*'|\[[^\]]*\]|\S+)"
)
_DIRECTIVE_RE = re.compile(
    r"^\s*(?P<verb>new|redirect|compile)\b\s*(?P<rest>.*)$", re.IGNORECASE
)
_NEW_OBJECT_RE = re.compile(
    r"^\s*(?:object\s*=\s*)?(?P<kind>[A-Za-z]+)\.(?P<name>[A-Za-z0-9_]+)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------


@dataclass
class BusAttributes:
    """Per-bus node attributes attached to the topology graph."""

    x: float = 0.0
    y: float = 0.0
    base_kv: float = DEFAULT_BASE_KV
    kw_load: float = 0.0
    kvar_load: float = 0.0


@dataclass
class EdgeAttributes:
    """Per-element edge attributes attached to the topology graph."""

    element_type: str  # "line" | "switch" | "transformer"
    name: str = ""
    length_ft: float = 0.0
    linecode: str = ""
    normamps: float | None = None
    phases: int = DEFAULT_PHASES


# ---------------------------------------------------------------------------
# Internal dataclasses
# ---------------------------------------------------------------------------


@dataclass
class _Directive:
    """A single logical DSS command — may span multiple ``~``-joined lines."""

    verb: str
    kind: str = ""
    name: str = ""
    kwargs: dict[str, str] = field(default_factory=dict)
    raw: str = ""


# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------


def _strip_comment(line: str) -> str:
    """Remove any ``!``-prefixed comment tail, preserving quoted strings."""
    # OpenDSS comments are not quoted in practice in the IEEE-123 bundle —
    # a simple trailing strip is sufficient.
    return _COMMENT_RE.sub("", line).rstrip()


def _iter_logical_lines(path: Path) -> Iterable[str]:
    """Yield logical lines from a .dss file, joining ``~`` continuations.

    Each yielded line is already stripped of comments and leading/trailing
    whitespace, and is never empty.
    """
    buffered = ""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            text = _strip_comment(raw).strip()
            if not text:
                continue
            if text.startswith("~"):
                # Continuation of previous logical line.
                buffered = (buffered + " " + text[1:].strip()).strip()
                continue
            if buffered:
                yield buffered
            buffered = text
    if buffered:
        yield buffered


def _parse_kwargs(rest: str) -> dict[str, str]:
    """Extract ``key=value`` pairs from a directive tail.

    Values may be bare tokens, quoted strings, or bracketed lists.
    Keys are lower-cased for case-insensitive lookup.
    """
    return {m.group("key").lower(): m.group("val") for m in _KEY_VALUE_RE.finditer(rest)}


def _parse_directive(line: str) -> _Directive | None:
    """Parse a single logical line into a :class:`_Directive`.

    Returns ``None`` if the line is not a recognised DSS directive (e.g.
    ``Clear``, ``Set ...``, ``Solve`` — we ignore those for topology).
    """
    match = _DIRECTIVE_RE.match(line)
    if not match:
        return None
    verb = match.group("verb").lower()
    rest = match.group("rest").strip()
    if verb == "redirect":
        # Redirect <path>  — path may or may not be quoted.
        target = rest.strip().strip('"').strip("'")
        # Only take the first token — some files have trailing comments already stripped.
        target = target.split()[0] if target else ""
        return _Directive(verb="redirect", raw=target)
    if verb == "compile":
        target = rest.strip().strip("()").strip('"').strip("'").split()[0]
        return _Directive(verb="compile", raw=target)
    # verb == "new"
    object_match = _NEW_OBJECT_RE.match(rest)
    if not object_match:
        return None
    kind = object_match.group("kind").lower()
    name = object_match.group("name").lower()
    # Remove the `object=kind.name` prefix when computing kwargs.
    tail = rest[object_match.end():]
    kwargs = _parse_kwargs(tail)
    return _Directive(verb="new", kind=kind, name=name, kwargs=kwargs, raw=rest)


# ---------------------------------------------------------------------------
# File-level iteration (with Redirect recursion)
# ---------------------------------------------------------------------------


def _iter_directives(master: Path) -> Iterable[_Directive]:
    """Walk ``master`` and recurse into every ``Redirect`` target.

    Cycle-safe: each absolute path is visited at most once.
    """
    visited: set[Path] = set()
    stack: list[Path] = [master.resolve()]
    while stack:
        current = stack.pop()
        if current in visited or not current.exists():
            if not current.exists():
                logger.warning("Redirect target not found: %s", current)
            continue
        visited.add(current)
        logger.debug("Parsing %s", current)
        for line in _iter_logical_lines(current):
            directive = _parse_directive(line)
            if directive is None:
                continue
            if directive.verb == "redirect":
                target = (current.parent / directive.raw).resolve()
                stack.append(target)
                continue
            if directive.verb == "compile":
                target = (current.parent / directive.raw).resolve()
                stack.append(target)
                continue
            yield directive


# ---------------------------------------------------------------------------
# Field helpers
# ---------------------------------------------------------------------------


def _strip_bus_name(bus: str) -> str:
    """Strip phase suffixes: ``150.1.2.3`` → ``150``."""
    return bus.split(".", 1)[0].strip().lower()


def _parse_buses_bracket(raw: str) -> list[str]:
    """Parse a ``buses=[<b1> <b2>]`` value — whitespace or comma separated."""
    inner = raw.strip().lstrip("[").rstrip("]")
    return [tok for tok in re.split(r"[,\s]+", inner) if tok]


def _as_float(raw: str | None, default: float = 0.0) -> float:
    if raw is None:
        return default
    try:
        return float(str(raw).strip().strip("[]").split()[0])
    except (ValueError, IndexError):
        return default


def _as_int(raw: str | None, default: int) -> int:
    try:
        return int(float(str(raw).strip()))
    except (ValueError, TypeError):
        return default


def _as_bool_yes(raw: str | None) -> bool:
    return str(raw).strip().lower() in {"yes", "true", "y", "t", "1"} if raw else False


# ---------------------------------------------------------------------------
# Bus coordinates
# ---------------------------------------------------------------------------


def _load_bus_coords(root: Path) -> dict[str, tuple[float, float]]:
    """Read ``BusCoords.dat`` (whitespace or comma delimited, one bus per line)."""
    coords_path = root / COORDS_FILENAME
    out: dict[str, tuple[float, float]] = {}
    if not coords_path.exists():
        logger.warning("BusCoords.dat not found at %s", coords_path)
        return out
    with coords_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            text = _strip_comment(raw).strip()
            if not text:
                continue
            parts = re.split(r"[,\s]+", text)
            if len(parts) < 3:
                continue
            name = parts[0].strip().lower()
            try:
                x, y = float(parts[1]), float(parts[2])
            except ValueError:
                continue
            out[name] = (x, y)
    return out


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------


def _apply_line(graph: nx.Graph, directive: _Directive) -> None:
    kwargs = directive.kwargs
    bus1_raw = kwargs.get("bus1")
    bus2_raw = kwargs.get("bus2")
    if not bus1_raw or not bus2_raw:
        logger.debug("Skipping malformed Line.%s: missing bus1/bus2", directive.name)
        return
    bus1 = _strip_bus_name(bus1_raw)
    bus2 = _strip_bus_name(bus2_raw)
    if bus1 == bus2:
        return

    units = kwargs.get("units", "kft").strip().lower()
    length = _as_float(kwargs.get("length"), default=0.0)
    length_ft = length * (KFT_TO_FT if units == "kft" else 1.0)
    if units not in {"kft", "ft"}:
        # mi, m, km — approximate conversions. IEEE-123 only uses kft.
        length_ft = length  # fall through; we don't validate further

    is_switch = _as_bool_yes(kwargs.get("switch"))
    attrs = EdgeAttributes(
        element_type="switch" if is_switch else "line",
        name=directive.name,
        length_ft=length_ft,
        linecode=kwargs.get("linecode", "").strip(),
        normamps=_as_float(kwargs.get("normamps")) or None,
        phases=_as_int(kwargs.get("phases"), default=DEFAULT_PHASES),
    )

    for bus in (bus1, bus2):
        if bus not in graph:
            graph.add_node(bus, **BusAttributes().__dict__)
    graph.add_edge(bus1, bus2, **attrs.__dict__)


def _apply_load(graph: nx.Graph, directive: _Directive) -> None:
    kwargs = directive.kwargs
    bus_raw = kwargs.get("bus1") or kwargs.get("bus")
    if not bus_raw:
        logger.debug("Skipping malformed Load.%s: no bus", directive.name)
        return
    bus = _strip_bus_name(bus_raw)
    kw = _as_float(kwargs.get("kw"))
    kvar = _as_float(kwargs.get("kvar"))
    kv = _as_float(kwargs.get("kv"), default=DEFAULT_BASE_KV)

    if bus not in graph:
        attrs = BusAttributes(base_kv=kv)
        graph.add_node(bus, **attrs.__dict__)
    # Aggregate multiple loads on the same bus.
    node = graph.nodes[bus]
    node["kw_load"] = node.get("kw_load", 0.0) + kw
    node["kvar_load"] = node.get("kvar_load", 0.0) + kvar
    if kv and not node.get("base_kv"):
        node["base_kv"] = kv


def _apply_transformer(graph: nx.Graph, directive: _Directive) -> None:
    kwargs = directive.kwargs
    buses: list[str] = []
    if "buses" in kwargs:
        raw = kwargs["buses"]
        parsed = _parse_buses_bracket(raw)
        buses = [_strip_bus_name(b) for b in parsed[:2]]
    else:
        # wdg=1 bus=X, wdg=2 bus=Y pattern — already continuation-joined.
        wdg_matches = re.findall(
            r"wdg\s*=\s*(\d+)\s+bus\s*=\s*(\S+)", directive.raw, flags=re.IGNORECASE
        )
        ordered = sorted(wdg_matches, key=lambda p: int(p[0]))
        buses = [_strip_bus_name(b) for _, b in ordered[:2]]

    buses = [b for b in buses if b]
    if len(buses) < 2 or buses[0] == buses[1]:
        return
    attrs = EdgeAttributes(
        element_type="transformer",
        name=directive.name,
        length_ft=0.0,
        linecode="",
        normamps=None,
        phases=_as_int(kwargs.get("phases"), default=DEFAULT_PHASES),
    )
    for bus in buses:
        if bus not in graph:
            graph.add_node(bus, **BusAttributes().__dict__)
    graph.add_edge(buses[0], buses[1], **attrs.__dict__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_ieee123(root: Path = DEFAULT_ROOT) -> nx.Graph:
    """Parse the IEEE 123 feeder into an undirected :class:`networkx.Graph`.

    Args:
        root: Directory containing ``IEEE123Master.dss`` and its includes
            (``IEEELineCodes.DSS``, ``IEEE123Loads.DSS``,
            ``IEEE123Regulators.DSS``) plus ``BusCoords.dat``.

    Returns:
        Undirected graph where nodes are bus names (strings) carrying the
        flat attribute set of :class:`BusAttributes`, and edges carry the
        flat attribute set of :class:`EdgeAttributes`.

    Raises:
        FileNotFoundError: If ``IEEE123Master.dss`` is missing under ``root``.
    """
    root = root.resolve()
    master = root / MASTER_FILENAME
    if not master.exists():
        raise FileNotFoundError(f"IEEE 123 master file not found: {master}")

    graph: nx.Graph = nx.Graph()
    for directive in _iter_directives(master):
        if directive.verb != "new":
            continue
        if directive.kind == "line":
            _apply_line(graph, directive)
        elif directive.kind == "load":
            _apply_load(graph, directive)
        elif directive.kind == "transformer":
            _apply_transformer(graph, directive)
        # Capacitors, regcontrols, circuits — not edges.

    # Attach bus coordinates.
    coords = _load_bus_coords(root)
    for bus, (x, y) in coords.items():
        if bus in graph:
            graph.nodes[bus]["x"] = x
            graph.nodes[bus]["y"] = y

    logger.info(
        "load_ieee123: nodes=%d edges=%d loaded_buses=%d",
        graph.number_of_nodes(),
        graph.number_of_edges(),
        sum(1 for _, attrs in graph.nodes(data=True) if attrs.get("kw_load", 0) > 0),
    )
    return graph


def to_pyg_data(graph: nx.Graph):  # -> torch_geometric.data.Data
    """Convert the topology graph into a :class:`torch_geometric.data.Data`.

    Node features (``data.x``, shape ``[N, 5]``) are standardised to unit
    scale along each column: ``[base_kv, kw_load, kvar_load, x, y]``.
    Columns with zero variance are left mean-centred only.

    Edge indices (``data.edge_index``, shape ``[2, 2*|E|]``) encode the
    undirected graph as two directed edges per original edge.
    Edge features (``data.edge_attr``, shape ``[2*|E|, 2]``) are
    ``[length_ft_scaled, normamps_scaled]``; missing ``normamps`` is
    imputed to the column mean before standardisation.

    Raw bus positions (``data.pos``, shape ``[N, 2]``) are preserved for
    dashboard rendering (pydeck expects un-standardised coordinates).
    The ordered list of bus names is returned as ``data.node_names`` so
    callers can round-trip predictions back to the topology.

    Args:
        graph: Output of :func:`load_ieee123`.

    Returns:
        ``torch_geometric.data.Data`` instance.
    """
    # Imports are local so the heavy torch stack is optional for pure-parsing use.
    import torch
    from torch_geometric.data import Data

    node_names: list[str] = sorted(graph.nodes())
    index_of = {name: i for i, name in enumerate(node_names)}
    num_nodes = len(node_names)

    # Raw node feature matrix [N, 5].
    raw_x = torch.zeros((num_nodes, 5), dtype=torch.float32)
    raw_pos = torch.zeros((num_nodes, 2), dtype=torch.float32)
    for i, name in enumerate(node_names):
        attrs = graph.nodes[name]
        raw_x[i, 0] = float(attrs.get("base_kv", DEFAULT_BASE_KV))
        raw_x[i, 1] = float(attrs.get("kw_load", 0.0))
        raw_x[i, 2] = float(attrs.get("kvar_load", 0.0))
        raw_x[i, 3] = float(attrs.get("x", 0.0))
        raw_x[i, 4] = float(attrs.get("y", 0.0))
        raw_pos[i, 0] = float(attrs.get("x", 0.0))
        raw_pos[i, 1] = float(attrs.get("y", 0.0))

    # Standardise columns (zero-mean, unit-std). Guard against std == 0.
    mean = raw_x.mean(dim=0, keepdim=True)
    std = raw_x.std(dim=0, keepdim=True)
    std = torch.where(std < 1e-12, torch.ones_like(std), std)
    x = (raw_x - mean) / std

    # Edge index + features.
    edge_list: list[tuple[int, int]] = []
    raw_lengths: list[float] = []
    raw_normamps: list[float] = []
    missing_normamps_mask: list[bool] = []
    for u, v, attrs in graph.edges(data=True):
        i, j = index_of[u], index_of[v]
        edge_list.append((i, j))
        edge_list.append((j, i))
        length = float(attrs.get("length_ft", 0.0))
        normamps = attrs.get("normamps")
        raw_lengths.extend([length, length])
        raw_normamps.extend([float(normamps) if normamps is not None else 0.0] * 2)
        missing_normamps_mask.extend([normamps is None, normamps is None])

    if not edge_list:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 2), dtype=torch.float32)
    else:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        lengths = torch.tensor(raw_lengths, dtype=torch.float32)
        normamps = torch.tensor(raw_normamps, dtype=torch.float32)
        # Impute missing normamps to the column mean of present values.
        missing = torch.tensor(missing_normamps_mask, dtype=torch.bool)
        if (~missing).any():
            present_mean = normamps[~missing].mean()
        else:
            present_mean = torch.tensor(0.0)
        normamps = torch.where(missing, present_mean, normamps)

        def _standardise(col: torch.Tensor) -> torch.Tensor:
            s = col.std()
            if s < 1e-12:
                return col - col.mean()
            return (col - col.mean()) / s

        edge_attr = torch.stack([_standardise(lengths), _standardise(normamps)], dim=1)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=raw_pos)
    data.node_names = node_names  # not a tensor — stored as python list
    return data
