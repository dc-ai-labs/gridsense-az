"use client";

import "leaflet/dist/leaflet.css";

import { useEffect, useMemo, useRef, useState } from "react";
import { useScenario } from "@/lib/context";
import { riskTier } from "@/lib/validate";
import type {
  CircleMarker,
  HeatLayer,
  Layer,
  LayerGroup,
  Map as LeafletMap,
} from "leaflet";
import type {
  PerBusMetric,
  ScenarioKind,
  TomorrowForecast,
} from "@/lib/types";

const SCENARIO_ACCENT: Record<ScenarioKind, string> = {
  baseline: "#4fdbc8",
  heat: "#ffb95f",
  ev: "#ffb3ad",
  custom: "#bbcac6",
};

const SCENARIO_LABEL_MAP: Record<ScenarioKind, string> = {
  baseline: "BASELINE",
  heat: "HEAT +10°F",
  ev: "EV SURGE",
  custom: "CUSTOM",
};

const TIER_COLOR = {
  error: "#ffb4ab",
  secondary: "#ffb95f",
  primary: "#4fdbc8",
} as const;

// Shared color stops for the leaflet.heat gradient AND the legend bar so
// they stay in visual parity. primary → secondary → tertiary → error.
const HEAT_GRADIENT: Record<number, string> = {
  0.0: "rgba(79, 219, 200, 0)",  // transparent teal — fades to nothing
  0.25: "#4fdbc8",                // teal
  0.5: "#ffc857",                 // saturated amber (richer than #ffb95f)
  0.75: "#ff6b4a",                // saturated coral
  1.0: "#ff3131",                 // deep red hotspot (punchy peak)
};

// CSS linear-gradient mirroring HEAT_GRADIENT for the legend ramp.
const HEAT_GRADIENT_CSS =
  "linear-gradient(to right, #4fdbc8 0%, #ffc857 40%, #ff6b4a 75%, #ff3131 100%)";

type HeatMode = "off" | "risk" | "load";

// IEEE-123 is a synthetic distribution feeder (~3.5 MW nameplate) and has no
// real geo coordinates. We pin its normalized [0, 1] layout into a fixed real
// Phoenix lat/lon bbox so the basemap and the topology share one projection
// and bus markers always overlap the same streets at any zoom level.
//
// Anchor: ~4.0 km E/W × 2.4 km N/S box centered on downtown Phoenix
// (≈ 33.4484°N, -112.0740°W, near the Capitol / City Hall block).
// 1° lat ≈ 110.95 km · 1° lng @ 33.45° ≈ 92.91 km
const PHX_BOUNDS = {
  lat_min: 33.4376,
  lat_max: 33.4592,
  lng_min: -112.09555,
  lng_max: -112.05245,
} as const;

const PHX_CENTER: [number, number] = [
  (PHX_BOUNDS.lat_min + PHX_BOUNDS.lat_max) / 2,
  (PHX_BOUNDS.lng_min + PHX_BOUNDS.lng_max) / 2,
];

function projectToLatLng(x_norm: number, y_norm: number): [number, number] {
  const lng =
    PHX_BOUNDS.lng_min + x_norm * (PHX_BOUNDS.lng_max - PHX_BOUNDS.lng_min);
  // y_norm grows downward in the source layout; flip so high-y → south.
  const lat =
    PHX_BOUNDS.lat_max - y_norm * (PHX_BOUNDS.lat_max - PHX_BOUNDS.lat_min);
  return [lat, lng];
}

interface HoverState {
  bus: string;
  lat: number;
  lng: number;
  metric: PerBusMetric;
}

export default function TacticalMap() {
  const {
    active,
    compareWith,
    current,
    baseline,
    heat,
    ev,
    topology,
    focus,
    setFocus,
  } = useScenario();
  const [hover, setHover] = useState<HoverState | null>(null);
  const [hoverPos, setHoverPos] = useState<{ x: number; y: number } | null>(
    null,
  );
  const [mapReady, setMapReady] = useState(false);
  const [heatMode, setHeatMode] = useState<HeatMode>("risk");

  const containerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<LeafletMap | null>(null);
  const leafletRef = useRef<typeof import("leaflet") | null>(null);

  const edgesLayerRef = useRef<LayerGroup | null>(null);
  const nodesLayerRef = useRef<LayerGroup | null>(null);
  const compareLayerRef = useRef<LayerGroup | null>(null);
  const evLayerRef = useRef<LayerGroup | null>(null);
  const focusLayerRef = useRef<LayerGroup | null>(null);
  const heatLayerRef = useRef<Layer | null>(null);
  const heatPluginReady = useRef(false);

  const compareScenario: TomorrowForecast | null =
    compareWith === null
      ? null
      : compareWith === "baseline"
        ? baseline
        : compareWith === "heat"
          ? heat
          : ev;
  const compareAccent =
    compareWith !== null ? SCENARIO_ACCENT[compareWith] : null;

  // Initialize Leaflet exactly once.
  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;
    let cancelled = false;
    (async () => {
      const L = (await import("leaflet")).default;
      // leaflet.heat augments the global L with L.heatLayer — side-effect import.
      await import("leaflet.heat");
      heatPluginReady.current = true;
      if (cancelled || !containerRef.current) return;

      const map = L.map(containerRef.current, {
        center: PHX_CENTER,
        zoom: 14,
        minZoom: 12,
        maxZoom: 17,
        zoomControl: false,
        attributionControl: false,
      });

      // Split base + labels so we can crank label opacity/contrast separately.
      L.tileLayer(
        "https://{s}.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}{r}.png",
        {
          subdomains: "abcd",
          maxZoom: 19,
          attribution: "OSM · CARTO",
        },
      ).addTo(map);

      L.tileLayer(
        "https://{s}.basemaps.cartocdn.com/dark_only_labels/{z}/{x}/{y}{r}.png",
        {
          subdomains: "abcd",
          maxZoom: 19,
          opacity: 0.5,
          pane: "tilePane",
          className: "tactical-map-labels",
        },
      ).addTo(map);

      L.control
        .attribution({ position: "bottomright", prefix: false })
        .addTo(map);
      L.control.zoom({ position: "bottomright" }).addTo(map);

      map.fitBounds(
        [
          [PHX_BOUNDS.lat_min, PHX_BOUNDS.lng_min],
          [PHX_BOUNDS.lat_max, PHX_BOUNDS.lng_max],
        ],
        { padding: [10, 10] },
      );

      // Heat layer added first so it sits below edges/nodes; the actual heat
      // overlay is installed by the heatMode effect below.
      const edgesLayer = L.layerGroup().addTo(map);
      const nodesLayer = L.layerGroup().addTo(map);
      const compareLayer = L.layerGroup().addTo(map);
      const evLayer = L.layerGroup().addTo(map);
      // Focus layer sits on top of everything else.
      const focusLayer = L.layerGroup().addTo(map);

      mapRef.current = map;
      leafletRef.current = L;
      edgesLayerRef.current = edgesLayer;
      nodesLayerRef.current = nodesLayer;
      compareLayerRef.current = compareLayer;
      evLayerRef.current = evLayer;
      focusLayerRef.current = focusLayer;
      setMapReady(true);

      // Force a redraw once the parent flex layout has settled.
      setTimeout(() => map.invalidateSize(), 50);
    })();

    return () => {
      cancelled = true;
      if (mapRef.current) {
        mapRef.current.remove();
        mapRef.current = null;
      }
      leafletRef.current = null;
      edgesLayerRef.current = null;
      nodesLayerRef.current = null;
      compareLayerRef.current = null;
      evLayerRef.current = null;
      focusLayerRef.current = null;
      heatLayerRef.current = null;
      heatPluginReady.current = false;
    };
  }, []);

  // Per-bus metric lookup for current scenario.
  const perBus = current.per_bus;
  const busNames = useMemo(
    () => topology.nodes.map((n) => n.bus),
    [topology.nodes],
  );

  const topRiskSet = useMemo(() => {
    const withMetric = busNames
      .map((b) => ({ bus: b, m: perBus[b] }))
      .filter(
        (x): x is { bus: string; m: PerBusMetric } => x.m !== undefined,
      );
    withMetric.sort((a, b) => b.m.risk_score - a.m.risk_score);
    return new Set(withMetric.slice(0, 10).map((x) => x.bus));
  }, [busNames, perBus]);

  const compareTopRiskSet = useMemo(() => {
    if (!compareScenario) return new Set<string>();
    const cmpPerBus = compareScenario.per_bus;
    const withMetric = busNames
      .map((b) => ({ bus: b, m: cmpPerBus[b] }))
      .filter(
        (x): x is { bus: string; m: PerBusMetric } => x.m !== undefined,
      );
    withMetric.sort((a, b) => b.m.risk_score - a.m.risk_score);
    return new Set(withMetric.slice(0, 10).map((x) => x.bus));
  }, [busNames, compareScenario]);

  const evBusSet = useMemo(() => {
    if (active !== "ev") return new Set<string>();
    const withMetric = busNames
      .map((b) => ({ bus: b, m: perBus[b] }))
      .filter(
        (x): x is { bus: string; m: PerBusMetric } => x.m !== undefined,
      );
    withMetric.sort((a, b) => b.m.peak_load_kw - a.m.peak_load_kw);
    return new Set(withMetric.slice(0, 20).map((x) => x.bus));
  }, [busNames, perBus, active]);

  // Edges layer (depends only on topology — stable between renders).
  useEffect(() => {
    const L = leafletRef.current;
    const layer = edgesLayerRef.current;
    if (!L || !layer || !mapReady) return;
    layer.clearLayers();

    const xy = new Map<string, [number, number]>();
    for (const n of topology.nodes) {
      xy.set(n.bus, projectToLatLng(n.x_norm, n.y_norm));
    }

    for (const e of topology.edges) {
      const a = xy.get(e.from);
      const b = xy.get(e.to);
      if (!a || !b) continue;
      L.polyline([a, b], {
        color: e.kind === "switch" ? "#5a6360" : "#7a8a87",
        weight: e.kind === "switch" ? 1 : 1.4,
        opacity: e.kind === "switch" ? 0.55 : 0.9,
        dashArray: e.kind === "switch" ? "3 3" : undefined,
        interactive: false,
      }).addTo(layer);
    }
  }, [topology, mapReady]);

  // Heat layer — a real geospatial continuous heatmap via leaflet.heat.
  // Rebuilds whenever heatMode, per-bus metrics, or topology change.
  useEffect(() => {
    const L = leafletRef.current;
    const map = mapRef.current;
    if (!L || !map || !mapReady || !heatPluginReady.current) return;

    // Always detach any existing heat layer before re-adding — prevents
    // stacked canvases and keeps renders flicker-free on slider drags.
    if (heatLayerRef.current) {
      map.removeLayer(heatLayerRef.current);
      heatLayerRef.current = null;
    }

    if (heatMode === "off") return;

    // Normalize load to 0..1 by bus-max; risk_score is already 0..1.
    let maxLoad = 0;
    if (heatMode === "load") {
      for (const n of topology.nodes) {
        const m = perBus[n.bus];
        if (m && m.peak_load_kw > maxLoad) maxLoad = m.peak_load_kw;
      }
      if (maxLoad <= 0) maxLoad = 1;
    }

    const points: [number, number, number][] = [];
    for (const n of topology.nodes) {
      const m = perBus[n.bus];
      if (!m) continue;
      const [lat, lng] = projectToLatLng(n.x_norm, n.y_norm);
      const intensity =
        heatMode === "risk" ? m.risk_score : m.peak_load_kw / maxLoad;
      points.push([lat, lng, intensity]);
    }

    // @types/leaflet.heat augments the Leaflet module with L.heatLayer; the
    // side-effect import above wires up the runtime implementation.
    const heat: HeatLayer = L.heatLayer(points, {
      radius: 26,
      blur: 14,
      minOpacity: 0.15,
      maxZoom: 17,
      max: 1.0,
      gradient: HEAT_GRADIENT,
    });
    heat.addTo(map);

    // Force heat canvas below the SVG overlay (edges/nodes/focus) so bus
    // markers stay crisp on top. Leaflet heat creates its own canvas inside
    // overlayPane; push it behind via CSS z-index tweak.
    const heatEl = (heat as unknown as { _canvas?: HTMLCanvasElement })._canvas;
    if (heatEl) {
      heatEl.style.zIndex = "1";
      heatEl.style.opacity = "0.85";
    }

    heatLayerRef.current = heat;
  }, [heatMode, perBus, topology, mapReady]);

  // Node layer (depends on perBus + topology + topRiskSet).
  useEffect(() => {
    const L = leafletRef.current;
    const layer = nodesLayerRef.current;
    if (!L || !layer || !mapReady) return;
    layer.clearLayers();

    for (const n of topology.nodes) {
      const ll = projectToLatLng(n.x_norm, n.y_norm);
      const metric = perBus[n.bus];
      const risk = metric?.risk_score ?? 0;
      const tier = riskTier(risk);
      const fill = TIER_COLOR[tier];
      const isTop = topRiskSet.has(n.bus);
      const r = isTop ? 7 : 4;

      if (isTop) {
        L.circleMarker(ll, {
          radius: r,
          color: fill,
          weight: 2,
          fill: false,
          className: "pulse-violation-svg",
          interactive: false,
        }).addTo(layer);
      }

      const marker: CircleMarker = L.circleMarker(ll, {
        radius: r,
        color: fill,
        weight: 0,
        fillColor: fill,
        fillOpacity: isTop ? 1 : 0.85,
        bubblingMouseEvents: false,
      }).addTo(layer);

      if (metric) {
        marker.on("mouseover", () => {
          setHover({ bus: n.bus, lat: ll[0], lng: ll[1], metric });
        });
        marker.on("mouseout", () => setHover(null));
      }
    }
  }, [topology, perBus, topRiskSet, mapReady]);

  // Compare-with ghost rings.
  useEffect(() => {
    const L = leafletRef.current;
    const layer = compareLayerRef.current;
    if (!L || !layer || !mapReady) return;
    layer.clearLayers();
    if (!compareAccent) return;
    for (const bus of compareTopRiskSet) {
      const node = topology.nodes.find((n) => n.bus === bus);
      if (!node) continue;
      const ll = projectToLatLng(node.x_norm, node.y_norm);
      L.circleMarker(ll, {
        radius: 10,
        color: compareAccent,
        weight: 1,
        opacity: 0.85,
        dashArray: "3 2",
        fill: false,
        interactive: false,
      }).addTo(layer);
    }
  }, [topology, compareTopRiskSet, compareAccent, mapReady]);

  // EV hotspot rings.
  useEffect(() => {
    const L = leafletRef.current;
    const layer = evLayerRef.current;
    if (!L || !layer || !mapReady) return;
    layer.clearLayers();
    if (active !== "ev") return;
    for (const bus of evBusSet) {
      const node = topology.nodes.find((n) => n.bus === bus);
      if (!node) continue;
      const ll = projectToLatLng(node.x_norm, node.y_norm);
      L.circleMarker(ll, {
        radius: 10,
        color: "#ffb95f",
        weight: 1,
        opacity: 0.7,
        dashArray: "2 2",
        fill: false,
        interactive: false,
      }).addTo(layer);
    }
  }, [topology, evBusSet, active, mapReady]);

  // Focus highlight (driven by PhysicsCheck row clicks via context).
  // Renders a bright halo + label for the focused bus, and/or a thick highlight
  // polyline for the focused OpenDSS line/switch element.
  useEffect(() => {
    const L = leafletRef.current;
    const map = mapRef.current;
    const layer = focusLayerRef.current;
    if (!L || !map || !layer || !mapReady) return;
    layer.clearLayers();
    if (!focus) return;

    const accent = focus.element ? "#ffb4ab" : "#4fdbc8";

    // Highlight a focused element (line/switch). Match by topology edge `name`.
    if (focus.element) {
      const edge = topology.edges.find(
        (e) => (e.name ?? "").toLowerCase() === focus.element,
      );
      if (edge) {
        const a = topology.nodes.find((n) => n.bus === edge.from);
        const b = topology.nodes.find((n) => n.bus === edge.to);
        if (a && b) {
          const aLL = projectToLatLng(a.x_norm, a.y_norm);
          const bLL = projectToLatLng(b.x_norm, b.y_norm);
          // Glow underlay
          L.polyline([aLL, bLL], {
            color: accent,
            weight: 9,
            opacity: 0.25,
            interactive: false,
          }).addTo(layer);
          // Crisp top stroke
          L.polyline([aLL, bLL], {
            color: accent,
            weight: 3,
            opacity: 1,
            interactive: false,
          }).addTo(layer);
          // Endpoint markers
          for (const ll of [aLL, bLL]) {
            L.circleMarker(ll, {
              radius: 6,
              color: accent,
              weight: 2,
              fill: false,
              interactive: false,
            }).addTo(layer);
          }
          // Center label
          const midLat = (aLL[0] + bLL[0]) / 2;
          const midLng = (aLL[1] + bLL[1]) / 2;
          L.marker([midLat, midLng], {
            interactive: false,
            keyboard: false,
            icon: L.divIcon({
              className: "tactical-focus-label",
              html: `<span>${focus.element.toUpperCase()} · ${edge.from}↔${edge.to}</span>`,
              iconSize: [0, 0],
              iconAnchor: [0, 0],
            }),
          }).addTo(layer);
          // Zoom to the midpoint of the highlighted line
          map.flyTo([midLat, midLng], 16, { animate: true, duration: 0.5 });
        }
      }
    }

    // Highlight a focused bus.
    if (focus.bus) {
      const node = topology.nodes.find((n) => n.bus === focus.bus);
      if (node) {
        const ll = projectToLatLng(node.x_norm, node.y_norm);
        const busAccent = focus.element ? "#ffb4ab" : "#4fdbc8";
        L.circleMarker(ll, {
          radius: 14,
          color: busAccent,
          weight: 2,
          opacity: 0.9,
          fill: false,
          interactive: false,
        }).addTo(layer);
        L.circleMarker(ll, {
          radius: 22,
          color: busAccent,
          weight: 1,
          opacity: 0.45,
          fill: false,
          dashArray: "4 3",
          interactive: false,
        }).addTo(layer);
        L.marker(ll, {
          interactive: false,
          keyboard: false,
          icon: L.divIcon({
            className: "tactical-focus-label",
            html: `<span>BUS_${focus.bus.toUpperCase()}</span>`,
            iconSize: [0, 0],
            iconAnchor: [0, -18],
          }),
        }).addTo(layer);

        map.panTo(ll, { animate: true, duration: 0.4 });
      }
    }
  }, [focus, topology, mapReady]);

  // Re-position hover tooltip on hover change OR on map pan/zoom.
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !hover) {
      setHoverPos(null);
      return;
    }
    const updatePos = () => {
      const pt = map.latLngToContainerPoint([hover.lat, hover.lng]);
      setHoverPos({ x: pt.x, y: pt.y });
    };
    updatePos();
    map.on("move", updatePos);
    map.on("zoom", updatePos);
    map.on("resize", updatePos);
    return () => {
      map.off("move", updatePos);
      map.off("zoom", updatePos);
      map.off("resize", updatePos);
    };
  }, [hover]);

  return (
    <div className="relative flex-1 overflow-hidden bg-surface-container-lowest tactical-map-leaflet">
      <div ref={containerRef} className="absolute inset-0" />


      {/* Layer label */}
      <div className="absolute top-3 left-3 bg-surface/85 p-2 font-mono text-[9px] uppercase border border-outline-variant z-[600] tracking-widest pointer-events-none">
        LAYER: IEEE_123_BUS_NETWORK // BASEMAP: PHX_AZ · SCEN:{" "}
        <span className="text-primary">{active.toUpperCase()}</span>
      </div>

      {/* Focus pill (visible when PhysicsCheck row is selected) */}
      {focus && (
        <div
          className="absolute top-12 left-3 z-[600] flex items-center gap-1 font-mono text-[9px] uppercase tracking-widest border bg-surface/90 pl-2 pr-1 py-1"
          style={{ borderColor: focus.element ? "#ffb4ab" : "#4fdbc8" }}
        >
          <span
            className="material-symbols-outlined text-[12px] leading-none"
            style={{ color: focus.element ? "#ffb4ab" : "#4fdbc8" }}
            aria-hidden
          >
            my_location
          </span>
          <span style={{ color: focus.element ? "#ffb4ab" : "#4fdbc8" }}>
            FOCUS:
          </span>
          <span className="text-on-surface">
            {focus.element
              ? focus.element.toUpperCase()
              : `BUS_${focus.bus?.toUpperCase()}`}
          </span>
          <button
            type="button"
            onClick={() => setFocus(null)}
            title="Clear focus (Esc)"
            className="ml-1 px-1 text-on-surface-variant hover:text-on-surface hover:bg-surface-container-low cursor-pointer"
          >
            ×
          </button>
        </div>
      )}

      {/* Heatmap legend + toggle (top-right) */}
      <div className="absolute top-3 right-3 z-[600] bg-surface/85 border border-outline-variant p-2 font-mono text-[9px] uppercase tracking-widest w-[186px]">
        <div className="flex items-center justify-between mb-1.5">
          <span className="text-on-surface-variant">HEATMAP</span>
          <span className="text-primary">
            {heatMode === "off" ? "—" : heatMode.toUpperCase()}
          </span>
        </div>
        <div className="flex gap-1 mb-2">
          {(["off", "risk", "load"] as const).map((mode) => {
            const on = heatMode === mode;
            return (
              <button
                key={mode}
                type="button"
                onClick={() => setHeatMode(mode)}
                className={
                  "flex-1 px-1 py-0.5 border font-mono text-[9px] uppercase tracking-widest cursor-pointer transition-colors " +
                  (on
                    ? "border-primary bg-primary/15 text-primary"
                    : "border-outline-variant text-on-surface-variant hover:border-outline hover:text-on-surface")
                }
              >
                {mode}
              </button>
            );
          })}
        </div>
        <div
          className={
            "h-2 border border-outline-variant " +
            (heatMode === "off" ? "opacity-30" : "")
          }
          style={{ backgroundImage: HEAT_GRADIENT_CSS }}
        />
        <div className="flex justify-between text-[8px] text-on-surface-variant mt-0.5 tracking-normal">
          <span>0.0</span>
          <span>0.5</span>
          <span>1.0</span>
        </div>
      </div>

      {/* Ghost-layer legend when comparing (stacked below heatmap legend) */}
      {compareWith && compareAccent && (
        <div
          className="absolute top-[120px] right-3 bg-surface/85 p-2 font-mono text-[9px] uppercase border z-[600] tracking-widest space-y-1 pointer-events-none"
          style={{ borderColor: compareAccent }}
        >
          <div style={{ color: compareAccent }}>GHOST_LAYER</div>
          <div className="flex items-center gap-2 text-on-surface-variant">
            <svg width="14" height="6">
              <circle
                cx="7"
                cy="3"
                r="2.5"
                fill="none"
                stroke={compareAccent}
                strokeWidth="1"
                strokeDasharray="3 2"
              />
            </svg>
            <span>{SCENARIO_LABEL_MAP[compareWith]} TOP-10</span>
          </div>
          <div className="text-[8px] text-on-surface-variant opacity-70">
            SOLID = ACTIVE · DASHED = COMPARE
          </div>
        </div>
      )}

      {/* Risk legend */}
      <div className="absolute bottom-3 left-3 bg-surface/85 border border-outline-variant p-2 font-mono text-[9px] uppercase z-[600] space-y-1 tracking-widest pointer-events-none">
        <div className="text-on-surface-variant">RISK_TIER</div>
        <div className="flex items-center gap-2">
          <span
            className="inline-block w-2 h-2"
            style={{ background: TIER_COLOR.primary }}
          />
          <span>LOW (&lt; 0.40)</span>
        </div>
        <div className="flex items-center gap-2">
          <span
            className="inline-block w-2 h-2"
            style={{ background: TIER_COLOR.secondary }}
          />
          <span>MED (0.40-0.70)</span>
        </div>
        <div className="flex items-center gap-2">
          <span
            className="inline-block w-2 h-2"
            style={{ background: TIER_COLOR.error }}
          />
          <span>HIGH (&gt; 0.70)</span>
        </div>
        {active === "ev" && (
          <div className="flex items-center gap-2 pt-1 mt-1 border-t border-outline-variant">
            <span
              className="inline-block w-2 h-2"
              style={{ background: "#ffb95f", outline: "1px solid #ffb95f" }}
            />
            <span>EV_HOTSPOT</span>
          </div>
        )}
      </div>

      {hover && hoverPos && (
        <div
          className="absolute pointer-events-none bg-surface-container-high border border-primary p-2 font-mono text-[10px] uppercase shadow-2xl z-[700]"
          style={{
            left: hoverPos.x,
            top: hoverPos.y,
            transform: "translate(12px, 12px)",
            minWidth: 200,
          }}
        >
          <div className="flex items-center gap-2 mb-1">
            <div
              className="w-1.5 h-1.5"
              style={{
                background: TIER_COLOR[riskTier(hover.metric.risk_score)],
              }}
            />
            <span className="font-bold text-primary tracking-widest">
              BUS_{hover.bus}
            </span>
          </div>
          <div className="space-y-0.5 text-[10px]">
            <p className="flex justify-between gap-4">
              <span className="text-on-surface-variant">RISK_SCORE</span>
              <span
                style={{
                  color: TIER_COLOR[riskTier(hover.metric.risk_score)],
                }}
              >
                {hover.metric.risk_score.toFixed(2)}
              </span>
            </p>
            <p className="flex justify-between gap-4">
              <span className="text-on-surface-variant">PEAK_LOAD</span>
              <span>{(hover.metric.peak_load_kw / 1000).toFixed(2)} MW</span>
            </p>
            <p className="flex justify-between gap-4">
              <span className="text-on-surface-variant">RATING</span>
              <span>
                {(hover.metric.rating_kw / 1000).toFixed(2)} MW
              </span>
            </p>
            <p className="flex justify-between gap-4">
              <span className="text-on-surface-variant">GEO</span>
              <span>
                {hover.lat.toFixed(4)}, {hover.lng.toFixed(4)}
              </span>
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
