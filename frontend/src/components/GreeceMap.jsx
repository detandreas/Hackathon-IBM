import { useState, useEffect, useRef, useCallback } from "react";
import DeckGL from "@deck.gl/react";
import { ScatterplotLayer, GeoJsonLayer, LineLayer } from "@deck.gl/layers";
import { Map } from "react-map-gl/maplibre";
import "maplibre-gl/dist/maplibre-gl.css";
import { greecePatches } from "../data/greeceData";

// ── Map style presets ──────────────────────────────────────────────────────────
const MAP_STYLES = {
  standard: "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
  satellite: {
    version: 8,
    sources: {
      satellite: {
        type: "raster",
        tiles: [
          "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        ],
        tileSize: 256,
        attribution: "© Esri",
      },
    },
    layers: [{ id: "satellite-tiles", type: "raster", source: "satellite" }],
  },
  heatmap: "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
};

const INITIAL_VIEW_STATE = {
  latitude: 38.5,
  longitude: 23.5,
  zoom: 5.8,
  pitch: 45,
  bearing: -10,
  transitionDuration: 800,
};

const GEO_URL =
  "https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson";

// ── Color helpers ──────────────────────────────────────────────────────────────
function scoreToRgba(score, alpha = 210) {
  if (score >= 76) return [239, 68, 68, alpha];
  if (score >= 51) return [245, 158, 11, alpha];
  if (score >= 26) return [234, 179, 8, alpha];
  return [0, 212, 170, alpha];
}
function scoreToLineRgba(score) {
  if (score >= 76) return [239, 68, 68, 255];
  if (score >= 51) return [245, 158, 11, 255];
  if (score >= 26) return [234, 179, 8, 255];
  return [0, 212, 170, 255];
}

// ── Haversine distance (km) ────────────────────────────────────────────────────
function haversineDist(lat1, lon1, lat2, lon2) {
  const R = 6371;
  const dLat = ((lat2 - lat1) * Math.PI) / 180;
  const dLon = ((lon2 - lon1) * Math.PI) / 180;
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos((lat1 * Math.PI) / 180) *
      Math.cos((lat2 * Math.PI) / 180) *
      Math.sin(dLon / 2) ** 2;
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

// ── Tooltip ───────────────────────────────────────────────────────────────────
const TIER_COLOR = {
  CRITICAL: "#EF4444",
  HIGH: "#F59E0B",
  MEDIUM: "#EAB308",
  LOW: "#00D4AA",
};

function MapTooltip({ info }) {
  if (!info || !info.object) return null;
  const patch = info.object;
  if (!patch.score && !patch.tier) return null; // asset pin without tier
  const color = TIER_COLOR[patch.tier] || "#00D4AA";
  return (
    <div
      className="pointer-events-none absolute z-50"
      style={{ left: info.x + 14, top: info.y - 36 }}
    >
      <div
        className="px-3 py-2 rounded-xl text-xs shadow-2xl"
        style={{
          background: "rgba(8, 12, 26, 0.97)",
          border: `1px solid ${color}50`,
          minWidth: 160,
          backdropFilter: "blur(12px)",
          boxShadow: `0 8px 32px rgba(0,0,0,0.6), 0 0 0 1px ${color}20`,
        }}
      >
        <div className="font-bold text-white text-sm leading-tight">{patch.name}</div>
        <div className="text-white/40 text-[10px] mt-0.5">{patch.region}</div>
        <div className="flex items-center gap-2 mt-1.5">
          <span className="font-mono font-bold text-base" style={{ color }}>
            {patch.score}
          </span>
          <span
            className="text-[10px] font-bold px-1.5 py-0.5 rounded-md uppercase"
            style={{ background: `${color}20`, color }}
          >
            {patch.tier}
          </span>
          <span className="text-white/30 text-[10px] ml-auto">
            {patch.trend === "rising" ? "↑" : patch.trend === "improving" ? "↓" : "→"}
          </span>
        </div>
      </div>
    </div>
  );
}

// ── Mode toggle button ─────────────────────────────────────────────────────────
function ModeButton({ label, active, onClick, icon }) {
  return (
    <button
      onClick={onClick}
      className="flex items-center gap-1.5 px-3 py-1.5 text-[11px] font-semibold rounded-lg transition-all"
      style={{
        background: active ? "rgba(0,212,170,0.2)" : "rgba(8,12,26,0.8)",
        color: active ? "#00D4AA" : "rgba(255,255,255,0.4)",
        border: active ? "1px solid rgba(0,212,170,0.4)" : "1px solid rgba(255,255,255,0.08)",
      }}
    >
      <span>{icon}</span>
      {label}
    </button>
  );
}

// ── Main component ─────────────────────────────────────────────────────────────
export default function GreeceMap({ onPatchClick, assetPins = [], selectedPatch }) {
  const [viewState, setViewState] = useState(INITIAL_VIEW_STATE);
  const [hoverInfo, setHoverInfo] = useState(null);
  const [greeceGeoJson, setGreeceGeoJson] = useState(null);
  const [geoLoading, setGeoLoading] = useState(true);
  const [pulseScale, setPulseScale] = useState(1);
  const [mapMode, setMapMode] = useState("standard"); // standard | heatmap | satellite
  const pulseRef = useRef(null);

  // ── Fetch Greece GeoJSON ─────────────────────────────────────────────────────
  useEffect(() => {
    fetch(GEO_URL)
      .then((r) => r.json())
      .then((data) => {
        const greece = data.features.find(
          (f) =>
            f.properties?.ISO_A3 === "GRC" ||
            f.properties?.ADMIN === "Greece" ||
            f.properties?.NAME === "Greece"
        );
        if (greece) setGreeceGeoJson(greece);
      })
      .catch(() => {})
      .finally(() => setGeoLoading(false));
  }, []);

  // ── Pulse animation ─────────────────────────────────────────────────────────
  useEffect(() => {
    let frame = 0;
    pulseRef.current = setInterval(() => {
      frame += 1;
      setPulseScale(1 + 0.35 * Math.abs(Math.sin(frame * 0.07)));
    }, 40);
    return () => clearInterval(pulseRef.current);
  }, []);

  // ── Compute asset → nearest critical patch lines ───────────────────────────
  const criticalPatches = greecePatches.filter((p) => p.score >= 76);
  const assetLines = assetPins
    .filter((a) => a.proximity_risk)
    .map((asset) => {
      let nearest = null;
      let minDist = Infinity;
      for (const cp of criticalPatches) {
        const d = haversineDist(asset.lat, asset.lon, cp.lat, cp.lon);
        if (d < minDist) {
          minDist = d;
          nearest = cp;
        }
      }
      return nearest
        ? {
            sourcePosition: [asset.lon, asset.lat],
            targetPosition: [nearest.lon, nearest.lat],
          }
        : null;
    })
    .filter(Boolean);

  // ── Layers ───────────────────────────────────────────────────────────────────
  const layers = [];

  // Greece boundary
  if (greeceGeoJson) {
    layers.push(
      new GeoJsonLayer({
        id: "greece-boundary-fill",
        data: greeceGeoJson,
        stroked: false,
        filled: true,
        getFillColor: [0, 212, 170, 18],
        pickable: false,
      }),
      new GeoJsonLayer({
        id: "greece-boundary",
        data: greeceGeoJson,
        stroked: true,
        filled: false,
        getLineColor: [0, 212, 170, 200],
        lineWidthMinPixels: 2,
        lineWidthMaxPixels: 3,
        pickable: false,
      })
    );
  }

  if (mapMode === "heatmap") {
    // ── HEATMAP mode: large blurred blobs by score ─────────────────────────
    // Outer glow layer — very large, very transparent
    layers.push(
      new ScatterplotLayer({
        id: "heat-outer",
        data: greecePatches,
        getPosition: (d) => [d.lon, d.lat],
        getRadius: 32000,
        radiusMinPixels: 20,
        radiusMaxPixels: 80,
        getFillColor: (d) => [...scoreToRgba(d.score, 0).slice(0, 3), 18],
        stroked: false,
        pickable: false,
      }),
      new ScatterplotLayer({
        id: "heat-mid",
        data: greecePatches,
        getPosition: (d) => [d.lon, d.lat],
        getRadius: 18000,
        radiusMinPixels: 12,
        radiusMaxPixels: 55,
        getFillColor: (d) => [...scoreToRgba(d.score, 0).slice(0, 3), 35],
        stroked: false,
        pickable: false,
      }),
      new ScatterplotLayer({
        id: "heat-core",
        data: greecePatches,
        getPosition: (d) => [d.lon, d.lat],
        getRadius: 8000,
        radiusMinPixels: 6,
        radiusMaxPixels: 30,
        getFillColor: (d) => scoreToRgba(d.score, 140),
        stroked: false,
        pickable: true,
        onClick: (info) => { if (info.object) onPatchClick(info.object); },
        onHover: (info) => setHoverInfo(info.object ? info : null),
      })
    );
  } else {
    // ── STANDARD / SATELLITE mode: scatter dots with pulse rings ────────────
    // Pulse rings — CRITICAL
    layers.push(
      new ScatterplotLayer({
        id: "critical-pulse",
        data: criticalPatches,
        getPosition: (d) => [d.lon, d.lat],
        getRadius: 10000 * pulseScale,
        radiusMinPixels: 6,
        radiusMaxPixels: 36,
        getFillColor: [239, 68, 68, 0],
        getLineColor: [239, 68, 68, Math.round(90 * (2 - pulseScale))],
        stroked: true,
        filled: false,
        lineWidthMinPixels: 1.5,
        pickable: false,
      })
    );

    // Pulse rings — HIGH
    const highPatches = greecePatches.filter((p) => p.score >= 51 && p.score < 76);
    layers.push(
      new ScatterplotLayer({
        id: "high-pulse",
        data: highPatches,
        getPosition: (d) => [d.lon, d.lat],
        getRadius: 9000 * (1 + 0.2 * Math.abs(Math.sin(pulseScale * 3))),
        radiusMinPixels: 5,
        radiusMaxPixels: 28,
        getFillColor: [245, 158, 11, 0],
        getLineColor: [245, 158, 11, Math.round(50 * (2 - pulseScale))],
        stroked: true,
        filled: false,
        lineWidthMinPixels: 1,
        pickable: false,
      })
    );

    // Glow halos
    layers.push(
      new ScatterplotLayer({
        id: "risk-patches-glow",
        data: greecePatches,
        getPosition: (d) => [d.lon, d.lat],
        getRadius: 12000,
        radiusMinPixels: 8,
        radiusMaxPixels: 32,
        getFillColor: (d) => [...scoreToRgba(d.score, 0).slice(0, 3), 40],
        stroked: false,
        pickable: false,
      })
    );

    // Main scatter dots
    layers.push(
      new ScatterplotLayer({
        id: "risk-patches",
        data: greecePatches,
        getPosition: (d) => [d.lon, d.lat],
        getRadius: (d) => {
          const base = 6000 + (d.score / 100) * 4000;
          return selectedPatch?.id === d.id ? base * 1.5 : base;
        },
        radiusMinPixels: 5,
        radiusMaxPixels: 22,
        getFillColor: (d) => scoreToRgba(d.score, selectedPatch?.id === d.id ? 255 : 210),
        getLineColor: (d) =>
          selectedPatch?.id === d.id ? [255, 255, 255, 255] : scoreToLineRgba(d.score),
        stroked: true,
        lineWidthMinPixels: selectedPatch ? 1.5 : 0,
        lineWidthMaxPixels: 3,
        pickable: true,
        autoHighlight: true,
        highlightColor: [255, 255, 255, 60],
        onClick: (info) => { if (info.object) onPatchClick(info.object); },
        onHover: (info) => setHoverInfo(info.object ? info : null),
        updateTriggers: {
          getRadius: [selectedPatch?.id],
          getFillColor: [selectedPatch?.id],
          getLineColor: [selectedPatch?.id],
        },
      })
    );
  }

  // Asset → critical zone connection lines
  if (assetLines.length > 0) {
    layers.push(
      new LineLayer({
        id: "asset-risk-lines",
        data: assetLines,
        getSourcePosition: (d) => d.sourcePosition,
        getTargetPosition: (d) => d.targetPosition,
        getColor: [245, 158, 11, 80],
        getWidth: 1.5,
        widthMinPixels: 1,
        pickable: false,
      })
    );
  }

  // Asset pins
  if (assetPins.length > 0) {
    const riskyAssets = assetPins.filter((a) => a.proximity_risk);
    if (riskyAssets.length > 0) {
      layers.push(
        new ScatterplotLayer({
          id: "asset-risk-halos",
          data: riskyAssets,
          getPosition: (d) => [d.lon, d.lat],
          getRadius: 9000 * (1 + 0.25 * Math.abs(Math.sin(pulseScale * 2.5))),
          radiusMinPixels: 10,
          radiusMaxPixels: 30,
          getFillColor: [245, 158, 11, 0],
          getLineColor: [245, 158, 11, 160],
          stroked: true,
          filled: false,
          lineWidthMinPixels: 2,
          pickable: false,
        })
      );
    }
    layers.push(
      new ScatterplotLayer({
        id: "asset-pins",
        data: assetPins,
        getPosition: (d) => [d.lon, d.lat],
        getRadius: 5000,
        radiusMinPixels: 5,
        radiusMaxPixels: 14,
        getFillColor: (d) =>
          d.proximity_risk ? [245, 158, 11, 230] : [255, 255, 255, 220],
        getLineColor: [20, 30, 50, 255],
        stroked: true,
        lineWidthMinPixels: 1.5,
        pickable: true,
        onHover: (info) => setHoverInfo(info.object ? info : null),
      })
    );
  }

  // ── Controls ──────────────────────────────────────────────────────────────────
  const zoomIn = useCallback(() =>
    setViewState((v) => ({ ...v, zoom: Math.min(v.zoom + 0.7, 14), transitionDuration: 300 })), []);
  const zoomOut = useCallback(() =>
    setViewState((v) => ({ ...v, zoom: Math.max(v.zoom - 0.7, 3), transitionDuration: 300 })), []);
  const resetView = useCallback(() =>
    setViewState({ ...INITIAL_VIEW_STATE, transitionDuration: 600 }), []);
  const tiltToggle = useCallback(() =>
    setViewState((v) => ({ ...v, pitch: v.pitch > 10 ? 0 : 45, transitionDuration: 600 })), []);

  const currentMapStyle =
    mapMode === "satellite" ? MAP_STYLES.satellite : MAP_STYLES.standard;

  return (
    <div className="relative w-full h-full overflow-hidden">
      <DeckGL
        viewState={viewState}
        onViewStateChange={({ viewState: vs }) => setViewState(vs)}
        controller={{ doubleClickZoom: true, touchRotate: true, dragRotate: true }}
        layers={layers}
        style={{ position: "absolute", inset: 0 }}
        getCursor={({ isHovering }) => (isHovering ? "pointer" : "grab")}
      >
        <Map
          mapStyle={currentMapStyle}
          style={{ width: "100%", height: "100%" }}
          attributionControl={false}
        />
      </DeckGL>

      {/* Hover tooltip */}
      {hoverInfo && hoverInfo.object && <MapTooltip info={hoverInfo} />}

      {/* ── Map Mode Toggle ────────────────────────────────────────────────── */}
      <div
        className="absolute top-4 left-1/2 -translate-x-1/2 flex gap-1.5 p-1 rounded-xl z-10"
        style={{
          background: "rgba(8,12,26,0.90)",
          border: "1px solid rgba(255,255,255,0.08)",
          backdropFilter: "blur(12px)",
          boxShadow: "0 4px 20px rgba(0,0,0,0.5)",
        }}
      >
        <ModeButton
          label="Standard"
          icon="◉"
          active={mapMode === "standard"}
          onClick={() => setMapMode("standard")}
        />
        <ModeButton
          label="Risk Heat"
          icon="⬡"
          active={mapMode === "heatmap"}
          onClick={() => setMapMode("heatmap")}
        />
        <ModeButton
          label="Satellite"
          icon="🛰"
          active={mapMode === "satellite"}
          onClick={() => setMapMode("satellite")}
        />
      </div>

      {/* ── Legend ────────────────────────────────────────────────────────── */}
      <div
        className="absolute bottom-16 left-4 p-3 rounded-xl text-xs z-10"
        style={{
          background: "rgba(8,12,26,0.88)",
          border: "1px solid rgba(0,212,170,0.18)",
          backdropFilter: "blur(10px)",
          boxShadow: "0 4px 24px rgba(0,0,0,0.4)",
        }}
      >
        <div className="text-white/40 font-semibold uppercase tracking-wider text-[10px] mb-2">
          Risk Level
        </div>
        {[
          { color: "#EF4444", label: "Critical  76–100" },
          { color: "#F59E0B", label: "High      51–75" },
          { color: "#EAB308", label: "Medium  26–50" },
          { color: "#00D4AA", label: "Low        0–25" },
        ].map(({ color, label }) => (
          <div key={label} className="flex items-center gap-2 mb-1.5 last:mb-0">
            <div
              className="w-2.5 h-2.5 rounded-full flex-shrink-0"
              style={{ background: color, boxShadow: `0 0 6px ${color}90` }}
            />
            <span className="text-white/55 font-mono text-[10px]">{label}</span>
          </div>
        ))}
        {assetPins.length > 0 && (
          <div className="border-t border-white/10 mt-2 pt-2 flex items-center gap-2">
            <div
              className="w-2.5 h-2.5 rounded-full flex-shrink-0"
              style={{ background: "rgba(255,255,255,0.85)", boxShadow: "0 0 4px rgba(255,255,255,0.5)" }}
            />
            <span className="text-white/55 font-mono text-[10px]">Asset Pin</span>
          </div>
        )}
      </div>

      {/* ── Zoom Controls ─────────────────────────────────────────────────── */}
      <div className="absolute right-4 bottom-16 flex flex-col gap-1.5 z-10">
        {[
          { label: "+", fn: zoomIn, title: "Zoom in" },
          { label: "−", fn: zoomOut, title: "Zoom out" },
          { label: "⌂", fn: resetView, title: "Reset view" },
          { label: "3D", fn: tiltToggle, title: "Toggle 3D pitch" },
        ].map(({ label, fn, title }) => (
          <button
            key={label}
            onClick={fn}
            title={title}
            className="w-9 h-9 rounded-xl font-bold text-white/70 hover:text-white transition-all text-sm flex items-center justify-center hover:scale-110 active:scale-95"
            style={{
              background: "rgba(8,12,26,0.88)",
              border: "1px solid rgba(255,255,255,0.1)",
              backdropFilter: "blur(10px)",
              boxShadow: "0 2px 12px rgba(0,0,0,0.4)",
            }}
          >
            {label}
          </button>
        ))}
      </div>

      {/* ── Compass ───────────────────────────────────────────────────────── */}
      <div
        className="absolute top-4 right-4 w-9 h-9 rounded-full flex items-center justify-center z-10 text-[10px] font-bold text-white/30"
        style={{
          background: "rgba(8,12,26,0.75)",
          border: "1px solid rgba(255,255,255,0.08)",
          backdropFilter: "blur(8px)",
          transform: `rotate(${viewState.bearing}deg)`,
          transition: "transform 0.1s linear",
        }}
      >
        N
      </div>

      {/* ── Pitch/zoom HUD ────────────────────────────────────────────────── */}
      <div
        className="absolute bottom-4 left-4 px-2.5 py-1.5 rounded-lg z-10 text-[10px] font-mono text-white/40"
        style={{
          background: "rgba(8,12,26,0.75)",
          border: "1px solid rgba(0,212,170,0.1)",
          backdropFilter: "blur(8px)",
        }}
      >
        <span className="text-[#00D4AA]/70">pitch</span> {Math.round(viewState.pitch)}°
        <span className="mx-1.5 text-white/20">·</span>
        <span className="text-[#00D4AA]/70">zoom</span> {viewState.zoom.toFixed(1)}
        <span className="mx-1.5 text-white/20">·</span>
        <span className="text-white/30 capitalize">{mapMode}</span>
      </div>

      {/* Attribution */}
      <div className="absolute bottom-2 right-4 text-[9px] text-white/15 z-10">
        © CARTO · © OpenStreetMap · © Esri · EarthRisk AI
      </div>

      {geoLoading && (
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 z-20 text-xs text-[#00D4AA]/60 animate-pulse pointer-events-none">
          Loading map data…
        </div>
      )}
    </div>
  );
}
