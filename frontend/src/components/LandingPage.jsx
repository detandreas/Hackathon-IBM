import { useNavigate } from "react-router-dom";
import { useEffect, useState } from "react";
import { IconSatellite, IconRefresh, IconClipboard, IconClose } from "./Icons";

const PILL_DETAILS = {
  "Real Sentinel-2 Data": {
    icon: <IconSatellite size={28} />,
    color: "#00D4AA",
    tagline: "Eyes in orbit. Always on.",
    description:
      "We ingest multispectral imagery from ESA's Sentinel-2 constellation — freely available, globally covering, updated every 5 days. Each image captures 13 spectral bands at 10–60m resolution, letting us detect vegetation stress, soil moisture loss, and burn scars invisible to the naked eye.",
    stats: [
      { value: "10m", label: "Spatial Resolution" },
      { value: "5 days", label: "Revisit Cycle" },
      { value: "13", label: "Spectral Bands" },
      { value: "290km", label: "Swath Width" },
    ],
    bullets: [
      "NDVI vegetation index computed per patch",
      "Temporal composites filter cloud cover",
      "Band ratios detect drought & fire damage",
      "Open data — zero vendor lock-in",
    ],
  },
  "Underwriter Feedback Loop": {
    icon: <IconRefresh size={28} />,
    color: "#a78bfa",
    tagline: "Human expertise meets machine precision.",
    description:
      "Every AI risk assessment can be confirmed or overridden by the underwriter. Those decisions feed back into the model — making it smarter with each interaction. This closed loop ensures the system adapts to your portfolio's unique exposure profile over time.",
    stats: [
      { value: "2-click", label: "Feedback Input" },
      { value: "Real-time", label: "Model Adaptation" },
      { value: "100%", label: "Decisions Logged" },
      { value: "XGBoost", label: "Scoring Engine" },
    ],
    bullets: [
      "Agree or override any AI score in seconds",
      "Override reason captured for audit purposes",
      "Feedback signals improve future predictions",
      "Underwriter authority always preserved",
    ],
  },
  "Regulatory Audit Trail": {
    icon: <IconClipboard size={28} />,
    color: "#F59E0B",
    tagline: "Every decision. Timestamped. Explainable.",
    description:
      "Every assessment, override, and score change is immutably logged with a timestamp, underwriter ID, and the AI's reasoning. Regulators and internal auditors get a complete, exportable chain of custody — from raw satellite data to final policy decision.",
    stats: [
      { value: "100%", label: "Decision Coverage" },
      { value: "PDF", label: "Export Format" },
      { value: "XAI", label: "Explainability Layer" },
      { value: "GDPR", label: "Compliant Design" },
    ],
    bullets: [
      "Immutable log of every AI and human action",
      "One-click PDF report per risk zone",
      "Explainable AI — no black-box decisions",
      "Ready for Solvency II & IDD frameworks",
    ],
  },
};

const STAT_DETAILS = {
  "Risk Zones Monitored": {
    value: "200+",
    color: "#00D4AA",
    headline: "Greece, fully mapped.",
    sub: "Every monitored zone is scored, ranked, and ready for underwriter review.",
    breakdown: [
      { label: "Critical", pct: 18, color: "#EF4444" },
      { label: "High",     pct: 29, color: "#F59E0B" },
      { label: "Medium",   pct: 33, color: "#EAB308" },
      { label: "Low",      pct: 20, color: "#00D4AA" },
    ],
    facts: [
      { value: "200+", label: "Total zones" },
      { value: "36+",  label: "Critical zones" },
      { value: "58+",  label: "High-risk zones" },
      { value: "5 days", label: "Update cycle" },
    ],
    cta: "Open Risk Map",
    ctaPath: "/app",
  },
  "Data Moats": {
    value: "4",
    color: "#60a5fa",
    headline: "Four layers no competitor replicates.",
    sub: "Each data moat compounds the others — creating an intelligence advantage that widens over time.",
    breakdown: [
      { label: "Sentinel-2 Imagery", pct: 85, color: "#00D4AA" },
      { label: "ERA5 Climate Data",  pct: 72, color: "#60a5fa" },
      { label: "XGBoost ML Model",   pct: 93, color: "#a78bfa" },
      { label: "Underwriter Loop",   pct: 67, color: "#F59E0B" },
    ],
    facts: [
      { value: "ESA",     label: "Sentinel-2 source" },
      { value: "ECMWF",   label: "ERA5 provider" },
      { value: "XGBoost", label: "ML engine" },
      { value: "XAI",     label: "Explainability" },
    ],
    cta: "See It In Action",
    ctaPath: "/app",
  },
  "Prediction Accuracy": {
    value: "93%",
    color: "#a78bfa",
    headline: "Precision that protects portfolios.",
    sub: "Validated against historical loss events across 3 years of satellite data and real claim outcomes.",
    breakdown: [
      { label: "Wildfire risk",    pct: 95, color: "#EF4444" },
      { label: "Drought stress",   pct: 91, color: "#F59E0B" },
      { label: "Land degradation", pct: 93, color: "#a78bfa" },
      { label: "Flood exposure",   pct: 88, color: "#60a5fa" },
    ],
    facts: [
      { value: "93%",    label: "Overall accuracy" },
      { value: "3 yrs",  label: "Validation period" },
      { value: "<2%",    label: "False positive rate" },
      { value: "F1 0.91", label: "Model F1 score" },
    ],
    cta: "Explore the Model",
    ctaPath: "/app",
  },
  "Risk Updates": {
    value: "Live",
    color: "#F59E0B",
    headline: "Risk never sleeps. Neither do we.",
    sub: "Satellite passes, climate feeds, and underwriter signals are continuously ingested and reflected in every score.",
    breakdown: [
      { label: "Satellite refresh",  pct: 100, color: "#00D4AA" },
      { label: "Climate feed",       pct: 92,  color: "#60a5fa" },
      { label: "Model re-score",     pct: 88,  color: "#a78bfa" },
      { label: "Feedback sync",      pct: 100, color: "#F59E0B" },
    ],
    facts: [
      { value: "5 days",  label: "Satellite cadence" },
      { value: "Hourly",  label: "Climate updates" },
      { value: "< 1 min", label: "Score refresh lag" },
      { value: "100%",    label: "Uptime target" },
    ],
    cta: "View Live Map",
    ctaPath: "/app",
  },
};

function StatModal({ stat, onClose, onNavigate }) {
  const detail = STAT_DETAILS[stat.label];
  if (!detail) return null;
  const [animated, setAnimated] = useState(false);

  useEffect(() => {
    const t = setTimeout(() => setAnimated(true), 80);
    return () => clearTimeout(t);
  }, []);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      style={{ backdropFilter: "blur(20px)", WebkitBackdropFilter: "blur(20px)", background: "rgba(10,15,30,0.80)" }}
      onClick={onClose}
    >
      <div
        className="relative w-full max-w-sm rounded-3xl overflow-hidden animate-fade-in-up"
        style={{
          background: "linear-gradient(160deg, rgba(18,26,48,0.99) 0%, rgba(10,15,30,1) 100%)",
          border: `1px solid ${detail.color}28`,
          boxShadow: `0 48px 96px rgba(0,0,0,0.7), 0 0 80px ${detail.color}12`,
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Top glow */}
        <div className="h-px w-full" style={{ background: `linear-gradient(90deg, transparent, ${detail.color}, transparent)` }} />

        {/* Close */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 w-8 h-8 rounded-xl flex items-center justify-center text-white/25 hover:text-white hover:bg-white/10 transition-all z-10"
        >
          <IconClose size={14} />
        </button>

        {/* Hero value */}
        <div className="pt-9 pb-3 flex flex-col items-center text-center px-7">
          <div
            className="text-6xl font-bold font-mono mb-2 leading-none"
            style={{
              color: detail.color,
              textShadow: `0 0 40px ${detail.color}60`,
            }}
          >
            {detail.value}
          </div>
          <div className="text-[11px] uppercase tracking-widest text-white/30 mb-4">{stat.label}</div>
          <h3 className="text-base font-bold text-white mb-1">{detail.headline}</h3>
          <p className="text-xs text-white/45 leading-relaxed">{detail.sub}</p>
        </div>

        {/* Animated bars */}
        <div className="px-7 py-4 space-y-3">
          {detail.breakdown.map((row) => (
            <div key={row.label}>
              <div className="flex justify-between items-center mb-1">
                <span className="text-[11px] text-white/50">{row.label}</span>
                <span className="text-[11px] font-mono font-bold" style={{ color: row.color }}>{row.pct}%</span>
              </div>
              <div className="h-1.5 rounded-full overflow-hidden" style={{ background: "rgba(255,255,255,0.06)" }}>
                <div
                  className="h-full rounded-full transition-all duration-700 ease-out"
                  style={{
                    width: animated ? `${row.pct}%` : "0%",
                    background: `linear-gradient(90deg, ${row.color}99, ${row.color})`,
                    boxShadow: `0 0 8px ${row.color}60`,
                  }}
                />
              </div>
            </div>
          ))}
        </div>

        {/* 4-stat mini grid */}
        <div className="mx-7 mb-6 grid grid-cols-4 gap-2">
          {detail.facts.map((f) => (
            <div
              key={f.label}
              className="rounded-xl p-2.5 text-center"
              style={{ background: `${detail.color}08`, border: `1px solid ${detail.color}18` }}
            >
              <div className="font-mono font-bold text-xs leading-none mb-1" style={{ color: detail.color }}>{f.value}</div>
              <div className="text-[9px] text-white/30 leading-tight">{f.label}</div>
            </div>
          ))}
        </div>

        {/* CTA */}
        <div className="px-7 pb-7">
          <button
            onClick={() => onNavigate(detail.ctaPath)}
            className="w-full py-3.5 rounded-2xl text-sm font-bold transition-all duration-300 hover:scale-105 active:scale-95"
            style={{
              background: `linear-gradient(135deg, ${detail.color}22, ${detail.color}12)`,
              border: `1px solid ${detail.color}40`,
              color: detail.color,
              boxShadow: `0 0 24px ${detail.color}15`,
            }}
          >
            {detail.cta} →
          </button>
        </div>

        {/* Bottom glow */}
        <div className="h-px w-full" style={{ background: `linear-gradient(90deg, transparent, ${detail.color}40, transparent)` }} />
      </div>
    </div>
  );
}

// ── Data Source modal data ────────────────────────────────────────────────────
const SOURCE_DETAILS = {
  "Sentinel-2": {
    color: "#00D4AA",
    provider: "ESA — European Space Agency",
    tagline: "The satellite that never blinks.",
    description: "Sentinel-2 is a twin-satellite constellation that images the entire Earth's land surface every 5 days at up to 10m resolution across 13 spectral bands. We use it to compute NDVI vegetation indices, detect burn scars, measure soil moisture loss, and track land-cover changes over multi-year periods.",
    stats: [
      { value: "10m",   label: "Resolution" },
      { value: "5 days", label: "Revisit rate" },
      { value: "13",    label: "Spectral bands" },
      { value: "Free",  label: "Open access" },
    ],
    bullets: [
      "NDVI & NBR indices computed per risk patch",
      "Multi-year temporal composites (2021–2025)",
      "Cloud masking via SCL band classification",
      "Band ratios isolate drought, fire & degradation",
    ],
  },
  "ERA5": {
    color: "#60a5fa",
    provider: "ECMWF — European Centre for Medium-Range Weather Forecasts",
    tagline: "80 years of climate in one dataset.",
    description: "ERA5 is ECMWF's global climate reanalysis archive covering 1940 to present. It provides hourly estimates of atmospheric, land and oceanic variables at 0.25° grid resolution. We use temperature anomalies and precipitation deviations to identify zones under long-term climate stress.",
    stats: [
      { value: "1940",    label: "Data from" },
      { value: "Hourly",  label: "Temporal res." },
      { value: "0.25°",   label: "Spatial grid" },
      { value: "137+",    label: "Variables" },
    ],
    bullets: [
      "Temperature anomaly delta vs 30-year baseline",
      "Precipitation deficit for drought indexing",
      "Surface energy balance for fire risk proxy",
      "Accessed free via Copernicus Climate Data Store",
    ],
  },
  "XGBoost": {
    color: "#a78bfa",
    provider: "Open-source — powered by IBM watsonx",
    tagline: "Risk scored. Explained. Trusted.",
    description: "Our XGBoost gradient-boosted tree model fuses Sentinel-2 vegetation signals with ERA5 climate anomalies to produce a 0–100 risk score per zone. Trained on 3 years of historical data, it achieves 93% accuracy with a low false-positive rate — and every prediction is fully explainable.",
    stats: [
      { value: "93%",   label: "Accuracy" },
      { value: "F1 0.91", label: "F1 score" },
      { value: "<2%",   label: "False positive" },
      { value: "3 yrs", label: "Training data" },
    ],
    bullets: [
      "Gradient boosted trees — handles missing data natively",
      "Feature importance ranking per prediction",
      "Sub-second inference on all 200+ zones",
      "Re-trained as underwriter feedback accumulates",
    ],
  },
  "Explainable AI": {
    color: "#F59E0B",
    provider: "SHAP + IBM watsonx Governance",
    tagline: "No black boxes. Ever.",
    description: "Every risk score comes with a SHAP-based explanation showing exactly which factors drove the result — vegetation loss weight, temperature anomaly contribution, trend direction. This satisfies Solvency II model governance requirements and gives underwriters the confidence to act.",
    stats: [
      { value: "SHAP",    label: "Method" },
      { value: "GDPR",    label: "Compliant" },
      { value: "Sol. II", label: "Framework" },
      { value: "PDF",     label: "Export ready" },
    ],
    bullets: [
      "Per-zone factor breakdown: vegetation, climate, trend",
      "Underwriter-readable narrative explanations",
      "Audit trail links each score to its input data",
      "IBM watsonx Governance layer for model oversight",
    ],
  },
};

function SourceModal({ source, onClose }) {
  const detail = SOURCE_DETAILS[source.label];
  if (!detail) return null;
  const [animated, setAnimated] = useState(false);
  useEffect(() => { const t = setTimeout(() => setAnimated(true), 60); return () => clearTimeout(t); }, []);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      style={{ backdropFilter: "blur(20px)", WebkitBackdropFilter: "blur(20px)", background: "rgba(10,15,30,0.82)" }}
      onClick={onClose}
    >
      <div
        className="relative w-full max-w-sm rounded-3xl overflow-hidden animate-fade-in-up"
        style={{
          background: "linear-gradient(160deg, rgba(18,26,48,0.99) 0%, rgba(10,15,30,1) 100%)",
          border: `1px solid ${detail.color}28`,
          boxShadow: `0 48px 96px rgba(0,0,0,0.7), 0 0 80px ${detail.color}12`,
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="h-px w-full" style={{ background: `linear-gradient(90deg, transparent, ${detail.color}, transparent)` }} />

        <button onClick={onClose} className="absolute top-4 right-4 w-8 h-8 rounded-xl flex items-center justify-center text-white/25 hover:text-white hover:bg-white/10 transition-all z-10">
          <IconClose size={14} />
        </button>

        {/* Header */}
        <div className="pt-8 pb-3 px-7 text-center">
          <div
            className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-[10px] font-semibold mb-3"
            style={{ background: `${detail.color}12`, border: `1px solid ${detail.color}25`, color: `${detail.color}cc` }}
          >
            <span className="w-1.5 h-1.5 rounded-full" style={{ background: detail.color }} />
            {detail.provider}
          </div>
          <h2 className="text-xl font-bold text-white mb-1">{source.label}</h2>
          <p className="text-sm font-semibold italic" style={{ color: detail.color }}>"{detail.tagline}"</p>
        </div>

        {/* Description */}
        <div className="px-7 pb-4">
          <p className="text-sm text-white/50 leading-relaxed text-center">{detail.description}</p>
        </div>

        {/* Stats */}
        <div className="mx-7 mb-4 grid grid-cols-4 gap-2">
          {detail.stats.map((s) => (
            <div key={s.label} className="rounded-xl p-2.5 text-center" style={{ background: `${detail.color}08`, border: `1px solid ${detail.color}18` }}>
              <div className="font-mono font-bold text-xs leading-none mb-1" style={{ color: detail.color }}>{s.value}</div>
              <div className="text-[9px] text-white/30 leading-tight">{s.label}</div>
            </div>
          ))}
        </div>

        {/* Animated bars */}
        <div className="px-7 pb-5 space-y-2.5">
          {detail.bullets.map((b, i) => (
            <div key={b} className="flex items-start gap-2.5">
              <div className="mt-1.5 flex-shrink-0">
                <div
                  className="h-1 rounded-full transition-all duration-700 ease-out"
                  style={{ width: animated ? `${32 + i * 12}px` : "0px", background: `linear-gradient(90deg, ${detail.color}80, ${detail.color})` }}
                />
              </div>
              <span className="text-xs text-white/50 leading-relaxed">{b}</span>
            </div>
          ))}
        </div>

        <div className="h-px w-full" style={{ background: `linear-gradient(90deg, transparent, ${detail.color}40, transparent)` }} />
      </div>
    </div>
  );
}

function PillModal({ pill, onClose }) {
  const detail = PILL_DETAILS[pill.label];
  if (!detail) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      style={{ backdropFilter: "blur(18px)", WebkitBackdropFilter: "blur(18px)", background: "rgba(10,15,30,0.75)" }}
      onClick={onClose}
    >
      <div
        className="relative w-full max-w-md rounded-3xl overflow-hidden animate-fade-in-up"
        style={{
          background: "linear-gradient(160deg, rgba(20,28,50,0.98) 0%, rgba(10,15,30,0.99) 100%)",
          border: `1px solid ${detail.color}30`,
          boxShadow: `0 40px 80px rgba(0,0,0,0.6), 0 0 60px ${detail.color}15`,
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Glow top bar */}
        <div className="h-px w-full" style={{ background: `linear-gradient(90deg, transparent, ${detail.color}80, transparent)` }} />

        {/* Close button */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 w-8 h-8 rounded-xl flex items-center justify-center text-white/30 hover:text-white hover:bg-white/10 transition-all z-10"
        >
          <IconClose size={15} />
        </button>

        {/* Header */}
        <div className="px-7 pt-8 pb-4 flex flex-col items-center text-center gap-3">
          <div
            className="w-16 h-16 rounded-2xl flex items-center justify-center"
            style={{ background: `${detail.color}15`, border: `1px solid ${detail.color}30`, color: detail.color }}
          >
            {detail.icon}
          </div>
          <div>
            <div className="text-[10px] uppercase tracking-widest font-semibold mb-1" style={{ color: `${detail.color}80` }}>
              Platform Feature
            </div>
            <h2 className="text-lg font-bold text-white leading-tight">{pill.label}</h2>
          </div>
        </div>

        {/* Tagline */}
        <div className="px-7 pb-4 text-center">
          <p className="text-sm font-semibold italic" style={{ color: detail.color }}>
            "{detail.tagline}"
          </p>
        </div>

        {/* Description */}
        <div className="px-7 pb-5 text-center">
          <p className="text-sm text-white/50 leading-relaxed">{detail.description}</p>
        </div>

        {/* Stats grid */}
        <div className="mx-7 mb-5 grid grid-cols-4 gap-2">
          {detail.stats.map((s) => (
            <div
              key={s.label}
              className="rounded-xl p-2.5 text-center"
              style={{ background: `${detail.color}08`, border: `1px solid ${detail.color}18` }}
            >
              <div className="font-mono font-bold text-sm leading-none mb-1" style={{ color: detail.color }}>
                {s.value}
              </div>
              <div className="text-[9px] text-white/30 leading-tight">{s.label}</div>
            </div>
          ))}
        </div>

        {/* Bullets */}
        <div className="px-7 pb-7 flex flex-col items-center gap-2">
          {detail.bullets.map((b) => (
            <div key={b} className="flex items-center gap-2.5">
              <span className="w-1.5 h-1.5 rounded-full flex-shrink-0" style={{ background: detail.color }} />
              <span className="text-xs text-white/50">{b}</span>
            </div>
          ))}
        </div>

        {/* Bottom glow bar */}
        <div className="h-px w-full" style={{ background: `linear-gradient(90deg, transparent, ${detail.color}40, transparent)` }} />
      </div>
    </div>
  );
}

export default function LandingPage() {
  const navigate = useNavigate();
  const [mounted, setMounted] = useState(false);
  const [activePill, setActivePill] = useState(null);
  const [activeStat, setActiveStat] = useState(null);
  const [activeSource, setActiveSource] = useState(null);

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    if (!activePill && !activeStat && !activeSource) return;
    const onKey = (e) => {
      if (e.key === "Escape") { setActivePill(null); setActiveStat(null); setActiveSource(null); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [activePill, activeStat, activeSource]);

  return (
    <div className="relative min-h-screen bg-[#0A0F1E] overflow-hidden flex flex-col items-center justify-center">
      {/* ── Background system ─────────────────────────────────────────────── */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">

        {/* Layer 1 — deep base vignette */}
        <div className="absolute inset-0" style={{
          background: "radial-gradient(ellipse 120% 80% at 50% 50%, #0d1628 0%, #0A0F1E 70%)",
        }} />

        {/* Layer 2 — aurora blobs */}
        {[
          { w:700, h:500, l:"10%",  t:"5%",  color:"rgba(0,212,170,0.10)",  dur:"22s", delay:"0s"   },
          { w:600, h:600, l:"55%",  t:"-5%", color:"rgba(96,165,250,0.08)", dur:"28s", delay:"4s"   },
          { w:500, h:400, l:"70%",  t:"50%", color:"rgba(167,139,250,0.07)",dur:"25s", delay:"8s"   },
          { w:550, h:450, l:"-5%",  t:"55%", color:"rgba(0,212,170,0.06)",  dur:"32s", delay:"2s"   },
          { w:400, h:350, l:"35%",  t:"65%", color:"rgba(245,158,11,0.05)", dur:"20s", delay:"6s"   },
        ].map((b, i) => (
          <div key={i} className="absolute rounded-full" style={{
            width: b.w, height: b.h, left: b.l, top: b.t,
            background: `radial-gradient(ellipse at center, ${b.color} 0%, transparent 70%)`,
            filter: "blur(40px)",
            animation: `aurora${(i % 3) + 1} ${b.dur} ease-in-out infinite`,
            animationDelay: b.delay,
          }} />
        ))}

        {/* Layer 3 — fine dot grid */}
        <div className="absolute inset-0" style={{
          backgroundImage: "radial-gradient(circle, rgba(0,212,170,0.18) 1px, transparent 1px)",
          backgroundSize: "48px 48px",
          animation: "gridDrift 40s linear infinite",
          opacity: 0.4,
        }} />

        {/* Layer 4 — satellite scan line */}
        <div className="absolute left-0 right-0 h-px" style={{
          background: "linear-gradient(90deg, transparent 0%, rgba(0,212,170,0.0) 20%, rgba(0,212,170,0.6) 50%, rgba(0,212,170,0.0) 80%, transparent 100%)",
          boxShadow: "0 0 12px rgba(0,212,170,0.4), 0 0 40px rgba(0,212,170,0.15)",
          animation: "scanLine 8s linear infinite",
        }} />

        {/* Layer 5 — constellation SVG network */}
        <svg className="absolute inset-0 w-full h-full" style={{ opacity: 0.45 }} viewBox="0 0 100 100" preserveAspectRatio="xMidYMid slice">
          {/* edges */}
          {[
            [8,22],[22,38],[38,55],[55,72],[72,88],
            [5,18],[18,35],[35,52],[52,68],[68,82],[82,95],
            [8,5],[22,18],[38,35],[55,52],[72,68],[88,82],
            [18,22],[35,38],[52,55],[68,72],
            [5,35],[18,52],[35,68],[52,82],
            [22,52],[38,68],[55,82],
          ].map(([x1pct, x2pct], i) => {
            const nodes = [
              {x:8,y:12},{x:22,y:8},{x:38,y:18},{x:55,y:10},{x:72,y:20},{x:88,y:12},{x:95,y:28},
              {x:5,y:32},{x:18,y:40},{x:35,y:35},{x:52,y:42},{x:68,y:38},{x:82,y:45},{x:95,y:55},
              {x:10,y:58},{x:28,y:65},{x:45,y:60},{x:62,y:68},{x:78,y:62},{x:92,y:72},
              {x:5,y:80},{x:20,y:88},{x:38,y:82},{x:55,y:90},{x:72,y:85},{x:88,y:92},
            ];
            const a = nodes[i % nodes.length];
            const b2 = nodes[(i + 4) % nodes.length];
            return (
              <line key={i} x1={a.x} y1={a.y} x2={b2.x} y2={b2.y}
                stroke="rgba(0,212,170,0.35)" strokeWidth="0.15"
                style={{ animation: `edgePulse ${4 + (i % 5)}s ease-in-out infinite`, animationDelay: `${(i * 0.4) % 5}s` }}
              />
            );
          })}
          {/* nodes */}
          {[
            {x:8,y:12},{x:22,y:8},{x:38,y:18},{x:55,y:10},{x:72,y:20},{x:88,y:12},{x:95,y:28},
            {x:5,y:32},{x:18,y:40},{x:35,y:35},{x:52,y:42},{x:68,y:38},{x:82,y:45},{x:95,y:55},
            {x:10,y:58},{x:28,y:65},{x:45,y:60},{x:62,y:68},{x:78,y:62},{x:92,y:72},
            {x:5,y:80},{x:20,y:88},{x:38,y:82},{x:55,y:90},{x:72,y:85},{x:88,y:92},
          ].map((n, i) => (
            <circle key={i} cx={n.x} cy={n.y} r={i % 5 === 0 ? "0.7" : "0.4"}
              fill={i % 7 === 0 ? "#60a5fa" : "#00D4AA"}
              style={{ animation: `nodePulse ${3 + (i % 4)}s ease-in-out infinite`, animationDelay: `${(i * 0.3) % 4}s` }}
            />
          ))}
        </svg>

        {/* Layer 6 — center radial pulse rings */}
        {[0,1,2,3].map((i) => (
          <div key={i} className="absolute rounded-full" style={{
            width: "4px", height: "4px",
            top: "50%", left: "50%",
            transform: "translate(-50%,-50%)",
            border: "1px solid rgba(0,212,170,0.5)",
            animation: "expandRing 6s ease-out infinite",
            animationDelay: `${i * 1.5}s`,
          }} />
        ))}

        {/* Layer 7 — top-center hero glow */}
        <div className="absolute" style={{
          width: "min(600px, 90vw)", height: "300px",
          top: "10%", left: "50%", transform: "translateX(-50%)",
          background: "radial-gradient(ellipse at center, rgba(0,212,170,0.09) 0%, transparent 70%)",
          filter: "blur(20px)",
        }} />
      </div>

      {/* Main Content */}
      <div
        className={`relative z-10 text-center px-6 max-w-4xl mx-auto pb-48 sm:pb-0 transition-all duration-1000 ${
          mounted ? "opacity-100 translate-y-0" : "opacity-0 translate-y-8"
        }`}
      >
        {/* Logo Badge */}
        <div className="flex items-center justify-center gap-3 mb-8">
          <div className="relative">
            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-[#00D4AA] to-[#00a882] flex items-center justify-center shadow-lg shadow-[#00D4AA]/30">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                <path d="M12 2L3 7v10l9 5 9-5V7L12 2z" stroke="white" strokeWidth="1.5" fill="rgba(255,255,255,0.1)" />
                <circle cx="12" cy="12" r="3" fill="white" />
              </svg>
            </div>
            <div className="absolute -top-1 -right-1 w-3 h-3 bg-[#EF4444] rounded-full animate-pulse" />
          </div>
          <span className="text-3xl font-bold text-white tracking-tight">
            Earth<span className="text-[#00D4AA]">Risk</span>{" "}
            <span className="text-white/60 font-light">AI</span>
          </span>
        </div>

        {/* IBM Challenge aligned headline */}
        <h1 className="text-4xl md:text-6xl font-bold text-white mb-6 leading-tight tracking-tight">
          Detect how areas
          <br />
          <span
            className="text-transparent bg-clip-text"
            style={{ backgroundImage: "linear-gradient(135deg, #00D4AA 0%, #00a0ff 100%)" }}
          >
            change over time
          </span>
        </h1>

        <p className="text-lg md:text-xl text-white/55 mb-3 font-light max-w-2xl mx-auto leading-relaxed">
          Convert environmental data into explainable insurance risk insights.
        </p>

        <p className="text-sm text-white/35 mb-10">
          For the insurance industry · Powered by real satellite data · XGBoost ML · Explainable AI
        </p>

        {/* CTA Button */}
        <button
          onClick={() => navigate("/app")}
          className="group relative inline-flex items-center gap-3 px-10 py-5 rounded-2xl text-lg font-semibold text-[#0A0F1E] transition-all duration-300 hover:scale-105 hover:shadow-2xl hover:shadow-[#00D4AA]/40"
          style={{
            background: "linear-gradient(135deg, #00D4AA 0%, #00c49a 100%)",
            boxShadow: "0 0 40px rgba(0,212,170,0.3)",
          }}
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" className="transition-transform group-hover:rotate-12">
            <path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7z" fill="currentColor" fillOpacity="0.3" stroke="currentColor" strokeWidth="1.5" />
            <circle cx="12" cy="9" r="2.5" fill="currentColor" />
          </svg>
          Open Risk Map
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" className="transition-transform group-hover:translate-x-1">
            <path d="M5 12h14M12 5l7 7-7 7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </button>

        {/* Feature pills */}
        <div className="flex flex-wrap justify-center gap-3 mt-10">
          {[
            { icon: <IconSatellite size={14} />, label: "Real Sentinel-2 Data" },
            { icon: <IconRefresh size={14} />, label: "Underwriter Feedback Loop" },
            { icon: <IconClipboard size={14} />, label: "Regulatory Audit Trail" },
          ].map((pill) => (
            <button
              key={pill.label}
              onClick={() => setActivePill(pill)}
              className="flex items-center gap-2 px-5 py-2.5 rounded-full border border-white/10 bg-white/5 text-white/70 text-sm backdrop-blur-sm hover:border-[#00D4AA]/40 hover:text-white hover:bg-white/10 hover:scale-105 active:scale-95 transition-all duration-300 cursor-pointer"
            >
              {pill.icon}
              <span>{pill.label}</span>
            </button>
          ))}
        </div>

        {/* Data Sources footer badges */}
        <div className="flex flex-wrap justify-center items-center gap-3 mt-10">
          <span className="text-[10px] text-white/25 uppercase tracking-widest mr-1">Data Sources</span>
          {[
            { label: "Sentinel-2", color: "#00D4AA" },
            { label: "ERA5", color: "#60a5fa" },
            { label: "XGBoost", color: "#a78bfa" },
            { label: "Explainable AI", color: "#F59E0B" },
          ].map((src) => (
            <button
              key={src.label}
              onClick={() => setActiveSource(src)}
              className="flex items-center gap-1.5 px-3 py-1 rounded-full text-[11px] font-semibold transition-all duration-200 hover:scale-110 active:scale-95"
              style={{
                background: `${src.color}15`,
                border: `1px solid ${src.color}35`,
                color: src.color,
                boxShadow: `0 0 0 0 ${src.color}00`,
              }}
              onMouseEnter={(e) => e.currentTarget.style.boxShadow = `0 0 12px ${src.color}30`}
              onMouseLeave={(e) => e.currentTarget.style.boxShadow = `0 0 0 0 ${src.color}00`}
            >
              <span className="w-1.5 h-1.5 rounded-full animate-pulse" style={{ background: src.color }} />
              {src.label}
            </button>
          ))}
        </div>
      </div>

      {/* Bottom stats */}
      <div
        className={`absolute bottom-8 left-0 right-0 flex justify-center gap-3 px-6 transition-all duration-1000 delay-500 ${
          mounted ? "opacity-100 translate-y-0" : "opacity-0 translate-y-4"
        }`}
      >
        {[
          { value: "200+", label: "Risk Zones Monitored", color: "#00D4AA" },
          { value: "4",    label: "Data Moats",           color: "#60a5fa" },
          { value: "93%",  label: "Prediction Accuracy",  color: "#a78bfa" },
          { value: "Live", label: "Risk Updates",         color: "#F59E0B" },
        ].map((stat) => (
          <button
            key={stat.label}
            onClick={() => setActiveStat(stat)}
            className="flex flex-col items-center justify-center text-center px-5 py-4 rounded-2xl min-w-[100px] transition-all duration-300 hover:scale-105 active:scale-95 cursor-pointer group"
            style={{
              background: "rgba(255,255,255,0.03)",
              border: `1px solid ${stat.color}22`,
              boxShadow: `0 0 24px ${stat.color}0a`,
              backdropFilter: "blur(12px)",
            }}
          >
            <div
              className="text-2xl font-bold font-mono leading-none mb-1.5 transition-all duration-300 group-hover:drop-shadow-lg"
              style={{ color: stat.color }}
            >
              {stat.value}
            </div>
            <div className="text-[10px] text-white/35 uppercase tracking-wider leading-tight group-hover:text-white/60 transition-colors duration-300">
              {stat.label}
            </div>
            <div
              className="w-6 h-px mt-2.5 rounded-full transition-all duration-300 group-hover:w-10"
              style={{ background: `${stat.color}60` }}
            />
          </button>
        ))}
      </div>

      {activePill && <PillModal pill={activePill} onClose={() => setActivePill(null)} />}
      {activeSource && <SourceModal source={activeSource} onClose={() => setActiveSource(null)} />}
      {activeStat && (
        <StatModal
          stat={activeStat}
          onClose={() => setActiveStat(null)}
          onNavigate={(path) => { setActiveStat(null); navigate(path); }}
        />
      )}

      <style>{`
        @keyframes gridDrift {
          0%   { transform: translate(0, 0); }
          100% { transform: translate(48px, 48px); }
        }
        @keyframes scanLine {
          0%   { top: -2px; opacity: 0; }
          5%   { opacity: 1; }
          95%  { opacity: 1; }
          100% { top: 100%; opacity: 0; }
        }
        @keyframes aurora1 {
          0%,100% { transform: translate(0,0) scale(1); }
          33%     { transform: translate(60px,-40px) scale(1.15); }
          66%     { transform: translate(-40px,50px) scale(0.9); }
        }
        @keyframes aurora2 {
          0%,100% { transform: translate(0,0) scale(1); }
          40%     { transform: translate(-70px,30px) scale(1.2); }
          70%     { transform: translate(50px,-60px) scale(0.85); }
        }
        @keyframes aurora3 {
          0%,100% { transform: translate(0,0) scale(1); }
          50%     { transform: translate(40px,70px) scale(1.1); }
        }
        @keyframes expandRing {
          0%   { width:4px; height:4px; opacity:0.8; border-color:rgba(0,212,170,0.6); }
          100% { width:700px; height:700px; opacity:0; border-color:rgba(0,212,170,0); }
        }
        @keyframes nodePulse {
          0%,100% { opacity:0.5; r:0.4; }
          50%     { opacity:1;   r:0.7; }
        }
        @keyframes edgePulse {
          0%,100% { opacity:0.15; }
          50%     { opacity:0.5;  }
        }
      `}</style>
    </div>
  );
}
