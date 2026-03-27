import { useState, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import GreeceMap from "./GreeceMap";
import ScorePanel from "./ScorePanel";
import StatsBar from "./StatsBar";
import PortfolioUploader from "./PortfolioUploader";
import HistoryDrawer from "./HistoryDrawer";
import { greecePatches } from "../data/greeceData";

// Top 5 highest-scoring patches for priority queue
const TOP5 = [...greecePatches]
  .sort((a, b) => b.score - a.score)
  .slice(0, 5);

const CRITICAL_COUNT = greecePatches.filter((p) => p.score >= 76).length;

// ── Priority Queue panel ───────────────────────────────────────────────────────
function PriorityQueue({ onAssess, onClose }) {
  return (
    <div
      className="absolute top-0 right-0 bottom-0 z-30 flex flex-col"
      style={{
        width: 340,
        background: "rgba(8,12,26,0.98)",
        borderLeft: "1px solid rgba(239,68,68,0.25)",
        boxShadow: "-12px 0 40px rgba(0,0,0,0.5)",
        backdropFilter: "blur(16px)",
      }}
    >
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-white/5 flex-shrink-0">
        <div>
          <div className="text-xs text-[#EF4444]/70 font-mono uppercase tracking-widest mb-0.5">
            Priority Queue
          </div>
          <div className="text-sm font-bold text-white">Areas Needing Immediate Attention</div>
        </div>
        <button
          onClick={onClose}
          className="w-7 h-7 rounded-lg bg-white/5 hover:bg-white/10 flex items-center justify-center text-white/40 hover:text-white text-xs transition-all"
        >
          ✕
        </button>
      </div>

      {/* IBM watsonx tag */}
      <div className="px-4 py-2 border-b border-white/5 flex-shrink-0">
        <div
          className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[10px] font-bold"
          style={{ background: "rgba(15,98,254,0.12)", border: "1px solid rgba(15,98,254,0.3)", color: "#0F62FE" }}
        >
          <span>⬡</span> Ranked by XGBoost risk model · IBM watsonx
        </div>
      </div>

      {/* Zone list */}
      <div className="flex-1 overflow-y-auto p-3 space-y-2">
        {TOP5.map((patch, i) => {
          const tierColor = patch.score >= 76 ? "#EF4444" : patch.score >= 51 ? "#F59E0B" : "#00D4AA";
          const trendArrow = patch.trend === "rising" ? "↑" : patch.trend === "improving" ? "↓" : "→";
          const trendColor = patch.trend === "rising" ? "#EF4444" : patch.trend === "improving" ? "#00D4AA" : "#EAB308";
          return (
            <div
              key={patch.id}
              className="flex items-center gap-3 p-3 rounded-xl transition-all hover:bg-white/5"
              style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)" }}
            >
              {/* Rank */}
              <div
                className="w-7 h-7 rounded-lg flex items-center justify-center text-xs font-black flex-shrink-0"
                style={{ background: `${tierColor}20`, color: tierColor }}
              >
                {i + 1}
              </div>

              {/* Zone info */}
              <div className="flex-1 min-w-0">
                <div className="text-xs font-semibold text-white truncate">{patch.name}</div>
                <div className="text-[10px] text-white/35 truncate">{patch.region}</div>
              </div>

              {/* Score + trend */}
              <div className="flex items-center gap-2 flex-shrink-0">
                <span className="text-sm font-bold font-mono" style={{ color: tierColor }}>
                  {patch.score}
                </span>
                <span className="text-sm font-bold" style={{ color: trendColor }}>{trendArrow}</span>
              </div>

              {/* Assess button */}
              <button
                onClick={() => { onAssess(patch); onClose(); }}
                className="px-2.5 py-1.5 rounded-lg text-[10px] font-bold transition-all hover:scale-105 active:scale-95 flex-shrink-0"
                style={{
                  background: `${tierColor}18`,
                  border: `1px solid ${tierColor}40`,
                  color: tierColor,
                }}
              >
                Assess
              </button>
            </div>
          );
        })}
      </div>

      <div className="px-4 py-3 border-t border-white/5 flex-shrink-0">
        <div className="text-[10px] text-white/25 text-center">
          Showing top 5 of {CRITICAL_COUNT} critical zones · Updated real-time
        </div>
      </div>
    </div>
  );
}

// ── Demo hint bar ─────────────────────────────────────────────────────────────
function DemoHintBar() {
  const [open, setOpen] = useState(false);

  return (
    <div
      className="flex-shrink-0 z-20"
      style={{
        background: open ? "rgba(0,212,170,0.06)" : "rgba(8,12,26,0.85)",
        borderTop: "1px solid rgba(0,212,170,0.12)",
      }}
    >
      <button
        onClick={() => setOpen((o) => !o)}
        className="w-full flex items-center justify-between px-4 py-2 text-left transition-colors hover:bg-white/5"
      >
        <div className="flex items-center gap-2">
          <span className="text-xs">🎯</span>
          <span className="text-[11px] font-semibold text-[#00D4AA]">Demo Guide</span>
          <span className="text-[10px] text-white/30">— 90-second walkthrough for judges</span>
        </div>
        <span
          className="text-[10px] text-[#00D4AA]/60"
          style={{ display: "inline-block", transform: open ? "rotate(180deg)" : "rotate(0deg)", transition: "transform 0.2s" }}
        >▾</span>
      </button>

      {open && (
        <div className="px-4 pb-3">
          <div className="flex flex-wrap items-center gap-x-1 gap-y-1.5">
            {[
              { num: "1", text: "Click Thessaly or Evia zone on the map" },
              { num: "2", text: "View Change Detection tab → see 2021–2025 evolution" },
              { num: "3", text: "Check Predicted Risk card (next 12 months)" },
              { num: "4", text: "Export PDF Report from action buttons" },
            ].map(({ num, text }) => (
              <div key={num} className="flex items-center gap-1.5">
                <span
                  className="w-4 h-4 rounded-full flex items-center justify-center text-[9px] font-black flex-shrink-0"
                  style={{ background: "rgba(0,212,170,0.2)", color: "#00D4AA" }}
                >
                  {num}
                </span>
                <span className="text-[11px] text-white/55">{text}</span>
                {num !== "4" && <span className="text-white/20 text-xs">→</span>}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Main App Page ─────────────────────────────────────────────────────────────
export default function AppPage() {
  const navigate = useNavigate();
  const [selectedPatch, setSelectedPatch] = useState(null);
  const [panelOpen, setPanelOpen] = useState(false);
  const [assetPins, setAssetPins] = useState([]);
  const [feedbackCount, setFeedbackCount] = useState(0);
  const [priorityOpen, setPriorityOpen] = useState(false);
  const [alertDismissed, setAlertDismissed] = useState(false);

  const handlePatchClick = useCallback((patch) => {
    setSelectedPatch(patch);
    setPanelOpen(true);
    setAlertDismissed(false);
    setPriorityOpen(false);
  }, []);

  const handleClosePanel = useCallback(() => {
    setPanelOpen(false);
    setTimeout(() => setSelectedPatch(null), 500);
  }, []);

  const handleFeedbackSubmit = useCallback(() => {
    setFeedbackCount((c) => c + 1);
  }, []);

  const handleAssetsLoaded = useCallback((assets) => {
    setAssetPins(assets);
  }, []);

  // FIX 4 — Critical alert: shown when selected patch score > 80 and not dismissed
  const showCriticalAlert = selectedPatch && selectedPatch.score > 80 && !alertDismissed;

  return (
    <div className="h-screen flex flex-col bg-[#0A0F1E] overflow-hidden">
      {/* ── Top header ──────────────────────────────────────────────────────── */}
      <header
        className="flex items-center justify-between flex-shrink-0 z-20"
        style={{
          background: "rgba(10,15,30,0.95)",
          borderBottom: "1px solid rgba(0,212,170,0.12)",
          backdropFilter: "blur(12px)",
        }}
      >
        {/* Logo */}
        <div className="flex items-center gap-2.5 px-4 py-3 cursor-pointer" onClick={() => navigate("/")}>
          <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-[#00D4AA] to-[#00a882] flex items-center justify-center shadow-md shadow-[#00D4AA]/20">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
              <path d="M12 2L3 7v10l9 5 9-5V7L12 2z" stroke="white" strokeWidth="2" fill="rgba(255,255,255,0.15)" />
              <circle cx="12" cy="12" r="3" fill="white" />
            </svg>
          </div>
          <span className="text-sm font-bold text-white">
            Earth<span className="text-[#00D4AA]">Risk</span>
            <span className="text-white/30 font-light"> AI</span>
          </span>
          <div
            className="text-[10px] px-2 py-0.5 rounded-full ml-1 font-bold tracking-wider"
            style={{ background: "rgba(0,212,170,0.1)", color: "#00D4AA", border: "1px solid rgba(0,212,170,0.2)" }}
          >
            LIVE
          </div>
        </div>

        {/* Stats bar center */}
        <div className="flex-1 flex justify-center overflow-hidden">
          <StatsBar refreshTrigger={feedbackCount} />
        </div>

        {/* Right actions */}
        <div className="flex items-center gap-2 px-4">
          {/* FIX 4 — Critical zones dot indicator */}
          <div className="hidden md:flex items-center gap-1.5 text-xs text-white/30">
            <span
              className="w-2 h-2 rounded-full bg-[#EF4444] flex-shrink-0"
              style={{ boxShadow: "0 0 5px #EF4444", animation: "pulse 1.5s infinite" }}
            />
            <span className="text-[#EF4444]/70 font-semibold">{CRITICAL_COUNT}</span>
            <span>critical</span>
          </div>

          {/* FIX 5 — Top Risk Zones button */}
          <button
            onClick={() => { setPriorityOpen((o) => !o); setPanelOpen(false); }}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all hover:scale-105"
            style={{
              background: priorityOpen ? "rgba(239,68,68,0.2)" : "rgba(239,68,68,0.1)",
              border: "1px solid rgba(239,68,68,0.35)",
              color: "#EF4444",
            }}
          >
            <span>⚠</span>
            <span className="hidden sm:inline">Top Risk Zones</span>
            <span
              className="text-[10px] px-1.5 py-0.5 rounded-full font-black"
              style={{ background: "rgba(239,68,68,0.2)" }}
            >
              {CRITICAL_COUNT}
            </span>
          </button>

          <button onClick={() => navigate("/")} className="text-xs text-white/30 hover:text-white transition-colors px-3 py-1.5">
            ← Home
          </button>
        </div>
      </header>

      {/* FIX 4 — CRITICAL ALERT BANNER (score > 80) */}
      {showCriticalAlert && (
        <div
          className="flex-shrink-0 flex items-center justify-between px-5 py-2 z-20"
          style={{
            background: "linear-gradient(90deg, rgba(239,68,68,0.2) 0%, rgba(239,68,68,0.06) 100%)",
            borderBottom: "1px solid rgba(239,68,68,0.4)",
          }}
        >
          <div className="flex items-center gap-2.5">
            <span
              className="text-xs font-black text-[#EF4444] flex-shrink-0"
              style={{ filter: "drop-shadow(0 0 4px #EF4444)" }}
            >
              ⚠
            </span>
            <span className="text-xs font-semibold text-[#EF4444]">
              CRITICAL ALERT — <span className="font-black">{selectedPatch.name}</span> (score {selectedPatch.score}/100) requires immediate risk reassessment
            </span>
          </div>
          <button
            onClick={() => setAlertDismissed(true)}
            className="text-[#EF4444]/50 hover:text-[#EF4444] text-xs ml-4 transition-colors flex-shrink-0"
          >
            ✕ Dismiss
          </button>
        </div>
      )}

      {/* ── Main content area ────────────────────────────────────────────────── */}
      <div className="flex flex-1 overflow-hidden relative">
        {/* Map */}
        <div
          className="flex-1 relative transition-all duration-500"
          style={{ marginRight: (panelOpen && !priorityOpen) ? "400px" : priorityOpen ? "340px" : "0" }}
        >
          <GreeceMap
            onPatchClick={handlePatchClick}
            assetPins={assetPins}
            selectedPatch={selectedPatch}
          />

          {!selectedPatch && !priorityOpen && (
            <div
              className="absolute top-16 left-1/2 -translate-x-1/2 flex items-center gap-2 px-4 py-2 rounded-full text-xs text-white/50 pointer-events-none animate-pulse"
              style={{
                background: "rgba(10,15,30,0.7)",
                border: "1px solid rgba(255,255,255,0.08)",
                backdropFilter: "blur(8px)",
              }}
            >
              <span>👆</span>
              Click any risk zone to open assessment
            </div>
          )}
        </div>

        {/* FIX 5 — Priority Queue panel (replaces/alongside score panel) */}
        {priorityOpen && (
          <PriorityQueue
            onAssess={handlePatchClick}
            onClose={() => setPriorityOpen(false)}
          />
        )}

        {/* Score Panel */}
        {panelOpen && selectedPatch && !priorityOpen && (
          <ScorePanel
            patch={selectedPatch}
            onClose={handleClosePanel}
            onFeedbackSubmit={handleFeedbackSubmit}
          />
        )}
      </div>

      {/* FIX 6 — Demo guide hint bar */}
      <DemoHintBar />

      <PortfolioUploader onAssetsLoaded={handleAssetsLoaded} />
      <HistoryDrawer refreshTrigger={feedbackCount} />
    </div>
  );
}
