import { useState, useEffect } from "react";
import { IconCheck, IconPencil, IconClose } from "./Icons";
import { apiUrl } from "../api";

const TIER_COLORS = {
  CRITICAL: "#EF4444",
  HIGH: "#F59E0B",
  MEDIUM: "#EAB308",
  LOW: "#00D4AA",
};

function formatDate(iso) {
  const d = new Date(iso);
  return d.toLocaleDateString("en-GB", { day: "2-digit", month: "short", year: "numeric" }) +
    " " + d.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" });
}

export default function HistoryDrawer({ refreshTrigger }) {
  const [open, setOpen] = useState(false);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState(false);

  useEffect(() => {
    if (open) fetchHistory();
  }, [open, refreshTrigger]);

  async function fetchHistory() {
    setLoading(true);
    setLoadError(false);
    try {
      const res = await fetch(apiUrl("/api/history"));
      const data = await res.json();
      setHistory(data.snapshots || data || []);
    } catch {
      setHistory([]);
      setLoadError(true);
    } finally {
      setLoading(false);
    }
  }

  const firstDate = history.length > 0
    ? new Date(history[history.length - 1].created_at).toLocaleDateString("en-GB", { day: "2-digit", month: "short", year: "numeric" })
    : "N/A";

  return (
    <>
      {/* Toggle button */}
      <button
        onClick={() => setOpen(!open)}
        className="fixed bottom-14 left-1/2 -translate-x-1/2 z-50 flex items-center gap-2 px-5 py-3 rounded-2xl text-sm font-semibold text-white transition-all hover:scale-105"
        style={{
          background: open ? "rgba(0,212,170,0.2)" : "rgba(30,41,59,0.95)",
          border: "1px solid rgba(0,212,170,0.25)",
          boxShadow: "0 8px 32px rgba(0,0,0,0.4)",
          backdropFilter: "blur(12px)",
        }}
      >
        <span style={{ display: "inline-block", transform: open ? "rotate(180deg)" : "rotate(0deg)", transition: "transform 0.2s" }}>
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="18 15 12 9 6 15" /></svg>
        </span>
        <span>Risk Archive</span>
        <span
          className="text-xs px-2 py-0.5 rounded-full font-bold"
          style={{ background: "rgba(0,212,170,0.2)", color: "#00D4AA" }}
        >
          {history.length}
        </span>
      </button>

      {/* Drawer — slides up from above the footer */}
      <div
        className="fixed left-0 right-0 z-40 transition-transform duration-400 ease-out"
        style={{
          bottom: "36px",
          transform: open ? "translateY(0)" : "translateY(110%)",
          maxHeight: "55vh",
          background: "rgba(8,12,24,0.98)",
          border: "1px solid rgba(0,212,170,0.12)",
          borderBottom: "none",
          borderRadius: "20px 20px 0 0",
          boxShadow: "0 -20px 60px rgba(0,0,0,0.6)",
          backdropFilter: "blur(20px)",
        }}
      >
        {/* Handle */}
        <div className="flex items-center justify-center pt-3 pb-2">
          <div className="w-8 h-1 rounded-full bg-white/20" />
        </div>

        {/* Header */}
        <div className="flex items-center justify-between px-6 py-2 border-b border-white/5">
          <div>
            <h3 className="text-sm font-bold text-white">Risk Assessment Archive</h3>
            <p className="text-xs text-white/30 mt-0.5">
              Archiving since {firstDate} · Regulatory audit trail
            </p>
          </div>
          <div className="flex items-center gap-2">
            <div
              className="text-xs px-3 py-1 rounded-full"
              style={{ background: "rgba(0,212,170,0.1)", color: "#00D4AA", border: "1px solid rgba(0,212,170,0.2)" }}
            >
              {history.length} snapshots
            </div>
            <button
              onClick={() => setOpen(false)}
              className="text-white/30 hover:text-white leading-none flex items-center"
            >
              <IconClose size={16} />
            </button>
          </div>
        </div>

        {/* Table */}
        <div className="overflow-y-auto" style={{ maxHeight: "calc(55vh - 80px)" }}>
          {loading ? (
            <div className="flex items-center justify-center py-12 text-white/40 text-sm animate-pulse">
              Loading archive…
            </div>
          ) : loadError ? (
            <div className="flex items-center justify-center py-12 text-white/40 text-sm">
              Failed to load archive from backend.
            </div>
          ) : history.length === 0 ? (
            <div className="flex items-center justify-center py-12 text-white/40 text-sm">
              No real risk snapshots yet. Run an assessment to populate the archive.
            </div>
          ) : (
            <table className="w-full text-xs">
              <thead className="sticky top-0" style={{ background: "rgba(8,12,24,0.95)" }}>
                <tr className="text-white/30 uppercase tracking-wider">
                  <th className="text-left px-6 py-3 font-semibold">Area</th>
                  <th className="text-left px-3 py-3 font-semibold">Score</th>
                  <th className="text-left px-3 py-3 font-semibold">Tier</th>
                  <th className="text-left px-3 py-3 font-semibold hidden md:table-cell">Timestamp</th>
                  <th className="text-left px-3 py-3 font-semibold">Action</th>
                </tr>
              </thead>
              <tbody>
                {history.map((snap, i) => {
                  const tier = snap.tier || (snap.score >= 76 ? "CRITICAL" : snap.score >= 51 ? "HIGH" : snap.score >= 26 ? "MEDIUM" : "LOW");
                  const color = TIER_COLORS[tier] || "#00D4AA";
                  return (
                    <tr
                      key={snap.id || i}
                      className="border-t border-white/[0.04] hover:bg-white/[0.02] transition-colors"
                    >
                      <td className="px-6 py-3 text-white font-medium">{snap.area_name}</td>
                      <td className="px-3 py-3">
                        <span className="font-bold font-mono" style={{ color }}>{snap.score}</span>
                      </td>
                      <td className="px-3 py-3">
                        <span
                          className="px-2 py-0.5 rounded-md text-[10px] font-bold uppercase"
                          style={{ background: `${color}20`, color }}
                        >
                          {tier}
                        </span>
                      </td>
                      <td className="px-3 py-3 text-white/30 hidden md:table-cell font-mono">
                        {formatDate(snap.created_at)}
                      </td>
                      <td className="px-3 py-3">
                        {snap.action ? (
                          <span
                            className="px-2 py-0.5 rounded-md text-[10px] font-semibold uppercase"
                            style={{
                              background: snap.action === "agree" ? "rgba(0,212,170,0.1)" : "rgba(245,158,11,0.1)",
                              color: snap.action === "agree" ? "#00D4AA" : "#F59E0B",
                            }}
                          >
                            <span className="flex items-center gap-1">
                              {snap.action === "agree"
                                ? <><IconCheck size={10} /> Confirmed</>
                                : <><IconPencil size={10} /> Override</>}
                            </span>
                          </span>
                        ) : (
                          <span className="text-white/20">—</span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </>
  );
}
