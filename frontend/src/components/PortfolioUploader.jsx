import { useState, useRef } from "react";
import { IconFolder, IconBarChart, IconDownload, IconClose } from "./Icons";
import {
  PieChart,
  Pie,
  Cell,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

const PIE_COLORS = {
  critical: "#EF4444",
  high: "#F59E0B",
  medium: "#EAB308",
  low: "#00D4AA",
};

// High-risk cluster centers (same as backend logic)
const HIGH_CENTERS = [
  { lat: 39.6, lon: 22.4 },
  { lat: 38.6, lon: 23.6 },
  { lat: 36.2, lon: 28.0 },
  { lat: 37.5, lon: 22.3 },
];

function formatEuro(n) {
  if (n >= 1_000_000) return `€${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `€${(n / 1_000).toFixed(0)}K`;
  return `€${n}`;
}

// Custom pie chart label
function PieLabel({ cx, cy, midAngle, innerRadius, outerRadius, percent }) {
  if (percent < 0.06) return null;
  const rad = (midAngle * Math.PI) / 180;
  const r = innerRadius + (outerRadius - innerRadius) * 0.5;
  const x = cx + r * Math.cos(-rad);
  const y = cy + r * Math.sin(-rad);
  return (
    <text x={x} y={y} fill="white" textAnchor="middle" dominantBaseline="central" fontSize={10} fontWeight="bold">
      {`${(percent * 100).toFixed(0)}%`}
    </text>
  );
}

export default function PortfolioUploader({ onAssetsLoaded, open: externalOpen, onToggle }) {
  const [internalOpen, setInternalOpen] = useState(false);
  const controlled = onToggle !== undefined;
  const open = controlled ? externalOpen : internalOpen;
  const setOpen = controlled ? onToggle : setInternalOpen;
  const [dragOver, setDragOver] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [stats, setStats] = useState(null);
  const [error, setError] = useState(null);
  const [reportLoading, setReportLoading] = useState(false);
  const fileRef = useRef(null);

  function classifyAsset(asset, criticalPatches) {
    // proximity_risk from backend already; use score if available
    if (asset.score !== undefined) {
      if (asset.score >= 76) return "critical";
      if (asset.score >= 51) return "high";
      if (asset.score >= 26) return "medium";
      return "low";
    }
    // Fallback: classify by distance to known high centers
    const inHighZone = HIGH_CENTERS.some((c) => {
      const d = Math.sqrt((asset.lat - c.lat) ** 2 + (asset.lon - c.lon) ** 2) * 111;
      return d < 30;
    });
    return inHighZone ? "high" : "low";
  }

  function buildStats(assets) {
    const classified = assets.map((a) => ({
      ...a,
      riskBucket: a.proximity_risk
        ? (a.score >= 76 ? "critical" : "high")
        : (a.score >= 26 ? "medium" : "low"),
    }));

    const highRisk = classified.filter((a) => a.proximity_risk || a.riskBucket === "critical");
    const valueAtRisk = highRisk.reduce((s, a) => s + (parseFloat(a.value) || 0), 0);
    const totalValue = classified.reduce((s, a) => s + (parseFloat(a.value) || 0), 0);

    // Counts per bucket
    const counts = { critical: 0, high: 0, medium: 0, low: 0 };
    classified.forEach((a) => { counts[a.riskBucket] = (counts[a.riskBucket] || 0) + 1; });

    const pieData = Object.entries(counts)
      .filter(([, v]) => v > 0)
      .map(([name, value]) => ({ name, value }));

    const maxRisk = highRisk.reduce(
      (m, a) => (parseFloat(a.value) > parseFloat(m?.value || 0) ? a : m),
      null
    );

    return {
      total: assets.length,
      highRisk: highRisk.length,
      pct: Math.round((highRisk.length / assets.length) * 100),
      topAsset: maxRisk?.name || "N/A",
      topAssetValue: maxRisk ? formatEuro(parseFloat(maxRisk.value) || 0) : "N/A",
      valueAtRisk: formatEuro(valueAtRisk),
      totalValue: formatEuro(totalValue),
      pieData,
    };
  }

  async function processFile(file) {
    if (!file || !file.name.endsWith(".csv")) {
      setError("Please upload a CSV file with columns: name, lat, lon, value");
      return;
    }
    setUploading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append("file", file);
      const res = await fetch("/api/assets/upload", { method: "POST", body: formData });
      const data = await res.json();

      if (data.features) {
        const assets = data.features.map((f) => ({
          ...f.properties,
          lat: f.geometry.coordinates[1],
          lon: f.geometry.coordinates[0],
        }));
        setStats(buildStats(assets));
        onAssetsLoaded(assets);
      }
    } catch {
      // Client-side CSV fallback
      const text = await file.text();
      const lines = text.trim().split("\n");
      const headers = lines[0].split(",").map((h) => h.trim().toLowerCase());
      const assets = [];
      for (let i = 1; i < lines.length; i++) {
        const vals = lines[i].split(",");
        const obj = {};
        headers.forEach((h, idx) => { obj[h] = vals[idx]?.trim(); });
        if (obj.lat && obj.lon) {
          const a = {
            name: obj.name || `Asset ${i}`,
            lat: parseFloat(obj.lat),
            lon: parseFloat(obj.lon),
            value: parseFloat(obj.value) || 0,
            proximity_risk: 0,
          };
          // Proximity check
          const inHigh = HIGH_CENTERS.some((c) => {
            const d = Math.sqrt((a.lat - c.lat) ** 2 + (a.lon - c.lon) ** 2) * 111;
            return d < 30;
          });
          if (inHigh) a.proximity_risk = 1;
          assets.push(a);
        }
      }
      setStats(buildStats(assets));
      onAssetsLoaded(assets);
    } finally {
      setUploading(false);
    }
  }

  async function handleDownloadReport() {
    if (!stats) return;
    setReportLoading(true);
    try {
      const params = new URLSearchParams({
        area_name: "Portfolio Summary",
        score: stats.pct,
        summary: `Portfolio of ${stats.total} insured assets. ${stats.highRisk} assets (${stats.pct}%) are located in HIGH or CRITICAL climate risk zones, representing ${stats.valueAtRisk} of total portfolio value at risk out of ${stats.totalValue}. Highest-risk asset: ${stats.topAsset} (${stats.topAssetValue}).`,
        snapshot_id: `portfolio-${Date.now()}`,
      });
      const link = document.createElement("a");
      link.href = `/api/report/pdf?${params}`;
      link.download = "earthrisk-portfolio-report.pdf";
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    } finally {
      setReportLoading(false);
    }
  }

  function handleDrop(e) {
    e.preventDefault();
    setDragOver(false);
    processFile(e.dataTransfer.files[0]);
  }

  function downloadSampleCSV() {
    const csv = `name,lat,lon,value\nAthens Office Complex,37.9838,23.7275,2500000\nThessaloniki Warehouse,40.6401,22.9444,1800000\nRhodes Resort Hotel,36.4341,28.2176,5200000\nVolos Industrial Park,39.3617,22.9417,3100000\nPatras Commercial,38.2466,21.7346,1400000\nChalkida Bridge Zone,38.5972,23.6028,4200000\nKymi Highland,38.6500,24.1000,900000`;
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "sample-portfolio.csv";
    a.click();
  }

  return (
    <>
      {/* Floating trigger — only when not controlled by footer */}
      {!controlled && (
        <button
          onClick={() => setOpen(!open)}
          className="fixed bottom-14 right-6 z-50 flex items-center gap-2 px-4 py-3 rounded-2xl text-sm font-semibold text-white transition-all hover:scale-105 shadow-xl"
          style={{
            background: open ? "rgba(0,212,170,0.25)" : "rgba(30,41,59,0.95)",
            border: "1px solid rgba(0,212,170,0.3)",
            boxShadow: "0 0 20px rgba(0,212,170,0.15)",
            backdropFilter: "blur(12px)",
          }}
        >
          <IconFolder size={18} />
          <span>Portfolio Upload</span>
          {stats && (
            <span
              className="text-xs px-2 py-0.5 rounded-full font-bold"
              style={{ background: "rgba(0,212,170,0.2)", color: "#00D4AA" }}
            >
              {stats.total}
            </span>
          )}
        </button>
      )}

      {/* Panel */}
      {open && (
        <div
          className="fixed bottom-12 left-4 z-50 w-80 rounded-2xl overflow-hidden"
          style={{
            background: "rgba(10,15,30,0.97)",
            border: "1px solid rgba(0,212,170,0.2)",
            boxShadow: "0 25px 50px rgba(0,0,0,0.5)",
            backdropFilter: "blur(20px)",
          }}
        >
          {/* Header */}
          <div className="flex items-center justify-between px-4 py-3 border-b border-white/5">
            <div>
              <div className="text-sm font-bold text-white">Asset Portfolio</div>
              <div className="text-xs text-white/30">Upload CSV to map insured assets</div>
            </div>
            <button onClick={() => setOpen(false)} className="text-white/30 hover:text-white flex items-center"><IconClose size={16} /></button>
          </div>

          {/* Drop zone */}
          <div className="p-4">
            <div
              onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={handleDrop}
              onClick={() => fileRef.current?.click()}
              className="rounded-xl border-2 border-dashed flex flex-col items-center justify-center py-6 cursor-pointer transition-all"
              style={{
                borderColor: dragOver ? "#00D4AA" : "rgba(255,255,255,0.1)",
                background: dragOver ? "rgba(0,212,170,0.05)" : "rgba(255,255,255,0.02)",
              }}
            >
              <input ref={fileRef} type="file" accept=".csv" className="hidden" onChange={(e) => processFile(e.target.files[0])} />
              {uploading ? (
                <div className="text-[#00D4AA] text-sm animate-pulse">Processing…</div>
              ) : (
                <>
                  <span className="mb-2 opacity-40"><IconBarChart size={32} /></span>
                  <p className="text-sm text-white/60 text-center">
                    Drop CSV file here
                    <br />
                    <span className="text-white/30 text-xs">or click to browse</span>
                  </p>
                  <p className="text-xs text-white/20 mt-1">Columns: name, lat, lon, value</p>
                </>
              )}
            </div>

            {error && <div className="mt-2 text-xs text-red-400 text-center">{error}</div>}

            <button
              onClick={downloadSampleCSV}
              className="mt-2 w-full text-xs text-white/30 hover:text-[#00D4AA] transition-colors py-1.5 text-center"
            >
              Download sample CSV →
            </button>
          </div>

          {/* Portfolio intelligence card */}
          {stats && (
            <div className="mx-4 mb-4 space-y-3">
              {/* Key metrics grid */}
              <div
                className="p-3 rounded-xl"
                style={{ background: "rgba(0,212,170,0.06)", border: "1px solid rgba(0,212,170,0.15)" }}
              >
                <div className="text-xs font-semibold text-[#00D4AA] mb-2 uppercase tracking-wider">
                  Portfolio Analysis
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <div className="text-xl font-bold text-white">{stats.total}</div>
                    <div className="text-xs text-white/40">Total Assets</div>
                  </div>
                  <div>
                    <div className="text-xl font-bold text-[#F59E0B]">{stats.pct}%</div>
                    <div className="text-xs text-white/40">In High-Risk Zones</div>
                  </div>
                  <div>
                    <div className="text-sm font-bold text-[#EF4444]">{stats.valueAtRisk}</div>
                    <div className="text-xs text-white/40">Value at Risk</div>
                  </div>
                  <div>
                    <div className="text-sm font-bold text-white/70">{stats.totalValue}</div>
                    <div className="text-xs text-white/40">Total Value</div>
                  </div>
                </div>
                {stats.topAsset !== "N/A" && (
                  <div className="mt-2 pt-2 border-t border-white/5">
                    <div className="text-xs text-white/40">Highest-Risk Asset</div>
                    <div className="text-sm text-[#EF4444] font-medium truncate">{stats.topAsset}</div>
                    <div className="text-xs text-white/30">{stats.topAssetValue}</div>
                  </div>
                )}
              </div>

              {/* Risk distribution pie chart */}
              {stats.pieData.length > 0 && (
                <div
                  className="p-3 rounded-xl"
                  style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.07)" }}
                >
                  <div className="text-xs font-semibold text-white/40 mb-2 uppercase tracking-wider">
                    Risk Distribution
                  </div>
                  <div className="h-32">
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={stats.pieData}
                          cx="50%"
                          cy="50%"
                          innerRadius={30}
                          outerRadius={55}
                          dataKey="value"
                          labelLine={false}
                          label={PieLabel}
                        >
                          {stats.pieData.map((entry) => (
                            <Cell key={entry.name} fill={PIE_COLORS[entry.name] || "#888"} />
                          ))}
                        </Pie>
                        <Tooltip
                          contentStyle={{
                            background: "rgba(10,15,30,0.95)",
                            border: "1px solid rgba(0,212,170,0.3)",
                            borderRadius: "8px",
                            fontSize: "11px",
                            color: "white",
                          }}
                          formatter={(v, name) => [`${v} assets`, name.charAt(0).toUpperCase() + name.slice(1)]}
                        />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                  {/* Legend */}
                  <div className="flex flex-wrap gap-x-3 gap-y-1 mt-1">
                    {stats.pieData.map((d) => (
                      <div key={d.name} className="flex items-center gap-1">
                        <div className="w-2 h-2 rounded-full" style={{ background: PIE_COLORS[d.name] }} />
                        <span className="text-[10px] text-white/40 capitalize">{d.name} ({d.value})</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Download report button */}
              <button
                onClick={handleDownloadReport}
                disabled={reportLoading}
                className="w-full flex items-center justify-center gap-2 py-2.5 rounded-xl text-xs font-semibold transition-all hover:scale-105 active:scale-95 disabled:opacity-50"
                style={{
                  background: "rgba(255,255,255,0.07)",
                  border: "1px solid rgba(255,255,255,0.15)",
                  color: "rgba(255,255,255,0.8)",
                }}
              >
                <IconDownload size={15} />
                {reportLoading ? "Generating…" : "Download Portfolio Report"}
              </button>
            </div>
          )}
        </div>
      )}
    </>
  );
}
