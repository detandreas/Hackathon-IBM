import { useEffect, useState } from "react";

const DEFAULT_STATS = {
  total_snapshots: 200,
  avg_score: 49,
  critical_count: 30,
  feedback_count: 0,
};

export default function StatsBar({ refreshTrigger }) {
  const [stats, setStats] = useState(DEFAULT_STATS);
  const [pulse, setPulse] = useState(false);

  useEffect(() => {
    fetchStats();
  }, [refreshTrigger]);

  async function fetchStats() {
    try {
      const res = await fetch("/api/stats");
      const data = await res.json();
      setStats({ ...DEFAULT_STATS, ...data });
    } catch {
      setStats(DEFAULT_STATS);
    }
    setPulse(true);
    setTimeout(() => setPulse(false), 600);
  }

  const statItems = [
    {
      label: "Areas Monitored",
      value: stats.total_snapshots,
      icon: "🗺️",
      color: "#00D4AA",
      suffix: "",
    },
    {
      label: "Avg Risk Score",
      value: stats.avg_score,
      icon: "📊",
      color: stats.avg_score >= 60 ? "#F59E0B" : "#00D4AA",
      suffix: "/100",
    },
    {
      label: "Critical Zones",
      value: stats.critical_count,
      icon: "⚠️",
      color: "#EF4444",
      suffix: "",
    },
    {
      label: "Feedback Signals",
      value: stats.feedback_count,
      icon: "🔄",
      color: "#a78bfa",
      suffix: "",
    },
  ];

  return (
    <div
      className="flex items-center gap-0 overflow-x-auto"
      style={{ scrollbarWidth: "none" }}
    >
      {statItems.map((item, i) => (
        <div
          key={item.label}
          className={`flex items-center gap-2.5 px-5 py-3 transition-all duration-300 ${
            i > 0 ? "border-l border-white/[0.06]" : ""
          }`}
        >
          <span className="text-sm">{item.icon}</span>
          <div>
            <div
              className={`text-base font-bold leading-none font-mono transition-all duration-300 ${
                pulse ? "scale-105" : "scale-100"
              }`}
              style={{ color: item.color }}
            >
              {item.value}{item.suffix}
            </div>
            <div className="text-[10px] text-white/30 mt-0.5 whitespace-nowrap">
              {item.label}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
