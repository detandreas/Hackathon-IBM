import { useEffect, useState } from "react";
import { IconMap, IconBarChart, IconAlert, IconRefresh } from "./Icons";
import { apiUrl } from "../api";

const EMPTY_STATS = {
  total_snapshots: 0,
  avg_score: 0,
  critical_count: 0,
  feedback_count: 0,
};

function normalizeStats(data) {
  if (!data || typeof data !== "object") return EMPTY_STATS;
  return {
    total_snapshots: Number(data.total_snapshots) || 0,
    avg_score: Number(data.avg_score) || 0,
    critical_count: Number(data.critical_count) || 0,
    feedback_count: Number(data.feedback_count) || 0,
  };
}

export default function StatsBar({ refreshTrigger }) {
  const [stats, setStats] = useState(EMPTY_STATS);
  const [pulse, setPulse] = useState(false);

  useEffect(() => {
    fetchStats();
  }, [refreshTrigger]);

  async function fetchStats() {
    try {
      const res = await fetch(apiUrl("/api/stats"));
      if (!res.ok) throw new Error("stats failed");
      const data = await res.json();
      setStats(normalizeStats(data));
    } catch {
      setStats(EMPTY_STATS);
    }
    setPulse(true);
    setTimeout(() => setPulse(false), 600);
  }

  const statItems = [
    {
      label: "Areas Monitored",
      value: stats.total_snapshots,
      icon: <IconMap size={14} />,
      color: "#00D4AA",
      suffix: "",
    },
    {
      label: "Avg Risk Score",
      value: stats.avg_score,
      icon: <IconBarChart size={14} />,
      color: stats.avg_score >= 60 ? "#F59E0B" : "#00D4AA",
      suffix: "/100",
    },
    {
      label: "Critical Zones",
      value: stats.critical_count,
      icon: <IconAlert size={14} />,
      color: "#EF4444",
      suffix: "",
    },
    {
      label: "Feedback Signals",
      value: stats.feedback_count,
      icon: <IconRefresh size={14} />,
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
          <span className="opacity-70" style={{ color: item.color }}>{item.icon}</span>
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
