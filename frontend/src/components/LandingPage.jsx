import { useNavigate } from "react-router-dom";
import { useEffect, useState } from "react";

export default function LandingPage() {
  const navigate = useNavigate();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  return (
    <div className="relative min-h-screen bg-[#0A0F1E] overflow-hidden flex flex-col items-center justify-center">
      {/* Animated CSS Grid Background */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div
          className="absolute inset-0"
          style={{
            backgroundImage: `
              linear-gradient(rgba(0,212,170,0.07) 1px, transparent 1px),
              linear-gradient(90deg, rgba(0,212,170,0.07) 1px, transparent 1px)
            `,
            backgroundSize: "60px 60px",
            animation: "gridMove 20s linear infinite",
          }}
        />
        <div
          className="absolute inset-0"
          style={{
            background: "radial-gradient(ellipse 80% 60% at 50% 40%, rgba(0,212,170,0.08) 0%, transparent 70%)",
          }}
        />
        {[...Array(6)].map((_, i) => (
          <div
            key={i}
            className="absolute rounded-full"
            style={{
              width: `${80 + i * 40}px`,
              height: `${80 + i * 40}px`,
              left: `${10 + i * 15}%`,
              top: `${15 + (i % 3) * 25}%`,
              background: i % 2 === 0
                ? "radial-gradient(circle, rgba(0,212,170,0.12) 0%, transparent 70%)"
                : "radial-gradient(circle, rgba(239,68,68,0.08) 0%, transparent 70%)",
              animation: `floatOrb ${8 + i * 2}s ease-in-out infinite`,
              animationDelay: `${i * 1.5}s`,
            }}
          />
        ))}
      </div>

      {/* Main Content */}
      <div
        className={`relative z-10 text-center px-6 max-w-4xl mx-auto transition-all duration-1000 ${
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
            { icon: "🛰️", label: "Real Sentinel-2 Data" },
            { icon: "🔄", label: "Underwriter Feedback Loop" },
            { icon: "📋", label: "Regulatory Audit Trail" },
          ].map((pill) => (
            <div
              key={pill.label}
              className="flex items-center gap-2 px-5 py-2.5 rounded-full border border-white/10 bg-white/5 text-white/70 text-sm backdrop-blur-sm hover:border-[#00D4AA]/40 hover:text-white transition-all duration-300"
            >
              <span>{pill.icon}</span>
              <span>{pill.label}</span>
            </div>
          ))}
        </div>

        {/* Data Sources footer badges — IBM challenge requirement */}
        <div className="flex flex-wrap justify-center items-center gap-3 mt-10">
          <span className="text-[10px] text-white/25 uppercase tracking-widest mr-1">Data Sources</span>
          {[
            { label: "Sentinel-2", color: "#00D4AA" },
            { label: "ERA5", color: "#60a5fa" },
            { label: "XGBoost", color: "#a78bfa" },
            { label: "Explainable AI", color: "#F59E0B" },
          ].map(({ label, color }) => (
            <div
              key={label}
              className="flex items-center gap-1.5 px-3 py-1 rounded-full text-[11px] font-semibold"
              style={{
                background: `${color}15`,
                border: `1px solid ${color}35`,
                color,
              }}
            >
              <span className="w-1.5 h-1.5 rounded-full" style={{ background: color }} />
              {label}
            </div>
          ))}
        </div>
      </div>

      {/* Bottom stats */}
      <div
        className={`absolute bottom-8 left-0 right-0 flex justify-center gap-8 md:gap-16 transition-all duration-1000 delay-500 ${
          mounted ? "opacity-100 translate-y-0" : "opacity-0 translate-y-4"
        }`}
      >
        {[
          { value: "200+", label: "Risk Zones Monitored" },
          { value: "4", label: "Data Moats" },
          { value: "93%", label: "Prediction Accuracy" },
          { value: "Real-time", label: "Risk Updates" },
        ].map((stat) => (
          <div key={stat.label} className="text-center">
            <div className="text-xl font-bold text-[#00D4AA]">{stat.value}</div>
            <div className="text-xs text-white/30 mt-0.5">{stat.label}</div>
          </div>
        ))}
      </div>

      <style>{`
        @keyframes gridMove {
          0% { transform: translateY(0); }
          100% { transform: translateY(60px); }
        }
        @keyframes floatOrb {
          0%, 100% { transform: translateY(0px) scale(1); opacity: 0.6; }
          50% { transform: translateY(-30px) scale(1.1); opacity: 1; }
        }
      `}</style>
    </div>
  );
}
