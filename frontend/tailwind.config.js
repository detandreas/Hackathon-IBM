/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        navy: "#0A0F1E",
        teal: {
          DEFAULT: "#00D4AA",
          dark: "#00a882",
        },
        slate: {
          card: "#1E293B",
        },
        risk: {
          low: "#00D4AA",
          medium: "#EAB308",
          high: "#F59E0B",
          critical: "#EF4444",
        },
      },
      fontFamily: {
        sans: ["Space Grotesk", "-apple-system", "BlinkMacSystemFont", "sans-serif"],
        mono: ["Space Mono", "monospace"],
      },
      backdropBlur: {
        xs: "2px",
      },
      animation: {
        "pulse-slow": "pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite",
        "fade-in": "fadeIn 0.5s ease-out",
        "slide-up": "slideUp 0.4s cubic-bezier(0.16, 1, 0.3, 1)",
      },
      keyframes: {
        fadeIn: {
          "0%": { opacity: "0" },
          "100%": { opacity: "1" },
        },
        slideUp: {
          "0%": { transform: "translateY(20px)", opacity: "0" },
          "100%": { transform: "translateY(0)", opacity: "1" },
        },
      },
      boxShadow: {
        teal: "0 0 30px rgba(0,212,170,0.3)",
        "teal-lg": "0 0 60px rgba(0,212,170,0.4)",
        card: "0 4px 24px rgba(0,0,0,0.4)",
      },
    },
  },
  plugins: [],
};
