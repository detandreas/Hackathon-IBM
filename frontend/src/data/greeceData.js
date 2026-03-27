// EarthRisk AI — Greece Risk Data Utilities
// Patches are now fetched from the backend /api/regions endpoint.

export const RISK_COLORS = {
  CRITICAL: "#EF4444",
  HIGH: "#F59E0B",
  MEDIUM: "#EAB308",
  LOW: "#00D4AA",
};

export function getRiskColor(score) {
  if (score >= 76) return "#EF4444";
  if (score >= 51) return "#F59E0B";
  if (score >= 26) return "#EAB308";
  return "#00D4AA";
}

export function getRiskTier(score) {
  if (score >= 76) return "CRITICAL";
  if (score >= 51) return "HIGH";
  if (score >= 26) return "MEDIUM";
  return "LOW";
}
