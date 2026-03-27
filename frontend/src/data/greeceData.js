// EarthRisk AI — Greece Risk Patches
// 200 patches across real Greek geographic clusters

function seededRandom(seed) {
  let s = seed;
  return function () {
    s = (s * 1664525 + 1013904223) & 0xffffffff;
    return (s >>> 0) / 0xffffffff;
  };
}

const clusters = [
  {
    name: "Thessaly", region: "Central Greece", centerLat: 39.6, centerLon: 22.4,
    spread: 0.5, tier: "HIGH", baseScore: 68, scoreVariance: 18, count: 20,
    areas: [
      {n:"Larissa Plains",lat:39.64,lon:22.42},{n:"Karditsa Valley",lat:39.37,lon:21.92},
      {n:"Trikala Basin",lat:39.56,lon:21.77},{n:"Volos Coastal",lat:39.36,lon:22.94},
      {n:"Magnesia Foothills",lat:39.30,lon:22.80},{n:"Almyros Wetlands",lat:39.18,lon:22.76},
      {n:"Farsala Agricultural",lat:39.29,lon:22.38},{n:"Tyrnavos Vineyard",lat:39.74,lon:22.29},
      {n:"Elassona Highland",lat:39.89,lon:22.19},{n:"Skiathos Northern",lat:39.16,lon:23.49},
      {n:"Pelion Peninsula",lat:39.40,lon:23.04},{n:"Zagora Orchards",lat:39.44,lon:23.10},
      {n:"Milies Village",lat:39.33,lon:23.16},{n:"Portaria Slopes",lat:39.39,lon:22.98},
      {n:"Agria Coastline",lat:39.35,lon:22.98},{n:"Nea Anchialos",lat:39.28,lon:22.84},
      {n:"Almyros Delta",lat:39.18,lon:22.76},{n:"Sophades Farmland",lat:39.34,lon:22.10},
      {n:"Palamas District",lat:39.47,lon:22.08},{n:"Mouzaki Gorge",lat:39.44,lon:21.67},
    ],
  },
  {
    name: "Attica", region: "Greater Athens", centerLat: 38.0, centerLon: 23.7,
    spread: 0.4, tier: "MEDIUM", baseScore: 45, scoreVariance: 20, count: 20,
    areas: [
      {n:"Athens City Center",lat:37.98,lon:23.73},{n:"Piraeus Port",lat:37.94,lon:23.65},
      {n:"Kifisia Suburb",lat:38.07,lon:23.81},{n:"Glyfada Coastal",lat:37.86,lon:23.75},
      {n:"Marousi Business",lat:38.05,lon:23.81},{n:"Halandri District",lat:38.02,lon:23.80},
      {n:"Nea Smyrni",lat:37.94,lon:23.71},{n:"Kallithea Urban",lat:37.95,lon:23.70},
      {n:"Peristeri West",lat:38.01,lon:23.69},{n:"Ilion Industrial",lat:38.03,lon:23.70},
      {n:"Acharnes North",lat:38.08,lon:23.73},{n:"Pallini East",lat:38.00,lon:23.88},
      {n:"Gerakas Hills",lat:38.02,lon:23.86},{n:"Koropi Agricultural",lat:37.90,lon:23.87},
      {n:"Lavrio Mining",lat:37.72,lon:24.05},{n:"Lauragais Slopes",lat:37.75,lon:23.90},
      {n:"Rafina Seaside",lat:38.02,lon:24.00},{n:"Marathon Plains",lat:38.15,lon:23.96},
      {n:"Nea Makri Beach",lat:38.09,lon:23.98},{n:"Penteli Forest",lat:38.05,lon:23.87},
    ],
  },
  {
    name: "Evia Island", region: "Central Greece Islands", centerLat: 38.6, centerLon: 23.6,
    spread: 0.45, tier: "CRITICAL", baseScore: 82, scoreVariance: 12, count: 20,
    areas: [
      {n:"Chalkida Bridge Zone",lat:38.46,lon:23.60},{n:"Eretria Ancient",lat:38.40,lon:23.79},
      {n:"Amarynthos Coastal",lat:38.39,lon:23.85},{n:"Aliveri Industrial",lat:38.37,lon:24.02},
      {n:"Kymi Highland",lat:38.63,lon:24.10},{n:"Limni Northern",lat:38.77,lon:23.29},
      {n:"Edipsos Springs",lat:38.85,lon:23.05},{n:"Istiaia Plains",lat:38.93,lon:23.15},
      {n:"Aidipsos Bay",lat:38.87,lon:23.05},{n:"Histiaiótis Forest",lat:38.90,lon:23.20},
      {n:"Psachna Valley",lat:38.56,lon:23.65},{n:"Nea Artaki",lat:38.51,lon:23.63},
      {n:"Vasiliko Industrial",lat:38.43,lon:23.90},{n:"Karystos Southern",lat:38.02,lon:24.42},
      {n:"Marmari Quarry",lat:38.05,lon:24.32},{n:"Styra Village",lat:38.16,lon:24.23},
      {n:"Platanistos Forest",lat:38.10,lon:24.20},{n:"Evia Central Ridge",lat:38.55,lon:23.80},
      {n:"Strofylia Wetlands",lat:38.48,lon:23.70},{n:"Leptokarya Slopes",lat:38.60,lon:23.50},
    ],
  },
  {
    name: "Rhodes", region: "Dodecanese", centerLat: 36.2, centerLon: 28.0,
    spread: 0.35, tier: "CRITICAL", baseScore: 85, scoreVariance: 10, count: 20,
    areas: [
      {n:"Rhodes Old Town",lat:36.45,lon:28.23},{n:"Ialyssos Resort",lat:36.42,lon:28.15},
      {n:"Kallithea Springs",lat:36.38,lon:28.24},{n:"Faliraki Beach",lat:36.35,lon:28.20},
      {n:"Afandou Village",lat:36.30,lon:28.17},{n:"Kolympia Coastal",lat:36.27,lon:28.16},
      {n:"Archangelos Historic",lat:36.21,lon:28.12},{n:"Lindos Acropolis",lat:36.09,lon:28.09},
      {n:"Lardos Bay",lat:36.06,lon:28.03},{n:"Gennadi Southern",lat:36.00,lon:27.93},
      {n:"Kattavia Tip",lat:35.88,lon:27.76},{n:"Monolithos Castle",lat:36.12,lon:27.72},
      {n:"Embonas Vineyard",lat:36.23,lon:27.85},{n:"Soroni Plains",lat:36.38,lon:28.06},
      {n:"Kremasti Airport Zone",lat:36.40,lon:28.09},{n:"Trianta Bay",lat:36.42,lon:28.10},
      {n:"Rhodes Airport Corridor",lat:36.41,lon:28.09},{n:"Theologos Village",lat:36.34,lon:27.95},
      {n:"Salakos Mountain",lat:36.30,lon:27.90},{n:"Profitis Ilias Peak",lat:36.26,lon:27.90},
    ],
  },
  {
    name: "Arcadia", region: "Peloponnese", centerLat: 37.5, centerLon: 22.3,
    spread: 0.5, tier: "HIGH", baseScore: 63, scoreVariance: 16, count: 20,
    areas: [
      {n:"Tripoli Highland",lat:37.51,lon:22.37},{n:"Megalopolis Industrial",lat:37.40,lon:22.14},
      {n:"Sparta Valley",lat:37.07,lon:22.43},{n:"Kalamata Coastal",lat:37.04,lon:22.11},
      {n:"Nafplio Historic",lat:37.57,lon:22.80},{n:"Argos Plains",lat:37.63,lon:22.72},
      {n:"Corinth Canal Zone",lat:37.94,lon:22.96},{n:"Nemea Vineyard",lat:37.82,lon:22.66},
      {n:"Mycenae Archaeological",lat:37.73,lon:22.76},{n:"Epidaurus Theater",lat:37.60,lon:23.08},
      {n:"Dimitsana Gorge",lat:37.59,lon:22.04},{n:"Stemnitsa Village",lat:37.55,lon:22.09},
      {n:"Vytina Resort",lat:37.65,lon:22.17},{n:"Langadia Canyon",lat:37.67,lon:22.10},
      {n:"Tropaia Highland",lat:37.62,lon:22.01},{n:"Levidi Plateau",lat:37.62,lon:22.28},
      {n:"Orchomenos Plains",lat:37.58,lon:22.33},{n:"Kandila Forest",lat:37.50,lon:22.10},
      {n:"Asea Valley",lat:37.43,lon:22.30},{n:"Tegea Ancient",lat:37.45,lon:22.42},
    ],
  },
  {
    name: "Crete East", region: "Crete", centerLat: 35.3, centerLon: 25.1,
    spread: 0.55, tier: "MEDIUM", baseScore: 42, scoreVariance: 22, count: 20,
    areas: [
      {n:"Heraklion Center",lat:35.34,lon:25.13},{n:"Knossos Archaeological",lat:35.30,lon:25.16},
      {n:"Archanes Vineyard",lat:35.24,lon:25.16},{n:"Peza Wine Region",lat:35.22,lon:25.20},
      {n:"Tylissos Ancient",lat:35.30,lon:25.00},{n:"Agios Nikolaos Bay",lat:35.19,lon:25.72},
      {n:"Elounda Resort",lat:35.26,lon:25.73},{n:"Spinalonga Island",lat:35.30,lon:25.73},
      {n:"Kritsa Village",lat:35.16,lon:25.65},{n:"Ierapetra Southern",lat:35.01,lon:25.73},
      {n:"Sitia Eastern",lat:35.21,lon:26.10},{n:"Zakros Gorge",lat:35.10,lon:26.26},
      {n:"Vai Palm Beach",lat:35.25,lon:26.26},{n:"Malia Ancient",lat:35.29,lon:25.49},
      {n:"Ammoudara Beach",lat:35.34,lon:25.08},{n:"Hersonissos Resort",lat:35.32,lon:25.38},
      {n:"Stalis Coastal",lat:35.31,lon:25.44},{n:"Kastelli Pediada",lat:35.22,lon:25.33},
      {n:"Arkalochori Village",lat:35.15,lon:25.27},{n:"Thrapsano Pottery",lat:35.18,lon:25.32},
    ],
  },
  {
    name: "Lesvos", region: "North Aegean", centerLat: 39.1, centerLon: 26.5,
    spread: 0.4, tier: "MEDIUM", baseScore: 48, scoreVariance: 18, count: 20,
    areas: [
      {n:"Mytilene Capital",lat:39.10,lon:26.55},{n:"Molyvos Castle",lat:39.37,lon:26.17},
      {n:"Petra Village",lat:39.35,lon:26.18},{n:"Eressos Ancient",lat:39.23,lon:25.93},
      {n:"Sigri Western",lat:39.21,lon:25.85},{n:"Plomari Ouzo",lat:38.97,lon:26.37},
      {n:"Agiassos Traditional",lat:39.07,lon:26.37},{n:"Kalloni Bay",lat:39.22,lon:26.20},
      {n:"Skala Kallonis",lat:39.20,lon:26.21},{n:"Mantamados Monastery",lat:39.28,lon:26.35},
      {n:"Thermi Hot Springs",lat:39.15,lon:26.57},{n:"Moria Camp Zone",lat:39.11,lon:26.50},
      {n:"Pamfylia Village",lat:39.14,lon:26.47},{n:"Vatera Beach",lat:38.99,lon:26.17},
      {n:"Skala Eresou",lat:39.22,lon:25.93},{n:"Agiasos Forest",lat:39.07,lon:26.37},
      {n:"Ipsilou Monastery",lat:39.25,lon:25.95},{n:"Antissa Village",lat:39.27,lon:26.05},
      {n:"Lisvori Village",lat:39.02,lon:26.20},{n:"Polichnitos Salt",lat:39.05,lon:26.17},
    ],
  },
  {
    name: "Macedonia", region: "Northern Greece", centerLat: 40.6, centerLon: 22.9,
    spread: 0.55, tier: "LOW", baseScore: 22, scoreVariance: 15, count: 20,
    areas: [
      {n:"Thessaloniki Port",lat:40.63,lon:22.94},{n:"Kalamaria Suburb",lat:40.58,lon:22.95},
      {n:"Panorama Hills",lat:40.59,lon:23.03},{n:"Pylaia East",lat:40.57,lon:22.99},
      {n:"Chortiatis Mountain",lat:40.59,lon:23.11},{n:"Lagkadas Lake",lat:40.68,lon:23.07},
      {n:"Langadikia Village",lat:40.72,lon:23.13},{n:"Asprovalta Beach",lat:40.73,lon:23.72},
      {n:"Nea Moudania",lat:40.24,lon:23.28},{n:"Polygyros Capital",lat:40.37,lon:23.44},
      {n:"Arnea Village",lat:40.50,lon:23.60},{n:"Chalkidiki Peninsula",lat:40.30,lon:23.50},
      {n:"Kassandra Coast",lat:40.05,lon:23.42},{n:"Sithonia Coast",lat:40.18,lon:23.78},
      {n:"Kavala Port",lat:40.94,lon:24.40},{n:"Drama Plains",lat:41.15,lon:24.15},
      {n:"Serres Agricultural",lat:41.09,lon:23.55},{n:"Kilkis Border",lat:40.99,lon:22.87},
      {n:"Katerini Coastal",lat:40.27,lon:22.50},{n:"Pieria Plains",lat:40.25,lon:22.45},
    ],
  },
  {
    name: "Epirus", region: "Northwestern Greece", centerLat: 39.6, centerLon: 20.8,
    spread: 0.5, tier: "LOW", baseScore: 20, scoreVariance: 14, count: 20,
    areas: [
      {n:"Ioannina Lake",lat:39.66,lon:20.85},{n:"Dodoni Ancient",lat:39.55,lon:20.79},
      {n:"Metsovo Mountain",lat:39.77,lon:21.18},{n:"Konitsa Bridge",lat:40.05,lon:20.75},
      {n:"Zagori Villages",lat:39.88,lon:20.75},{n:"Papingo Rock",lat:39.95,lon:20.68},
      {n:"Vikos Gorge",lat:39.90,lon:20.75},{n:"Monodendri View",lat:39.87,lon:20.75},
      {n:"Kipi Bridges",lat:39.88,lon:20.72},{n:"Aristi Village",lat:39.93,lon:20.68},
      {n:"Preveza Coastal",lat:38.95,lon:20.75},{n:"Nikopolis Ancient",lat:39.01,lon:20.72},
      {n:"Arta Bridge",lat:39.16,lon:20.98},{n:"Parga Seaside",lat:39.28,lon:20.40},
      {n:"Sivota Bay",lat:39.40,lon:20.23},{n:"Filiates Border",lat:39.60,lon:20.32},
      {n:"Igoumenitsa Port",lat:39.50,lon:20.27},{n:"Margariti Village",lat:39.38,lon:20.35},
      {n:"Paramythia Hill",lat:39.43,lon:20.49},{n:"Thesprotia Plains",lat:39.50,lon:20.38},
    ],
  },
  {
    name: "Dodecanese", region: "South Aegean Islands", centerLat: 36.4, centerLon: 27.2,
    spread: 0.6, tier: "MEDIUM", baseScore: 50, scoreVariance: 20, count: 20,
    areas: [
      {n:"Kos Ancient Town",lat:36.89,lon:27.09},{n:"Kardamena Resort",lat:36.79,lon:27.02},
      {n:"Kefalos Bay",lat:36.73,lon:26.97},{n:"Tingaki Beach",lat:36.88,lon:27.06},
      {n:"Mastichari Port",lat:36.86,lon:27.00},{n:"Patmos Monastery",lat:37.31,lon:26.55},
      {n:"Skala Port",lat:37.32,lon:26.56},{n:"Chora Village",lat:37.31,lon:26.55},
      {n:"Grikos Bay",lat:37.30,lon:26.56},{n:"Leros Island",lat:37.15,lon:26.85},
      {n:"Lakki Port",lat:37.12,lon:26.83},{n:"Agia Marina",lat:37.15,lon:26.87},
      {n:"Kalymnos Sponge",lat:36.95,lon:26.98},{n:"Pothia Capital",lat:36.95,lon:26.98},
      {n:"Myrties Bay",lat:36.96,lon:26.94},{n:"Nisyros Volcano",lat:36.58,lon:27.16},
      {n:"Mandraki Crater",lat:36.61,lon:27.13},{n:"Tilos Wildlife",lat:36.42,lon:27.38},
      {n:"Symi Harbor",lat:36.62,lon:27.84},{n:"Panormitis Monastery",lat:36.56,lon:27.85},
    ],
  },
];

function generatePatches() {
  const patches = [];
  let id = 1;

  clusters.forEach((cluster) => {
    const rng = seededRandom(cluster.centerLat * 1000 + cluster.centerLon * 100);

    cluster.areas.forEach((area) => {
      const areaName = area.n;
      const lat = area.lat;
      const lon = area.lon;

      let rawScore = cluster.baseScore + (rng() - 0.5) * cluster.scoreVariance * 2;
      rawScore = Math.max(5, Math.min(99, rawScore));
      const score = Math.round(rawScore);

      let tier;
      if (score >= 76) tier = "CRITICAL";
      else if (score >= 51) tier = "HIGH";
      else if (score >= 26) tier = "MEDIUM";
      else tier = "LOW";

      let trend;
      if (score > 65) trend = "rising";
      else if (score >= 35) trend = "stable";
      else trend = "improving";

      const ndvi_drop = Math.round((score * 0.3 + rng() * 20) * 10) / 10;
      const temp_increase = Math.round((score * 0.025 + rng() * 1.5) * 10) / 10;
      const land_stress = Math.round((score * 0.008 + rng() * 0.3) * 100) / 100;
      const asset_proximity = Math.round((score * 0.6 + rng() * 30) * 10) / 10;

      const trendData = [];
      let baseVal = score - (trend === "rising" ? 15 : trend === "improving" ? -10 : 0);
      for (let month = 0; month < 48; month++) {
        const noise = (rng() - 0.5) * 8;
        const drift = trend === "rising" ? month * 0.3 : trend === "improving" ? -month * 0.2 : 0;
        const val = Math.max(5, Math.min(99, Math.round(baseVal + drift + noise)));
        const year = 2021 + Math.floor(month / 12);
        const mo = (month % 12) + 1;
        trendData.push({
          date: `${year}-${String(mo).padStart(2, "0")}`,
          score: val,
        });
      }

      patches.push({
        id: `patch-${String(id).padStart(3, "0")}`,
        name: areaName,
        region: cluster.region,
        cluster: cluster.name,
        lat,
        lon,
        score,
        tier,
        trend,
        trendData,
        factors: {
          ndvi_drop: Math.min(95, ndvi_drop),
          temp_increase: Math.min(4.5, temp_increase),
          land_stress: Math.min(0.95, land_stress),
          asset_proximity: Math.min(95, asset_proximity),
        },
      });
      id++;
    });
  });

  return patches;
}

export const greecePatches = generatePatches();

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
