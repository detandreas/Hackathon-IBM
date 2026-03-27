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
    name: "Thessaly",
    region: "Central Greece",
    centerLat: 39.6,
    centerLon: 22.4,
    spread: 0.5,
    tier: "HIGH",
    baseScore: 68,
    scoreVariance: 18,
    count: 20,
    areas: [
      "Larissa Plains","Karditsa Valley","Trikala Basin","Volos Coastal","Magnesia Foothills",
      "Almyros Wetlands","Farsala Agricultural","Tyrnavos Vineyard","Elassona Highland","Skiathos Northern",
      "Pelion Peninsula","Zagora Orchards","Milies Village","Portaria Slopes","Agria Coastline",
      "Nea Anchialos","Almyros Delta","Sophades Farmland","Palamas District","Mouzaki Gorge",
    ],
  },
  {
    name: "Attica",
    region: "Greater Athens",
    centerLat: 38.0,
    centerLon: 23.7,
    spread: 0.4,
    tier: "MEDIUM",
    baseScore: 45,
    scoreVariance: 20,
    count: 20,
    areas: [
      "Athens City Center","Piraeus Port","Kifisia Suburb","Glyfada Coastal","Marousi Business",
      "Halandri District","Nea Smyrni","Kallithea Urban","Peristeri West","Ilion Industrial",
      "Acharnes North","Pallini East","Gerakas Hills","Koropi Agricultural","Lavrio Mining",
      "Lauragais Slopes","Rafina Seaside","Marathon Plains","Nea Makri Beach","Penteli Forest",
    ],
  },
  {
    name: "Evia Island",
    region: "Central Greece Islands",
    centerLat: 38.6,
    centerLon: 23.6,
    spread: 0.45,
    tier: "CRITICAL",
    baseScore: 82,
    scoreVariance: 12,
    count: 20,
    areas: [
      "Chalkida Bridge Zone","Eretria Ancient","Amarynthos Coastal","Aliveri Industrial","Kymi Highland",
      "Limni Northern","Edipsos Springs","Istiaia Plains","Aidipsos Bay","Histiaiótis Forest",
      "Psachna Valley","Nea Artaki","Vasiliko Industrial","Karystos Southern","Marmari Quarry",
      "Styra Village","Platanistos Forest","Evia Central Ridge","Strofylia Wetlands","Leptokarya Slopes",
    ],
  },
  {
    name: "Rhodes",
    region: "Dodecanese",
    centerLat: 36.2,
    centerLon: 28.0,
    spread: 0.35,
    tier: "CRITICAL",
    baseScore: 85,
    scoreVariance: 10,
    count: 20,
    areas: [
      "Rhodes Old Town","Ialyssos Resort","Kallithea Springs","Faliraki Beach","Afandou Village",
      "Kolympia Coastal","Archangelos Historic","Lindos Acropolis","Lardos Bay","Gennadi Southern",
      "Kattavia Tip","Monolithos Castle","Embonas Vineyard","Soroni Plains","Kremasti Airport Zone",
      "Trianta Bay","Rhodes Airport Corridor","Theologos Village","Salakos Mountain","Profitis Ilias Peak",
    ],
  },
  {
    name: "Arcadia",
    region: "Peloponnese",
    centerLat: 37.5,
    centerLon: 22.3,
    spread: 0.5,
    tier: "HIGH",
    baseScore: 63,
    scoreVariance: 16,
    count: 20,
    areas: [
      "Tripoli Highland","Megalopolis Industrial","Sparta Valley","Kalamata Coastal","Nafplio Historic",
      "Argos Plains","Corinth Canal Zone","Nemea Vineyard","Mycenae Archaeological","Epidaurus Theater",
      "Dimitsana Gorge","Stemnitsa Village","Vytina Resort","Langadia Canyon","Tropaia Highland",
      "Levidi Plateau","Orchomenos Plains","Kandila Forest","Asea Valley","Tegea Ancient",
    ],
  },
  {
    name: "Crete East",
    region: "Crete",
    centerLat: 35.3,
    centerLon: 25.1,
    spread: 0.55,
    tier: "MEDIUM",
    baseScore: 42,
    scoreVariance: 22,
    count: 20,
    areas: [
      "Heraklion Center","Knossos Archaeological","Archanes Vineyard","Peza Wine Region","Tylissos Ancient",
      "Agios Nikolaos Bay","Elounda Resort","Spinalonga Island","Kritsa Village","Ierapetra Southern",
      "Sitia Eastern","Zakros Gorge","Vai Palm Beach","Malia Ancient","Ammoudara Beach",
      "Hersonissos Resort","Stalis Coastal","Kastelli Pediada","Arkalochori Village","Thrapsano Pottery",
    ],
  },
  {
    name: "Lesvos",
    region: "North Aegean",
    centerLat: 39.1,
    centerLon: 26.5,
    spread: 0.4,
    tier: "MEDIUM",
    baseScore: 48,
    scoreVariance: 18,
    count: 20,
    areas: [
      "Mytilene Capital","Molyvos Castle","Petra Village","Eressos Ancient","Sigri Western",
      "Plomari Ouzo","Agiassos Traditional","Kalloni Bay","Skala Kallonis","Mantamados Monastery",
      "Thermi Hot Springs","Moria Camp Zone","Pamfylia Village","Vatera Beach","Skala Eresou",
      "Agiasos Forest","Ipsilou Monastery","Antissa Village","Lisvori Village","Polichnitos Salt",
    ],
  },
  {
    name: "Macedonia",
    region: "Northern Greece",
    centerLat: 40.6,
    centerLon: 22.9,
    spread: 0.55,
    tier: "LOW",
    baseScore: 22,
    scoreVariance: 15,
    count: 20,
    areas: [
      "Thessaloniki Port","Kalamaria Suburb","Panorama Hills","Pylaia East","Chortiatis Mountain",
      "Lagkadas Lake","Langadikia Village","Asprovalta Beach","Nea Moudania","Polygyros Capital",
      "Arnea Village","Chalkidiki Peninsula","Kassandra Coast","Sithonia Coast","Kavala Port",
      "Drama Plains","Serres Agricultural","Kilkis Border","Katerini Coastal","Pieria Plains",
    ],
  },
  {
    name: "Epirus",
    region: "Northwestern Greece",
    centerLat: 39.6,
    centerLon: 20.8,
    spread: 0.5,
    tier: "LOW",
    baseScore: 20,
    scoreVariance: 14,
    count: 20,
    areas: [
      "Ioannina Lake","Dodoni Ancient","Metsovo Mountain","Konitsa Bridge","Zagori Villages",
      "Papingo Rock","Vikos Gorge","Monodendri View","Kipi Bridges","Aristi Village",
      "Preveza Coastal","Nikopolis Ancient","Arta Bridge","Parga Seaside","Sivota Bay",
      "Filiates Border","Igoumenitsa Port","Margariti Village","Paramythia Hill","Thesprotia Plains",
    ],
  },
  {
    name: "Dodecanese",
    region: "South Aegean Islands",
    centerLat: 36.4,
    centerLon: 27.2,
    spread: 0.6,
    tier: "MEDIUM",
    baseScore: 50,
    scoreVariance: 20,
    count: 20,
    areas: [
      "Kos Ancient Town","Kardamena Resort","Kefalos Bay","Tingaki Beach","Mastichari Port",
      "Patmos Monastery","Skala Port","Chora Village","Grikos Bay","Leros Island",
      "Lakki Port","Agia Marina","Kalymnos Sponge","Pothia Capital","Myrties Bay",
      "Nisyros Volcano","Mandraki Crater","Tilos Wildlife","Symi Harbor","Panormitis Monastery",
    ],
  },
];

function generatePatches() {
  const patches = [];
  let id = 1;

  clusters.forEach((cluster) => {
    const rng = seededRandom(cluster.centerLat * 1000 + cluster.centerLon * 100);

    cluster.areas.forEach((areaName, i) => {
      const latOffset = (rng() - 0.5) * cluster.spread * 2;
      const lonOffset = (rng() - 0.5) * cluster.spread * 2;
      const lat = cluster.centerLat + latOffset;
      const lon = cluster.centerLon + lonOffset;

      // Deterministic score generation
      let rawScore = cluster.baseScore + (rng() - 0.5) * cluster.scoreVariance * 2;
      rawScore = Math.max(5, Math.min(99, rawScore));
      const score = Math.round(rawScore);

      // Determine tier
      let tier;
      if (score >= 76) tier = "CRITICAL";
      else if (score >= 51) tier = "HIGH";
      else if (score >= 26) tier = "MEDIUM";
      else tier = "LOW";

      // Trend based on score
      let trend;
      if (score > 65) trend = "rising";
      else if (score >= 35) trend = "stable";
      else trend = "improving";

      // Generate factors deterministically
      const ndvi_drop = Math.round((score * 0.3 + rng() * 20) * 10) / 10;
      const temp_increase = Math.round((score * 0.025 + rng() * 1.5) * 10) / 10;
      const land_stress = Math.round((score * 0.008 + rng() * 0.3) * 100) / 100;
      const asset_proximity = Math.round((score * 0.6 + rng() * 30) * 10) / 10;

      // Generate 4-year monthly trend data (48 data points)
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
