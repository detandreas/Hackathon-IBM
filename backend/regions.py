"""Region definitions for the 10 Greek areas covered by EarthRisk AI.
Each area now has real-world coordinates.
Shared between the data pipeline and the API server.
"""

import numpy as np

REGIONS = {
    "thessaly": {
        "name": "Thessaly",
        "display_region": "Central Greece",
        "center_lat": 39.6, "center_lon": 22.4,
        "bbox": [21.9, 39.1, 22.9, 40.1],
        "spread": 0.5,
        "areas": [
            {"name": "Larissa Plains", "lat": 39.64, "lon": 22.42},
            {"name": "Karditsa Valley", "lat": 39.37, "lon": 21.92},
            {"name": "Trikala Basin", "lat": 39.56, "lon": 21.77},
            {"name": "Volos Coastal", "lat": 39.36, "lon": 22.94},
            {"name": "Magnesia Foothills", "lat": 39.30, "lon": 22.80},
            {"name": "Almyros Wetlands", "lat": 39.18, "lon": 22.76},
            {"name": "Farsala Agricultural", "lat": 39.29, "lon": 22.38},
            {"name": "Tyrnavos Vineyard", "lat": 39.74, "lon": 22.29},
            {"name": "Elassona Highland", "lat": 39.89, "lon": 22.19},
            {"name": "Skiathos Northern", "lat": 39.16, "lon": 23.49},
            {"name": "Pelion Peninsula", "lat": 39.40, "lon": 23.04},
            {"name": "Zagora Orchards", "lat": 39.44, "lon": 23.10},
            {"name": "Milies Village", "lat": 39.33, "lon": 23.16},
            {"name": "Portaria Slopes", "lat": 39.39, "lon": 22.98},
            {"name": "Agria Coastline", "lat": 39.35, "lon": 22.98},
            {"name": "Nea Anchialos", "lat": 39.28, "lon": 22.84},
            {"name": "Almyros Delta", "lat": 39.18, "lon": 22.76},
            {"name": "Sophades Farmland", "lat": 39.34, "lon": 22.10},
            {"name": "Palamas District", "lat": 39.47, "lon": 22.08},
            {"name": "Mouzaki Gorge", "lat": 39.44, "lon": 21.67},
        ],
    },
    "attica": {
        "name": "Attica",
        "display_region": "Greater Athens",
        "center_lat": 38.0, "center_lon": 23.7,
        "bbox": [23.3, 37.6, 24.1, 38.4],
        "spread": 0.4,
        "areas": [
            {"name": "Athens City Center", "lat": 37.98, "lon": 23.73},
            {"name": "Piraeus Port", "lat": 37.94, "lon": 23.65},
            {"name": "Kifisia Suburb", "lat": 38.07, "lon": 23.81},
            {"name": "Glyfada Coastal", "lat": 37.86, "lon": 23.75},
            {"name": "Marousi Business", "lat": 38.05, "lon": 23.81},
            {"name": "Halandri District", "lat": 38.02, "lon": 23.80},
            {"name": "Nea Smyrni", "lat": 37.94, "lon": 23.71},
            {"name": "Kallithea Urban", "lat": 37.95, "lon": 23.70},
            {"name": "Peristeri West", "lat": 38.01, "lon": 23.69},
            {"name": "Ilion Industrial", "lat": 38.03, "lon": 23.70},
            {"name": "Acharnes North", "lat": 38.08, "lon": 23.73},
            {"name": "Pallini East", "lat": 38.00, "lon": 23.88},
            {"name": "Gerakas Hills", "lat": 38.02, "lon": 23.86},
            {"name": "Koropi Agricultural", "lat": 37.90, "lon": 23.87},
            {"name": "Lavrio Mining", "lat": 37.72, "lon": 24.05},
            {"name": "Lauragais Slopes", "lat": 37.75, "lon": 23.90},
            {"name": "Rafina Seaside", "lat": 38.02, "lon": 24.00},
            {"name": "Marathon Plains", "lat": 38.15, "lon": 23.96},
            {"name": "Nea Makri Beach", "lat": 38.09, "lon": 23.98},
            {"name": "Penteli Forest", "lat": 38.05, "lon": 23.87},
        ],
    },
    "evia": {
        "name": "Evia Island",
        "display_region": "Central Greece Islands",
        "center_lat": 38.6, "center_lon": 23.6,
        "bbox": [23.15, 38.15, 24.05, 39.05],
        "spread": 0.45,
        "areas": [
            {"name": "Chalkida Bridge Zone", "lat": 38.46, "lon": 23.60},
            {"name": "Eretria Ancient", "lat": 38.40, "lon": 23.79},
            {"name": "Amarynthos Coastal", "lat": 38.39, "lon": 23.85},
            {"name": "Aliveri Industrial", "lat": 38.37, "lon": 24.02},
            {"name": "Kymi Highland", "lat": 38.63, "lon": 24.10},
            {"name": "Limni Northern", "lat": 38.77, "lon": 23.29},
            {"name": "Edipsos Springs", "lat": 38.85, "lon": 23.05},
            {"name": "Istiaia Plains", "lat": 38.93, "lon": 23.15},
            {"name": "Aidipsos Bay", "lat": 38.87, "lon": 23.05},
            {"name": "Histiaiótis Forest", "lat": 38.90, "lon": 23.20},
            {"name": "Psachna Valley", "lat": 38.56, "lon": 23.65},
            {"name": "Nea Artaki", "lat": 38.51, "lon": 23.63},
            {"name": "Vasiliko Industrial", "lat": 38.43, "lon": 23.90},
            {"name": "Karystos Southern", "lat": 38.02, "lon": 24.42},
            {"name": "Marmari Quarry", "lat": 38.05, "lon": 24.32},
            {"name": "Styra Village", "lat": 38.16, "lon": 24.23},
            {"name": "Platanistos Forest", "lat": 38.10, "lon": 24.20},
            {"name": "Evia Central Ridge", "lat": 38.55, "lon": 23.80},
            {"name": "Strofylia Wetlands", "lat": 38.48, "lon": 23.70},
            {"name": "Leptokarya Slopes", "lat": 38.60, "lon": 23.50},
        ],
    },
    "rhodes": {
        "name": "Rhodes",
        "display_region": "Dodecanese",
        "center_lat": 36.2, "center_lon": 28.0,
        "bbox": [27.65, 35.85, 28.35, 36.55],
        "spread": 0.35,
        "areas": [
            {"name": "Rhodes Old Town", "lat": 36.45, "lon": 28.23},
            {"name": "Ialyssos Resort", "lat": 36.42, "lon": 28.15},
            {"name": "Kallithea Springs", "lat": 36.38, "lon": 28.24},
            {"name": "Faliraki Beach", "lat": 36.35, "lon": 28.20},
            {"name": "Afandou Village", "lat": 36.30, "lon": 28.17},
            {"name": "Kolympia Coastal", "lat": 36.27, "lon": 28.16},
            {"name": "Archangelos Historic", "lat": 36.21, "lon": 28.12},
            {"name": "Lindos Acropolis", "lat": 36.09, "lon": 28.09},
            {"name": "Lardos Bay", "lat": 36.06, "lon": 28.03},
            {"name": "Gennadi Southern", "lat": 36.00, "lon": 27.93},
            {"name": "Kattavia Tip", "lat": 35.88, "lon": 27.76},
            {"name": "Monolithos Castle", "lat": 36.12, "lon": 27.72},
            {"name": "Embonas Vineyard", "lat": 36.23, "lon": 27.85},
            {"name": "Soroni Plains", "lat": 36.38, "lon": 28.06},
            {"name": "Kremasti Airport Zone", "lat": 36.40, "lon": 28.09},
            {"name": "Trianta Bay", "lat": 36.42, "lon": 28.10},
            {"name": "Rhodes Airport Corridor", "lat": 36.41, "lon": 28.09},
            {"name": "Theologos Village", "lat": 36.34, "lon": 27.95},
            {"name": "Salakos Mountain", "lat": 36.30, "lon": 27.90},
            {"name": "Profitis Ilias Peak", "lat": 36.26, "lon": 27.90},
        ],
    },
    "arcadia": {
        "name": "Arcadia",
        "display_region": "Peloponnese",
        "center_lat": 37.5, "center_lon": 22.3,
        "bbox": [21.8, 37.0, 22.8, 38.0],
        "spread": 0.5,
        "areas": [
            {"name": "Tripoli Highland", "lat": 37.51, "lon": 22.37},
            {"name": "Megalopolis Industrial", "lat": 37.40, "lon": 22.14},
            {"name": "Sparta Valley", "lat": 37.07, "lon": 22.43},
            {"name": "Kalamata Coastal", "lat": 37.04, "lon": 22.11},
            {"name": "Nafplio Historic", "lat": 37.57, "lon": 22.80},
            {"name": "Argos Plains", "lat": 37.63, "lon": 22.72},
            {"name": "Corinth Canal Zone", "lat": 37.94, "lon": 22.96},
            {"name": "Nemea Vineyard", "lat": 37.82, "lon": 22.66},
            {"name": "Mycenae Archaeological", "lat": 37.73, "lon": 22.76},
            {"name": "Epidaurus Theater", "lat": 37.60, "lon": 23.08},
            {"name": "Dimitsana Gorge", "lat": 37.59, "lon": 22.04},
            {"name": "Stemnitsa Village", "lat": 37.55, "lon": 22.09},
            {"name": "Vytina Resort", "lat": 37.65, "lon": 22.17},
            {"name": "Langadia Canyon", "lat": 37.67, "lon": 22.10},
            {"name": "Tropaia Highland", "lat": 37.62, "lon": 22.01},
            {"name": "Levidi Plateau", "lat": 37.62, "lon": 22.28},
            {"name": "Orchomenos Plains", "lat": 37.58, "lon": 22.33},
            {"name": "Kandila Forest", "lat": 37.50, "lon": 22.10},
            {"name": "Asea Valley", "lat": 37.43, "lon": 22.30},
            {"name": "Tegea Ancient", "lat": 37.45, "lon": 22.42},
        ],
    },
    "crete": {
        "name": "Crete East",
        "display_region": "Crete",
        "center_lat": 35.3, "center_lon": 25.1,
        "bbox": [24.55, 34.75, 25.65, 35.85],
        "spread": 0.55,
        "areas": [
            {"name": "Heraklion Center", "lat": 35.34, "lon": 25.13},
            {"name": "Knossos Archaeological", "lat": 35.30, "lon": 25.16},
            {"name": "Archanes Vineyard", "lat": 35.24, "lon": 25.16},
            {"name": "Peza Wine Region", "lat": 35.22, "lon": 25.20},
            {"name": "Tylissos Ancient", "lat": 35.30, "lon": 25.00},
            {"name": "Agios Nikolaos Bay", "lat": 35.19, "lon": 25.72},
            {"name": "Elounda Resort", "lat": 35.26, "lon": 25.73},
            {"name": "Spinalonga Island", "lat": 35.30, "lon": 25.73},
            {"name": "Kritsa Village", "lat": 35.16, "lon": 25.65},
            {"name": "Ierapetra Southern", "lat": 35.01, "lon": 25.73},
            {"name": "Sitia Eastern", "lat": 35.21, "lon": 26.10},
            {"name": "Zakros Gorge", "lat": 35.10, "lon": 26.26},
            {"name": "Vai Palm Beach", "lat": 35.25, "lon": 26.26},
            {"name": "Malia Ancient", "lat": 35.29, "lon": 25.49},
            {"name": "Ammoudara Beach", "lat": 35.34, "lon": 25.08},
            {"name": "Hersonissos Resort", "lat": 35.32, "lon": 25.38},
            {"name": "Stalis Coastal", "lat": 35.31, "lon": 25.44},
            {"name": "Kastelli Pediada", "lat": 35.22, "lon": 25.33},
            {"name": "Arkalochori Village", "lat": 35.15, "lon": 25.27},
            {"name": "Thrapsano Pottery", "lat": 35.18, "lon": 25.32},
        ],
    },
    "lesvos": {
        "name": "Lesvos",
        "display_region": "North Aegean",
        "center_lat": 39.1, "center_lon": 26.5,
        "bbox": [26.1, 38.7, 26.9, 39.5],
        "spread": 0.4,
        "areas": [
            {"name": "Mytilene Capital", "lat": 39.10, "lon": 26.55},
            {"name": "Molyvos Castle", "lat": 39.37, "lon": 26.17},
            {"name": "Petra Village", "lat": 39.35, "lon": 26.18},
            {"name": "Eressos Ancient", "lat": 39.23, "lon": 25.93},
            {"name": "Sigri Western", "lat": 39.21, "lon": 25.85},
            {"name": "Plomari Ouzo", "lat": 38.97, "lon": 26.37},
            {"name": "Agiassos Traditional", "lat": 39.07, "lon": 26.37},
            {"name": "Kalloni Bay", "lat": 39.22, "lon": 26.20},
            {"name": "Skala Kallonis", "lat": 39.20, "lon": 26.21},
            {"name": "Mantamados Monastery", "lat": 39.28, "lon": 26.35},
            {"name": "Thermi Hot Springs", "lat": 39.15, "lon": 26.57},
            {"name": "Moria Camp Zone", "lat": 39.11, "lon": 26.50},
            {"name": "Pamfylia Village", "lat": 39.14, "lon": 26.47},
            {"name": "Vatera Beach", "lat": 38.99, "lon": 26.17},
            {"name": "Skala Eresou", "lat": 39.22, "lon": 25.93},
            {"name": "Agiasos Forest", "lat": 39.07, "lon": 26.37},
            {"name": "Ipsilou Monastery", "lat": 39.25, "lon": 25.95},
            {"name": "Antissa Village", "lat": 39.27, "lon": 26.05},
            {"name": "Lisvori Village", "lat": 39.02, "lon": 26.20},
            {"name": "Polichnitos Salt", "lat": 39.05, "lon": 26.17},
        ],
    },
    "macedonia": {
        "name": "Macedonia",
        "display_region": "Northern Greece",
        "center_lat": 40.6, "center_lon": 22.9,
        "bbox": [22.35, 40.05, 23.45, 41.15],
        "spread": 0.55,
        "areas": [
            {"name": "Thessaloniki Port", "lat": 40.63, "lon": 22.94},
            {"name": "Kalamaria Suburb", "lat": 40.58, "lon": 22.95},
            {"name": "Panorama Hills", "lat": 40.59, "lon": 23.03},
            {"name": "Pylaia East", "lat": 40.57, "lon": 22.99},
            {"name": "Chortiatis Mountain", "lat": 40.59, "lon": 23.11},
            {"name": "Lagkadas Lake", "lat": 40.68, "lon": 23.07},
            {"name": "Langadikia Village", "lat": 40.72, "lon": 23.13},
            {"name": "Asprovalta Beach", "lat": 40.73, "lon": 23.72},
            {"name": "Nea Moudania", "lat": 40.24, "lon": 23.28},
            {"name": "Polygyros Capital", "lat": 40.37, "lon": 23.44},
            {"name": "Arnea Village", "lat": 40.50, "lon": 23.60},
            {"name": "Chalkidiki Peninsula", "lat": 40.30, "lon": 23.50},
            {"name": "Kassandra Coast", "lat": 40.05, "lon": 23.42},
            {"name": "Sithonia Coast", "lat": 40.18, "lon": 23.78},
            {"name": "Kavala Port", "lat": 40.94, "lon": 24.40},
            {"name": "Drama Plains", "lat": 41.15, "lon": 24.15},
            {"name": "Serres Agricultural", "lat": 41.09, "lon": 23.55},
            {"name": "Kilkis Border", "lat": 40.99, "lon": 22.87},
            {"name": "Katerini Coastal", "lat": 40.27, "lon": 22.50},
            {"name": "Pieria Plains", "lat": 40.25, "lon": 22.45},
        ],
    },
    "epirus": {
        "name": "Epirus",
        "display_region": "Northwestern Greece",
        "center_lat": 39.6, "center_lon": 20.8,
        "bbox": [20.3, 39.1, 21.3, 40.1],
        "spread": 0.5,
        "areas": [
            {"name": "Ioannina Lake", "lat": 39.66, "lon": 20.85},
            {"name": "Dodoni Ancient", "lat": 39.55, "lon": 20.79},
            {"name": "Metsovo Mountain", "lat": 39.77, "lon": 21.18},
            {"name": "Konitsa Bridge", "lat": 40.05, "lon": 20.75},
            {"name": "Zagori Villages", "lat": 39.88, "lon": 20.75},
            {"name": "Papingo Rock", "lat": 39.95, "lon": 20.68},
            {"name": "Vikos Gorge", "lat": 39.90, "lon": 20.75},
            {"name": "Monodendri View", "lat": 39.87, "lon": 20.75},
            {"name": "Kipi Bridges", "lat": 39.88, "lon": 20.72},
            {"name": "Aristi Village", "lat": 39.93, "lon": 20.68},
            {"name": "Preveza Coastal", "lat": 38.95, "lon": 20.75},
            {"name": "Nikopolis Ancient", "lat": 39.01, "lon": 20.72},
            {"name": "Arta Bridge", "lat": 39.16, "lon": 20.98},
            {"name": "Parga Seaside", "lat": 39.28, "lon": 20.40},
            {"name": "Sivota Bay", "lat": 39.40, "lon": 20.23},
            {"name": "Filiates Border", "lat": 39.60, "lon": 20.32},
            {"name": "Igoumenitsa Port", "lat": 39.50, "lon": 20.27},
            {"name": "Margariti Village", "lat": 39.38, "lon": 20.35},
            {"name": "Paramythia Hill", "lat": 39.43, "lon": 20.49},
            {"name": "Thesprotia Plains", "lat": 39.50, "lon": 20.38},
        ],
    },
    "dodecanese": {
        "name": "Dodecanese",
        "display_region": "South Aegean Islands",
        "center_lat": 36.4, "center_lon": 27.2,
        "bbox": [26.6, 35.8, 27.8, 37.0],
        "spread": 0.6,
        "areas": [
            {"name": "Kos Ancient Town", "lat": 36.89, "lon": 27.09},
            {"name": "Kardamena Resort", "lat": 36.79, "lon": 27.02},
            {"name": "Kefalos Bay", "lat": 36.73, "lon": 26.97},
            {"name": "Tingaki Beach", "lat": 36.88, "lon": 27.06},
            {"name": "Mastichari Port", "lat": 36.86, "lon": 27.00},
            {"name": "Patmos Monastery", "lat": 37.31, "lon": 26.55},
            {"name": "Skala Port", "lat": 37.32, "lon": 26.56},
            {"name": "Chora Village", "lat": 37.31, "lon": 26.55},
            {"name": "Grikos Bay", "lat": 37.30, "lon": 26.56},
            {"name": "Leros Island", "lat": 37.15, "lon": 26.85},
            {"name": "Lakki Port", "lat": 37.12, "lon": 26.83},
            {"name": "Agia Marina", "lat": 37.15, "lon": 26.87},
            {"name": "Kalymnos Sponge", "lat": 36.95, "lon": 26.98},
            {"name": "Pothia Capital", "lat": 36.95, "lon": 26.98},
            {"name": "Myrties Bay", "lat": 36.96, "lon": 26.94},
            {"name": "Nisyros Volcano", "lat": 36.58, "lon": 27.16},
            {"name": "Mandraki Crater", "lat": 36.61, "lon": 27.13},
            {"name": "Tilos Wildlife", "lat": 36.42, "lon": 27.38},
            {"name": "Symi Harbor", "lat": 36.62, "lon": 27.84},
            {"name": "Panormitis Monastery", "lat": 36.56, "lon": 27.85},
        ],
    },
}

GREECE_BBOX = [20.0, 34.5, 28.5, 41.5]


def nearest_region_name(lat: float, lon: float) -> str:
    """Return the name of the nearest region to the given coordinates."""
    best_name = "Unknown"
    best_dist = float("inf")
    for info in REGIONS.values():
        d = (lat - info["center_lat"]) ** 2 + (lon - info["center_lon"]) ** 2
        if d < best_dist:
            best_dist = d
            best_name = info["name"]
    return best_name


def nearest_region_key(lat: float, lon: float) -> str:
    """Return the key of the nearest region to the given coordinates."""
    best_key = ""
    best_dist = float("inf")
    for key, info in REGIONS.items():
        d = (lat - info["center_lat"]) ** 2 + (lon - info["center_lon"]) ** 2
        if d < best_dist:
            best_dist = d
            best_key = key
    return best_key


_REGION_CENTERS = np.array(
    [[info["center_lat"], info["center_lon"]] for info in REGIONS.values()]
)
_REGION_NAMES = np.array([info["name"] for info in REGIONS.values()])


def nearest_region_name_vectorized(
    lats: np.ndarray, lons: np.ndarray
) -> np.ndarray:
    """Vectorized nearest-region assignment using numpy broadcasting."""
    coords = np.column_stack([lats, lons])                     # (N, 2)
    dists = np.sum((coords[:, None, :] - _REGION_CENTERS[None, :, :]) ** 2, axis=2)
    return _REGION_NAMES[np.argmin(dists, axis=1)]
