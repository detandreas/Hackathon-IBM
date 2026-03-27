"""Region definitions for the 10 Greek areas covered by EarthRisk AI.
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
            "Larissa Plains", "Karditsa Valley", "Trikala Basin", "Volos Coastal",
            "Magnesia Foothills", "Almyros Wetlands", "Farsala Agricultural",
            "Tyrnavos Vineyard", "Elassona Highland", "Skiathos Northern",
            "Pelion Peninsula", "Zagora Orchards", "Milies Village",
            "Portaria Slopes", "Agria Coastline", "Nea Anchialos",
            "Almyros Delta", "Sophades Farmland", "Palamas District", "Mouzaki Gorge",
        ],
    },
    "attica": {
        "name": "Attica",
        "display_region": "Greater Athens",
        "center_lat": 38.0, "center_lon": 23.7,
        "bbox": [23.3, 37.6, 24.1, 38.4],
        "spread": 0.4,
        "areas": [
            "Athens City Center", "Piraeus Port", "Kifisia Suburb", "Glyfada Coastal",
            "Marousi Business", "Halandri District", "Nea Smyrni", "Kallithea Urban",
            "Peristeri West", "Ilion Industrial", "Acharnes North", "Pallini East",
            "Gerakas Hills", "Koropi Agricultural", "Lavrio Mining",
            "Lauragais Slopes", "Rafina Seaside", "Marathon Plains",
            "Nea Makri Beach", "Penteli Forest",
        ],
    },
    "evia": {
        "name": "Evia Island",
        "display_region": "Central Greece Islands",
        "center_lat": 38.6, "center_lon": 23.6,
        "bbox": [23.15, 38.15, 24.05, 39.05],
        "spread": 0.45,
        "areas": [
            "Chalkida Bridge Zone", "Eretria Ancient", "Amarynthos Coastal",
            "Aliveri Industrial", "Kymi Highland", "Limni Northern", "Edipsos Springs",
            "Istiaia Plains", "Aidipsos Bay", "Histiaiótis Forest",
            "Psachna Valley", "Nea Artaki", "Vasiliko Industrial",
            "Karystos Southern", "Marmari Quarry", "Styra Village",
            "Platanistos Forest", "Evia Central Ridge", "Strofylia Wetlands",
            "Leptokarya Slopes",
        ],
    },
    "rhodes": {
        "name": "Rhodes",
        "display_region": "Dodecanese",
        "center_lat": 36.2, "center_lon": 28.0,
        "bbox": [27.65, 35.85, 28.35, 36.55],
        "spread": 0.35,
        "areas": [
            "Rhodes Old Town", "Ialyssos Resort", "Kallithea Springs", "Faliraki Beach",
            "Afandou Village", "Kolympia Coastal", "Archangelos Historic",
            "Lindos Acropolis", "Lardos Bay", "Gennadi Southern",
            "Kattavia Tip", "Monolithos Castle", "Embonas Vineyard",
            "Soroni Plains", "Kremasti Airport Zone", "Trianta Bay",
            "Rhodes Airport Corridor", "Theologos Village", "Salakos Mountain",
            "Profitis Ilias Peak",
        ],
    },
    "arcadia": {
        "name": "Arcadia",
        "display_region": "Peloponnese",
        "center_lat": 37.5, "center_lon": 22.3,
        "bbox": [21.8, 37.0, 22.8, 38.0],
        "spread": 0.5,
        "areas": [
            "Tripoli Highland", "Megalopolis Industrial", "Sparta Valley",
            "Kalamata Coastal", "Nafplio Historic", "Argos Plains",
            "Corinth Canal Zone", "Nemea Vineyard", "Mycenae Archaeological",
            "Epidaurus Theater", "Dimitsana Gorge", "Stemnitsa Village",
            "Vytina Resort", "Langadia Canyon", "Tropaia Highland",
            "Levidi Plateau", "Orchomenos Plains", "Kandila Forest",
            "Asea Valley", "Tegea Ancient",
        ],
    },
    "crete": {
        "name": "Crete East",
        "display_region": "Crete",
        "center_lat": 35.3, "center_lon": 25.1,
        "bbox": [24.55, 34.75, 25.65, 35.85],
        "spread": 0.55,
        "areas": [
            "Heraklion Center", "Knossos Archaeological", "Archanes Vineyard",
            "Peza Wine Region", "Tylissos Ancient", "Agios Nikolaos Bay",
            "Elounda Resort", "Spinalonga Island", "Kritsa Village",
            "Ierapetra Southern", "Sitia Eastern", "Zakros Gorge",
            "Vai Palm Beach", "Malia Ancient", "Ammoudara Beach",
            "Hersonissos Resort", "Stalis Coastal", "Kastelli Pediada",
            "Arkalochori Village", "Thrapsano Pottery",
        ],
    },
    "lesvos": {
        "name": "Lesvos",
        "display_region": "North Aegean",
        "center_lat": 39.1, "center_lon": 26.5,
        "bbox": [26.1, 38.7, 26.9, 39.5],
        "spread": 0.4,
        "areas": [
            "Mytilene Capital", "Molyvos Castle", "Petra Village",
            "Eressos Ancient", "Sigri Western", "Plomari Ouzo",
            "Agiassos Traditional", "Kalloni Bay", "Skala Kallonis",
            "Mantamados Monastery", "Thermi Hot Springs", "Moria Camp Zone",
            "Pamfylia Village", "Vatera Beach", "Skala Eresou",
            "Agiasos Forest", "Ipsilou Monastery", "Antissa Village",
            "Lisvori Village", "Polichnitos Salt",
        ],
    },
    "macedonia": {
        "name": "Macedonia",
        "display_region": "Northern Greece",
        "center_lat": 40.6, "center_lon": 22.9,
        "bbox": [22.35, 40.05, 23.45, 41.15],
        "spread": 0.55,
        "areas": [
            "Thessaloniki Port", "Kalamaria Suburb", "Panorama Hills",
            "Pylaia East", "Chortiatis Mountain", "Lagkadas Lake",
            "Langadikia Village", "Asprovalta Beach", "Nea Moudania",
            "Polygyros Capital", "Arnea Village", "Chalkidiki Peninsula",
            "Kassandra Coast", "Sithonia Coast", "Kavala Port",
            "Drama Plains", "Serres Agricultural", "Kilkis Border",
            "Katerini Coastal", "Pieria Plains",
        ],
    },
    "epirus": {
        "name": "Epirus",
        "display_region": "Northwestern Greece",
        "center_lat": 39.6, "center_lon": 20.8,
        "bbox": [20.3, 39.1, 21.3, 40.1],
        "spread": 0.5,
        "areas": [
            "Ioannina Lake", "Dodoni Ancient", "Metsovo Mountain",
            "Konitsa Bridge", "Zagori Villages", "Papingo Rock",
            "Vikos Gorge", "Monodendri View", "Kipi Bridges",
            "Aristi Village", "Preveza Coastal", "Nikopolis Ancient",
            "Arta Bridge", "Parga Seaside", "Sivota Bay",
            "Filiates Border", "Igoumenitsa Port", "Margariti Village",
            "Paramythia Hill", "Thesprotia Plains",
        ],
    },
    "dodecanese": {
        "name": "Dodecanese",
        "display_region": "South Aegean Islands",
        "center_lat": 36.4, "center_lon": 27.2,
        "bbox": [26.6, 35.8, 27.8, 37.0],
        "spread": 0.6,
        "areas": [
            "Kos Ancient Town", "Kardamena Resort", "Kefalos Bay",
            "Tingaki Beach", "Mastichari Port", "Patmos Monastery",
            "Skala Port", "Chora Village", "Grikos Bay", "Leros Island",
            "Lakki Port", "Agia Marina", "Kalymnos Sponge",
            "Pothia Capital", "Myrties Bay", "Nisyros Volcano",
            "Mandraki Crater", "Tilos Wildlife", "Symi Harbor",
            "Panormitis Monastery",
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
