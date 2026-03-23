"""Project configuration and constants."""
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

# Finike region coordinates
FINIKE_LAT = 36.2975
FINIKE_LON = 30.1483

# Bounding box for Finike citrus region (approximate)
FINIKE_BBOX = {
    "min_lat": 36.20,
    "max_lat": 36.40,
    "min_lon": 30.00,
    "max_lon": 30.30,
}

# Hal market codes/names
HAL_MARKETS = {
    "finike": "Finike",
    "antalya": "Antalya",
    "mersin": "Mersin",
    "adana": "Adana",
    "istanbul": "İstanbul",
}

# Orange varieties in Finike
ORANGE_VARIETIES = [
    "Washington Navel",
    "Valencia",
    "Yafa (Jaffa)",
    "Kan Portakalı (Blood Orange)",
]

# Harvest season (approximate)
HARVEST_START_MONTH = 11  # November
HARVEST_END_MONTH = 5     # May

# Competitor countries
COMPETITOR_COUNTRIES = {
    "egypt": {"harvest_start": 11, "harvest_end": 5, "currency": "EGP"},
    "morocco": {"harvest_start": 11, "harvest_end": 6, "currency": "MAD"},
    "spain": {"harvest_start": 11, "harvest_end": 6, "currency": "EUR"},
    "south_africa": {"harvest_start": 6, "harvest_end": 10, "currency": "ZAR"},
    "italy": {"harvest_start": 12, "harvest_end": 4, "currency": "EUR"},
    "greece": {"harvest_start": 11, "harvest_end": 5, "currency": "EUR"},
    "argentina": {"harvest_start": 5, "harvest_end": 10, "currency": "ARS"},
    "brazil": {"harvest_start": 6, "harvest_end": 12, "currency": "BRL"},
}

# Key importing countries for Turkish oranges
IMPORT_MARKETS = ["russia", "iraq", "eu", "saudi_arabia", "uae", "ukraine", "belarus"]

# Prediction horizons (days)
PREDICTION_HORIZONS = [30, 60, 90]

# Feature engineering
ROLLING_WINDOWS = [7, 14, 30, 60, 90]
LAG_DAYS = [1, 7, 14, 30, 60, 90]

# Weather API
OPEN_METEO_BASE_URL = "https://api.open-meteo.com/v1"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1"
WEATHER_VARIABLES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "relative_humidity_2m_mean",
    "wind_speed_10m_max",
]

# Scenario alert thresholds
ALERT_THRESHOLDS = {
    "frost_mild": -2.0,       # °C — mild frost
    "frost_severe": -5.0,     # °C — severe frost
    "drought_days": 30,       # consecutive days without rain
    "ndvi_drop_pct": 15.0,    # % drop vs seasonal average
    "fx_spike_pct": 10.0,     # % monthly FX move
    "competitor_export_surge_pct": 20.0,  # % YoY export increase
}
