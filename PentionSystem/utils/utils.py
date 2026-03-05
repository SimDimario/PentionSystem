import gc
import os
from datetime import datetime
import numpy as np
import requests

from config import OPEN_METEO_URL


def random_position(free_cells):
    idx = np.random.choice(len(free_cells))
    y, x = free_cells[idx]
    return float(y), float(x)


def clean_tmp_files():
    tmp_files = ["/tmp/C1.npy", "/tmp/binary_map.npy"]

    for f in tmp_files:
        try:
            if os.path.exists(f):
                os.remove(f)
        except Exception as e:
            print(f"[cleanup] Cannot remove {f}: {e}")

    gc.collect()


def get_meteo(lat, lon):
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ["windspeed_10m", "winddirection_10m", "relativehumidity_2m"],
        "timezone": "UTC",
    }

    response = requests.get(OPEN_METEO_URL, params=params)
    response.raise_for_status()

    data = response.json()
    hourly = data["hourly"]

    # ora corrente in UTC, arrotondata all’ora
    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)

    # parse robusto degli orari API
    times = [datetime.fromisoformat(t.replace("Z", "")) for t in hourly["time"]]

    # trova l’ora più vicina (niente ValueError)
    time_index = min(range(len(times)), key=lambda i: abs(times[i] - now))

    return {
        "wind_speed": hourly["windspeed_10m"][time_index],
        "wind_dir": hourly["winddirection_10m"][time_index],
        "RH": hourly["relativehumidity_2m"][time_index],
    }


def grid_index_to_coords(idx_x, idx_y, bounds, grid_size):
    """
    Converte indici della griglia in coordinate geografiche.

    Args:
        idx_x, idx_y: Indici nella griglia (0 a grid_size-1)
        bounds: (min_lon, min_lat, max_lon, max_lat)
        grid_size: Dimensione griglia (es. 500)

    Returns:
        (lon, lat): Coordinate geografiche
    """
    min_lon, min_lat, max_lon, max_lat = bounds

    lon = min_lon + (idx_x / (grid_size - 1)) * (max_lon - min_lon)
    lat = min_lat + (idx_y / (grid_size - 1)) * (max_lat - min_lat)

    return lon, lat
