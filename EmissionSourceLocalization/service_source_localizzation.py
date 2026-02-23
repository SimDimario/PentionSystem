# service_source_localizzation.py
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import joblib
import logging
import os

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "models/")

# Caricamento modello e scaler
logger.info(f"Loading model from {MODEL_PATH}")
model = joblib.load(os.path.join(MODEL_PATH, "emission_source_model.pkl"))
logger.info("Model loaded successfully.")

logger.info(f"Loading scaler from {MODEL_PATH}")
scaler = joblib.load(os.path.join(MODEL_PATH, "scaler.pkl"))
logger.info("Scaler loaded successfully.")

def estrai_feature(time, conc, window=None, spike_height=None):
   
    time = np.array(time)
    conc = np.array(conc)

    # Concentrazione massima e tempo del massimo
    C_max = np.max(conc)
    idx_max = np.argmax(conc)
    t_peak = time[idx_max]

    # Primo picco
    if spike_height is None:
        spike_height = 0.1 * C_max
    peaks, _ = find_peaks(conc, height=spike_height)
    t_first_peak = time[peaks[0]] if len(peaks) > 0 else np.nan

    # Rise / fall rate
    rise_rate = (C_max - conc[0]) / (t_peak - time[0] + 1e-6)
    fall_rate = (C_max - conc[-1]) / (time[-1] - t_peak + 1e-6)

    # Media e deviazione standard
    if window is not None:
        mask = (time >= window[0]) & (time <= window[1])
        conc_window = conc[mask]
    else:
        conc_window = conc
    mean_val = np.mean(conc_window)
    std_val = np.std(conc_window)

    # Area sotto la curva
    auc = np.trapezoid(conc, time)

    # Durata plume
    above = np.where(conc > spike_height)[0]
    if len(above) > 0:
        t_plume_start = time[above[0]]
        t_plume_end = time[above[-1]]
        plume_duration = t_plume_end - t_plume_start
    else:
        plume_duration = 0.0

    # Spike count / frequency
    spike_count = len(peaks)
    total_duration = time[-1] - time[0]
    spike_freq = spike_count / total_duration if total_duration > 0 else 0.0

    return {
        "C_max": C_max,
        "t_peak": t_peak,
        "t_first_peak": t_first_peak,
        "mean": mean_val,
        "std": std_val,
        "AUC": auc,
        "rise_rate": rise_rate,
        "fall_rate": fall_rate,
        "plume_duration": plume_duration,
        "spike_count": spike_count,
        "spike_frequency": spike_freq
    }

def predict_source(sensors: list, n_sensor_operating: int):

    logger.info(f"Predicting source for {len(sensors)} sensor readings")

    df = pd.DataFrame([s.dict() for s in sensors])
    logger.info(df.info)

    if df.empty:
        logger.warning("Nessun dato sensori disponibile!")
        return {"predicted_location": [(None, None)]}

    agg_features_list = []
    for sensor_id, group in df.groupby("sensor_id"):
        times = group["time"].to_list()
        concs = group["conc"].to_list()

        feat = estrai_feature(times, concs)
        first_row = group.iloc[0]
        feat.update({
            "wind_dir_x": first_row["wind_dir_x"],
            "wind_dir_y": first_row["wind_dir_y"],
            "wind_speed": first_row["wind_speed"],
            "wind_type": first_row["wind_type"],
            "n_sens_valid": n_sensor_operating,
        })
        agg_features_list.append(feat)

    X_input = pd.DataFrame(agg_features_list).fillna(0)

    cols_scaler = scaler.feature_names_in_
    logger.info(cols_scaler)
    X_input = X_input.reindex(columns=cols_scaler, fill_value=0)

    X_scaled = scaler.transform(X_input)
    X_scaled_df = pd.DataFrame(X_scaled, columns=cols_scaler)

    logger.info(f"Input features shape after scaling: {X_scaled_df.shape}")

    for col in ["simulation_id", "source_x", "source_y"]:
        if col not in X_scaled_df.columns:
            X_scaled_df[col] = 0

    X_scaled_df = X_scaled_df[model.feature_names_in_]

    y_pred = model.predict(X_scaled_df)

    x,y = y_pred[0]

    logger.info(f"Prediction completed: {x},{y}")
    return x, y