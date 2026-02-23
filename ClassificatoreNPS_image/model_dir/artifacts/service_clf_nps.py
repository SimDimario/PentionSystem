import numpy as np
import joblib
import os
import logging
from tensorflow.keras.models import load_model  # type: ignore

# Configurazione logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

base_dir = os.path.dirname(__file__)
dnn_path = os.path.join(base_dir, 'model', 'dnn_spectra_version.keras')
brf_path = os.path.join(base_dir, 'model', 'balanced_random_forest_brf.pkl')
scaler_path = os.path.join(base_dir, 'model', 'scale_dnn.pkl')

logger.info("Caricamento modelli...")
dnn_clf = load_model(dnn_path)
logger.info("DNN caricata")
brf_clf = joblib.load(brf_path)
logger.info("BRF caricato")
scaler_dnn = joblib.load(scaler_path)
logger.info("Scaler DNN caricato")

mz_range = np.arange(1, 601)

legends = {
    0: 'Cathinone analogues',
    1: 'Cannabinoid analogues',
    2: 'Phenethylamine analogues',
    3: 'Piperazine analogues',
    4: 'Tryptamine analogues',
    5: 'Fentanyl analogues',
    6: 'Other compounds'
}


def _compute_features(spectrum):
    """Estrae 13 caratteristiche dallo spettro di massa."""
    logger.debug("Calcolo feature dello spettro")

    peaks = [(mz, intensity) for mz, intensity in zip(mz_range, spectrum) if intensity > 0]

    if not peaks:
        logger.warning("Spettro senza picchi")
        return [np.nan] * 13

    mz_values, intensities = zip(*peaks)
    mz_values = np.array(mz_values)
    intensities = np.array(intensities)

    base_peak_idx = np.argmax(intensities)
    base_peak_mass = mz_values[base_peak_idx]

    base_prox = np.min(np.abs(mz_values - base_peak_mass)[np.abs(mz_values - base_peak_mass) != 0]) if len(
        mz_values) > 1 else 0.0
    max_mass = np.max(mz_values)
    max_prox = np.min(np.abs(mz_values - max_mass)[np.abs(mz_values - max_mass) != 0]) if len(mz_values) > 1 else 0.0
    num_peaks = len(peaks)
    intensity_mean = np.mean(intensities)
    intensity_std = np.std(intensities)
    intensity_density = np.max(intensities) / num_peaks
    mass_mean = np.mean(mz_values)
    mass_std = np.std(mz_values)
    mass_density = max_mass / num_peaks

    diffs = np.abs(np.subtract.outer(mz_values, mz_values))
    diffs = diffs[np.triu_indices(len(diffs), k=1)]
    diff_counts = np.bincount(np.round(diffs).astype(int))
    ppmd = np.argmax(diff_counts) if len(diff_counts) > 0 else 0
    mean_ppmd = np.mean(diffs) if len(diffs) > 0 else 0

    return [
        base_peak_mass, base_prox, max_mass, max_prox,
        num_peaks, intensity_mean, intensity_std, intensity_density,
        mass_mean, mass_std, mass_density, ppmd, mean_ppmd
    ]


def pipe_clf_dnn(spectra: np.ndarray):
    if spectra is None or len(spectra) == 0:
        raise ValueError("Input spectra is empty or None")
    if spectra.ndim != 2:
        raise ValueError(f"Expected 2D array (n_samples, n_features), got shape {spectra.shape}")

    try:
        logger.info("Inizio predizione DNN")
        spectra_scaled = scaler_dnn.transform(spectra)
        logger.debug("Scaler applicato")
        predictions_raw = dnn_clf.predict(spectra_scaled, verbose=0)
        predictions = [legends.get(int(np.argmax(p)), f"Classe {int(np.argmax(p))}") for p in predictions_raw]
        logger.info("Predizione DNN completata")
    except Exception as e:
        logger.exception("Errore durante la predizione DNN")
        raise RuntimeError(f"Error during DNN prediction: {str(e)}")

    return np.array(predictions)


def pipe_clf_brf(spectra: np.ndarray):
    if spectra is None or len(spectra) == 0:
        raise ValueError("Input spectra is empty or None")
    if spectra.ndim != 2:
        raise ValueError(f"Expected 2D array (n_samples, n_features), got shape {spectra.shape}")

    predictions = []
    try:
        logger.info("Inizio predizione BRF")
        for idx, spectrum in enumerate(spectra):
            logger.debug(f"Calcolo feature spettro {idx}")
            features = np.array(_compute_features(spectrum))
            selected_indices = [0, 1, 3, 5, 6, 7, 8, 10, 11]
            features = features[selected_indices].reshape(1, -1)
            prediction = brf_clf.predict(features)
            predictions.append(legends.get(prediction[0], f"Classe {prediction[0]}"))
        logger.info("Predizione BRF completata")
    except Exception as e:
        logger.exception("Errore durante la predizione BRF")
        raise RuntimeError(f"Error during BRF prediction: {str(e)}")

    return np.array(predictions)
