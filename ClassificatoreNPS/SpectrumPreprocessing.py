import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import logging

logger = logging.getLogger(__name__)

mz_range = np.arange(1, 601)


# ---------------------------------------------------------------------------
# Modelli di fitting
# ---------------------------------------------------------------------------

def _gaussian(x, amplitude, mu, sigma):
    """Gaussiana standard."""
    return amplitude * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _exponential_tail(x, a, b):
    """Coda esponenziale crescente (interferente di fondo)."""
    return a * np.exp(b * x)


# ---------------------------------------------------------------------------
# Step 1 — Stima e rimozione della coda destra
# ---------------------------------------------------------------------------

def _remove_right_tail(mz: np.ndarray, intensities: np.ndarray,
                       tail_percentile: float = 85.0):
    """
    Identifica la coda destra dello spettro (interferente) tramite fit
    esponenziale sulla regione ad alto m/z e la sottrae.

    Parametri
    ----------
    mz : array m/z (asse x)
    intensities : array intensità (asse y)
    tail_percentile : percentile di m/z oltre il quale si cerca la coda

    Ritorna
    -------
    cleaned : intensità dopo sottrazione della coda (clip a 0)
    tail_model : valore del modello esponenziale su tutto il range
    tail_start_idx : indice da cui inizia la coda
    """
    # Individua dove inizia la crescita esponenziale (coda destra)
    tail_start_mz = np.percentile(mz, tail_percentile)
    tail_mask = mz >= tail_start_mz

    if tail_mask.sum() < 5:
        logger.debug("Punti insufficienti per il fit della coda, skip")
        return intensities.copy(), np.zeros_like(intensities), len(mz)

    mz_tail = mz[tail_mask]
    int_tail = intensities[tail_mask]

    # Evita valori zero/negativi per il fit esponenziale
    int_tail_safe = np.clip(int_tail, 1e-6, None)

    try:
        # Stima iniziale: regressione lineare su log(y) = log(a) + b*x
        log_y = np.log(int_tail_safe)
        coeffs = np.polyfit(mz_tail, log_y, 1)
        b0, log_a0 = coeffs
        a0 = np.exp(log_a0)

        popt, _ = curve_fit(
            _exponential_tail, mz_tail, int_tail_safe,
            p0=[a0, b0], maxfev=5000
        )
        tail_model = _exponential_tail(mz, *popt)

        # Sottrai la coda solo nella regione della coda
        tail_contribution = np.zeros_like(intensities, dtype=float)
        tail_contribution[tail_mask] = tail_model[tail_mask]

        cleaned = intensities.astype(float) - tail_contribution
        cleaned = np.clip(cleaned, 0, None)

        tail_start_idx = int(np.searchsorted(mz, tail_start_mz))
        logger.debug(f"Coda stimata: a={popt[0]:.4f}, b={popt[1]:.6f}, "
                     f"inizio a m/z={tail_start_mz:.1f}")
        return cleaned, tail_model, tail_start_idx

    except Exception as e:
        logger.warning(f"Fit coda fallito: {e} — restituisco spettro originale")
        return intensities.copy(), np.zeros_like(intensities), len(mz)


# ---------------------------------------------------------------------------
# Step 2 — Fit gaussiano sul residuo pulito
# ---------------------------------------------------------------------------

def _fit_gaussian_peak(mz: np.ndarray, intensities: np.ndarray):
    """
    Applica un fit gaussiano allo spettro pulito per identificare
    il picco principale (base peak reale).

    Ritorna
    -------
    popt : (amplitude, mu, sigma) oppure None se il fit fallisce
    fitted_curve : array con la gaussiana fittata su tutto il range
    """
    if intensities.max() == 0:
        return None, np.zeros_like(intensities)

    # Stima iniziale: picco più alto come centro gaussiana
    amp0 = intensities.max()
    mu0 = mz[np.argmax(intensities)]
    sigma0 = 10.0  # stima iniziale larghezza ~10 Da

    try:
        popt, _ = curve_fit(
            _gaussian, mz, intensities,
            p0=[amp0, mu0, sigma0],
            bounds=([0, mz.min(), 1], [np.inf, mz.max(), 200]),
            maxfev=10000
        )
        fitted_curve = _gaussian(mz, *popt)
        logger.debug(f"Gaussiana fittata: μ={popt[1]:.2f}, σ={popt[2]:.2f}, "
                     f"amp={popt[0]:.1f}")
        return popt, fitted_curve

    except Exception as e:
        logger.warning(f"Fit gaussiano fallito: {e}")
        return None, np.zeros_like(intensities)


# ---------------------------------------------------------------------------
# Step 3 — Rimozione rumore sotto soglia relativa
# ---------------------------------------------------------------------------

def _remove_noise(intensities: np.ndarray,
                  threshold_pct: float = 1.0) -> np.ndarray:
    """
    Azzera tutti i picchi con intensità < threshold_pct % del base peak.
    Corrisponde alla soglia standard in spettrometria di massa.

    Parametri
    ----------
    threshold_pct : soglia percentuale (default 1% del picco massimo)
    """
    if intensities.max() == 0:
        return intensities.copy()
    threshold = (threshold_pct / 100.0) * intensities.max()
    cleaned = intensities.copy()
    cleaned[cleaned < threshold] = 0.0
    return cleaned


# ---------------------------------------------------------------------------
# Pipeline completa di preprocessing
# ---------------------------------------------------------------------------

def preprocess_spectrum(spectrum: np.ndarray,
                        noise_threshold_pct: float = 1.0,
                        tail_percentile: float = 85.0,
                        apply_gaussian_fit: bool = True) -> dict:
    """
    Pipeline completa di preprocessing per uno spettro di massa:

      1. Rimozione rumore sotto soglia relativa
      2. Stima e sottrazione della coda destra (interferente esponenziale)
      3. Fit gaussiano sul residuo per validare il picco principale

    Parametri
    ----------
    spectrum : array 1D di lunghezza 600 (intensità per m/z 1-600)
    noise_threshold_pct : % del base peak sotto cui azzerare (default 1%)
    tail_percentile : percentile m/z per iniziare la ricerca della coda
    apply_gaussian_fit : se True applica il fit gaussiano finale

    Ritorna
    -------
    dict con:
      - 'cleaned_spectrum'   : spettro pulito (array 600)
      - 'tail_model'         : modello della coda stimata (array 600)
      - 'gaussian_params'    : (amplitude, mu, sigma) o None
      - 'gaussian_curve'     : curva gaussiana fittata (array 600)
      - 'tail_start_idx'     : indice di inizio coda
      - 'noise_threshold'    : soglia di rumore usata (valore assoluto)
    """
    if len(spectrum) != 600:
        raise ValueError(f"Lo spettro deve avere 600 elementi (m/z 1-600), "
                         f"ricevuto: {len(spectrum)}")

    mz = mz_range.astype(float)
    intensities = np.array(spectrum, dtype=float)

    # --- Step 1: rimozione rumore ---
    denoised = _remove_noise(intensities, noise_threshold_pct)
    noise_threshold_abs = (noise_threshold_pct / 100.0) * intensities.max()
    logger.debug(f"Rumore rimosso: soglia={noise_threshold_abs:.2f}")

    # --- Step 2: rimozione coda destra ---
    cleaned, tail_model, tail_start_idx = _remove_right_tail(
        mz, denoised, tail_percentile
    )

    # --- Step 3: fit gaussiano sul residuo ---
    gaussian_params, gaussian_curve = None, np.zeros(600)
    if apply_gaussian_fit:
        gaussian_params, gaussian_curve = _fit_gaussian_peak(mz, cleaned)

    return {
        'cleaned_spectrum': cleaned,
        'tail_model': tail_model,
        'gaussian_params': gaussian_params,   # (amplitude, mu, sigma)
        'gaussian_curve': gaussian_curve,
        'tail_start_idx': tail_start_idx,
        'noise_threshold': noise_threshold_abs,
    }


# ---------------------------------------------------------------------------
# Funzione leggera — solo denoising (per DNN)
# ---------------------------------------------------------------------------

def denoise_spectrum(spectrum: np.ndarray,
                     noise_threshold_pct: float = 1.0) -> np.ndarray:
    """
    Rimuove solo il rumore sotto soglia relativa, senza toccare la coda.
    Usare per la DNN: gli spettri NPS hanno picchi caratteristici ad alto
    m/z che NON devono essere rimossi dalla sottrazione della coda.

    Parametri
    ----------
    spectrum : array 1D di lunghezza 600
    noise_threshold_pct : % del base peak sotto cui azzerare (default 1%)

    Ritorna
    -------
    array 1D denoised (stessa lunghezza dell'input)
    """
    if len(spectrum) != 600:
        raise ValueError(f"Lo spettro deve avere 600 elementi, ricevuto: {len(spectrum)}")
    return _remove_noise(np.array(spectrum, dtype=float), noise_threshold_pct)


def denoise_batch(spectra: np.ndarray,
                  noise_threshold_pct: float = 1.0) -> np.ndarray:
    """
    Applica solo il denoising a un batch di spettri (n_samples, 600).
    Versione efficiente per la DNN.

    Ritorna
    -------
    np.ndarray (n_samples, 600) — spettri denoised
    """
    if spectra.ndim != 2 or spectra.shape[1] != 600:
        raise ValueError(f"Atteso array (n_samples, 600), ricevuto: {spectra.shape}")

    result = spectra.astype(float).copy()
    for i, spectrum in enumerate(result):
        if spectrum.max() > 0:
            threshold = (noise_threshold_pct / 100.0) * spectrum.max()
            result[i][result[i] < threshold] = 0.0

    logger.info(f"Denoising completato per {len(spectra)} spettri")
    return result


# ---------------------------------------------------------------------------
# Pipeline completa con coda — mantenuta per uso diagnostico
# ---------------------------------------------------------------------------

def preprocess_batch(spectra: np.ndarray,
                     noise_threshold_pct: float = 1.0,
                     tail_percentile: float = 85.0,
                     apply_gaussian_fit: bool = False) -> np.ndarray:
    """
    Applica il preprocessing a un batch di spettri (n_samples, 600).

    Per il batch, il fit gaussiano è disabilitato di default per efficienza;
    viene usato solo il denoising + sottrazione coda.

    Ritorna
    -------
    cleaned_batch : np.ndarray (n_samples, 600) — spettri puliti
    """
    if spectra.ndim != 2 or spectra.shape[1] != 600:
        raise ValueError(f"Atteso array (n_samples, 600), ricevuto: {spectra.shape}")

    cleaned_batch = np.zeros_like(spectra, dtype=float)
    for i, spectrum in enumerate(spectra):
        try:
            result = preprocess_spectrum(
                spectrum,
                noise_threshold_pct=noise_threshold_pct,
                tail_percentile=tail_percentile,
                apply_gaussian_fit=apply_gaussian_fit
            )
            cleaned_batch[i] = result['cleaned_spectrum']
        except Exception as e:
            logger.warning(f"Preprocessing fallito per spettro {i}: {e} "
                           f"— uso spettro originale")
            cleaned_batch[i] = spectrum

    logger.info(f"Preprocessing completato per {len(spectra)} spettri")
    return cleaned_batch