from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.responses import JSONResponse
import os
import sys
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import service_clf_nps as service_clf_nps
import uvicorn


# ---------------------------------------------------------------------------
# Schema input
# ---------------------------------------------------------------------------

class Spectra(BaseModel):
    spectra: list[list[float]]

    # Parametri preprocessing interferenti (opzionali, con default)
    apply_preprocessing: bool = Field(
        default=True,
        description="Se True, applica rimozione rumore + sottrazione coda gaussiana"
    )
    noise_threshold_pct: float = Field(
        default=1.0,
        ge=0.0, le=10.0,
        description="Soglia di rumore come % del base peak (default 1%)"
    )
    tail_percentile: float = Field(
        default=85.0,
        ge=50.0, le=99.0,
        description="Percentile m/z da cui inizia la ricerca della coda destra (default 85%)"
    )

    def to_numpy(self) -> np.ndarray:
        logger.debug("Converto input JSON in numpy array")
        return np.array(self.spectra, dtype=float)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="NPS Classifier API",
    description=(
        "Classificazione di spettri di massa per New Psychoactive Substances. "
        "Supporta preprocessing automatico degli interferenti tramite "
        "fit gaussiano e sottrazione della coda destra."
    ),
    version="2.0.0"
)


@app.post(
    "/predict_dnn",
    summary="Classificazione DNN",
    description=(
        "Classifica gli spettri di massa tramite Deep Neural Network. "
        "Il preprocessing degli interferenti è applicato di default."
    )
)
def predict_dnn(input_data: Spectra):
    logger.info("Ricevuta richiesta su /predict_dnn")
    try:
        mass_spectrum = input_data.to_numpy()
        logger.info(f"Shape input: {mass_spectrum.shape} | "
                    f"preprocessing={input_data.apply_preprocessing} | "
                    f"noise_thr={input_data.noise_threshold_pct}% | "
                    f"tail_pct={input_data.tail_percentile}")

        predictions = service_clf_nps.pipe_clf_dnn(
            mass_spectrum,
            apply_preprocessing=input_data.apply_preprocessing,
            noise_threshold_pct=input_data.noise_threshold_pct,
            tail_percentile=input_data.tail_percentile
        )
        logger.info(f"Predizioni DNN: {predictions}")

        return JSONResponse(
            content={"predictions": predictions.tolist()},
            status_code=200
        )
    except Exception as e:
        logger.exception("Errore in /predict_dnn")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post(
    "/predict_brf",
    summary="Classificazione BRF",
    description=(
        "Classifica gli spettri di massa tramite Balanced Random Forest. "
        "Il preprocessing include fit gaussiano per identificare il picco reale "
        "prima dell'estrazione delle feature."
    )
)
def predict_brf(input_data: Spectra):
    logger.info("Ricevuta richiesta su /predict_brf")
    try:
        mass_spectrum = input_data.to_numpy()
        logger.info(f"Shape input: {mass_spectrum.shape} | "
                    f"preprocessing={input_data.apply_preprocessing} | "
                    f"noise_thr={input_data.noise_threshold_pct}% | "
                    f"tail_pct={input_data.tail_percentile}")

        predictions = service_clf_nps.pipe_clf_brf(
            mass_spectrum,
            apply_preprocessing=input_data.apply_preprocessing,
            noise_threshold_pct=input_data.noise_threshold_pct,
            tail_percentile=input_data.tail_percentile
        )
        logger.info(f"Predizioni BRF: {predictions}")

        return JSONResponse(
            content={"predictions": predictions.tolist()},
            status_code=200
        )
    except Exception as e:
        logger.exception("Errore in /predict_brf")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post(
    "/preprocess_only",
    summary="Solo preprocessing",
    description=(
        "Restituisce gli spettri dopo rimozione degli interferenti, "
        "senza classificare. Utile per debug e validazione del preprocessing."
    )
)
def preprocess_only(input_data: Spectra):
    """
    Endpoint di diagnostica: restituisce gli spettri puliti e i parametri
    della gaussiana fittata (mu, sigma) per ogni spettro.
    Utile per verificare il comportamento del preprocessing.
    """
    logger.info("Ricevuta richiesta su /preprocess_only")
    try:
        from SpectrumPreprocessing import preprocess_spectrum

        mass_spectra = input_data.to_numpy()
        results = []

        for i, spectrum in enumerate(mass_spectra):
            result = preprocess_spectrum(
                spectrum,
                noise_threshold_pct=input_data.noise_threshold_pct,
                tail_percentile=input_data.tail_percentile,
                apply_gaussian_fit=True
            )
            gp = result['gaussian_params']
            results.append({
                "spectrum_index": i,
                "cleaned_spectrum": result['cleaned_spectrum'].tolist(),
                "tail_start_idx": result['tail_start_idx'],
                "noise_threshold": result['noise_threshold'],
                "gaussian_mu": float(gp[1]) if gp is not None else None,
                "gaussian_sigma": float(gp[2]) if gp is not None else None,
                "gaussian_amplitude": float(gp[0]) if gp is not None else None,
            })

        return JSONResponse(content={"results": results}, status_code=200)

    except Exception as e:
        logger.exception("Errore in /preprocess_only")
        return JSONResponse(content={"error": str(e)}, status_code=500)


"""if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)"""