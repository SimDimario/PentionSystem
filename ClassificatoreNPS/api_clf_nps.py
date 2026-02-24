from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi import FastAPI, Depends
from auth import verify_token
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

# Aggiungo al path la directory superiore
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import service_clf_nps
import uvicorn

class Spectra(BaseModel):
    spectra: list[list[float]]

    def to_numpy(self) -> np.ndarray:
        """Converte le liste JSON in un array numpy."""
        logger.debug("Converto input JSON in numpy array")
        return np.array(self.spectra, dtype=float)


app = FastAPI()

@app.post("/predict_dnn")
def predict_dnn(input_data: Spectra,user=Depends(verify_token)):
    logger.info("Ricevuta richiesta su /predict_dnn")
    try:
        mass_spectrum = input_data.to_numpy()
        logger.info(f"Shape input numpy: {mass_spectrum.shape}")

        logger.info("Invoco service_clf_nps.pipe_clf_dnn()")
        predictions = service_clf_nps.pipe_clf_dnn(mass_spectrum)
        logger.info("Classificazione DNN completata")

        logger.info(f"predictions: {predictions}")

        return JSONResponse(
            content={"predictions": predictions.tolist()},
            status_code=200
        )
    except Exception as e:
        logger.exception("Errore in /predict_dnn")
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )


@app.post("/predict_brf")
def predict_brf(input_data: Spectra,user=Depends(verify_token)):
    logger.info("Ricevuta richiesta su /predict_brf")
    try:
        mass_spectrum = input_data.to_numpy()
        logger.info(f"Shape input numpy: {mass_spectrum.shape}")

        logger.info("Invoco service_clf_nps.pipe_clf_brf()")
        predictions = service_clf_nps.pipe_clf_brf(mass_spectrum)
        logger.info("Classificazione BRF completata")

        return JSONResponse(
            content={"predictions": predictions.tolist()},
            status_code=200
        )
    except Exception as e:
        logger.exception("Errore in /predict_brf")
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )

"""if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)"""
