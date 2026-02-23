from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import os
import sys
import logging
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from service_source_localizzation import predict_source

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

class SensorData(BaseModel):
    sensor_id: int
    sensor_is_fault: bool
    time: float | None
    conc: float | None
    wind_dir_x: float | None
    wind_dir_y: float | None
    wind_speed: float | None
    wind_type: int | None

class PredictRequest(BaseModel):
    payload_sensors: List[SensorData]
    n_sensor_operating: int

app = FastAPI()

@app.post("/predict_source_raw")
def predict_source_raw(request: PredictRequest):
    logger.info(f"Ricevuta richiesta /predict_source_raw con {len(request.payload_sensors)} record")

    try:
        x, y = predict_source(request.payload_sensors, request.n_sensor_operating)
        logger.info("Predizione sorgente completata")


        return  {
                "status": 200,
                "x": x,
                "y": y,
        }

    except Exception as e:
        logger.exception("Errore durante la predizione della sorgente")
        raise e

"""if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8003)"""
