from fastapi import FastAPI,Depends
import os, sys

from pydantic import BaseModel, Field
from typing import List, Tuple, Optional
import logging
import numpy as np
from auth import verify_token

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from gaussianPuff.gaussianModel import run_dispersion_model
from gaussianPuff.config import ModelConfig, WindType, StabilityType, PasquillGiffordStability, NPS, OutputType
from gaussianPuff.plot_utils import plot_plan_view
import uvicorn

# Configurazione logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

class ModelConfigRequest(BaseModel):
    days: int
    RH: float
    aerosol_type: str
    humidify: bool
    stability_profile: str
    stability_value: str
    wind_type: str
    wind_speed: float
    output: str
    stacks: List[Tuple[float, float, float, float]]
    dry_size: Optional[float] = 60e-9
    x_slice: Optional[int] = 26
    y_slice: Optional[int] = 1
    grid_size: Optional[int] = 500
    dispersion_model: str
    config_puff: Optional[dict] = None

class Payload(BaseModel):
    config: ModelConfigRequest
    bounds: List[float] = Field(..., min_items=4, max_items=4)

app = FastAPI()

@app.post("/start_simulation")
def start_simulation(payload: dict,user=Depends(verify_token)):
    logger.info("Ricevuta richiesta /start_simulation")
    try:
        raw_config = payload.get("config", {})
        logger.info(raw_config)

        wind_type= WindType.from_string(raw_config["wind_type"])
        stability_type = StabilityType.from_string(raw_config["stability_profile"])
        output_type= OutputType.from_string(raw_config["output"])
        stability_value=PasquillGiffordStability.from_string(raw_config["stability_value"])
        nps_type= NPS.from_string(raw_config["aerosol_type"])

        config = ModelConfig(
            days=raw_config["days"],
            RH=raw_config["RH"],
            aerosol_type=nps_type,
            humidify=raw_config["humidify"],
            stability_profile=stability_type,
            stability_value=stability_value,
            wind_type=wind_type,
            wind_speed=raw_config["wind_speed"],
            output=output_type,
            stacks=raw_config["stacks"],
            dry_size=raw_config["dry_size"],
            x_slice=raw_config["x_slice"],
            y_slice=raw_config["y_slice"],
            dispersion_model=raw_config["config_puff"],
            config_puff=raw_config["config_puff"]
        )

        bounds = payload.get("bounds", None)
        logger.info(f"Configurazione modello creata: {config}")
        logger.info(f"Bounds ricevuti: {bounds}")

        result = run_dispersion_model(config, bounds)
        logger.info("Simulazione completata")

        C1, (x, y, z), times, stability, wind_dir, stab_label, wind_label, puff = result
        logger.info("End gaussian model simulation")

        try:
            response = {
                "status": 200,
                "concentration": C1.tolist() if isinstance(C1, np.ndarray) else C1,
                "x": x.tolist() if isinstance(x, np.ndarray) else x,
                "y": y.tolist() if isinstance(y, np.ndarray) else y,
                "z": z.tolist() if isinstance(z, np.ndarray) else z,
                "times": times.tolist() if isinstance(times, np.ndarray) else times,
                "stability": stability.tolist() if isinstance(stability, np.ndarray) else str(stability),
                "wind_dir": wind_dir.tolist() if isinstance(wind_dir, np.ndarray) else wind_dir,
                "stab_label": str(stab_label),
                "wind_label": str(wind_label),
                "puff": puff.tolist() if isinstance(puff, np.ndarray) else (str(puff) if puff is not None else None)
            }

            for k, v in response.items():
                logger.info(f"{k}: {type(v)}")

        except Exception as e:
            logger.exception("Errore nella conversione dei dati")
            raise e

        logger.info(f"Simulazione completata: ritorno")
        return response

    except Exception as error:
        logger.exception("Errore durante la simulazione")
        raise error

"""if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8002)"""
