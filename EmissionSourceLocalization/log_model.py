import os
from pathlib import Path
import sys
import logging
from typing import List, Optional
from pydantic import BaseModel
import numpy as np
import mlflow
from mlflow.pyfunc import PythonModel


MLFLOW_TRACKING_URI = "http://localhost:5000"
MLFLOW_MODEL_NAME = "emission_source_localization_model"
MLFLOW_EXPERIMENT_NAME = "emission_source_localization_experiment"


class SensorData(BaseModel):
    sensor_id: int
    sensor_is_fault: bool
    time: Optional[float] = None
    conc: Optional[float] = None
    wind_dir_x: Optional[float] = None
    wind_dir_y: Optional[float] = None
    wind_speed: Optional[float] = None
    wind_type: Optional[int] = None


class PredictRequest(BaseModel):
    payload_sensors: List[SensorData]
    n_sensor_operating: int


class PredictResponse(BaseModel):
    status: int
    x: float
    y: float


class EmissionSourceLocalizationModel(PythonModel):

    def __init__(self):

        self.model_dir = None
        self.logger = None
        self.predict_source = None

    def load_context(self, context):

        self.model_dir = Path(
            context.artifacts["main_script"].replace("\\", "/")
        ).parent  # "/opt/ml/model/artifacts"

        os.chdir(self.model_dir)
        sys.path.append(os.path.dirname(os.getcwd()))

        sys.path.append(
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        )

        from service_source_localizzation import predict_source

        self.predict_source = predict_source

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
        )
        self.logger = logging.getLogger(__name__)

    def predict(
        self, context, model_input: list[PredictRequest], params=None
    ) -> PredictResponse:

        request = model_input[0]
        self.logger.info(
            "Ricevuta richiesta /predict_source_raw con %s record",
            len(request.payload_sensors),
        )

        try:
            x, y = self.predict_source(
                request.payload_sensors, request.n_sensor_operating
            )
            self.logger.info("Predizione sorgente completata")

            return PredictResponse(
                status=200,
                x=x,
                y=y,
            )

        except Exception as e:
            self.logger.exception("Errore durante la predizione della sorgente")
            raise e


def log_model():

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    print("Logging model to MLflow...")

    with mlflow.start_run() as run:

        model = EmissionSourceLocalizationModel()

        model_info = mlflow.pyfunc.log_model(
            python_model=model,
            registered_model_name=MLFLOW_MODEL_NAME,
            artifacts={
                "main_script": "service_source_localizzation.py",
                "model_dir": "models",
            },
        )

    print(
        f"Model logged successfully. Run ID: {run.info.run_id}, Model URI: {model_info.model_uri}"
    )


def test_model():

    print("Testing model prediction...")

    model = mlflow.pyfunc.load_model(model_uri=f"models:/{MLFLOW_MODEL_NAME}/latest")

    input = [
        PredictRequest(
            payload_sensors=[
                SensorData(
                    sensor_id=1,
                    sensor_is_fault=False,
                    time=0.0,
                    conc=0.0,
                    wind_dir_x=0.0,
                    wind_dir_y=0.0,
                    wind_speed=0.0,
                    wind_type=0,
                )
            ],
            n_sensor_operating=1,
        )
    ]

    result = model.predict(input)
    print("Test result:", result)


if __name__ == "__main__":
    log_model()
    test_model()
