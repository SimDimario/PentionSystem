import os
from pathlib import Path
import sys
from typing import Optional
from pydantic import BaseModel
import numpy as np
import mlflow
from mlflow.pyfunc import PythonModel


MLFLOW_TRACKING_URI = "http://localhost:5000"
MLFLOW_MODEL_NAME = "correction_dispertion_model"
MLFLOW_EXPERIMENT_NAME = "correction_dispertion_experiment"


class DispersionInput(BaseModel):
    wind_speed: float
    wind_dir: list[float]
    concentration_map: str
    building_map: str
    global_features: Optional[list[float]] = None


class CorrectionDispersionModel(PythonModel):

    def __init__(self):

        self.model_dir = None
        self.correct_dispersion = None

    def load_context(self, context):

        # path where the model artifacts are stored
        self.model_dir = Path(
            context.artifacts["project_folder"].replace("\\", "/")
        ).parent

        os.chdir(self.model_dir)
        sys.path.append(os.path.dirname(os.getcwd()))

        print(f"Current working directory: {os.getcwd()}")

        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

        from CorrectionDispersion.service_correction import correct_dispersion

        self.correct_dispersion = correct_dispersion

    def predict(
        self, context, model_input: list[DispersionInput], params=None
    ) -> list[list[float]]:

        payload = model_input[0]

        conc_map = np.load(payload.concentration_map)
        build_map = np.load(payload.building_map)
        glob_feat = (
            np.array(payload.global_features, dtype=np.float32)
            if payload.global_features
            else None
        )

        correction_map = self.correct_dispersion(
            payload.wind_dir, payload.wind_speed, conc_map, build_map, glob_feat
        )

        return correction_map.tolist()


def log_model():

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    print("Logging model to MLflow...")

    with mlflow.start_run() as run:

        model = CorrectionDispersionModel()

        # to make it work, create a folder named CorrectionDispersion with
        # only the required files to make it work
        model_info = mlflow.pyfunc.log_model(
            python_model=model,
            registered_model_name=MLFLOW_MODEL_NAME,
            artifacts={
                "project_folder": "CorrectionDispersion",
            },
        )

    print(
        f"Model logged successfully. Run ID: {run.info.run_id}, Model URI: {model_info.model_uri}"
    )


def test_model():

    print("Testing model prediction...")

    model = mlflow.pyfunc.load_model(model_uri=f"models:/{MLFLOW_MODEL_NAME}/latest")

    conc_map_path = os.path.abspath("test_simulation_data/C1.npy")
    build_map_path = os.path.abspath("test_simulation_data/binary_map.npy")

    input = [
        DispersionInput(
            wind_speed=5.0,
            wind_dir=[1.0, 0.0],
            concentration_map=conc_map_path,
            building_map=build_map_path,
        )
    ]

    result = model.predict(input)
    print("Test result:", result)


if __name__ == "__main__":
    log_model()
    # test_model()
