import os
from pathlib import Path
import sys
import logging
from pydantic import BaseModel
import numpy as np
import mlflow
from mlflow.pyfunc import PythonModel


MLFLOW_TRACKING_URI = "http://localhost:5000"
MLFLOW_MODEL_NAME = "nps_classifier_model"
MLFLOW_EXPERIMENT_NAME = "nps_classifier_experiment"


class Spectra(BaseModel):
    spectra: list[list[float]]

    def to_numpy(self) -> np.ndarray:
        """Converte le liste JSON in un array numpy."""
        return np.array(self.spectra, dtype=float)


class PredictParams(BaseModel):
    predict_type: str


class PredictionRequest(BaseModel):
    spectra: Spectra
    params: PredictParams


class NPSClassifierModel(PythonModel):

    def __init__(self):

        self.model_dir = None
        self.logger = None
        self.service_clf_nps = None

    def load_context(self, context):

        # Normalize path to handle mixed separators from MLflow artifacts
        artifact_path = Path(context.artifacts["model"].replace("\\", "/"))
        self.model_dir = artifact_path.parent  # "/opt/ml/model/artifacts"

        os.chdir(self.model_dir)
        sys.path.append(os.path.dirname(os.getcwd()))

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
        )
        self.logger = logging.getLogger(__name__)

        self.logger.debug("Model directory: %s", self.model_dir)

        # Aggiungo al path la directory superiore
        sys.path.append(
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        )

        import service_clf_nps

        self.service_clf_nps = service_clf_nps

    def predict(
        self, context, model_input: list[PredictionRequest], params=None
    ) -> list[str]:

        api_input = model_input[0]
        input_data = api_input.spectra
        predict_type = api_input.params.predict_type

        if predict_type == "dnn":

            self.logger.info("Ricevuta richiesta su /predict_dnn")
            try:
                mass_spectrum = input_data.to_numpy()
                self.logger.info("Shape input numpy: %s", mass_spectrum.shape)

                self.logger.info("Invoco service_clf_nps.pipe_clf_dnn()")
                predictions = self.service_clf_nps.pipe_clf_dnn(mass_spectrum)
                self.logger.info("Classificazione DNN completata")

                self.logger.info("predictions: %s", predictions)

                return predictions.tolist()
            except Exception:
                self.logger.exception("Errore in /predict_dnn")
                raise Exception("Errore durante la classificazione DNN")

        elif predict_type == "brf":

            self.logger.info("Ricevuta richiesta su /predict_brf")
            try:
                mass_spectrum = input_data.to_numpy()
                self.logger.info("Shape input numpy: %s", mass_spectrum.shape)

                self.logger.info("Invoco service_clf_nps.pipe_clf_brf()")
                predictions = self.service_clf_nps.pipe_clf_brf(mass_spectrum)
                self.logger.info("Classificazione BRF completata")

                return predictions.tolist()
            except Exception:
                self.logger.exception("Errore in /predict_brf")
                raise Exception("Errore durante la classificazione BRF")

        else:
            self.logger.warning(
                "Valore non supportato per predict_type: %s", predict_type
            )
            raise ValueError(f"Valore non supportato per predict_type: {predict_type}")


def log_model():

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    print("Logging model to MLflow...")

    with mlflow.start_run() as run:

        model = NPSClassifierModel()

        model_info = mlflow.pyfunc.log_model(
            python_model=model,
            registered_model_name=MLFLOW_MODEL_NAME,
            artifacts={
                "service_clf_nps.py": "service_clf_nps.py",
                "model": "model",
            },
        )

    print(
        f"Model logged successfully. Run ID: {run.info.run_id}, Model URI: {model_info.model_uri}"
    )


def test_model():

    print("Testing model prediction...")

    model = mlflow.pyfunc.load_model(model_uri=f"models:/{MLFLOW_MODEL_NAME}/latest")

    input = [
        PredictionRequest(
            spectra=Spectra(spectra=np.random.rand(60, 600).tolist()),
            params=PredictParams(predict_type="dnn"),
        )
    ]

    result = model.predict(input)
    print("Test result:", result)


if __name__ == "__main__":
    log_model()
    test_model()
