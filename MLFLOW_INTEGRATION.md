# MLflow deployment and Python environments setup

## Python Environments Setup

### ClassificatoreNPS

```bash
conda create -n ClassificatoreNPS python=3.10 -y
conda activate ClassificatoreNPS
pip install mlflow==3.9.0
pip install -r ClassificatoreNPS/requirements.txt
conda deactivate
```

### CorrectionDispersion

```bash
conda create -n CorrectionDispersion python=3.10 -y
conda activate CorrectionDispersion
pip install mlflow==3.9.0
pip install -r CorrectionDispersion/requirements.txt
conda deactivate
```

### EmissionSourceLocalization

```bash
conda create -n EmissionSourceLocalization python=3.10 -y
conda activate EmissionSourceLocalization
pip install mlflow==3.9.0
pip install -r EmissionSourceLocalization/requirements.txt
conda deactivate
```

## MLflow

Start MLflow server:

```bash
conda activate ClassificatoreNPS
mlflow server
```

Then, open the MLflow UI in your browser at [http://localhost:5000](http://localhost:5000).

### Start model logging scripts and export model as dockerfile

* ClassificatoreNPS:

  * log model:

    ```bash
    conda activate ClassificatoreNPS
    cd ClassificatoreNPS
    python log_model.py
    cd ..
    conda deactivate
    ```

  * generate dockerfile and run the model in a container:

    ```bash
    set MLFLOW_TRACKING_URI=http://localhost:5000
    conda activate ClassificatoreNPS
    mlflow models generate-dockerfile -m "models:/nps_classifier_model/latest" -d ClassificatoreNPS_image

    docker build -t nps_classifier_model:latest ClassificatoreNPS_image
    docker run --rm -it -p 8080:8080 -e MLFLOW_MODELS_WORKERS=1 nps_classifier_model:latest
    conda deactivate
    ```

  * test the model:

    ```bash
    conda activate ClassificatoreNPS
    python ClassificatoreNPS/test_mlflow_api.py
    conda deactivate
    ```

* CorrectionDispersion:

  * log model (for this one create a folder called CorrectionDispersion inside CorrectionDispersion with only the code needed and the weights):

    ```bash
    conda activate CorrectionDispersion
    cd CorrectionDispersion
    python log_model.py
    cd ..
    conda deactivate
    ```

  * generate dockerfile and run the model in a container:

    ```bash
    set MLFLOW_TRACKING_URI=http://localhost:5000
    conda activate CorrectionDispersion
    mlflow models generate-dockerfile -m "models:/correction_dispertion_model/latest" -d CorrectionDispersion_image

    docker build -t correction_dispertion_model:latest CorrectionDispersion_image
    docker run --rm -it -p 8080:8080 -e MLFLOW_MODELS_WORKERS=1 -v pentionsystem_simulation_data:/simulation_data correction_dispertion_model:latest
    conda deactivate
    ```

  * test the model:

    ```bash
    conda activate CorrectionDispersion
    python CorrectionDispersion/test_mlflow_api.py
    conda deactivate
    ```

* EmissionSourceLocalization:

  * log model:

    ```bash
    conda activate EmissionSourceLocalization
    cd EmissionSourceLocalization
    python log_model.py
    cd ..
    conda deactivate
    ```

  * generate dockerfile and run the model in a container:

    ```bash
    set MLFLOW_TRACKING_URI=http://localhost:5000
    conda activate EmissionSourceLocalization
    mlflow models generate-dockerfile -m "models:/emission_source_localization_model/latest" -d EmissionSourceLocalization_image

    docker build -t emission_source_localization_model:latest EmissionSourceLocalization_image
    docker run --rm -it -p 8080:8080 -e MLFLOW_MODELS_WORKERS=1 emission_source_localization_model:latest
    conda deactivate
    ```

  * test the model:

    ```bash
    conda activate EmissionSourceLocalization
    python EmissionSourceLocalization/test_mlflow_api.py
    conda deactivate
    ```

## TODOs

* CorrectionDispersion: avoid loading the model on each request but load it at the start of the service, like the others, to reduce the response time
* PentionSystem: it is based on code from gaussianPuff, it should be refactored to make the two services more independent
* code cleaning:
  * more logging, all the services except the PentionSystem have logs
  * formatting, apply black formatter and isort to all .py files
  * resolve pylint warnings
  * remove AI symbols from code and UI
  * translate all comments, logs and UI text in english
  * transition to a more traditional FE-BE architecture?
    * an idea could be to create a backend service that contains /generate_binary_map and /start_simulation endpoints
    * keycloak integration
* docker:
  * remove unused files from images (articles, etc.) to make them lighter, so avoid "COPY . ." in the dockerfile
  * don't expose ports except the frontend one
  * give a hostname to each container that expose API for PentionSystem, this way you can avoid using <http://host.docker.internal>

Application refactor keeping streamlit:

```bash
my_app/
├── app.py                  # Entry point - just UI and routing
├── core/
│   ├── __init__.py
│   └── logic.py            # Pure business logic (no Streamlit)
├── services/
│   ├── __init__.py
│   └── service_logic.py    # code from gaussianPuff and other services, refactored to be inside the same container
├── ui/
│   ├── __init__.py
│   ├── components.py       # reusable UI components
│   └── state.py            # session_state management
├── config.py               # Constants and configuration
└── utils/
    ├── __init__.py
    └── helpers.py          # Utility functions
```
