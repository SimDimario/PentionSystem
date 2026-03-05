import gc
from collections import Counter
import requests
import streamlit as st
import numpy as np

from config import nps_classes, API_URL
from utils.plot_functions import (
    plot_binary_map,
    plot_plan_view,
    plot_wind_rose,
    plot_dispersion_on_map,
)
from utils.utils import (
    get_meteo,
    random_position,
    grid_index_to_coords,
)

from gaussianPuff.Sensor import SensorSubstance, SensorAir
from gaussianPuff.config import NPS, OutputType, DispersionModelType, ModelConfig


def run_application(
    payload,
    progress_bar,
    status_text,
    weather_placeholder,
    sensors_placeholder,
    nps_placeholder,
    source_placeholder,
    wind_rose_placeholder,
    dispersion_placeholder,
    map_section,
    metadata_section,
    metadata_placeholder,
    weather_section,
    sensors_section,
    nps_section,
    source_section,
):

    n_sensors = payload.get("Number of sensors", 10)
    payload.pop("Number of sensors", None)

    progress = 0
    progress_bar.progress(progress)

    # --- Binary map generation
    status_text.text("Binary map generation...")

    response = requests.post(
        f"{API_URL}8001/generate_binary_map", json=payload, timeout=60
    )
    if response.status_code != 200:
        st.error("Error in binary map generation.")
        return None

    data = response.json()
    if data.get("status_code") != "success":
        st.error("Error in binary map generation.")
        return None

    binary_map = np.array(data.get("map"), dtype=np.float32)
    metadata = data.get("metadata", {})
    free_cells = np.argwhere(binary_map == 1)
    building_cells = np.sum(binary_map == 0)

    if len(free_cells) == 0:
        st.error(
            "❌ Mappa binaria invalida: nessuna cella libera trovata. "
            "Verifica le coordinate della bounding box."
        )
        return None

    mean_height = metadata.get("mean_height")

    mean_height_str = (
        f"{float(mean_height):.1f} m"
        if mean_height is not None and not np.isnan(mean_height)
        else "N/A"
    )

    with metadata_section:
        metadata_placeholder.markdown(
            f"**Griglia**: {metadata.get('grid_size', 'N/A')}×{metadata.get('grid_size', 'N/A')}\n"
            f"**Edifici totali**: {metadata.get('total_buildings', 'N/A')}\n"
            f"**Celle edifici**: {int(np.sum(building_cells)) if isinstance(building_cells, np.ndarray) else building_cells:,}\n"
            f"**Celle libere**: {int(np.sum(free_cells)) if isinstance(free_cells, np.ndarray) else free_cells:,}\n"
            f"**CRS**: {metadata.get('crs', 'N/A')}\n"
            f"**Risoluzione**: {metadata.get('resolution (m)', 'N/A')} m\n"
            f"**Densità edifici**: {float(metadata.get('building_density', np.nan)):.1f}%\n"
            f"**Altezza media edifici**: {mean_height_str}\n"
            f"**Città**: {metadata.get('city', 'N/A')}"
        )

    progress += 20
    progress_bar.progress(progress)

    # --- Meteo condition
    status_text.text("Sample meteo condition...")

    lat_center = (payload["min_lat"] + payload["max_lat"]) / 2
    lon_center = (payload["min_lon"] + payload["max_lon"]) / 2

    meteo = get_meteo(lat_center, lon_center)

    sensor_air = SensorAir(sensor_id=00, x=lon_center, y=lat_center, z=2.0)

    # sensor_air = SensorAir(sensor_id=00, x=0.0, y=0.0, z=2.0)
    wind_speed, wind_type, stability_type, stability_value, humidify, dry_size, RH = (
        sensor_air.sample_meteorology()
    )

    if weather_section is not None:
        weather_placeholder.markdown(
            f"💨 **Wind speed (m/s):** {wind_speed}  \n"
            f"💨 **Wind type:** {wind_type}  \n"
            f"📈 **Stability:** {stability_type}  \n"
            f"♒︎ **Relative Humidity (%):** {RH}"
        )

    # --- Sensor substance
    status_text.text("Air sampling...")
    sensors_substance = []

    for i in range(n_sensors):
        x, y = random_position(free_cells)
        sensor_substance = SensorSubstance(
            i, x=x, y=y, z=2.0, noise_level=round(np.random.uniform(0.0, 0.0005), 4)
        )
        sensors_substance.append(sensor_substance)

    plot_binary_map(binary_map, metadata["bounds"], map_section, sensors_substance)

    mass_spectrum = []
    for sensor in sensors_substance:
        recording = sensor.run_sensor(wind_speed, stability_type, RH, wind_type)
        recording = [rec for rec in recording if not np.isnan(rec).any()]
        mass_spectrum.extend(recording)

    print(f"1->{type(mass_spectrum)}")  # list
    print(f"2->{type(mass_spectrum[0])}")  # numpy.ndarray

    if sensors_section is not None:
        sensor_info = [
            {
                "ID": s.id,
                "x": s.x,
                "y": s.y,
                "Status": "Operating" if not s.is_fault else "Faulty",
            }
            for s in sensors_substance
        ]
        sensors_placeholder.table(sensor_info)

    progress += 20
    progress_bar.progress(progress)

    # --- NPS classification
    status_text.text("NPS classification...")
    substance_nps = []

    if mass_spectrum:
        spectra_json = [m.tolist() for m in mass_spectrum]
        print(f"spectra_json: {type(spectra_json)}")
        json = {
            "inputs": [
                {
                    "spectra": {"spectra": spectra_json},
                    "params": {"predict_type": "dnn"},
                }
            ]
        }
        # replaced old api with mlflow serve
        response_dnn = requests.post(
            "http://classificatore_nps_mlflow:8080/invocations", json=json, timeout=10
        )

        if response_dnn.status_code == 200:
            predictions = response_dnn.json().get("predictions", [])
            print(len(predictions))
            substance_nps = [pred for pred in predictions if pred in nps_classes]
        else:
            st.error(f"Errore API {response_dnn.status_code}")

    print(type(substance_nps))
    print(len(substance_nps))

    if substance_nps:
        most_common_substance = Counter(substance_nps).most_common(1)[0][0]
        print(most_common_substance)
        nps = NPS.from_string(most_common_substance)
    else:
        most_common_substance = None
        print("Nessuna sostanza presente")

    if nps_section is not None:
        if substance_nps:
            nps_placeholder.markdown(most_common_substance)
        else:
            nps_placeholder.warning("No NPS identified.")

    del mass_spectrum, spectra_json
    if "response_dnn" in locals():
        del response_dnn
    gc.collect()

    progress += 20
    progress_bar.progress(progress)

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    x_src, y_src = random_position(free_cells)
    h_src = round(np.random.uniform(1, 10), 2)  # altezza del pennacchio
    Q = round(np.random.uniform(0.0001, 0.01), 4)  # tasso di emissione
    stacks = [(x_src, y_src, Q, h_src)]

    print(stability_value)
    print(wind_speed)
    print(wind_type)

    param_gaussian_model = ModelConfig(
        days=10,
        RH=RH,
        aerosol_type=NPS(nps),
        humidify=humidify,
        stability_profile=stability_type,
        stability_value=stability_value,
        wind_type=wind_type,
        wind_speed=wind_speed,
        output=OutputType.PLAN_VIEW,
        stacks=stacks,
        dry_size=dry_size,
        x_slice=26,
        y_slice=1,
        dispersion_model=DispersionModelType.PLUME,
    )

    bounds = (
        payload["min_lon"],
        payload["min_lat"],
        payload["max_lon"],
        payload["max_lat"],
    )

    response_gauss = requests.post(
        f"{API_URL}8002/start_simulation",
        json={"config": param_gaussian_model.to_dict(), "bounds": bounds},
        timeout=60,
    )

    print("risposta ottenuta")
    print(f"code: {response_gauss.status_code}")
    print(response_gauss)

    if response_gauss.status_code != 200:
        st.error("Error in Gaussian puff simulation 01.")
        return sensors_substance, substance_nps, None, None, None, metadata

    gauss_data = response_gauss.json()
    x_raw = gauss_data.get("x", [])
    y_raw = gauss_data.get("y", [])
    times_raw = gauss_data.get("times", [])
    wind_dir_raw = gauss_data.get("wind_dir")
    C1_raw = gauss_data.get("concentration", [])

    x = np.array(x_raw)
    y = np.array(y_raw)
    times = np.array(times_raw)
    wind_dir = np.array(wind_dir_raw)
    C1 = np.array(C1_raw)

    print(type(C1))
    print(C1.shape)
    print(type(wind_dir))
    print(wind_dir.shape)
    print(type(wind_speed))
    print(type(x))
    print(x.shape)
    print(type(y))
    print(y.shape)

    status_text.text("Dispersion map generation...")
    plot_plan_view(C1, x, y, dispersion_placeholder)
    status_text.text("Wind rose graph generation...")
    plot_wind_rose(wind_dir, wind_speed, wind_rose_placeholder)

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # --- Localizzazione sorgente
    status_text.text("Source estimation...")

    payload_sensors = []
    for s in sensors_substance:

        if not s.is_fault:

            s.sample_substance(C1, x, y, times)
            # s.sample_substance_synthetic()

            for idx, (t_idx, conc) in enumerate(zip(s.times, s.noisy_concentrations)):
                if idx >= len(wind_dir):
                    break
                wd = wind_dir[idx]

                payload_sensors.append(
                    {
                        "sensor_id": s.id,
                        "sensor_is_fault": s.is_fault,
                        "time": t_idx,
                        "conc": conc if not s.is_fault else None,
                        "wind_dir_x": (
                            np.cos(np.deg2rad(wd)) if not s.is_fault else None
                        ),
                        "wind_dir_y": (
                            np.sin(np.deg2rad(wd)) if not s.is_fault else None
                        ),
                        "wind_speed": wind_speed if not s.is_fault else None,
                        "wind_type": wind_type.value if not s.is_fault else None,
                    }
                )

    n_sensor_operating = len([s for s in sensors_substance if not s.is_fault])

    status_text.text("Start the prediction of the source...")
    response_loc = requests.post(
        "http://loc_emission_source_mlflow:8080/invocations",
        json={
            "inputs": [
                {
                    "payload_sensors": payload_sensors,
                    "n_sensor_operating": n_sensor_operating,
                }
            ]
        },
        timeout=10,
    )

    if response_loc.status_code != 200:
        st.error("Error in prediction of source.")

    data = response_loc.json().get("predictions", {})
    lon = data["x"]
    lat = data["y"]

    x, y = grid_index_to_coords(lon, lat, bounds, 500)

    if source_section is not None:
        if x is not None and y is not None:
            source_placeholder.markdown(f"Lat: {x}, Long: {y}")
        else:
            source_placeholder.warning("Source not estimated.")

    progress += 20
    progress_bar.progress(progress)

    # --- gaussian plume dispersion (raw simulation)
    status_text.text("Raw dispersion simulation...")

    stacks = [(x, y, Q, h_src)]

    param_gaussian_model = ModelConfig(
        days=10,
        RH=RH,
        aerosol_type=NPS(nps),
        humidify=humidify,
        stability_profile=stability_type,
        stability_value=stability_value,
        wind_type=wind_type,
        wind_speed=wind_speed,
        output=OutputType.PLAN_VIEW,
        stacks=stacks,
        dry_size=dry_size,
        x_slice=26,
        y_slice=1,
        dispersion_model=DispersionModelType.PLUME,
    )

    bounds = (
        payload["min_lon"],
        payload["min_lat"],
        payload["max_lon"],
        payload["max_lat"],
    )

    response_gauss = requests.post(
        f"{API_URL}8002/start_simulation",
        json={"config": param_gaussian_model.to_dict(), "bounds": bounds},
        timeout=60,
    )

    if response_gauss.status_code != 200:
        st.error("Error in Gaussian puff simulation.")
        return sensors_substance, substance_nps, None, None, None, metadata

    gauss_data = response_gauss.json()
    x_raw = gauss_data.get("x", [])
    y_raw = gauss_data.get("y", [])
    times_raw = gauss_data.get("times", [])
    wind_dir_raw = gauss_data.get("wind_dir")
    C1_raw = gauss_data.get("concentration", [])

    x_grid = np.array(x_raw)
    y_grid = np.array(y_raw)
    times = np.array(times_raw)
    wind_dir = np.array(wind_dir_raw)
    C1 = np.array(C1_raw)

    status_text.text("Dispersion map generation...")
    plot_plan_view(C1, x_grid, y_grid, dispersion_placeholder)
    status_text.text("Wind rose graph generation...")
    plot_wind_rose(wind_dir, wind_speed, wind_rose_placeholder)

    progress += 20
    progress_bar.progress(progress)

    # --- Dispersion simulation + correction
    status_text.text("Dispersion simulation...")

    np.save("/tmp/C1.npy", C1)
    np.save("/tmp/binary_map.npy", binary_map)

    response_mcxm = requests.post(
        "http://correction_dispersion_mlflow:8080/invocations",
        json={
            "inputs": [
                {
                    "wind_speed": wind_speed,
                    "wind_dir": wind_dir.tolist(),
                    "concentration_map": "/simulation_data/C1.npy",
                    "building_map": "/simulation_data/binary_map.npy",
                }
            ]
        },
        timeout=10,
    )

    if response_mcxm.status_code != 200:
        st.error("Errore nella correzione della dispersione.")
        return sensors_substance, substance_nps, x, y, C1, metadata

    real_dispersion_map = response_mcxm.json().get("predictions", [])
    real_dispersion_map = np.array(real_dispersion_map)
    del response_mcxm
    gc.collect()
    print(f"mapp finale {type(real_dispersion_map)}")
    print(real_dispersion_map.shape)
    # progress += 20
    # progress_bar.progress(progress)

    if real_dispersion_map.ndim == 3:
        # integrazione temporale (coerente con plot_plan_view)
        tmp = real_dispersion_map
        real_dispersion_map = np.trapezoid(tmp, axis=2)
        del tmp
        gc.collect()
    # assert real_dispersion_map.ndim == 2

    from streamlit_folium import st_folium

    if map_section is not None:
        m = plot_dispersion_on_map(
            payload["min_lat"],
            payload["min_lon"],
            payload["max_lat"],
            payload["max_lon"],
            sensors_substance,
            real_dispersion_map,
            x,
            y,
        )
        map_section.subheader("🗺️ Dispersion map")
        st_folium(m, width=700, height=500)
        m.save("dispersion_map.html")

        np.save("/tmp/dispersion.npy", real_dispersion_map)

        st.session_state.simulation_results["dispersion_map"] = {
            "min_lat": payload["min_lat"],
            "min_lon": payload["min_lon"],
            "max_lat": payload["max_lat"],
            "max_lon": payload["max_lon"],
            "sensors": [
                {"id": s.id, "x": s.x, "y": s.y, "is_fault": s.is_fault}
                for s in sensors_substance
            ],
            "dispersion": "/tmp/dispersion.npy",
            "x_src": x,
            "y_src": y,
        }

    print("C1 shape:", C1.shape)
    print("C1 ndim:", C1.ndim)

    progress = 100
    progress_bar.progress(progress)
    status_text.text("Simulation completed ✅")
    print("END")

    np.save("/tmp/dispersion.npy", real_dispersion_map)

    st.session_state.simulation_results.update(
        {
            "weather": {
                "wind_speed": wind_speed,
                "wind_type": wind_type,
                "stability": stability_type,
                "RH": RH,
            },
            "sensors": [
                {"id": s.id, "x": s.x, "y": s.y, "is_fault": s.is_fault}
                for s in sensors_substance
            ],
            "nps": most_common_substance,
            "source": (x, y),
            "metadata": metadata,
            "wind_dir": wind_dir,
        }
    )
