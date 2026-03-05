import gc
import numpy as np
from streamlit_folium import st_folium

from .logic import run_application
from utils.utils import clean_tmp_files
from utils.plot_functions import plot_dispersion_on_map, plot_wind_rose


def start_button_logic(
    status_text,
    progress_bar,
    min_lon,
    min_lat,
    max_lon,
    max_lat,
    place,
    n_sensors,
    weather_placeholder,
    sensors_placeholder,
    nps_placeholder,
    source_placeholder,
    wind_rose_placeholder,
    dispersion_placeholder,
    map_section,
    metadata_section,
    metadata_placeholder,
    sensors_section,
    nps_section,
    source_section,
    weather_section,
):

    status_text.success("Simulation started ✅")

    payload = {
        "min_lon": min_lon,
        "min_lat": min_lat,
        "max_lon": max_lon,
        "max_lat": max_lat,
        "grid_size": 500,
        "place": place,
        "Number of sensors": n_sensors,
    }

    clean_tmp_files()
    gc.collect()
    run_application(
        payload=payload,
        status_text=status_text,
        progress_bar=progress_bar,
        weather_placeholder=weather_placeholder,
        sensors_placeholder=sensors_placeholder,
        nps_placeholder=nps_placeholder,
        source_placeholder=source_placeholder,
        wind_rose_placeholder=wind_rose_placeholder,
        dispersion_placeholder=dispersion_placeholder,
        map_section=map_section,
        metadata_section=metadata_section,
        metadata_placeholder=metadata_placeholder,
        weather_section=weather_section,
        sensors_section=sensors_section,
        nps_section=nps_section,
        source_section=source_section,
    )


def stop_button_logic(
    st,
    status_text,
    progress_bar,
    weather_placeholder,
    sensors_placeholder,
    nps_placeholder,
    source_placeholder,
    wind_rose_placeholder,
    dispersion_placeholder,
    map_section,
):

    st.session_state.simulation_results = {
        "weather": None,
        "sensors": None,
        "nps": None,
        "source": None,
        "dispersion_map_path": None,
        "metadata": None,
    }
    progress_bar.progress(0)
    status_text.text("Simulation stopped ❌")

    weather_placeholder.markdown(
        "💨 **Wind speed (m/s):** N/A  \n"
        "💨 **Wind type:** N/A  \n"
        "📈 **Stability:** N/A  \n"
        "♒︎ **Relative Humidity (%):** N/A"
    )
    sensors_placeholder.write("No data available.")
    nps_placeholder.write("N/A")
    source_placeholder.write("N/A")
    wind_rose_placeholder.empty()
    dispersion_placeholder.empty()
    map_section.empty()


def runtime_logic(
    st,
    weather_placeholder,
    sensors_placeholder,
    nps_placeholder,
    source_placeholder,
    wind_rose_placeholder,
    map_section,
):
    results = st.session_state.simulation_results

    if results["weather"] is not None:
        weather_placeholder.markdown(
            f"- **Wind speed (m/s):** {results['weather']['wind_speed']}  \n"
            f"- **Wind type:** {results['weather']['wind_type']}  \n"
            f"- **Stability:** {results['weather']['stability']}  \n"
            f"- **Relative Humidity (%):** {results['weather']['RH']}"
        )

    if results["sensors"] is not None:
        sensor_info = [
            {
                "ID": s["id"],
                "x": s["x"],
                "y": s["y"],
                "Status": "Operating" if not s["is_fault"] else "Faulty",
            }
            for s in results["sensors"]
        ]
        sensors_placeholder.table(sensor_info)

    if results["nps"] is not None:
        if results["nps"]:
            nps_placeholder.write(results["nps"])
        else:
            nps_placeholder.warning("No NPS identified.")

    if results["source"] is not None:
        origin_lat, origin_lon = results["source"]
        if origin_lat is not None and origin_lon is not None:
            source_placeholder.write(f"Lat: {origin_lat}, Long: {origin_lon}")
        else:
            source_placeholder.warning("Source not estimated.")

    dm = results.get("dispersion_map")

    if isinstance(dm, dict):
        map_section.subheader("🗺️ Dispersion map")

        # Carica l'array dal file
        dispersion_data = np.load(dm["dispersion"])

        m = plot_dispersion_on_map(
            dm["min_lat"],
            dm["min_lon"],
            dm["max_lat"],
            dm["max_lon"],
            dm["sensors"],
            dispersion_data,
            dm["x_src"],
            dm["y_src"],
        )

        st_folium(m, width=700, height=500)

    if results.get("wind_dir") is not None and results.get("weather") is not None:
        plot_wind_rose(
            np.array(results["wind_dir"]),
            results["weather"]["wind_speed"],
            wind_rose_placeholder,
        )
