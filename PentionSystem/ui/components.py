import streamlit as st

from core.ui_invoked import start_button_logic, stop_button_logic, runtime_logic


def fixed_ui_init():

    st.set_page_config(page_title="PentionSystem", layout="wide")

    st.markdown(
        """
        <div style="
            position: sticky; 
            top: 0; 
            background-color: white; 
            padding: 20px; 
            z-index: 999; 
            font-size: 36px; 
            font-weight: bold;
            text-align: center;
        ">
            💊 PENTION - NPS Source emission identification system
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar input
    st.sidebar.header("Insert simulation parameters")

    st.sidebar.markdown(
        """
        <style>
        .start-btn > button {
            background-color: #28a745 !important;
            color: white !important;
            font-weight: bold;
            border-radius: 8px;
            width: 100%;
            padding: 0.5em 0;
        }
        .stop-btn > button {
            background-color: #dc3545 !important;
            color: white !important;
            font-weight: bold;
            border-radius: 8px;
            width: 100%;
            padding: 0.5em 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def input_parameters_setup():

    min_lat = st.sidebar.number_input("Min Lat", value=41.89, format="%.5f")
    min_lon = st.sidebar.number_input("Min Lon", value=12.48, format="%.5f")
    max_lat = st.sidebar.number_input("Max Lat", value=41.91, format="%.5f")
    max_lon = st.sidebar.number_input("Max Lon", value=12.50, format="%.5f")
    place = st.sidebar.text_input("Place", value="Insert place name")
    n_sensors = st.sidebar.slider(
        "Number of sensors", min_value=5, max_value=50, value=10, step=1
    )

    return min_lat, min_lon, max_lat, max_lon, place, n_sensors


def start_button(col1):

    with col1:
        st.markdown('<div class="start-btn">', unsafe_allow_html=True)
        start = st.button("▶ Start")
        st.markdown("</div>", unsafe_allow_html=True)

    return start


def stop_button(col2):

    with col2:
        st.markdown('<div class="stop-btn">', unsafe_allow_html=True)
        stop = st.button("⏹ Stop")
        st.markdown("</div>", unsafe_allow_html=True)

    return stop


def ui_init():

    fixed_ui_init()

    min_lat, min_lon, max_lat, max_lon, place, n_sensors = input_parameters_setup()
    col1, col2 = st.sidebar.columns(2)
    start = start_button(col1)
    stop = stop_button(col2)

    # Layout colonne: lato-sinistra, centro (mappa), lato-destra
    col_left, col_center, col_right = st.columns([1, 3, 1])

    with col_left:
        weather_section = st.container()
        dispersion_section = st.container()
        metadata_section = st.container()

    with col_center:
        map_section = st.container()

    with col_right:
        nps_section = st.container()
        source_section = st.container()
        wind_rose_section = st.container()

    with weather_section:
        st.markdown("**⛅Meteo conditions**")
        weather_placeholder = st.empty()
        weather_placeholder.markdown(
            "💨 **Wind speed (m/s):** N/A  \n"
            "💨 **Wind type:** N/A  \n"
            "📈 **Stability:** N/A  \n"
            "♒︎ **Relative Humidity (%):** N/A"
        )

    with dispersion_section:
        st.markdown("**🗺️ Dispersion map**")
        dispersion_placeholder = st.empty()

    sensors_section = st.container()

    with metadata_section:
        st.markdown("**🏙️ Info city map**")
        metadata_placeholder = st.empty()

    with sensors_section:
        st.markdown("**🛰️ Sensor**")
        sensors_placeholder = st.empty()
        sensors_placeholder.write("No data available.")

    with nps_section:
        st.markdown("**🧪 Nps predicted by sensor**")
        nps_placeholder = st.empty()
        nps_placeholder.write("N/A")

    with source_section:
        st.markdown("**📍 Source estimated**")
        source_placeholder = st.empty()
        source_placeholder.write("N/A")

    with wind_rose_section:
        st.markdown("🧭 **Wind rose**")
        wind_rose_placeholder = st.empty()

    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()

    # Button click logic
    if start:
        start_button_logic(
            status_text=status_text,
            progress_bar=progress_bar,
            min_lon=min_lon,
            min_lat=min_lat,
            max_lon=max_lon,
            max_lat=max_lat,
            place=place,
            n_sensors=n_sensors,
            weather_placeholder=weather_placeholder,
            sensors_placeholder=sensors_placeholder,
            nps_placeholder=nps_placeholder,
            source_placeholder=source_placeholder,
            wind_rose_placeholder=wind_rose_placeholder,
            dispersion_placeholder=dispersion_placeholder,
            map_section=map_section,
            metadata_section=metadata_section,
            metadata_placeholder=metadata_placeholder,
            sensors_section=sensors_section,
            nps_section=nps_section,
            source_section=source_section,
            weather_section=weather_section,
        )
    elif stop:
        stop_button_logic(
            st=st,
            status_text=status_text,
            progress_bar=progress_bar,
            weather_placeholder=weather_placeholder,
            sensors_placeholder=sensors_placeholder,
            nps_placeholder=nps_placeholder,
            source_placeholder=source_placeholder,
            wind_rose_placeholder=wind_rose_placeholder,
            dispersion_placeholder=dispersion_placeholder,
            map_section=map_section,
        )
    else:
        runtime_logic(
            st=st,
            weather_placeholder=weather_placeholder,
            sensors_placeholder=sensors_placeholder,
            nps_placeholder=nps_placeholder,
            source_placeholder=source_placeholder,
            wind_rose_placeholder=wind_rose_placeholder,
            map_section=map_section,
        )
