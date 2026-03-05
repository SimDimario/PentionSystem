import streamlit as st


def init_state():

    if "simulation_results" not in st.session_state:
        st.session_state.simulation_results = {
            "weather": None,
            "sensors": [],
            "nps": None,
            "source": None,
            "dispersion_map": None,
            "metadata": None,
        }
