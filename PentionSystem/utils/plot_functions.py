import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from windrose import WindroseAxes
import folium
from folium.plugins import HeatMap


def plot_plan_view(C1, x, y, dispersion_placeholder, stability_class=1):
    with dispersion_placeholder:
        fig, ax_main = plt.subplots(figsize=(8, 6))

        # Integra la concentrazione nel tempo lungo l'asse 2 (T)
        data = np.trapezoid(C1, axis=2) * 1e6  # µg/m³ #type:ignore
        vmin = np.percentile(data, 5)
        vmax = np.percentile(data, 95)

        # Plot della concentrazione integrata
        pcm = ax_main.pcolor(
            x, y, data, cmap="jet", shading="auto", vmin=vmin, vmax=vmax
        )
        fig.colorbar(pcm, ax=ax_main, label=r"$\mu g \cdot m^{-3}$")
        ax_main.set_xlabel("x (m)")
        ax_main.set_ylabel("y (m)")
        ax_main.axis("equal")

        st.pyplot(fig, clear_figure=False)


def plot_wind_rose(wind_dir, wind_speed, wind_rose_placeholder):
    with wind_rose_placeholder:
        if wind_dir is not None and wind_speed is not None:
            fig = plt.figure(figsize=(6, 6))

            ax_inset = WindroseAxes.from_ax(fig=fig)

            # Plot rosa dei venti con direzioni e velocità
            wind_dir = np.array(wind_dir) % 360
            wind_speed = np.full_like(wind_dir, fill_value=wind_speed, dtype=float)
            ax_inset.bar(
                wind_dir, wind_speed, normed=True, opening=0.8, edgecolor="white"
            )
            ax_inset.set_legend(loc="lower right", title="Wind speed (m/s)")
            ax_inset.set_title("Rosa dei venti")

            st.pyplot(fig, clear_figure=True)


def plot_binary_map(binary_map, bounds, map_section, sensors=None):
    with map_section:

        fig, ax = plt.subplots(figsize=(8, 8))

        x_min, y_min, x_max, y_max = bounds
        im = ax.imshow(
            binary_map, cmap="gray", extent=(x_min, x_max, y_min, y_max), origin="lower"
        )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        ax.set_xlabel("Coordinate X (grid)")
        ax.set_ylabel("Coordinate Y (grid)")
        ax.grid(False)

        if sensors is not None:
            for s in sensors:
                x = s.x
                y = s.y
                faulty = s.is_fault

                if not faulty:
                    ax.scatter(
                        x,
                        y,
                        c="green",
                        marker="o",
                        edgecolors="black",
                        s=80,
                        label="Operating",
                    )
                else:
                    ax.scatter(
                        x,
                        y,
                        c="red",
                        marker="X",
                        edgecolors="black",
                        s=100,
                        label="Faulty",
                    )

            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc="upper right")

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Occupazione")

        st.pyplot(fig, clear_figure=True)


def plot_dispersion_on_map(
    min_lat,
    min_lon,
    max_lat,
    max_lon,
    sensors,
    dispersion_map,
    source_lat=None,
    source_lon=None,
    title="Mappa Dispersione",
    wind_dir=None,
    wind_speed=None,
    puff_list=None,
    stability_class=1,
    n_show=10,
):
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2

    m = folium.Map(
        location=[center_lat, center_lon], zoom_start=14, tiles="cartodbpositron"
    )

    """
    # Sensori
    for s in sensors:
        x,y=sensor_xy(s)
        sid=sensor_get(s,"id")
        folium.Marker(
            [x,y],
            popup=f"Sensor {sid}",
            icon=folium.Icon(color="blue", icon="info-sign")
        ).add_to(m)
    """
    for s in sensors:
        row, col = sensor_xy(s)  # row, col (non x,y!)
        sid = sensor_get(s, "id")

        lat = max_lat - (row / (len(dispersion_map) - 1)) * (max_lat - min_lat)
        lon = min_lon + (col / (len(dispersion_map[0]) - 1)) * (max_lon - min_lon)

        folium.Marker(
            [lat, lon],
            popup=f"Sensor {sid}",
            icon=folium.Icon(color="blue", icon="info-sign"),
        ).add_to(m)

    # Sorgente stimata
    if source_lat is not None and source_lon is not None:
        folium.Marker(
            [source_lon, source_lat],
            popup="Sorgente Stimata",
            icon=folium.Icon(color="red", icon="fire"),
        ).add_to(m)

    # Heatmap della dispersione
    heat_data = []
    rows = len(dispersion_map)
    cols = len(dispersion_map[0]) if rows > 0 else 0

    for i in range(rows):
        for j in range(cols):
            # lat = min_lat + (max_lat - min_lat) * (i / max(rows-1, 1))
            lat = max_lat - (max_lat - min_lat) * (i / max(rows - 1, 1))
            lon = min_lon + (max_lon - min_lon) * (j / max(cols - 1, 1))
            conc = float(dispersion_map[i][j])
            heat_data.append([lat, lon, conc])

    if heat_data:
        HeatMap(heat_data, radius=25, blur=15, min_opacity=0.2).add_to(m)

    title_html = f"""
         <h3 align="center" style="font-size:18px"><b>{title}</b></h3>
         """
    m.get_root().html.add_child(folium.Element(title_html))

    return m


def sensor_xy(s):
    if isinstance(s, dict):
        return s["x"], s["y"]
    return s.x, s.y


def sensor_get(s, key):
    return s[key] if isinstance(s, dict) else getattr(s, key)
