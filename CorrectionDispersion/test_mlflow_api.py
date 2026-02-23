import requests
import numpy as np

URL = "http://localhost:8080/invocations"

input_data = {
    "inputs": [
        {
            "wind_speed": 5.0,
            "wind_dir": [1.0, 0.0],
            "concentration_map": "/simulation_data/C1.npy",
            "building_map": "/simulation_data/binary_map.npy",
        }
    ]
}

response = requests.post(URL, json=input_data, timeout=10)

if response.status_code == 200:

    print("Request successful!")
    print("Response:", np.array(response.json()["predictions"]).shape)
else:

    print(f"Request failed with status code: {response.status_code}")
    print(response.text[:200])
    print(response.text[-200:])
