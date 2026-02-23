import requests
import numpy as np

URL = "http://localhost:8080/invocations"

input_data = {
    "inputs": [
        {
            "payload_sensors": [
                {
                    "sensor_id": 1,
                    "sensor_is_fault": False,
                    "time": 0.0,
                    "conc": 0.0,
                    "wind_dir_x": 0.0,
                    "wind_dir_y": 0.0,
                    "wind_speed": 0.0,
                    "wind_type": 0,
                }
            ],
            "n_sensor_operating": 1,
        }
    ]
}

response = requests.post(URL, json=input_data, timeout=10)

if response.status_code == 200:

    print("Request successful!")
    print("Response:", response.json())
else:

    print(f"Request failed with status code: {response.status_code}")
    print(response.text[:200])
    print(response.text[-200:])
