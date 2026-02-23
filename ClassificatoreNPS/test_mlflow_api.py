import requests
import numpy as np

URL = "http://localhost:8080/invocations"

input_data = {
    "inputs": [
        {
            "spectra": {"spectra": np.random.rand(60, 600).tolist()},
            "params": {"predict_type": "dnn"},
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
