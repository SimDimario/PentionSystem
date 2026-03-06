from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional
import random
from datetime import datetime, timezone

app = FastAPI()


class BoundingBox(BaseModel):
    min_lon: float
    max_lon: float
    min_lat: float
    max_lat: float
    count: int = Field(3, description="Number of sample points to generate")


class Coordinates(BaseModel):
    latitude: float
    longitude: float


class MeasuredConcentrations(BaseModel):
    PFOS: float
    PFOA: float
    GenX: float


class SamplingPoint(BaseModel):
    id: str
    coordinates: Coordinates
    measured_concentrations: MeasuredConcentrations
    sample_time: str


class SamplingResponse(BaseModel):
    sampling_points: List[SamplingPoint]
    units: dict


def random_id() -> str:
    # e.g. one uppercase letter followed by three digits
    return f"{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{random.randint(0,999):03d}"


def random_conc() -> float:
    # generate a random concentration value, for example range 0-300
    return round(random.uniform(0, 300), 1)


@app.post("/samples", response_model=SamplingResponse)
def generate_samples(bbox: BoundingBox):
    points = []
    for _ in range(bbox.count):
        lat = random.uniform(bbox.min_lat, bbox.max_lat)
        lon = random.uniform(bbox.min_lon, bbox.max_lon)
        sample = SamplingPoint(
            id=random_id(),
            coordinates=Coordinates(latitude=lat, longitude=lon),
            measured_concentrations=MeasuredConcentrations(
                PFOS=random_conc(),
                PFOA=random_conc(),
                GenX=random_conc(),
            ),
            sample_time=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        )
        points.append(sample)

    return SamplingResponse(
        sampling_points=points,
        units={
            "concentration": "ppq",
            "coordinates": "WGS84",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8005)
