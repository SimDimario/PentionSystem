FROM python:3.10-slim-bookworm

WORKDIR /SamplingService

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --verbose

COPY . .

EXPOSE 8005

CMD ["uvicorn", "api_sampler:app", "--host", "0.0.0.0", "--port", "8005"]
