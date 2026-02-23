FROM python:3.10-slim-bookworm

WORKDIR /EmissionSourceLocalization

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --verbose

COPY . .

EXPOSE 8003

CMD ["uvicorn", "api_source_localization:app", "--host", "0.0.0.0", "--port", "8003"]
