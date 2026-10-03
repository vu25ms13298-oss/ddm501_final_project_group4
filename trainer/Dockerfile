# Reproducible OCR training environment.
# Uses the same pinned ML library versions as the API and Airflow images so the
# pickled model loads identically in every service.
FROM python:3.10-slim

# fonts-dejavu-core: TrueType fonts for the synthetic character generator.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 fonts-dejavu-core && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-ci.txt .
RUN pip install --no-cache-dir -r requirements-ci.txt

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

CMD ["python", "scripts/train_with_mlflow.py"]
