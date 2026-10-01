FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    mlflow==2.17.2 \
    psycopg2-binary==2.9.9 \
    gunicorn==22.0.0

RUN mkdir -p /mlflow-artifacts

EXPOSE 5000

CMD ["mlflow", "server", \
     "--backend-store-uri", "postgresql://mlflow:mlflow@postgres:5432/mlflow", \
     "--default-artifact-root", "mlflow-artifacts:/", \
     "--serve-artifacts", \
     "--artifacts-destination", "/mlflow-artifacts", \
     "--host", "0.0.0.0", \
     "--port", "5000"]
