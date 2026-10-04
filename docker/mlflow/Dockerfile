FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# boto3: S3 client used by the server to store artifacts in MinIO.
RUN pip install --no-cache-dir \
    mlflow==2.17.2 \
    psycopg2-binary==2.9.9 \
    gunicorn==22.0.0 \
    boto3==1.35.36

EXPOSE 5000

# Artifacts are proxied: clients use mlflow-artifacts:/ over HTTP and only this
# server talks to MinIO (needs AWS_* and MLFLOW_S3_ENDPOINT_URL env vars).
CMD ["mlflow", "server", \
     "--backend-store-uri", "postgresql://mlflow:mlflow@postgres:5432/mlflow", \
     "--default-artifact-root", "mlflow-artifacts:/", \
     "--serve-artifacts", \
     "--artifacts-destination", "s3://mlflow-artifacts", \
     "--host", "0.0.0.0", \
     "--port", "5000"]
