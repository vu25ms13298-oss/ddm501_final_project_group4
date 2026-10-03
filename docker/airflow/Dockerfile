FROM apache/airflow:2.8.4-python3.11

# TrueType fonts are required by the synthetic character generator
# (src/classifier.py); without them it silently falls back to a tiny bitmap font.
USER root
RUN apt-get update && apt-get install -y --no-install-recommends fonts-dejavu-core && \
    rm -rf /var/lib/apt/lists/*

USER airflow

ARG AIRFLOW_VERSION=2.8.4
ARG PYTHON_VERSION=3.11
RUN pip install --no-cache-dir \
      --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt" \
      "pandas==2.1.4" \
      "scikit-learn==1.5.2" \
      "scikit-image==0.24.0" \
      "opencv-python-headless==4.10.0.84" \
      "mlflow==2.17.2" \
      "joblib==1.4.2"
