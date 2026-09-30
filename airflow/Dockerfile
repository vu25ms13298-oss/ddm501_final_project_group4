FROM apache/airflow:2.8.4-python3.11

USER airflow

ARG AIRFLOW_VERSION=2.8.4
ARG PYTHON_VERSION=3.11
RUN pip install --no-cache-dir \
      --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt" \
      "pandas==2.1.4" \
      "scikit-learn==1.6.0" \
      "scikit-image==0.22.0" \
      "opencv-python-headless==4.9.0.80" \
      "mlflow==2.17.2" \
      "joblib==1.3.2"
