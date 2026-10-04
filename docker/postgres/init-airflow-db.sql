-- Airflow metadata database (MLflow uses the default "mlflow" database).
-- Executed by the postgres image only when the data volume is empty.
CREATE DATABASE airflow;
