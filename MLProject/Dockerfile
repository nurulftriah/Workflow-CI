FROM python:3.12-slim

WORKDIR /app

COPY conda.yaml .
COPY modelling.py .
COPY namadataset_preprocessing/ ./namadataset_preprocessing/

RUN pip install --no-cache-dir \
    mlflow>=2.13.0 \
    scikit-learn>=1.4.0 \
    pandas>=2.2.0 \
    numpy>=1.26.0 \
    dagshub>=0.3.0

CMD ["python", "modelling.py"]
