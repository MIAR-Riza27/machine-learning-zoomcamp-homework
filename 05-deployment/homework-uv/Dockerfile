FROM agrigorev/zoomcamp-model:2025

WORKDIR /code

COPY pyproject.toml .
RUN pip install --upgrade pip
RUN pip install .
RUN pip install scikit-learn==1.6.1
RUN pip install fastapi uvicorn

COPY app.py .
COPY pipeline_v1.bin .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]