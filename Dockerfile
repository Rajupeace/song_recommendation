FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV OPENBLAS_NUM_THREADS=1
ENV OMP_NUM_THREADS=1

RUN apt-get update &&     apt-get install -y --no-install-recommends libgomp1 &&     rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt .

RUN python -m pip install --no-cache-dir --upgrade pip &&     python -m pip install --no-cache-dir -r requirements-docker.txt

COPY src/ ./src/

EXPOSE 8000

CMD ["uvicorn", "src.song_recommendation_api:app", "--host", "0.0.0.0", "--port", "8000"]
