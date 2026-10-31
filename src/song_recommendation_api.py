
import os

# Configure numerical library threads before imports
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import json
import time
import logging
import pandas as pd
import mlflow

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST
)
from threadpoolctl import threadpool_limits


# ==========================================
# CONFIGURATION
# ==========================================

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MODEL_URI = os.getenv("MLFLOW_MODEL_URI", "models:/Song_Recommendation_ALS@candidate")

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_registry_uri(MLFLOW_URI)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

recommendation_model = None

threadpool_limits(limits=1, user_api="blas")


# ==========================================
# PROMETHEUS METRICS
# ==========================================

HTTP_REQUESTS = Counter(
    "song_api_http_requests",
    "Total HTTP requests",
    ["method", "endpoint", "status"]
)

HTTP_LATENCY = Histogram(
    "song_api_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"]
)

RECOMMENDATION_REQUESTS = Counter(
    "song_api_recommendation_requests",
    "Total recommendation requests",
    ["status"]
)

MODEL_LOADED = Gauge(
    "song_api_model_loaded",
    "Whether the recommendation model is loaded"
)


# ==========================================
# APPLICATION LIFECYCLE
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    global recommendation_model

    logger.info("Loading recommendation model...")

    recommendation_model = mlflow.pyfunc.load_model(
        MODEL_URI
    )

    MODEL_LOADED.set(1)

    logger.info("Recommendation model loaded!")

    yield

    recommendation_model = None
    MODEL_LOADED.set(0)

    logger.info("API application stopped.")


app = FastAPI(
    title="Song Recommendation API",
    description="MLOps Song Recommendation Service with Monitoring",
    version="1.1.0",
    lifespan=lifespan
)


# ==========================================
# HTTP REQUEST MONITORING
# ==========================================

@app.middleware("http")
async def monitor_requests(request: Request, call_next):

    start_time = time.perf_counter()
    status = "500"

    try:
        response = await call_next(request)
        status = str(response.status_code)
        return response

    finally:
        duration = time.perf_counter() - start_time

        HTTP_REQUESTS.labels(
            request.method,
            request.url.path,
            status
        ).inc()

        HTTP_LATENCY.labels(
            request.method,
            request.url.path
        ).observe(duration)


# ==========================================
# REQUEST SCHEMA
# ==========================================

class RecommendationRequest(BaseModel):

    user_id: str = Field(min_length=1)

    top_n: int = Field(
        default=10,
        ge=1,
        le=50
    )


# ==========================================
# HOME ENDPOINT
# ==========================================

@app.get("/")
def home():

    return {
        "project": "Song Recommendation System",
        "status": "running",
        "version": "1.1.0"
    }


# ==========================================
# HEALTH CHECK
# ==========================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "model_loaded": recommendation_model is not None
    }


# ==========================================
# RECOMMENDATION ENDPOINT
# ==========================================

@app.post("/recommend")
def recommend_songs(request: RecommendationRequest):

    if recommendation_model is None:

        RECOMMENDATION_REQUESTS.labels("error").inc()

        raise HTTPException(
            status_code=503,
            detail="Model is not loaded."
        )

    try:

        input_data = pd.DataFrame({
            "user_id": [request.user_id]
        })

        predictions = recommendation_model.predict(
            input_data,
            params={"top_n": request.top_n}
        )

        recommendations = json.loads(
            predictions.iloc[0]["recommendations_json"]
        )

        RECOMMENDATION_REQUESTS.labels("success").inc()

        return {
            "status": "success",
            "user_id": request.user_id,
            "total_recommendations": len(recommendations),
            "recommendations": recommendations
        }

    except ValueError as error:

        RECOMMENDATION_REQUESTS.labels("error").inc()

        raise HTTPException(
            status_code=404,
            detail=str(error)
        )

    except Exception:

        RECOMMENDATION_REQUESTS.labels("error").inc()

        logger.exception("Recommendation failed")

        raise HTTPException(
            status_code=500,
            detail="Recommendation generation failed."
        )


# ==========================================
# PROMETHEUS METRICS ENDPOINT
# ==========================================

@app.get("/metrics", include_in_schema=False)
def metrics():

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )
