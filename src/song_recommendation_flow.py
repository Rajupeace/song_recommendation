
import json
from pathlib import Path

import pandas as pd
import mlflow

from prefect import flow, task


# ==========================================
# PROJECT CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs"

USER_MAPPING_FILE = (
    PROCESSED_DIR / "user_index_train.csv"
)

MODEL_URI = "models:/Song_Recommendation_ALS@candidate"

OUTPUT_FILE = (
    OUTPUT_DIR / "prefect_song_recommendations.csv"
)

MLFLOW_URI = "http://127.0.0.1:5000"


# ==========================================
# TASK 1: VALIDATE PROJECT FILES
# ==========================================

@task(name="validate-project-files", log_prints=True)
def validate_project_files():

    print("Validating project files...")

    if not USER_MAPPING_FILE.exists():
        raise FileNotFoundError(USER_MAPPING_FILE)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("User mapping found!")
    print("Output directory ready!")

    return str(USER_MAPPING_FILE), str(OUTPUT_FILE)


# ==========================================
# TASK 2: SELECT USER
# ==========================================

@task(name="select-user", log_prints=True)
def select_user(mapping_file):

    users = pd.read_csv(
        mapping_file,
        dtype={"user_id": str}
    )

    if users.empty:
        raise ValueError("No users available!")

    user_id = users.iloc[0]["user_id"]

    print("Selected user:", user_id)

    return user_id


# ==========================================
# TASK 3: GENERATE RECOMMENDATIONS
# ==========================================

@task(name="generate-recommendations", log_prints=True)
def generate_recommendations(user_id):

    print("Connecting to MLflow...")

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_registry_uri(MLFLOW_URI)

    print("Loading registered ALS model...")

    model = mlflow.pyfunc.load_model(MODEL_URI)

    input_data = pd.DataFrame({
        "user_id": [str(user_id)]
    })

    predictions = model.predict(input_data)

    print("Recommendations generated successfully!")

    return predictions.to_dict(orient="records")


# ==========================================
# TASK 4: SAVE RECOMMENDATIONS
# ==========================================

@task(name="save-recommendations", log_prints=True)
def save_recommendations(prediction_records, output_file):

    if not prediction_records:
        raise ValueError("No predictions received!")

    record = prediction_records[0]

    recommendations = json.loads(
        record["recommendations_json"]
    )

    result = pd.DataFrame(recommendations)

    result.insert(0, "user_id", record["user_id"])

    result.to_csv(
        output_file,
        index=False
    )

    print("Recommendation report saved!")
    print("Output:", output_file)
    print("Recommended songs:", len(result))

    return str(output_file)


# ==========================================
# MAIN PREFECT FLOW
# ==========================================

@flow(
    name="song-recommendation-mlops-flow",
    log_prints=True
)
def song_recommendation_flow():

    mapping_file, output_file = validate_project_files()

    user_id = select_user(mapping_file)

    predictions = generate_recommendations(user_id)

    saved_file = save_recommendations(
        predictions,
        output_file
    )

    print("\nSONG RECOMMENDATION FLOW COMPLETED!")

    return {
        "user_id": user_id,
        "output_file": saved_file
    }


# ==========================================
# EXECUTE FLOW
# ==========================================

if __name__ == "__main__":

    song_recommendation_flow()
