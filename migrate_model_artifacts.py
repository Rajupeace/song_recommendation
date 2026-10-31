
import mlflow
from mlflow import MlflowClient
from pathlib import Path

TRACKING_URI = "http://127.0.0.1:5000"
MODEL_NAME = "Song_Recommendation_ALS"
MODEL_ID = "m-209630954b3a4095b8429b8bc7b3df89"

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_registry_uri(TRACKING_URI)

client = MlflowClient(tracking_uri=TRACKING_URI)

print("\n1. Inspecting the existing MLflow 3 model...")

logged_model = client.get_logged_model(MODEL_ID)

print("Model ID:", logged_model.model_id)
print("Artifact location:", logged_model.artifact_location)

artifacts = client.list_logged_model_artifacts(MODEL_ID)

print("\nExisting model files:")
for item in artifacts:
    print(item.path, "| Directory:", item.is_dir)

print("\n2. Downloading the original model on Windows...")

source_uri = f"models:/{MODEL_ID}"

local_model_dir = Path(
    mlflow.artifacts.download_artifacts(
        artifact_uri=source_uri,
        tracking_uri=TRACKING_URI
    )
)

print("Downloaded to:", local_model_dir)

if not (local_model_dir / "MLmodel").exists():
    raise RuntimeError(
        "MLmodel file not found. Original model download is incomplete."
    )

print("\n3. Creating an MLflow experiment with proxied artifacts...")

experiment_name = "Song_Recommendation_Docker_Proxy_20261031"

existing = client.get_experiment_by_name(experiment_name)

if existing is None:
    experiment_id = client.create_experiment(experiment_name)
    experiment = client.get_experiment(experiment_id)
else:
    experiment = existing

artifact_location = experiment.artifact_location

print("Experiment ID:", experiment.experiment_id)
print("Artifact location:", artifact_location)

if not artifact_location.startswith("mlflow-artifacts:"):
    raise RuntimeError(
        "This experiment is not using MLflow artifact proxy. "
        "Restart the server with the correct command before continuing."
    )

print("\n4. Uploading model artifacts to the MLflow server...")

with mlflow.start_run(
    experiment_id=experiment.experiment_id,
    run_name="Song_ALS_Docker_Proxy_Migration"
) as run:
    mlflow.log_artifacts(
        str(local_model_dir),
        artifact_path="model"
    )
    run_id = run.info.run_id

print("New run ID:", run_id)

print("\n5. Registering the migrated model...")

model_source = f"runs:/{run_id}/model"

new_version = mlflow.register_model(
    model_uri=model_source,
    name=MODEL_NAME
)

print("New model version:", new_version.version)
print("New model status:", new_version.status)

print("\n6. Updating the candidate alias...")

client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="candidate",
    version=new_version.version
)

print("Candidate alias now points to version:", new_version.version)

print("\n7. Testing the newly registered model...")

new_uri = f"models:/{MODEL_NAME}/{new_version.version}"

test_model = mlflow.pyfunc.load_model(new_uri)

print("Model loaded successfully!")
print("Model URI:", new_uri)
print("\nMIGRATION COMPLETED SUCCESSFULLY!")
