
from mlflow import MlflowClient

TRACKING_URI = "http://127.0.0.1:5000"
MODEL_NAME = "Song_Recommendation_ALS"

client = MlflowClient(tracking_uri=TRACKING_URI)

print("\n===== REGISTERED MODEL INSPECTION =====")

model_version = client.get_model_version_by_alias(
    name=MODEL_NAME,
    alias="candidate"
)

print("Model name:", MODEL_NAME)
print("Version:", model_version.version)
print("Source:", repr(model_version.source))
print("Run ID:", repr(model_version.run_id))
print("Status:", model_version.status)

download_uri = client.get_model_version_download_uri(
    MODEL_NAME,
    model_version.version
)
print("Download URI:", repr(download_uri))

if model_version.run_id:
    run = client.get_run(model_version.run_id)

    print("\n===== SOURCE RUN =====")
    print("Run artifact URI:", repr(run.info.artifact_uri))
    print("Run status:", run.info.status)

    print("\n===== TOP-LEVEL ARTIFACTS =====")
    artifacts = client.list_artifacts(model_version.run_id)

    for artifact in artifacts:
        print(
            "Path:", artifact.path,
            "| Directory:", artifact.is_dir
        )

print("\n===== INSPECTION COMPLETED =====")
