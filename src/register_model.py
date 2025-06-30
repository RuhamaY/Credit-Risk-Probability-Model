import argparse
import mlflow
import os

def register_model(run_id, model_name):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    model_uri = f"runs:/{run_id}/model"
    mlflow.register_model(model_uri, model_name)
    print(f"Registered model {model_name} from run {run_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-name", required=True)
    args = parser.parse_args()
    
    register_model(args.run_id, args.model_name)