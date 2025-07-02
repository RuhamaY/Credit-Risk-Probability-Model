from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import logging
import os
import mlflow # Import mlflow to load from registry
import numpy as np # Needed for potential numpy array checks in prediction output

# Import Pydantic models from the new file
from .pydantic_models import CustomerFeatures, PredictionResult

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Credit Risk Scoring API",
              description="API for predicting customer credit risk")

# --- Constants for Credit Score and Risk Categories ---
MIN_CREDIT_SCORE = 300
MAX_CREDIT_SCORE = 850
HIGH_RISK_THRESHOLD = 0.7
MEDIUM_RISK_THRESHOLD = 0.3

# --- Model and Preprocessor Loading ---
PREPROCESSOR_PATH = '../models/preprocessor.pkl'

model = None
preprocessor = None

@app.on_event("startup")
async def load_resources():
    global model, preprocessor

    # --- Load Preprocessor ---
    try:
        if not os.path.exists(PREPROCESSOR_PATH):
            raise FileNotFoundError(f"Preprocessor file not found at {PREPROCESSOR_PATH}. "
                                    "Ensure your train.py saves it after fitting.")
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        logger.info(f"Successfully loaded preprocessor from {PREPROCESSOR_PATH}")
    except FileNotFoundError as e:
        logger.error(f"Failed to load preprocessor: {e}. Raw data cannot be processed.")
        raise RuntimeError(f"Application cannot start without preprocessor: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading the preprocessor: {e}")
        raise RuntimeError(f"Application cannot start due to preprocessor loading error: {e}")

    # --- Load Model from MLflow Model Registry ---
    # Using 'Production' stage, or you could use 'latest' if preferred.
    # Make sure this name ('CreditRiskClassifier') matches what you registered in train.py.
    mlflow_model_uri = "models:/CreditRiskClassifier/Production"
    try:
        model = mlflow.pyfunc.load_model(mlflow_model_uri)
        logger.info(f"Successfully loaded model from MLflow Registry: {mlflow_model_uri}")
    except Exception as e:
        logger.error(f"Failed to load model from MLflow Registry ({mlflow_model_uri}): {e}")
        raise RuntimeError(f"Application cannot start due to MLflow model loading error: {e}")


# --- API Endpoints ---
@app.post("/predict", response_model=PredictionResult)
async def predict(features: CustomerFeatures):
    """
    Predict credit risk for a given customer based on their transactional features,
    and provide recommended credit limit and term.
    """
    if model is None or preprocessor is None:
        logger.error("Model or preprocessor not loaded. Application is not ready.")
        raise HTTPException(status_code=503, detail="Service not ready: Model or preprocessor not loaded.")

    # 1. Convert incoming Pydantic model to a Pandas DataFrame
    # Use .model_dump() for Pydantic v2+; use .dict() for Pydantic v1.
    input_df = pd.DataFrame([features.model_dump()])

    # 2. Apply the SAME preprocessing steps as used during training
    try:
        processed_input_data = preprocessor.transform(input_df)
        logger.info(f"Successfully preprocessed input data for customer {features.customer_id}.")
    except Exception as e:
        logger.error(f"Error during preprocessing for customer {features.customer_id}: {e}")
        raise HTTPException(status_code=400, detail=f"Data preprocessing failed: {e}")

    # 3. Predict probability
    try:
        # mlflow.pyfunc.load_model returns a pyfunc model.
        # Its predict method might return probabilities directly or require .predict_proba.
        # We need to adapt based on how the underlying model (sklearn) was logged.
        # If it was logged as mlflow.sklearn.log_model, it usually supports predict_proba.
        if hasattr(model, 'predict_proba') and callable(model.predict_proba):
            proba = model.predict_proba(processed_input_data)[0, 1]
        elif hasattr(model, 'predict') and callable(model.predict):
            # If predict returns a single value or an array, try to interpret it as probability
            # This path is less common for classifiers if predict_proba is available
            prediction_output = model.predict(processed_input_data)
            if isinstance(prediction_output, (list, np.ndarray)) and len(prediction_output.shape) > 1 and prediction_output.shape[1] > 1:
                proba = prediction_output[0, 1] # Assumed output like [[prob_class_0, prob_class_1]]
            elif isinstance(prediction_output, (list, np.ndarray)) and prediction_output.ndim == 1:
                proba = prediction_output[0] # Assumed direct probability output for single sample
            else:
                raise ValueError("Model prediction output format is ambiguous for probability extraction.")
        else:
            raise ValueError("Model does not have 'predict_proba' or 'predict' method.")

        logger.info(f"Prediction for customer {features.customer_id}: Probability = {proba:.4f}")
    except Exception as e:
        logger.error(f"Error during model prediction for customer {features.customer_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")

    # 4. Convert to credit score (300-850 scale)
    score = min(MAX_CREDIT_SCORE, max(MIN_CREDIT_SCORE, int(MAX_CREDIT_SCORE - (proba * (MAX_CREDIT_SCORE - MIN_CREDIT_SCORE)))))

    # 5. Determine risk category
    if proba > HIGH_RISK_THRESHOLD:
        category = "High"
    elif proba > MEDIUM_RISK_THRESHOLD:
        category = "Medium"
    else:
        category = "Low"


    # 6. --- Example Logic for Recommended Limit and Term (CUSTOMIZE THIS!) ---
    recommended_limit = None
    recommended_term = None

    if score >= 750: # Excellent Credit
        recommended_limit = 10000.0
        recommended_term = 60 # months
    elif score >= 650: # Good Credit
        recommended_limit = 5000.0
        recommended_term = 36 # months
    elif score >= 550: # Fair Credit
        recommended_limit = 2000.0
        recommended_term = 24 # months
    else: # Poor Credit
        recommended_limit = 500.0
        recommended_term = 12 # months

    return {
        "customer_id": features.customer_id,
        "risk_probability": float(proba),
        "risk_category": category,
        "credit_score": score,
        "recommended_limit": recommended_limit,
        "recommended_term": recommended_term
    }

@app.get("/", summary="Health Check")
async def health_check():
    """
    Checks the health of the API.
    """
    return {"status": "healthy"}
