import pandas as pd
import numpy as np
import os
import mlflow 
import mlflow.sklearn 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier 
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix)
import joblib
import warnings


# Suppress warnings for cleaner output during training.
warnings.filterwarnings("ignore")

def load_processed_data(file_path):
    """Loads the model-ready data from the specified path."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Processed data file not found at {file_path}")
    return pd.read_csv(file_path)


def load_data():
    """
    Loads preprocessed data and separates features (X) and target (y).
    Includes robust handling for the target column name.
    """
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        processed_data_path = os.path.normpath(os.path.join(script_dir, '../data/processed/model_ready_data.csv'))
        
        data = load_processed_data(processed_data_path)
        
        # Define expected target column names, including the one from ColumnTransformer.
        possible_target_cols = ['target__is_high_risk', 'is_high_risk', 'high_risk', 'risk_flag']
        target_col = None
        for col in possible_target_cols:
            if col in data.columns:
                target_col = col
                break
                
        if not target_col:
            raise ValueError(
                f"No target column found. Expected one of: {', '.join(possible_target_cols)}"
            )
            
        X = data.drop(columns=[target_col])
        y = data[target_col]
        
        print(f"Loaded data with {X.shape[0]} rows and {X.shape[1]} features. Target column: '{target_col}'.")
        return X, y
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

def evaluate_model(y_true, y_pred, y_proba):
    """Calculates comprehensive classification evaluation metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0), 
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_proba),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }
    return metrics

def train_models():
    """Trains, evaluates, and tracks multiple models using MLflow."""
    try:
        # Load preprocessed data
        X, y = load_data()
        
        # Ensure target variable has at least two classes for classification.
        if len(np.unique(y)) < 2:
            raise ValueError("Target variable must have at least 2 classes for classification.")
            
        # Split data into training and testing sets, maintaining target distribution.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"Data split into {X_train.shape[0]} training and {X_test.shape[0]} testing samples.")
        print(f"Training target distribution:\n{y_train.value_counts(normalize=True)}")
        print(f"Testing target distribution:\n{y_test.value_counts(normalize=True)}")

        # --- REMOVED TEMPORARY: REDUCE DATASET SIZE FOR TESTING ---
        # The script will now train on the full X_train, y_train
        # --- END REMOVED TEMPORARY SECTION ---

        mlflow.set_tracking_uri("./mlruns") 
        mlflow.set_experiment("CreditRiskModeling")
        print("DEBUG: MLflow tracking URI and experiment set.")
            
        # Define models and their hyperparameter grids for GridSearchCV.
        models_to_train = {
            "LogisticRegression": {
                'model': LogisticRegression(max_iter=1000, random_state=42),
                'params': {
                    'C': [0.001, 0.01, 0.1, 1, 10],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear'] 
                }
            },
            "RandomForestClassifier": {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10]
                }
            },
             "GradientBoostingClassifier": {
                "model": GradientBoostingClassifier(random_state=42),
                "params": {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1],
                    'max_depth': [3, 5]
                }
            }
        }
        
        best_model_overall = None
        best_roc_auc_score = -1.0 
        
        print("DEBUG: About to start iterating through models_to_train (all models active).")

        for model_name, config in models_to_train.items():
            with mlflow.start_run(run_name=model_name) as run:
                print(f"\n--- Training {model_name} ---")
                
                # Hyperparameter tuning using GridSearchCV.
                grid_search = GridSearchCV(
                    config['model'],
                    config['params'],
                    cv=5, 
                    scoring='roc_auc', 
                    n_jobs=1, # Keeping n_jobs=1 for stability with full dataset
                    verbose=1 
                )
                
                print(f"DEBUG: Starting GridSearchCV fit for {model_name}...")
                grid_search.fit(X_train, y_train)
                print(f"DEBUG: GridSearchCV fit completed for {model_name}.")
                
                # Get the best estimator from GridSearchCV.
                model = grid_search.best_estimator_
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1] 
                
                # Evaluate model performance.
                metrics = evaluate_model(y_test, y_pred, y_proba)
                
                mlflow.log_params(grid_search.best_params_)
                mlflow.log_metrics({k: v for k, v in metrics.items() if k != 'confusion_matrix'})
                mlflow.log_dict(metrics['confusion_matrix'], "confusion_matrix.json")
                mlflow.sklearn.log_model(model, name=model_name,
                                         input_example=X_train.head(1))
                
                print(f"Best Params: {grid_search.best_params_}")
                print(f"ROC AUC: {metrics['roc_auc']:.4f}")
                print(f"F1 Score: {metrics['f1']:.4f}")
                print(f"Confusion Matrix:\n{np.array(metrics['confusion_matrix'])}") 

                # Track the best performing model overall.
                if metrics['roc_auc'] > best_roc_auc_score:
                    best_roc_auc_score = metrics['roc_auc']
                    best_model_overall = {
                        "name": model_name,
                        "estimator": model,
                        "metrics": metrics,
                        "run_id": run.info.run_id 
                    }
        
        # Save the best model locally and register it in MLflow Model Registry.
        if best_model_overall:
            os.makedirs('../models', exist_ok=True)
            joblib.dump(best_model_overall['estimator'], '../models/best_model.pkl')
            
            with mlflow.start_run(run_name="Register_Best_Overall_Model") as reg_run:
                mlflow.sklearn.log_model(
                    best_model_overall['estimator'],
                    name="best_credit_risk_model_artifact",
                    registered_model_name="CreditRiskClassifier",
                    input_example=X_train.head(1)
                )
                mlflow.log_params(best_model_overall['estimator'].get_params())
                mlflow.log_metrics({k: v for k, v in best_model_overall['metrics'].items() if k != 'confusion_matrix'}) 
                mlflow.log_dict(best_model_overall['metrics']['confusion_matrix'], "best_confusion_matrix.json") 

                print(f"\nBest overall model '{best_model_overall['name']}' saved locally "
                      f"and registered as 'CreditRiskClassifier' in MLflow Model Registry.")
                print(f"Best overall model ROC AUC: {best_roc_auc_score:.4f}")
        else:
            print("\nNo models were trained or evaluated successfully.")
            
    except Exception as e:
        print(f"\nAn error occurred during the model training process: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    train_models()
