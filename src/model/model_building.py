import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
import yaml
from src.logger import logging


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logging.info('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the data: %s', e)
        raise

def train_model(X_train: np.ndarray, y_train: np.ndarray, C: float, 
                penalty: str, solver: str, max_iter: int) -> LogisticRegression:
    """Train the Logistic Regression model."""
    try:
        clf = LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=max_iter)
        clf.fit(X_train, y_train)
        logging.info('Model training completed with C=%s, penalty=%s, solver=%s, max_iter=%s', 
                     C, penalty, solver, max_iter)
        return clf
    except Exception as e:
        logging.error('Error during model training: %s', e)
        raise

def save_model(model, file_path: str) -> None:
    """Save the trained model to a file."""
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logging.info('Model saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the model: %s', e)
        raise

def main():
    try:
        # Load parameters from params.yaml
        with open('params.yaml', 'r') as file:
            params = yaml.safe_load(file)
        
        if 'model_building' not in params:
            raise ValueError("'model_building' section not found in params.yaml")
        
        model_params = params['model_building']
        
        # Load all parameters from YAML (no defaults - must be in YAML)
        C = model_params['C']
        penalty = model_params['penalty']
        solver = model_params['solver']
        max_iter = model_params.get('max_iter', 1000)  # Optional with reasonable default

        train_data = load_data('./data/processed/train_bow.csv')
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

        clf = train_model(X_train, y_train, C=C, penalty=penalty, solver=solver, max_iter=max_iter)
        
        save_model(clf, 'models/model.pkl')
    except KeyError as e:
        logging.error('Required parameter missing in params.yaml: %s', e)
        raise
    except Exception as e:
        logging.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")
        raise

if __name__ == '__main__':
    main()