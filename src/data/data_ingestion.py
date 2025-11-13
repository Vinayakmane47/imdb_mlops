# data ingestion
import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)

import os
from sklearn.model_selection import train_test_split
import yaml
import logging
from src.logger import logging
from src.connections import s3_connection
from dotenv import load_dotenv
load_dotenv()

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logging.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logging.error('YAML error: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error: %s', e)
        raise

def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(data_url)
        logging.info('Data loaded from %s', data_url)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the data: %s', e)
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data."""
    try:
        # df.drop(columns=['tweet_id'], inplace=True)
        logging.info("pre-processing...")
        final_df = df[df['sentiment'].isin(['positive', 'negative'])]
        final_df['sentiment'] = final_df['sentiment'].replace({'positive': 1, 'negative': 0})
        logging.info('Data preprocessing completed')
        return final_df
    except KeyError as e:
        logging.error('Missing column in the dataframe: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error during preprocessing: %s', e)
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the train and test datasets."""
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        logging.debug('Train and test data saved to %s', raw_data_path)
    except Exception as e:
        logging.error('Unexpected error occurred while saving the data: %s', e)
        raise

def load_data_from_s3() -> pd.DataFrame:
    """Try to load data from S3, returns None if fails or credentials not available."""
    try:
        bucket_name = os.getenv("AWS_BUCKET_NAME")
        aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        region_name = os.getenv("AWS_REGION", "us-east-1")
        
        # Check if all required env vars are set
        if not all([bucket_name, aws_access_key, aws_secret_key]):
            logging.warning("AWS credentials not fully configured. Skipping S3 fetch.")
            return None
        
        s3 = s3_connection.s3_operations(
            bucket_name=bucket_name,
            aws_access_key=aws_access_key,
            aws_secret_key=aws_secret_key,
            region_name=region_name,
        )
        df = s3.fetch_file_from_s3("data.csv")
        return df
    except Exception as e:
        logging.warning(f"Failed to load data from S3: {e}")
        return None

def main():
    try:
        params = load_params(params_path='params.yaml')
        test_size = params['data_ingestion']['test_size']
        
        # Try to load from S3 first
        df = load_data_from_s3()
        
        # Fallback to public URL if S3 fails or credentials not available
        if df is None or df.empty:
            logging.info("S3 fetch failed or returned empty. Using fallback URL...")
            fallback_url = 'https://raw.githubusercontent.com/vikashishere/Datasets/refs/heads/main/data.csv'
            df = load_data(data_url=fallback_url)
        
        if df is None or df.empty:
            raise ValueError("Failed to load data from both S3 and fallback URL")
        
        final_df = preprocess_data(df)
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
        save_data(train_data, test_data, data_path='./data')
        
    except Exception as e:
        logging.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")
        raise

if __name__ == '__main__':
    main()

    