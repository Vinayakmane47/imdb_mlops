import setuptools
import os
import re
import string
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)

import numpy as np
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import scipy.sparse

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# ========================== CONFIGURATION ==========================
CONFIG = {
    "data_path": "notebooks/data.csv",
    "test_size": 0.2,
    "mlflow_tracking_uri": "https://dagshub.com/Vinayakmane47/imdb_mlops.mlflow/",
    "dagshub_repo_owner": "Vinayakmane47",
    "dagshub_repo_name": "imdb_mlops",
    "experiment_name": "High-Impact Parameter Tuning"
}

# ========================== SETUP MLflow & DAGSHUB ==========================
mlflow.set_tracking_uri(CONFIG["mlflow_tracking_uri"])
dagshub.init(repo_owner=CONFIG["dagshub_repo_owner"], repo_name=CONFIG["dagshub_repo_name"], mlflow=True)
mlflow.set_experiment(CONFIG["experiment_name"])

# ========================== TEXT PREPROCESSING ==========================
def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

def remove_stop_words(text):
    stop_words = set(stopwords.words("english"))
    return " ".join([word for word in text.split() if word not in stop_words])

def removing_numbers(text):
    return ''.join([char for char in text if not char.isdigit()])

def lower_case(text):
    return text.lower()

def removing_punctuations(text):
    return re.sub(f"[{re.escape(string.punctuation)}]", ' ', text)

def removing_urls(text):
    return re.sub(r'https?://\S+|www\.\S+', '', text)

def normalize_text(df):
    try:
        df['review'] = df['review'].apply(lower_case)
        df['review'] = df['review'].apply(remove_stop_words)
        df['review'] = df['review'].apply(removing_numbers)
        df['review'] = df['review'].apply(removing_punctuations)
        df['review'] = df['review'].apply(removing_urls)
        df['review'] = df['review'].apply(lemmatization)
        return df
    except Exception as e:
        print(f"Error during text normalization: {e}")
        raise

# ========================== LOAD & PREPROCESS DATA ==========================
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        df = normalize_text(df)
        df = df[df['sentiment'].isin(['positive', 'negative'])]
        df['sentiment'] = df['sentiment'].replace({'negative': 0, 'positive': 1}).infer_objects(copy=False)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

# ========================== HIGH-IMPACT PARAMETER CONFIGURATIONS ==========================
# N-gram ranges - Most impactful for capturing context
NGRAM_RANGES = {
    'Unigrams': (1, 1),
    'Unigrams+Bigrams': (1, 2),
    'Unigrams+Trigrams': (1, 3)
}

# max_features - CRITICAL for performance and speed
# Limiting features reduces noise and improves generalization
# OPTIMIZED: Reduced from 5 to 3 options (most impactful values)
MAX_FEATURES_OPTIONS = [10000, 20000, None]  # None = use all features

# min_df - Filter rare words that appear in very few documents
# OPTIMIZED: Reduced from 3 to 2 options
MIN_DF_OPTIONS = [1, 2]  # Minimum document frequency

# max_df - Filter very common words (stopwords-like)
# OPTIMIZED: Reduced from 3 to 2 options
MAX_DF_OPTIONS = [0.95, 1.0]  # Maximum document frequency

# sublinear_tf - Apply sublinear scaling to term frequency (log(1 + tf))
# This helps reduce the impact of very frequent terms
SUBLINEAR_TF_OPTIONS = [False, True]

# Model hyperparameters - These significantly impact accuracy
# OPTIMIZED: Reduced hyperparameter ranges for faster execution
ALGORITHM_CONFIGS = {
    'LogisticRegression': {
        'C': [0.1, 1, 10, 100],  # Regularization strength (lower = stronger)
        'penalty': ['l2'],  # L2 regularization typically works best
        'solver': ['lbfgs']  # Good for small-medium datasets
    },
    'MultinomialNB': {
        'alpha': [0.5, 1.0, 2.0]  # Smoothing parameter (reduced from 4 to 3)
    },
    'RandomForest': {
        'n_estimators': [100, 200],  # Reduced from 3 to 2
        'max_depth': [10, 20],  # Reduced from 3 to 2 (removed None)
        'min_samples_split': [2, 5]
    },
    'GradientBoosting': {
        'n_estimators': [100, 200],
        'learning_rate': [0.1, 0.2],  # Reduced from 3 to 2 (removed 0.01)
        'max_depth': [3, 5]  # Reduced from 3 to 2 (removed 7)
    }
}

# ========================== TRAIN & EVALUATE MODELS ==========================
def train_and_evaluate(df):
    """
    Train and evaluate models with high-impact parameters for accuracy improvement.
    Tests: n-grams, max_features, min_df, max_df, sublinear_tf, and model hyperparameters.
    """
    experiment_count = 0
    best_accuracy = 0
    best_config = None
    
    with mlflow.start_run(run_name="High-Impact Parameter Tuning") as parent_run:
        # Test each algorithm with its hyperparameter grid
        for algo_name, algo_params in ALGORITHM_CONFIGS.items():
            for ngram_name, ngram_range in NGRAM_RANGES.items():
                for max_feat in MAX_FEATURES_OPTIONS:
                    for min_df in MIN_DF_OPTIONS:
                        for max_df in MAX_DF_OPTIONS:
                            for sublinear_tf in SUBLINEAR_TF_OPTIONS:
                                # Get hyperparameter combinations for current algorithm
                                hyperparam_combos = get_hyperparameter_combinations(algo_name, algo_params)
                                
                                for hyperparams in hyperparam_combos:
                                    experiment_count += 1
                                    
                                    # Create descriptive run name
                                    max_feat_str = f"{max_feat//1000}K" if max_feat else "All"
                                    run_name = f"{algo_name} | {ngram_name} | {max_feat_str}feat | min_df{min_df} | max_df{max_df} | sublinear{sublinear_tf}"
                                    for key, val in hyperparams.items():
                                        run_name += f" | {key}{val}"
                                    
                                    with mlflow.start_run(run_name=run_name[:250], nested=True) as child_run:  # MLflow has 250 char limit
                                        try:
                                            # Create vectorizer with optimized parameters
                                            vectorizer = TfidfVectorizer(
                                                ngram_range=ngram_range,
                                                max_features=max_feat,
                                                min_df=min_df,
                                                max_df=max_df,
                                                sublinear_tf=sublinear_tf,
                                                lowercase=True,
                                                strip_accents='unicode'
                                            )
                                            
                                            # Feature extraction
                                            X = vectorizer.fit_transform(df['review'])
                                            y = df['sentiment']
                                            
                                            # Get feature statistics
                                            n_features = X.shape[1]
                                            
                                            X_train, X_test, y_train, y_test = train_test_split(
                                                X, y, test_size=CONFIG["test_size"], random_state=42
                                            )
                                            
                                            # Create model with hyperparameters
                                            model = create_model(algo_name, hyperparams)
                                            
                                            # Log all parameters
                                            mlflow.log_params({
                                                "vectorizer": "TF-IDF",
                                                "algorithm": algo_name,
                                                "ngram_range": str(ngram_range),
                                                "ngram_name": ngram_name,
                                                "max_features": max_feat if max_feat else "None",
                                                "min_df": min_df,
                                                "max_df": max_df,
                                                "sublinear_tf": sublinear_tf,
                                                "n_features": n_features,
                                                "test_size": CONFIG["test_size"],
                                                **hyperparams
                                            })
                                            
                                            # Train model
                                            model.fit(X_train, y_train)
                                            
                                            # Evaluate model
                                            y_pred = model.predict(X_test)
                                            metrics = {
                                                "accuracy": accuracy_score(y_test, y_pred),
                                                "precision": precision_score(y_test, y_pred),
                                                "recall": recall_score(y_test, y_pred),
                                                "f1_score": f1_score(y_test, y_pred)
                                            }
                                            mlflow.log_metrics(metrics)
                                            
                                            # Log model (only for top performers to save space)
                                            if metrics['accuracy'] > best_accuracy - 0.01:  # Log if within 1% of best
                                                input_example = X_test[:5] if not scipy.sparse.issparse(X_test) else X_test[:5].toarray()
                                                mlflow.sklearn.log_model(model, "model", input_example=input_example)
                                            
                                            # Track best configuration
                                            if metrics['accuracy'] > best_accuracy:
                                                best_accuracy = metrics['accuracy']
                                                best_config = {
                                                    'algorithm': algo_name,
                                                    'ngram': ngram_name,
                                                    'max_features': max_feat,
                                                    'min_df': min_df,
                                                    'max_df': max_df,
                                                    'sublinear_tf': sublinear_tf,
                                                    'hyperparams': hyperparams,
                                                    'accuracy': metrics['accuracy'],
                                                    'f1': metrics['f1_score']
                                                }
                                            
                                            # Print results
                                            print(f"\n[{experiment_count}] {algo_name} | {ngram_name} | Features:{n_features} | Acc:{metrics['accuracy']:.4f} | F1:{metrics['f1_score']:.4f}")
                                            
                                        except Exception as e:
                                            print(f"Error in {run_name}: {e}")
                                            mlflow.log_param("error", str(e))
    
    # Print best configuration
    print("\n" + "=" * 80)
    print("BEST CONFIGURATION FOUND:")
    print("=" * 80)
    if best_config:
        print(f"Algorithm: {best_config['algorithm']}")
        print(f"N-gram: {best_config['ngram']}")
        print(f"Max Features: {best_config['max_features']}")
        print(f"Min DF: {best_config['min_df']}")
        print(f"Max DF: {best_config['max_df']}")
        print(f"Sublinear TF: {best_config['sublinear_tf']}")
        print(f"Hyperparameters: {best_config['hyperparams']}")
        print(f"Accuracy: {best_config['accuracy']:.4f}")
        print(f"F1 Score: {best_config['f1_score']:.4f}")
    print("=" * 80)

def get_hyperparameter_combinations(algo_name, algo_params):
    """Generate all combinations of hyperparameters for an algorithm."""
    from itertools import product
    
    param_names = list(algo_params.keys())
    param_values = list(algo_params.values())
    
    combinations = []
    for combo in product(*param_values):
        combinations.append(dict(zip(param_names, combo)))
    
    return combinations

def create_model(algo_name, hyperparams):
    """Create a model instance with given hyperparameters."""
    if algo_name == 'LogisticRegression':
        return LogisticRegression(
            max_iter=1000,
            random_state=42,
            **hyperparams
        )
    elif algo_name == 'MultinomialNB':
        return MultinomialNB(**hyperparams)
    elif algo_name == 'RandomForest':
        return RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
            **hyperparams
        )
    elif algo_name == 'GradientBoosting':
        return GradientBoostingClassifier(
            random_state=42,
            **hyperparams
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")


# ========================== EXECUTION ==========================
if __name__ == "__main__":
    print("Starting High-Impact Parameter Tuning Experiment...")
    print("=" * 80)
    df = load_data(CONFIG["data_path"])
    print(f"Loaded {len(df)} samples")
    print(f"\nTesting Parameters:")
    print(f"  - N-gram ranges: {len(NGRAM_RANGES)} configurations")
    print(f"  - Max features: {len(MAX_FEATURES_OPTIONS)} options")
    print(f"  - Min DF: {len(MIN_DF_OPTIONS)} options")
    print(f"  - Max DF: {len(MAX_DF_OPTIONS)} options")
    print(f"  - Sublinear TF: {len(SUBLINEAR_TF_OPTIONS)} options")
    print(f"  - Algorithms: {len(ALGORITHM_CONFIGS)} with hyperparameter tuning")
    
    # Calculate total experiments
    total_experiments = 0
    for algo_name, algo_params in ALGORITHM_CONFIGS.items():
        algo_combos = 1
        for param_values in algo_params.values():
            algo_combos *= len(param_values)
        total_experiments += algo_combos
    
    total_experiments *= len(NGRAM_RANGES) * len(MAX_FEATURES_OPTIONS) * len(MIN_DF_OPTIONS) * len(MAX_DF_OPTIONS) * len(SUBLINEAR_TF_OPTIONS)
    
    print(f"\nTotal experiments: {total_experiments}")
    print("=" * 80)
    print("\nThis experiment tests the most impactful parameters for accuracy:")
    print("  - N-gram ranges (capturing word context)")
    print("  - Max features (reducing noise, improving generalization)")
    print("  - Min/Max document frequency (filtering rare/common words)")
    print("  - Sublinear TF scaling (reducing impact of frequent terms)")
    print("  - Model hyperparameters (C, alpha, n_estimators, learning_rate, etc.)")
    print("=" * 80)
    print()
    
    train_and_evaluate(df)
    print("\nExperiment completed!")

