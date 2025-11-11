import os
import re
import string
import warnings
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
import scipy.sparse

from sklearn.model_selection import train_test_split, ParameterSampler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Optional: mac-friendly gradient boosters (skip if not available)
try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except (ImportError, OSError) as e:
    HAS_LIGHTGBM = False
    print(f"Warning: LightGBM not available ({type(e).__name__}), skipping LightGBM experiments")

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except (ImportError, OSError) as e:
    HAS_CATBOOST = False
    print(f"Warning: CatBoost not available ({type(e).__name__}), skipping CatBoost experiments")

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

warnings.filterwarnings("ignore")
pd.set_option('future.no_silent_downcasting', True)

# ========================== CONFIGURATION ==========================
CONFIG = {
    "data_path": "notebooks/data.csv",
    "test_size": 0.2,
    "subset_fraction": 0.25,  # faster tuning subset
    "mlflow_tracking_uri": "https://dagshub.com/Vinayakmane47/imdb_mlops.mlflow/",
    "dagshub_repo_owner": "Vinayakmane47",
    "dagshub_repo_name": "imdb_mlops",
    "experiment_name": "Adaptive Parameter Tuning v3",
    "random_search_iter": 5,
    "skip_threshold": 0.005,  # 0.5% improvement filter
    "early_stop_limit": 5,
}

# ========================== MLflow & DAGsHub ==========================
mlflow.set_tracking_uri(CONFIG["mlflow_tracking_uri"])
dagshub.init(repo_owner=CONFIG["dagshub_repo_owner"],
             repo_name=CONFIG["dagshub_repo_name"], mlflow=True)
mlflow.set_experiment(CONFIG["experiment_name"])

# ========================== TEXT PREPROCESSING ==========================
def lemmatization(text):
    lem = WordNetLemmatizer()
    return " ".join([lem.lemmatize(w) for w in text.split()])

def remove_stop_words(text):
    stop_words = set(stopwords.words("english"))
    return " ".join([w for w in text.split() if w not in stop_words])

def normalize_text(df):
    df['review'] = df['review'].str.lower()
    df['review'] = df['review'].apply(remove_stop_words)
    df['review'] = df['review'].apply(lambda x: re.sub(r'https?://\S+|www\.\S+', '', x))
    df['review'] = df['review'].apply(lambda x: re.sub(r'\d+', '', x))
    df['review'] = df['review'].apply(lambda x: re.sub(f"[{re.escape(string.punctuation)}]", ' ', x))
    df['review'] = df['review'].apply(lemmatization)
    return df

# ========================== LOAD DATA ==========================
def load_data(file_path):
    df = pd.read_csv(file_path)
    df = normalize_text(df)
    df = df[df['sentiment'].isin(['positive', 'negative'])]
    df['sentiment'] = df['sentiment'].replace({'negative': 0, 'positive': 1})
    df['sentiment'] = df['sentiment'].astype(int)  # Ensure integer type
    df = df.sample(frac=CONFIG["subset_fraction"], random_state=42).reset_index(drop=True)
    return df

# ========================== PARAM GRIDS ==========================
NGRAM_RANGES = {'Unigrams': (1, 1), 'Uni+Bigrams': (1, 2)}
MAX_FEATURES_OPTIONS = [5000, 10000, None]

# Build algorithm configs dynamically based on available libraries
ALGORITHM_CONFIGS = {
    'LogisticRegression': {'C': [0.1, 1, 10, 100]},
    'MultinomialNB': {'alpha': [0.1, 0.5, 1.0, 2.0]},
    'RandomForest': {'n_estimators': [100, 200], 'max_depth': [10, 20, None]},
    'GradientBoosting': {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5],
    },
}

# Add LightGBM if available
if HAS_LIGHTGBM:
    ALGORITHM_CONFIGS['LightGBM'] = {
        'n_estimators': [100, 200],
        'num_leaves': [31, 63, 127],
        'learning_rate': [0.01, 0.1],
    }

# Add CatBoost if available
if HAS_CATBOOST:
    ALGORITHM_CONFIGS['CatBoost'] = {
        'iterations': [100, 200],
        'depth': [4, 6, 8],
        'learning_rate': [0.01, 0.1],
    }

# ========================== MODEL FACTORY ==========================
def create_model(name, params):
    if name == 'LogisticRegression':
        return LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42, **params)
    elif name == 'MultinomialNB':
        return MultinomialNB(**params)
    elif name == 'RandomForest':
        return RandomForestClassifier(random_state=42, n_jobs=-1, **params)
    elif name == 'GradientBoosting':
        return GradientBoostingClassifier(random_state=42, **params)
    elif name == 'LightGBM':
        if not HAS_LIGHTGBM:
            raise ValueError("LightGBM is not available")
        return LGBMClassifier(random_state=42, n_jobs=-1, **params)
    elif name == 'CatBoost':
        if not HAS_CATBOOST:
            raise ValueError("CatBoost is not available")
        # Silent mode avoids clutter on Mac terminals
        return CatBoostClassifier(verbose=0, random_state=42, **params)
    else:
        raise ValueError(f"Unknown algorithm: {name}")

# ========================== BASELINES ==========================
def compute_baselines(df):
    print("\nComputing baselines...")
    baselines = {}
    vect = TfidfVectorizer(ngram_range=(1, 1), max_features=10000)
    X = vect.fit_transform(df['review'])
    y = df['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=CONFIG["test_size"], random_state=42)
    for name in ALGORITHM_CONFIGS.keys():
        model = create_model(name, {})
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        baselines[name] = acc
        print(f"  {name}: {acc:.4f}")
    return baselines

# ========================== MAIN TRAINING ==========================
def train_and_evaluate(df):
    baselines = compute_baselines(df)
    best_global = {"accuracy": 0}
    run_count = 0

    with mlflow.start_run(run_name="Adaptive Parameter Tuning v3") as parent:
        for algo, params in ALGORITHM_CONFIGS.items():
            best_acc = baselines[algo]
            no_improve = 0
            print(f"\n=== {algo} ===")

            for ng_name, ng_range in NGRAM_RANGES.items():
                for max_feat in MAX_FEATURES_OPTIONS:
                    sampler = list(ParameterSampler(params, n_iter=CONFIG["random_search_iter"], random_state=42))

                    for hp in sampler:
                        run_count += 1
                        run_name = f"{algo}_{ng_name}_{max_feat or 'All'}_{'_'.join([f'{k}{v}' for k,v in hp.items()])}"[:250]

                        with mlflow.start_run(run_name=run_name, nested=True):
                            try:
                                vect = TfidfVectorizer(ngram_range=ng_range, max_features=max_feat)
                                X = vect.fit_transform(df['review'])
                                y = df['sentiment']
                                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=CONFIG["test_size"], random_state=42)

                                model = create_model(algo, hp)
                                model.fit(X_train, y_train)
                                y_pred = model.predict(X_test)

                                metrics = {
                                    "accuracy": accuracy_score(y_test, y_pred),
                                    "precision": precision_score(y_test, y_pred),
                                    "recall": recall_score(y_test, y_pred),
                                    "f1_score": f1_score(y_test, y_pred),
                                }

                                mlflow.log_params({
                                    "algorithm": algo,
                                    "ngram_range": str(ng_range),
                                    "max_features": max_feat,
                                    **hp
                                })
                                mlflow.log_metrics(metrics)

                                # Skip low-impact runs
                                if metrics["accuracy"] < baselines[algo] + CONFIG["skip_threshold"]:
                                    print(f" ⏭️  Skipped (no improvement): {run_name}")
                                    continue

                                # Log model only if strong
                                if metrics["accuracy"] >= best_acc - 0.01:
                                    mlflow.sklearn.log_model(model, "model")

                                # Track bests
                                if metrics["accuracy"] > best_acc:
                                    best_acc = metrics["accuracy"]
                                    no_improve = 0
                                    best_global = {
                                        "algorithm": algo,
                                        "ngram": ng_name,
                                        "max_features": max_feat,
                                        "hyperparams": hp,
                                        **metrics,
                                    }
                                else:
                                    no_improve += 1

                                print(f" [{run_count}] {algo} | Acc:{metrics['accuracy']:.4f} | F1:{metrics['f1_score']:.4f}")

                                if no_improve >= CONFIG["early_stop_limit"]:
                                    print(f" 🛑 Early stopping {algo} (no improvement in {CONFIG['early_stop_limit']} runs).")
                                    break
                            except Exception as e:
                                print(f"Error in {run_name}: {e}")
                                mlflow.log_param("error", str(e))

    print("\n" + "="*80)
    print("BEST CONFIGURATION FOUND:")
    print("="*80)
    for k, v in best_global.items():
        print(f"{k}: {v}")
    print("="*80)

# ========================== EXECUTION ==========================
if __name__ == "__main__":
    print("Starting Adaptive Parameter Tuning v3...")
    print("=" * 80)
    df = load_data(CONFIG["data_path"])
    print(f"Loaded {len(df)} samples (subset for tuning).")
    train_and_evaluate(df)
    print("\nExperiment completed successfully ✅")
