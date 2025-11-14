# IMDB Sentiment Analysis - MLOps Pipeline

A production-ready MLOps pipeline for sentiment analysis of IMDB movie reviews, demonstrating end-to-end machine learning operations from data ingestion to cloud deployment.

## Overview

This project implements a complete MLOps workflow featuring automated data pipelines, model versioning, continuous integration/deployment, and cloud-based serving infrastructure. The system processes IMDB movie reviews to classify sentiment (positive/negative) using a Logistic Regression model with TF-IDF feature engineering.

## Key Features

- **Automated Data Pipeline**: Version-controlled data processing using DVC
- **Model Versioning**: MLflow-based experiment tracking and model registry
- **CI/CD Integration**: Automated testing, building, and deployment via GitHub Actions
- **Cloud Deployment**: Containerized Flask application on AWS EKS
- **Production Monitoring**: Prometheus metrics and Grafana dashboards
- **Reproducible Experiments**: Parameterized pipeline with YAML configuration

## Technology Stack

**MLOps & Data**
- DVC (Data Version Control) for data and pipeline versioning
- MLflow for experiment tracking and model registry
- DAGSHub for MLflow backend hosting

**Machine Learning**
- Scikit-learn (Logistic Regression)
- TF-IDF vectorization with n-grams (1-3)
- NLTK for text preprocessing

**Infrastructure**
- Docker for containerization
- AWS ECR for container registry
- AWS EKS for Kubernetes orchestration
- AWS S3 for data storage

**CI/CD & Monitoring**
- GitHub Actions for automation
- Prometheus for metrics collection
- Grafana for visualization

## Architecture

### Data Pipeline Flow

```
Data Ingestion → Preprocessing → Feature Engineering → Model Training → Evaluation → Registration
     │                │                  │                  │              │            │
  S3/URL          Text Cleaning      TF-IDF Vector      Logistic      Metrics     MLflow
  Download        Lemmatization      (20K features)     Regression    (Acc/Prec)  Registry
  Train/Test      Stopword Removal   N-grams (1-3)      (C=10, L2)    (Recall)    Production
  Split (80/20)   URL/Number Remove                     (lbfgs)       (AUC)
```

### System Components

1. **Development Environment**: Local development with DVC pipeline execution
2. **CI/CD Pipeline**: Automated testing and deployment on code push
3. **Cloud Infrastructure**: AWS EKS cluster hosting containerized Flask application
4. **Monitoring Stack**: Prometheus scraping application metrics, visualized in Grafana
5. **Model Registry**: MLflow tracking experiments and managing model versions

## Project Structure

```
imdb_mlops/
├── .github/workflows/        # CI/CD pipeline configuration
├── data/                     # Data directories (DVC tracked)
│   ├── raw/                  # Original datasets
│   ├── interim/              # Preprocessed data
│   └── processed/            # Feature-engineered data
├── flask_app/                # Production Flask application
│   ├── app.py               # Main application with Prometheus metrics
│   └── templates/           # Web UI templates
├── models/                   # Trained models (DVC tracked)
├── notebooks/                # Experimental notebooks
├── reports/                  # Evaluation metrics
├── scripts/                  # Utility scripts
├── src/                      # Source code modules
│   ├── data/                # Data ingestion and preprocessing
│   ├── features/            # Feature engineering
│   ├── model/               # Model training and evaluation
│   └── connections/         # AWS connection utilities
├── tests/                    # Unit and integration tests
├── dvc.yaml                  # DVC pipeline definition
├── params.yaml               # Pipeline parameters
└── deployment.yaml           # Kubernetes deployment configuration
```

## Model Configuration

**Algorithm**: Logistic Regression  
**Features**: TF-IDF vectorization
- Maximum features: 20,000
- N-gram range: (1, 3) - Unigrams, Bigrams, Trigrams
- Minimum document frequency: 1
- Maximum document frequency: 1.0
- Sublinear TF: Enabled

**Hyperparameters**:
- Regularization (C): 10
- Penalty: L2
- Solver: lbfgs
- Maximum iterations: 1000

## Setup

### Prerequisites

- Python 3.10+
- Conda
- Docker
- AWS CLI configured
- kubectl configured
- DVC installed

### Installation

```bash
# Clone repository
git clone <repository-url>
cd imdb_mlops

# Create conda environment
conda create -n atlas python=3.10
conda activate atlas

# Install dependencies
pip install -r requirements.txt
pip install -r flask_app/requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"

# Configure environment variables
# Create .env file with:
# DAGSHUB_TOKEN=your_token
# AWS_ACCESS_KEY_ID=your_key
# AWS_SECRET_ACCESS_KEY=your_secret
```

## Usage

### Running the Pipeline

```bash
# Execute complete pipeline
dvc repro

# Run specific stage
dvc repro model_building

# Pull data from remote storage
dvc pull

# Push artifacts to remote storage
dvc push
```

### Local Development

```bash
# Start Flask application
cd flask_app
python app.py

# Access application
# http://localhost:5005

# View metrics
# http://localhost:5005/metrics
```

### Testing

```bash
# Run all tests
python -m unittest discover tests

# Run specific test suite
python -m unittest tests.test_model
python -m unittest tests.test_flask_app
```

## CI/CD Pipeline

The GitHub Actions workflow automates the following:

1. **Code Validation**: Checkout and dependency installation
2. **Pipeline Execution**: Run complete DVC pipeline (data → model)
3. **Testing**: Execute unit and integration tests
4. **Model Promotion**: Promote validated model to Production in MLflow
5. **Container Build**: Build Docker image with application
6. **Registry Push**: Push image to AWS ECR
7. **Kubernetes Deployment**: Deploy to EKS cluster with automatic rollout

The pipeline triggers on push to main branch and pull requests.

## Deployment

The application is deployed as a containerized service on AWS EKS:

- **Container Registry**: AWS ECR
- **Orchestration**: Kubernetes (EKS)
- **Service Type**: LoadBalancer
- **Replicas**: 2 pods for high availability
- **Resource Limits**: 512Mi memory, 1 CPU per pod

Deployment is automated via CI/CD. Manual deployment:

```bash
kubectl apply -f deployment.yaml
kubectl rollout restart deployment flask-app
```

## Monitoring

### Application Metrics

The Flask application exposes Prometheus metrics at `/metrics`:

- `app_request_count`: Total HTTP requests (by method and endpoint)
- `app_request_latency_seconds`: Request latency distribution
- `model_prediction_count`: Prediction counts by class
- `app_cost_per_request_usd`: Estimated compute cost per request
- `app_total_cost_usd`: Cumulative compute cost

### Model Metrics

Model evaluation metrics are logged to MLflow and stored in `reports/metrics.json`:
- Accuracy
- Precision
- Recall
- AUC (Area Under Curve)

### Visualization

- **MLflow UI**: `https://dagshub.com/Vinayakmane47/imdb_mlops.mlflow`
- **Grafana**: Configured to visualize Prometheus metrics
- **Local Reports**: `reports/metrics.json` for programmatic access

## Configuration

Pipeline parameters are centralized in `params.yaml`. Modify parameters and run `dvc repro` to retrain with new configurations:

```yaml
data_ingestion:
  test_size: 0.20

feature_engineering:
  max_features: 20000
  ngram_range: [1, 3]
  min_df: 1
  max_df: 1.0
  sublinear_tf: true

model_building:
  C: 10
  penalty: l2
  solver: lbfgs
  max_iter: 1000
```

## Data Management

- **Local Development**: Data stored in `data/` directory, versioned by DVC
- **CI/CD**: Data downloaded fresh from source, processed on ephemeral runner
- **Production**: Models in MLflow registry, vectorizer in container image
- **Remote Storage**: DVC artifacts stored in AWS S3

## License

See [LICENSE](LICENSE) file for details.

---

**Built with industry-standard MLOps practices**
