# IMDB Sentiment Analysis - MLOps Pipeline

A production-ready MLOps pipeline for sentiment analysis of IMDB movie reviews, featuring automated CI/CD, model versioning, and cloud deployment.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Features](#features)
- [Setup](#setup)
- [Usage](#usage)
- [CI/CD Pipeline](#cicd-pipeline)
- [Deployment](#deployment)
- [Monitoring](#monitoring)
- [Model Metrics](#model-metrics)

## 🎯 Overview

This project implements an end-to-end MLOps pipeline for sentiment analysis using IMDB movie reviews. It demonstrates best practices in:

- **Data Versioning**: DVC for data and model versioning
- **Model Tracking**: MLflow for experiment tracking and model registry
- **CI/CD**: Automated testing, building, and deployment via GitHub Actions
- **Cloud Deployment**: Containerized Flask app on AWS EKS
- **Monitoring**: Prometheus metrics and Grafana dashboards
- **Reproducibility**: Parameterized pipeline with YAML configuration

## 🏗️ Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         DEVELOPMENT                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Code       │───▶│   DVC        │───▶│   MLflow     │      │
│  │   Changes    │    │   Pipeline   │    │   Tracking   │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      CI/CD PIPELINE                              │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  GitHub Actions (ubuntu-latest)                          │   │
│  │  ├── Checkout Code                                       │   │
│  │  ├── Install Dependencies                               │   │
│  │  ├── Run DVC Pipeline (dvc repro)                       │   │
│  │  │   ├── Data Ingestion                                 │   │
│  │  │   ├── Data Preprocessing                             │   │
│  │  │   ├── Feature Engineering (TF-IDF)                   │   │
│  │  │   ├── Model Training (LogisticRegression)           │   │
│  │  │   ├── Model Evaluation                               │   │
│  │  │   └── Model Registration (MLflow)                   │   │
│  │  ├── Run Tests                                          │   │
│  │  ├── Build Docker Image                                 │   │
│  │  ├── Push to AWS ECR                                    │   │
│  │  └── Deploy to EKS                                      │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      CLOUD INFRASTRUCTURE                        │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   AWS S3     │    │   AWS ECR    │    │   AWS EKS    │      │
│  │   (Data)     │    │   (Images)   │    │   (Pods)     │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                  │                    │               │
│         └──────────────────┴────────────────────┘               │
│                              │                                  │
│                              ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Flask Application (Containerized)                        │  │
│  │  ├── Load Model from MLflow                              │  │
│  │  ├── Serve Predictions                                   │  │
│  │  └── Expose Prometheus Metrics                            │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      MONITORING                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  Prometheus  │───▶│   Grafana    │    │   MLflow     │      │
│  │  (Metrics)   │    │  (Dashboards)│    │  (Tracking)  │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

### Data Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    DVC PIPELINE STAGES                          │
└─────────────────────────────────────────────────────────────────┘

1. DATA INGESTION
   Input:  S3/Public URL → data.csv
   Output: data/raw/train.csv, test.csv
   Process: Download, split train/test (80/20)

2. DATA PREPROCESSING
   Input:  data/raw/
   Output: data/interim/train_processed.csv, test_processed.csv
   Process: Text cleaning, lemmatization, stopword removal

3. FEATURE ENGINEERING
   Input:  data/interim/
   Output: data/processed/train_bow.csv, test_bow.csv
          models/vectorizer.pkl
   Process: TF-IDF vectorization (20K features, ngrams 1-3)

4. MODEL BUILDING
   Input:  data/processed/
   Output: models/model.pkl
   Process: Train LogisticRegression (C=10, solver=lbfgs)

5. MODEL EVALUATION
   Input:  models/model.pkl, data/processed/test_bow.csv
   Output: reports/metrics.json, reports/experiment_info.json
   Process: Calculate accuracy, precision, recall, AUC
           Log to MLflow

6. MODEL REGISTRATION
   Input:  reports/experiment_info.json
   Output: MLflow Model Registry
   Process: Register model version, promote to Production
```

### Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    AWS EKS CLUSTER                              │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  LoadBalancer Service (flask-app-service)                │  │
│  │  └── Routes traffic to pods                               │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Deployment (flask-app) - 2 Replicas                     │  │
│  │  ├── Pod 1: Flask App Container                         │  │
│  │  │   ├── Port: 5005                                      │  │
│  │  │   ├── Model: MLflow (models:/my_model/Production)    │  │
│  │  │   ├── Vectorizer: models/vectorizer.pkl               │  │
│  │  │   └── Metrics: /metrics (Prometheus)                  │  │
│  │  └── Pod 2: Flask App Container                         │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MONITORING STACK                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│  │  Prometheus  │───▶│   Grafana    │    │   EC2        │     │
│  │  (Scrapes    │    │  (Dashboards)│    │  (Hosts      │     │
│  │   /metrics)  │    │              │    │   Monitoring)│     │
│  └──────────────┘    └──────────────┘    └──────────────┘     │
└─────────────────────────────────────────────────────────────────┘
```

## 🛠️ Tech Stack

### Core Technologies
- **Python 3.10**: Main programming language
- **Scikit-learn**: Machine learning library
- **Flask**: Web framework for API
- **Gunicorn**: WSGI HTTP server

### MLOps Tools
- **DVC (Data Version Control)**: Data and pipeline versioning
- **MLflow**: Experiment tracking and model registry
- **DAGSHub**: MLflow backend hosting

### Infrastructure
- **Docker**: Containerization
- **AWS ECR**: Container registry
- **AWS EKS**: Kubernetes orchestration
- **AWS S3**: Data storage (DVC remote)

### CI/CD
- **GitHub Actions**: Continuous integration/deployment
- **Kubectl**: Kubernetes deployment

### Monitoring
- **Prometheus**: Metrics collection
- **Grafana**: Visualization and dashboards

## 📁 Project Structure

```
imdb_mlops/
├── .github/
│   └── workflows/
│       └── ci.yaml              # CI/CD pipeline configuration
├── data/
│   ├── raw/                     # Raw data (tracked by DVC)
│   ├── interim/                 # Preprocessed data (tracked by DVC)
│   └── processed/               # Feature-engineered data (tracked by DVC)
├── flask_app/
│   ├── app.py                   # Flask application
│   ├── templates/
│   │   └── index.html          # Web UI
│   └── requirements.txt        # Flask dependencies
├── models/                      # Trained models (tracked by DVC)
│   ├── model.pkl
│   └── vectorizer.pkl
├── notebooks/                   # Jupyter notebooks for experimentation
├── reports/                     # Evaluation metrics and reports
│   └── metrics.json
├── scripts/
│   └── promote_model.py        # Model promotion script
├── src/
│   ├── connections/            # AWS connection utilities
│   ├── data/                   # Data ingestion and preprocessing
│   ├── features/               # Feature engineering
│   ├── model/                  # Model training and evaluation
│   └── logger/                 # Logging utilities
├── tests/                       # Unit and integration tests
├── .dvc/                        # DVC configuration
├── dvc.yaml                     # DVC pipeline definition
├── params.yaml                  # Pipeline parameters
├── deployment.yaml              # Kubernetes deployment config
├── Dockerfile                   # Container image definition
└── requirements.txt             # Python dependencies
```

## ✨ Features

### Model Configuration
- **Algorithm**: Logistic Regression
- **Features**: TF-IDF with Unigrams + Trigrams (1-3 ngrams)
- **Max Features**: 20,000
- **Hyperparameters**:
  - C: 10
  - Penalty: L2
  - Solver: lbfgs
  - Max Iterations: 1000

### Monitoring Metrics
- **Application Metrics**:
  - Request count (by method and endpoint)
  - Request latency (by endpoint)
  - Prediction count (by class)
  - Cost per request (USD)
  - Total accumulated cost (USD)

### Model Metrics
- Accuracy
- Precision
- Recall
- AUC (Area Under Curve)

## 🚀 Setup

### Prerequisites
- Python 3.10+
- Conda (recommended)
- Docker
- AWS CLI configured
- kubectl configured
- DVC installed

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd imdb_mlops
```

2. **Create conda environment**
```bash
conda create -n atlas python=3.10
conda activate atlas
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
pip install -r flask_app/requirements.txt
```

4. **Download NLTK data**
```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
```

5. **Configure DVC remote (if using S3)**
```bash
dvc remote add -d myremote s3://your-bucket-name/dvc-cache
```

6. **Set up environment variables**
Create a `.env` file:
```bash
DAGSHUB_TOKEN=your_dagshub_token
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
AWS_BUCKET_NAME=your_bucket_name
AWS_REGION=us-east-1
```

## 📖 Usage

### Running the Pipeline Locally

1. **Run the complete pipeline**
```bash
dvc repro
```

2. **Run specific stages**
```bash
dvc repro data_ingestion
dvc repro model_building
```

3. **Pull data from DVC remote**
```bash
dvc pull
```

4. **Push data to DVC remote**
```bash
dvc push
```

### Running the Flask App Locally

1. **Start the Flask application**
```bash
cd flask_app
python app.py
```

2. **Access the web interface**
```
http://localhost:5005
```

3. **Check metrics endpoint**
```
http://localhost:5005/metrics
```

### Running Tests

```bash
# Run all tests
python -m unittest discover tests

# Run specific test
python -m unittest tests.test_model
python -m unittest tests.test_flask_app
```

## 🔄 CI/CD Pipeline

### Pipeline Stages

1. **Code Checkout**: Clone repository
2. **Environment Setup**: Install Python and dependencies
3. **Pipeline Execution**: Run `dvc repro` (full ML pipeline)
4. **Testing**: Run unit and integration tests
5. **Model Promotion**: Promote model to Production in MLflow
6. **Docker Build**: Build container image
7. **ECR Push**: Push image to AWS ECR
8. **EKS Deployment**: Deploy to Kubernetes cluster

### Triggering CI/CD

The pipeline automatically runs on:
- Push to `main` branch
- Pull requests

### Manual Trigger

You can also trigger manually from GitHub Actions tab.

## 🚢 Deployment

### Prerequisites
- AWS EKS cluster created
- AWS ECR repository created
- kubectl configured for EKS
- AWS credentials configured in GitHub Secrets

### Deployment Process

The CI/CD pipeline automatically:
1. Builds Docker image with latest code
2. Pushes to AWS ECR
3. Updates Kubernetes deployment
4. Restarts pods to pull new image

### Manual Deployment

```bash
# Build and push Docker image
docker build -t your-ecr-repo:latest .
docker push your-ecr-repo:latest

# Deploy to EKS
kubectl apply -f deployment.yaml
kubectl rollout restart deployment flask-app
```

### Accessing the Application

After deployment, get the LoadBalancer URL:
```bash
kubectl get service flask-app-service
```

Access the application at the EXTERNAL-IP:5005

## 📊 Monitoring

### Prometheus Metrics

The Flask app exposes metrics at `/metrics` endpoint:
- `app_request_count`: Total requests
- `app_request_latency_seconds`: Request latency
- `model_prediction_count`: Prediction counts
- `app_cost_per_request_usd`: Cost per request
- `app_total_cost_usd`: Total cost

### Grafana Dashboards

Configure Prometheus to scrape metrics from:
- Service: `flask-app-service:5005`
- Path: `/metrics`

Create dashboards in Grafana to visualize:
- Request rates and latency
- Prediction distribution
- Cost metrics
- Model performance

### MLflow Tracking

View experiments and models at:
- **DAGSHub MLflow**: `https://dagshub.com/Vinayakmane47/imdb_mlops.mlflow`

Track:
- Model versions
- Training metrics
- Hyperparameters
- Model artifacts

## 📈 Model Metrics

Model evaluation metrics are stored in:
- **Local**: `reports/metrics.json`
- **MLflow**: Logged during evaluation stage

Current model metrics (example):
```json
{
  "accuracy": 0.85,
  "precision": 0.84,
  "recall": 0.86,
  "auc": 0.92
}
```

View metrics:
- **MLflow UI**: Latest experiment run
- **Local file**: `cat reports/metrics.json`
- **Grafana**: If exposed as Prometheus metrics

## 🔧 Configuration

### Pipeline Parameters (`params.yaml`)

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

Modify parameters in `params.yaml` and run `dvc repro` to retrain with new settings.

## 🧪 Testing

### Test Structure
- `tests/test_model.py`: Model validation tests
- `tests/test_flask_app.py`: Flask application tests

### Running Tests
```bash
# All tests
python -m unittest discover tests

# Specific test file
python -m unittest tests.test_model
```

## 📝 Data Storage

### Local Development
- Data stored in `data/` directory
- Tracked by DVC (not Git)
- Versioned in DVC cache

### CI/CD
- Data downloaded fresh from source
- Processed on CI runner
- Artifacts not persisted (ephemeral)

### Production
- Models stored in MLflow
- Vectorizer in Docker image
- Data in S3 (DVC remote)

## 🔐 Security

- **Secrets Management**: GitHub Secrets for sensitive data
- **Kubernetes Secrets**: DAGSHUB_TOKEN stored as K8s secret
- **Environment Variables**: `.env` file (not committed)
- **IAM Roles**: AWS credentials via IAM

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `python -m unittest discover tests`
5. Commit and push
6. Create a pull request

## 📄 License

See [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- DVC for data versioning
- MLflow for experiment tracking
- DAGSHub for MLflow hosting
- AWS for cloud infrastructure

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Built with ❤️ using MLOps best practices**
