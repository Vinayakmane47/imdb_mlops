from flask import Flask, render_template, request
import mlflow
import pickle
import os
import pandas as pd
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
import time
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
import re
import dagshub
from dotenv import load_dotenv
import warnings

# -------------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------------
load_dotenv()
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# -------------------------------------------------------------------------
# TEXT CLEANING HELPERS
# -------------------------------------------------------------------------
def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

def remove_stop_words(text):
    stop_words = set(stopwords.words("english"))
    return " ".join([word for word in str(text).split() if word not in stop_words])

def removing_numbers(text):
    return ''.join([char for char in text if not char.isdigit()])

def lower_case(text):
    return " ".join([word.lower() for word in text.split()])

def removing_punctuations(text):
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.replace('؛', "")
    text = re.sub('\s+', ' ', text).strip()
    return text

def removing_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def normalize_text(text):
    text = lower_case(text)
    text = remove_stop_words(text)
    text = removing_numbers(text)
    text = removing_punctuations(text)
    text = removing_urls(text)
    text = lemmatization(text)
    return text

# -------------------------------------------------------------------------
# MLflow & DAGSHub setup
# -------------------------------------------------------------------------
dagshub_token = os.getenv("DAGSHUB_TOKEN")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_TOKEN environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "Vinayakmane47"
repo_name = "imdb_mlops"

mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

# -------------------------------------------------------------------------
# FLASK APP
# -------------------------------------------------------------------------
app = Flask(__name__)

# -------------------------------------------------------------------------
# PROMETHEUS METRICS SETUP
# -------------------------------------------------------------------------
registry = CollectorRegistry()

REQUEST_COUNT = Counter(
    "app_request_count",
    "Total number of requests to the app",
    ["method", "endpoint"],
    registry=registry
)

REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds",
    "Latency of requests in seconds",
    ["endpoint"],
    registry=registry
)

PREDICTION_COUNT = Counter(
    "model_prediction_count",
    "Count of predictions for each class",
    ["prediction"],
    registry=registry
)

# ✅ New cost metrics
COST_PER_REQUEST = Gauge(
    "app_cost_per_request_usd",
    "Estimated EC2/EKS compute cost per inference request in USD",
    ["endpoint"],
    registry=registry
)

TOTAL_COST = Counter(
    "app_total_cost_usd",
    "Total accumulated EC2/EKS compute cost in USD",
    registry=registry
)

# Adjust this according to your instance type
INSTANCE_COST_PER_HOUR = 0.0416   # Example: t3.medium
SECONDS_IN_HOUR = 3600

# -------------------------------------------------------------------------
# MODEL SETUP
# -------------------------------------------------------------------------
model_name = "my_model"

def get_latest_model_version(model_name):
    try:
        client = mlflow.MlflowClient()
        latest_version = client.get_latest_versions(model_name, stages=["Production"])
        if not latest_version:
            latest_version = client.get_latest_versions(model_name, stages=["Staging"])
        if not latest_version:
            latest_version = client.get_latest_versions(model_name, stages=["None"])
        return latest_version[0].version if latest_version else None
    except Exception as e:
        print(f"Error getting model version: {e}")
        return None

model_version = get_latest_model_version(model_name)
model_uri = f'models:/{model_name}/{model_version}'
print(f"Fetching model from: {model_uri}")
model = mlflow.pyfunc.load_model(model_uri)
vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

# -------------------------------------------------------------------------
# ROUTES
# -------------------------------------------------------------------------
@app.route("/")
def home():
    REQUEST_COUNT.labels(method="GET", endpoint="/").inc()
    start_time = time.time()
    response = render_template("index.html", result=None)
    REQUEST_LATENCY.labels(endpoint="/").observe(time.time() - start_time)
    return response

@app.route("/predict", methods=["POST"])
def predict():
    REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()
    start_time = time.time()

    text = request.form["text"]
    text = normalize_text(text)
    features = vectorizer.transform([text])
    features_df = pd.DataFrame(features.toarray(), columns=[str(i) for i in range(features.shape[1])])

    result = model.predict(features_df)
    prediction = result[0]
    PREDICTION_COUNT.labels(prediction=str(prediction)).inc()

    latency = time.time() - start_time
    REQUEST_LATENCY.labels(endpoint="/predict").observe(latency)

    # Compute cost based on EC2/EKS instance pricing
    cost = (latency * INSTANCE_COST_PER_HOUR) / SECONDS_IN_HOUR
    COST_PER_REQUEST.labels(endpoint="/predict").set(cost)
    TOTAL_COST.inc(cost)

    return render_template("index.html", result=prediction, latency=latency, cost=cost)

@app.route("/metrics", methods=["GET"])
def metrics():
    """Expose only custom Prometheus metrics."""
    return generate_latest(registry), 200, {"Content-Type": CONTENT_TYPE_LATEST}

# -------------------------------------------------------------------------
# MAIN ENTRY
# -------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5005)
