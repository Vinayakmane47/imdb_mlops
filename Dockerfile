FROM python:3.10-slim

WORKDIR /app

# Copy requirements first for better caching
COPY flask_app/requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -m nltk.downloader stopwords wordnet

# Copy Flask application
COPY flask_app/ /app/

# Create models directory and copy model files
RUN mkdir -p /app/models
COPY models/vectorizer.pkl /app/models/vectorizer.pkl

EXPOSE 5005

# Local development
# CMD ["python", "app.py"]  

# Production (uncomment when deploying)
CMD ["gunicorn", "--bind", "0.0.0.0:5005", "--timeout", "120", "app:app"]