FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY bank_term_deposit_model.joblib .

EXPOSE 8000

# IMPORTANT: just run Flask app via Python
CMD ["python", "app.py"]
