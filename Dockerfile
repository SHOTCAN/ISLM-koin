FROM python:3.12-slim

WORKDIR /app

# System deps for numpy/scikit-learn (slim image lacks build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Railway injects PORT env var
CMD ["python", "bot_standalone.py"]
