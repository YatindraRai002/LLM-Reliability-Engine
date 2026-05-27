FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose ports for both Streamlit and FastAPI
EXPOSE 8501 8000

# Default command runs Streamlit
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
