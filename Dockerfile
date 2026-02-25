# Usa Python 3.11 como imagen base
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV CHROME_BIN=/usr/bin/chromium
ENV CHROME_PATH=/usr/lib/chromium/

WORKDIR /app

# Dependencias del sistema
# chomium y chromium-driver para el scraping con Selenium
RUN apt-get update && apt-get install -y \
    build-essential \
    poppler-utils \
    curl \
    wget \
    chromium \
    chromium-driver \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Dependencias de Python
COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY corpus ./corpus
COPY vector_store ./vector_store
COPY streamlit_app.py .
COPY .streamlit ./.streamlit

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]