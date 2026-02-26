# Usa Python 3.11 como imagen base
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Instalar dependencias del sistema, de Python y limpiar todo
COPY requirements.txt .
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /root/.cache

COPY app ./app
COPY corpus ./corpus
COPY vector_store ./vector_store
COPY streamlit_app.py .
COPY .streamlit ./.streamlit

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]