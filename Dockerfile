# python:3.11-slim matches setup.py python_requires=">=3.11".
# The full conda environment (statsmodels, prophet, xgboost, etc.) is only
# needed for training scripts — the Streamlit app requires only streamlit,
# pandas, numpy, and plotly (see requirements.txt).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Copy requirements first so this layer is cached as long as dependencies
# don't change, even when application code is updated.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# data/ and models/ are mounted as read-only volumes at runtime
# (see docker-compose.yml) — they are not copied into the image.
COPY src/     src/
COPY app/     app/
COPY setup.py .

# Install the project package so src/ is importable.
RUN pip install --no-cache-dir -e .

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c \
        "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0"]
