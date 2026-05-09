# Optional local/container deployment only.
# Streamlit Community Cloud deploys directly from GitHub and does not use Docker.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY app_data/ app_data/
COPY .streamlit/ .streamlit/

EXPOSE 8501

CMD ["streamlit", "run", "app/model_results_dashboard.py", "--server.address=0.0.0.0", "--server.port=8501"]
