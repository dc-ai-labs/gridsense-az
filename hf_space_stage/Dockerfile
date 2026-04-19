# HF Space (docker SDK) runtime for GridSense-AZ Streamlit dashboard.
FROM python:3.11-slim

WORKDIR /app

# System deps for OpenDSS (libgomp) + curl (healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Sources
COPY app.py ./
COPY app ./app
COPY src ./src
COPY data ./data

# HF Spaces expect the app on $PORT (defaults to 7860); for streamlit docker sdk
# the convention is 7860.
ENV PORT=7860
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLE_CORS=false
ENV STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

EXPOSE 7860

CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
