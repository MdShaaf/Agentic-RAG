FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# libgomp1: required by faiss-cpu / torch; ffmpeg: needed if the video reader extracts audio
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libgomp1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch first (much smaller than the default CUDA build)
RUN pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the reranker weights into the image so cold starts don't download ~1GB
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-base')"

COPY . .

EXPOSE 8501

# CHANGE THIS to match your entry point:
#   Streamlit : streamlit run main.py --server.port=8501 --server.address=0.0.0.0
#   FastAPI   : uvicorn main:app --host 0.0.0.0 --port 8501
#   Gradio    : python main.py   (with server_name="0.0.0.0", server_port=8501)
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
