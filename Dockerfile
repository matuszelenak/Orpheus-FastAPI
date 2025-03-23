FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    build-essential

WORKDIR /app

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY static static
COPY templates templates
COPY tts_engine tts_engine
COPY app.py app.py

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]