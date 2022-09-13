#app/Dockerfile

FROM python:3.9-slim

EXPOSE 8501

WORKDIR /codegen-1

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/nelis-tech/codegen-2.git .
    
RUN pip3 install streamlit \
    pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116 \
    pip install datasets \
    pip install accelerate \
    pip install sentencepiece \
    pip install git+https://github.com/huggingface/transformers

ENTRYPOINT ["streamlit", "run", "app.py", "model.py", "--server.port=8501", "--server.address=0.0.0.0"]