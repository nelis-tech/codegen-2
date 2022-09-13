#app/Dockerfile

FROM python:3.9-slim

EXPOSE 8500

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone 
    
RUN conda create --name my_env python=3.9 -y \
    conda activate my_env \
    pip install streamlit \
    pip install torch \
    pip install datasets \
    pip install accelerate \
    pip install sentencepiece \
    pip install git+https://github.com/huggingface/transformers.git 

ENTRYPOINT ["streamlit", "run", "app.py", "model.py", "--server.port=8500", "--server.address=0.0.0.0"]