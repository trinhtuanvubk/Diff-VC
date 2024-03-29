FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime

WORKDIR /workspace 

RUN apt-get update \ 
  && apt-get install curl libcurl4-openssl-dev libb64-dev -y \
  && apt-get install libsndfile1-dev -y \
  && pip install --upgrade pip
RUN pip install torchaudio==0.8.1
# setup for librosa 
RUN apt-get install libsndfile1

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt --no-cache-dir

COPY . .
# CMD ["python3", "app_gradio.py"]
