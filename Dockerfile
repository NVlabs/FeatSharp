FROM nvcr.io/nvidia/pytorch:25.04-py3

WORKDIR /app

RUN yes | pip install --upgrade pip

RUN apt-get update && apt-get install -y webp
RUN yes | pip install webp

RUN yes | pip install webdataset tqdm wandb opencv-python openmim timm transformers omegaconf hydra-core
RUN yes | pip install open_clip_torch

RUN mim install mmengine
RUN mim install "mmcv==2.1.0"
RUN mim install mmsegmentation

# Optional: For ReSFU support
RUN git clone https://github.com/zmhhmz/ReSFU.git
WORKDIR /app/ReSFU/FNS_Attn
RUN python setup.py develop
WORKDIR /app

