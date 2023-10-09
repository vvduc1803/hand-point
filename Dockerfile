FROM nvidia/cuda:11.1.1-devel-ubuntu20.04

RUN apt update

RUN apt-get clean -y

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 libglfw3-dev libgles2-mesa-dev -y

RUN  apt-get update

RUN apt-get install -y python3.9 python3-pip sudo

RUN pip3 install --upgrade pip

RUN pip3 install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html

RUN  apt-get update \
  && apt-get install -y wget \
  && rm -rf /var/lib/apt/lists/*

RUN sudo apt-get update

RUN useradd -m ana

RUN chown -R ana:ana /home/ana

COPY --chown=ana . /home/ana/Study/CVPR/handpoint

USER ana

RUN cd /home/ana/Study/CVPR/handpoint && pip3 install -r requirements.txt

WORKDIR /home/ana/Study/CVPR/handpoint