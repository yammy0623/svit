FROM nvidia/cuda:11.6.1-devel-ubuntu20.04

RUN apt-get update && \
    apt-get install -y python3 python3-pip vim libglib2.0-0 libsm6 libxrender1 libxext6 nvtop tmux

WORKDIR /workspace

RUN pip3 install --no-cache-dir numpy opencv-python

CMD ["bash"]