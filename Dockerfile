FROM nvcr.io/nvidia/tensorrt:24.01-py3

RUN apt update && \
    apt update -y cmake libgl1-mesa-dev libopencv-dev libeigen3-dev libyaml-cpp-dev

RUN pip install notebook python3-opencv

