FROM nvidia/cuda:11.2.0-cudnn8-runtime-ubuntu18.04

RUN apt-get update && apt-get install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa && apt-get update && apt-get install -y python3.6 && apt-get install -y python3-pip sudo

RUN python3 --version

RUN apt-get update && apt-get install -y gcc python3-dev musl-dev libjpeg-dev zlib1g zlib1g-dev ffmpeg libsm6 libxext6 git

RUN apt-get update && apt-get install -y libprotobuf-dev protobuf-compiler

RUN apt-get update && apt-get -y install cmake

RUN useradd -m sabuj

RUN chown -R sabuj:sabuj /home/sabuj/

RUN chmod 755 /home/sabuj/

USER sabuj

WORKDIR /home/sabuj

COPY requirements.txt requirements.txt

RUN pip3 install --upgrade pip

RUN pip3 install -r requirements.txt

WORKDIR /home/sabuj

COPY --chown=sabuj . /home/sabuj
