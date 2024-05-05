FROM ubuntu:noble-20240429

RUN apt update

RUN add-apt-repository ppa:deadsnakes/ppa && apt-get update && apt install -y python3.9 python3.9-distutils