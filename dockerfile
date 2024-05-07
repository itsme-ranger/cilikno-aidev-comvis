FROM ubuntu:noble-20240429

RUN apt update

RUN add-apt-repository ppa:deadsnakes/ppa && apt-get update && apt install -y python3.9 python3.9-distutils git wget

RUN mkdir /analytic

WORKDIR /analytic

RUN mkdir weights

RUN git clone https://github.com/IDEA-Research/GroundingDINO.git && cd GroundingDINO && python3.9 -m pip install -q -e . \
    && python3.9 -m pip install -q roboflow && cd ../weights \
    && wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

WORKDIR /analytic

COPY . .

RUN python3.9 -m pip install -r requirements.txt

ENTRYPOINT [ "main.py" ]