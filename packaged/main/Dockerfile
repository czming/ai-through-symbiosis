FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=nointeractive
RUN apt update && apt install -y python3 python3-pip
RUN apt install -y curl wget git vim
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
WORKDIR /app
COPY ./requirements.txt ./
RUN python3 -m pip install -r requirements.txt
COPY ./main.py ./
CMD python3 -u  main.py && sleep infinity
