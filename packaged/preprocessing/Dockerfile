FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=nointeractive
WORKDIR /app
RUN apt update && apt install -y python3 python3-pip python3-opencv swig3.0 wget unzip cmake libopencv-dev
RUN python3 -m pip install numpy
WORKDIR /app
COPY ./requirements.txt ./
RUN python3 -m pip install -r requirements.txt
WORKDIR /service-wrapper
COPY ./service-wrapper/ ./service-wrapper
RUN python3 -m pip install ./service-wrapper
WORKDIR /app
COPY ./ ./
CMD gunicorn --worker-connections=32 --bind 0.0.0.0:5000 --access-logfile - --log-level debug main:service

