FROM ubuntu:22.04@sha256:6d7b5d3317a71adb5e175640150e44b8b9a9401a7dd394f44840626aff9fa94d
ENV DEBIAN_FRONTEND=nointeractive
WORKDIR /app
RUN apt update && apt install -y python3 python3-pip
RUN mkdir /service-wrapper
COPY ./service-wrapper/ /service-wrapper
RUN python3 -m pip install gunicorn
RUN python3 -m pip install /service-wrapper
CMD gunicorn --worker-connections=32 --bind 0.0.0.0:5000 --access-logfile - --log-level debug --timeout 300 main:service
