FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=nointeractive
RUN apt update && apt install -y  libc6-dev-i386 python3 python3-pip
WORKDIR /
COPY ./htk /htk
COPY ./gt2k /gt2k
WORKDIR /htk
RUN ./configure --without-x --disable-hslab
RUN sed -i "s/        /\t/g" HLMTools/Makefile
RUN make clean && make all && make install
WORKDIR /service-wrapper
COPY ./service-wrapper/ ./service-wrapper
RUN python3 -m pip install ./service-wrapper
WORKDIR /app
COPY ./requirements.txt ./requirements.txt
RUN python3 -m pip install -r requirements.txt
COPY ./symbiosis ./symbiosis
COPY ./main.py ./main.py
CMD gunicorn --worker-connections=32 --bind 0.0.0.0:5000 --access-logfile - main:service
