FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=nointeractive
RUN apt update && apt install -y  libc6-dev-i386 python3 python3-pip bc ksh
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
WORKDIR /app/symbiosis/trainsets 
WORKDIR /app/symbiosis/testsets
WORKDIR /app/symbiosis/data 
WORKDIR /app/symbiosis/label
WORKDIR /app/symbiosis/ext
WORKDIR /app/symbiosis/models
WORKDIR /app
COPY ./symbiosis ./symbiosis
COPY ./models/. /app/symbiosis/models/.
COPY ./mlf/. /app/symbiosis/mlf/.
COPY ./main.py ./main.py
CMD gunicorn --worker-connections=32 --timeout 120 --bind 0.0.0.0:5000 --access-logfile - main:service
