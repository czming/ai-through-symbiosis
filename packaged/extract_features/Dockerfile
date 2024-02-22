FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=nointeractive
WORKDIR /app
RUN apt update && apt install -y python3 python3-pip python3-opencv swig3.0 wget unzip cmake libopencv-dev
WORKDIR /aruco
RUN python3 -m pip install numpy
RUN wget -O aruco.zip https://downloads.sourceforge.net/project/aruco/3.1.12/aruco-3.1.12.zip?r=https%3A%2F%2Fsourceforge.net%2Fprojects%2Faruco%2Ffiles%2Flatest%2Fdownload \
    && ls -l \
    && unzip aruco.zip \
    && cd aruco-3.1.12 \
    && ls -l \
    && mkdir build \
    && cd build \
    && cmake -DCMAKE_BUILD_TYPE:STRING=RELEASE .. \
    && make -j$(grep -c ^processor /proc/cpuinfo) \
    && make install  \
    && cd .. \
    && for x in `ls python/dist/ | grep *.whl`; do python3 -m pip install $x; done
WORKDIR /app
COPY ./requirements.txt ./
RUN python3 -m pip install -r requirements.txt
WORKDIR /service-wrapper
COPY ./service-wrapper/ ./service-wrapper
RUN python3 -m pip install ./service-wrapper
WORKDIR /app
COPY ./ ./
CMD gunicorn --worker-connections=32 --bind 0.0.0.0:5000 --access-logfile - main:service

