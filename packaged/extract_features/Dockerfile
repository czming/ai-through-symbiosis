FROM aits-common
RUN apt update && apt install -y python3-opencv swig3.0 wget unzip cmake libopencv-dev
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
COPY ./ ./
