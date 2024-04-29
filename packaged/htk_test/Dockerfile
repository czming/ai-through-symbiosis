FROM aits-common
WORKDIR /app/symbiosis/trainsets 
WORKDIR /app/symbiosis/testsets
WORKDIR /app/symbiosis/data 
WORKDIR /app/symbiosis/label
WORKDIR /app/symbiosis/ext
WORKDIR /app/symbiosis/models
RUN apt update && apt install -y  libc6-dev-i386 python3 python3-pip bc ksh
WORKDIR /
COPY ./htk /htk
COPY ./gt2k /gt2k
WORKDIR /htk
RUN ./configure --without-x --disable-hslab
RUN sed -i "s/        /\t/g" HLMTools/Makefile
RUN make clean && make all && make install
WORKDIR /app
COPY ./requirements.txt ./requirements.txt
RUN python3 -m pip install -r requirements.txt
WORKDIR /app
COPY ./symbiosis ./symbiosis
COPY ./models/. /app/symbiosis/models/.
COPY ./mlf/. /app/symbiosis/mlf/.
COPY ./main.py ./main.py
