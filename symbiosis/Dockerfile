FROM ubuntu:18.04
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git
COPY ./third-party/htk ./htk
COPY ./third-party/gt2k ./gt2k
COPY ./scripts/prepare_container.sh ./htk/prepare_container.sh
RUN apt-get install -y build-essential
RUN apt-get install -y vim
RUN apt-get install -y libc6-dev-i386
RUN apt-get install -y ksh
RUN apt-get install -y bc
RUN apt install -y libx11-dev
RUN apt-get install -y moreutils
RUN apt-get install -y python3 python3-pip
WORKDIR ./htk
RUN chmod +x prepare_container.sh
RUN ./prepare_container.sh
WORKDIR ../tmp
