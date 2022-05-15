FROM ubuntu:18.04
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git
COPY ./htk ./htk
COPY ./gt2k ./gt2k
COPY ./PhraseLevel ./PhraseLevel
COPY ./prepare ./htk/prepare
COPY ./WordLevel ./WordLevel
RUN apt-get install -y build-essential
RUN apt-get install -y vim
RUN apt-get install -y libc6-dev-i386
RUN apt-get install -y ksh
RUN apt-get install -y bc
RUN apt install -y libx11-dev
WORKDIR ./htk
RUN chmod +x prepare
RUN ./prepare
