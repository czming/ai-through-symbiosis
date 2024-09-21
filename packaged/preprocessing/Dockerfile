FROM aits-common
RUN apt update && apt install -y python3-opencv
COPY ./requirements.txt ./
RUN python3 -m pip install -r requirements.txt
COPY ./ ./
