FROM aits-common
COPY ./requirements.txt ./
RUN python3 -m pip install -r requirements.txt
COPY ./ ./
CMD python3 -m flask --app main.py run --host 0.0.0.0
