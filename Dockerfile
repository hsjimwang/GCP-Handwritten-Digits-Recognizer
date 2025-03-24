FROM python:3.10-slim

WORKDIR /app

COPY app app
COPY ml ml
COPY requirements.txt requirements.txt
COPY ml/emnist_digits_recognizer.ckpt /emnist_digits_recognizer.ckpt

RUN apt-get update && \
    apt-get install -y gcc libffi-dev && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install -r requirements.txt

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
