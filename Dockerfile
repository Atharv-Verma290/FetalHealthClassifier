FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt /app/

RUN pip install -r /app/requirements.txt

COPY . .

EXPOSE 4000

ENV FLASK_APP=src.app

CMD [ "flask", "run", "--host=0.0.0.0", "--port=4000", "--reload" ]