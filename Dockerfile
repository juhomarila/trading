FROM python:3.10-slim

ENV APP_HOME /app
WORKDIR $APP_HOME

ENV PYTHONUNBUFFERED 1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD exec gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 0 trading.wsgi:application