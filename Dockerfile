FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt notebook

COPY . .

RUN useradd -m -u 1000 app && chown -R app:app /workspace
USER app

EXPOSE 8888

ENV JUPYTER_TOKEN=change_me

CMD ["sh", "-c", "jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --NotebookApp.token=${JUPYTER_TOKEN}"]
