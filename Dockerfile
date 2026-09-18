FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements-deploy.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt \
	&& useradd --create-home --uid 10001 apil

COPY backend ./backend

RUN chown -R apil:apil /app
USER apil

EXPOSE 8000

ENV WEB_CONCURRENCY=1

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
	CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --workers ${WEB_CONCURRENCY}"]
