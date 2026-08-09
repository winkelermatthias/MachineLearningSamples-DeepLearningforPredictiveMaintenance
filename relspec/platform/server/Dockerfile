# build context is the relspec/ directory (see docker-compose.yml)
FROM python:3.12-slim

WORKDIR /srv
COPY platform/server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# the signal-processing package and the service
COPY src/relspec /usr/local/lib/python3.12/site-packages/relspec
COPY platform/server/relspec_service /srv/relspec_service
COPY platform/web /srv/web

ENV RELSPEC_SRC=""
EXPOSE 8000
CMD ["python3", "-m", "uvicorn", "relspec_service.app:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
