FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY relspec_gateway ./relspec_gateway
# spool lives on a volume so buffered data survives restarts
VOLUME /data
ENV RS_SPOOL=/data/gateway-spool.db RS_MAPPING=/data/devices.json
EXPOSE 47700/udp 47701 4840 8080
CMD ["python", "-m", "relspec_gateway.main"]
