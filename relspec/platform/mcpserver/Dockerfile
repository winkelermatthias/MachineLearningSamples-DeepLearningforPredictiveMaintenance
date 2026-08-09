# build context is the relspec/ directory
FROM python:3.12-slim

WORKDIR /srv
RUN pip install --no-cache-dir "mcp>=1.9" httpx
COPY platform/mcpserver/relspec_mcp /srv/relspec_mcp

EXPOSE 8765
CMD ["python3", "-m", "relspec_mcp.server"]
