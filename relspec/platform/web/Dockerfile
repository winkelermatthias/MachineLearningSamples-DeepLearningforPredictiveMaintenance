# build context is the relspec/ directory
FROM python:3.12-slim AS build
WORKDIR /build
RUN pip install --no-cache-dir numpy scipy
COPY src/relspec /usr/local/lib/python3.12/site-packages/relspec
COPY viz/explorer_template.html viz/explorer_template.html
COPY platform/web platform/web
RUN python3 platform/web/build.py

FROM nginx:1.27-alpine
COPY platform/web/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /build/platform/web/app /usr/share/nginx/html
