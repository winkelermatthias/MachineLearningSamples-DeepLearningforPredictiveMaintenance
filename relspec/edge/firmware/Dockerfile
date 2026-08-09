# Builds the firmware core tests + POSIX device simulator.
FROM gcc:13 AS build
WORKDIR /fw
COPY core core
COPY ports/posix ports/posix
COPY tests tests
COPY Makefile .
RUN make test device-sim
FROM debian:bookworm-slim
COPY --from=build /fw/build/device-sim /usr/local/bin/device-sim
ENTRYPOINT ["device-sim"]
