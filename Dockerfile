# Shopping Sessions: C++ AVX2 pipeline.
FROM debian:bookworm AS build
RUN apt-get update && apt-get install -y --no-install-recommends g++ make && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY Makefile ./
COPY src ./src
RUN make CXXFLAGS="-O3 -march=x86-64-v3 -std=c++17 -pipe"

FROM debian:bookworm-slim
RUN useradd --uid 1000 --no-create-home --home-dir /scratch app
COPY --from=build /src/pipeline /app/pipeline
WORKDIR /app
CMD ["/app/pipeline"]
