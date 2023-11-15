FROM mcr.microsoft.com/devcontainers/universal:2

# Install pdflatex
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends texlive-xetex \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
