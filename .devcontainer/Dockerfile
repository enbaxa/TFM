FROM mcr.microsoft.com/devcontainers/universal:2

# Install pdflatex
RUN sudo -u root apt-get update \
    && sudo -u root apt-get install texlive-xetex \
    && sudo -u root apt-get clean \
    && sudo -u root rm -rf /var/lib/apt/lists/*