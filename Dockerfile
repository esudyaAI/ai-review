FROM python:3.12-slim

# gitleaks is the reviewer's only deterministic check. Language linters run as a
# separate job in each consuming repository's own pipeline, not inside this image.
ARG GITLEAKS_VERSION=8.18.4
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates git \
    && curl -sSL "https://github.com/gitleaks/gitleaks/releases/download/v${GITLEAKS_VERSION}/gitleaks_${GITLEAKS_VERSION}_linux_x64.tar.gz" \
       | tar -xz -C /usr/local/bin gitleaks \
    && apt-get purge -y curl && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/reviewer
COPY pyproject.toml ./
COPY reviewer ./reviewer
RUN pip install --no-cache-dir .

WORKDIR /workspace
ENTRYPOINT ["python", "-m", "reviewer"]
