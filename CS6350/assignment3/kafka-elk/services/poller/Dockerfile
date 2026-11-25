FROM ghcr.io/astral-sh/uv:python3.15-trixie

# Copy the project into the image
COPY . /app

# Disable development dependencies
ENV UV_NO_DEV=1

# Sync the project into a new environment, asserting the lockfile is up to date
WORKDIR /app

RUN uv sync
RUN rm .env
RUN mv .env_docker .env

CMD ["uv", "run", "poller"]
