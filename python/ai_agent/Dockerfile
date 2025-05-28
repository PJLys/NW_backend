FROM python:3.9-slim

# System dependencies
RUN apt-get update && \
    apt-get install -y libportaudio2 portaudio19-dev ffmpeg curl alsa-utils && \
    apt-get clean

# Set working directory
WORKDIR /app
ENV PYTHONPATH=/app

# Copy poetry files first for caching
COPY pyproject.toml poetry.lock ./
RUN pip install --upgrade pip && pip install poetry

# Don't use virtualenvs inside container
RUN poetry config virtualenvs.create false && poetry install --no-root

# Copy the rest of the app
COPY . .

COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh
CMD ["/app/entrypoint.sh"]