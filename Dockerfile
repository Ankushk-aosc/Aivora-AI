# Production deployment (continuous-engineering priority #29).
#
# HONESTY NOTE: this Dockerfile has NOT been built or run - no Docker
# daemon is available in the environment that authored it (`docker`
# is not on PATH here). It reuses the exact pip-install and server-
# start commands already verified working directly on this machine
# throughout this project, so it should build, but "should" is not
# "verified" - do not treat this as tested until it has actually been
# built and run somewhere with Docker available. See ai_platform
# registry entry PRODUCTION_DEPLOYMENT for the honest status.

FROM python:3.11-slim

WORKDIR /app

# System deps for pypdf/pillow-adjacent wheels and sqlite (already stdlib,
# listed for clarity of what this image actually needs at the OS level).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Same command already verified working on the development machine
# throughout this project (pip install -r requirements.txt).
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# checkpoints/ and data/ hold real trained weights and prepared
# datasets/databases - mount these as volumes rather than baking them
# into the image, so a container restart doesn't lose trained state
# and the image itself stays small.
VOLUME ["/app/checkpoints", "/app/data", "/app/experiments"]

EXPOSE 8000

# Same entrypoint already verified working directly on this machine:
# `python app/backend/server.py --port 8000 --checkpoint <path>`.
# No default checkpoint is baked in - set CHECKPOINT_PATH or override
# the command, since which checkpoint to serve is a deployment decision,
# not something to hardcode into the image.
ENV CHECKPOINT_PATH=""
ENV HOST="0.0.0.0"
ENV PORT="8000"

CMD ["sh", "-c", "python app/backend/server.py --host $HOST --port $PORT ${CHECKPOINT_PATH:+--checkpoint $CHECKPOINT_PATH}"]
