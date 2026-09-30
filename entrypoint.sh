#!/bin/sh

set -e

echo "Running migrations..."

python -m app.core.wait_for_dependencies

alembic upgrade head

echo "Starting API..."

exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
