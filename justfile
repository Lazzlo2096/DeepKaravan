# Docker Compose reads .env itself, including COMPOSE_FILE and device settings.
[windows]
set shell := ["powershell.exe", "-NoLogo", "-NoProfile", "-Command"]

[unix]
set shell := ["sh", "-eu", "-c"]

# Show available commands.
default:
    @just --list

# Create .env from the example if it does not exist.
[windows]
init:
    @if (-not (Test-Path -LiteralPath .env)) { Copy-Item -LiteralPath .env.example -Destination .env }

# Create .env from the example if it does not exist.
[unix]
init:
    @test -e .env || cp .env.example .env

# Prepare Python 3.12 and install development dependencies (requires uv).
setup: init
    uv sync --locked --extra api

# Update uv.lock after changing dependencies in pyproject.toml.
lock:
    uv lock

# Verify that uv.lock matches pyproject.toml without modifying it.
lock-check:
    uv lock --check

# Build images for the device selected in .env.
build: init
    docker compose build

# Build and start the API and inference process in one container.
up: init
    docker compose up -d --build

# Stop and remove containers, preserving the model cache.
down:
    docker compose down

# Restart all services or specified services, e.g. just restart api.
restart *services:
    docker compose restart {{ services }}

# Show service status.
ps:
    docker compose ps

# Follow the AMD GPU worker running in Docker Desktop / WSL.
logs-gpu:
    docker compose logs --follow --tail 100 api

# Verify real GPU computation inside the Windows / WSL worker image.
gpu-check:
    docker compose run --rm --no-deps api python scripts/check_amd.py

# Follow logs, optionally for specific services, e.g. just logs api.
logs *services:
    docker compose logs --follow --tail 100 {{ services }}

# Validate the Compose configuration.
config: init
    docker compose config --quiet

# Run unit tests without downloading the model; extra pytest arguments are accepted.
test *args:
    uv run --locked --extra api pytest -q {{ args }}

# Check Python code with Ruff.
lint:
    uv run --locked --extra api ruff check .

# Format Python code and the justfile.
format:
    uv run --locked --extra api ruff format .
    just --fmt

# Check Python and justfile formatting without changing files.
format-check:
    uv run --locked --extra api ruff format --check .
    just --fmt --check

# Check uv.lock, lint, formatting, unit tests and Compose configuration.
check: lock-check lint format-check test config

# Check an already running service with the real model.
smoke base_url="http://localhost:8000" timeout="600":
    uv run --locked --extra api python scripts/smoke.py --base-url "{{ base_url }}" --timeout "{{ timeout }}"

# Check restart semantics: in-memory tasks disappear after container restart.
recovery base_url="http://localhost:8000":
    uv run --locked --extra api python scripts/check_recovery.py --base-url "{{ base_url }}"
