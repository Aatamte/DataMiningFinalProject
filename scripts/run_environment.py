"""Start the execution environment server.

Usage:
    uv run python scripts/run_environment.py  # Stop, start fresh, wait for ready
"""

import subprocess
import sys
import time
from pathlib import Path

import httpx


DOCKER_COMPOSE = Path(__file__).parent.parent / "docker" / "docker-compose.yml"
HEALTH_URL = "http://localhost:8080/health"


def check_health() -> dict | None:
    """Check if server is healthy."""
    try:
        response = httpx.get(HEALTH_URL, timeout=5.0)
        if response.status_code == 200:
            return response.json()
    except httpx.RequestError:
        pass
    return None


def stop_server():
    """Stop the execution environment."""
    print("Stopping execution environment...")
    subprocess.run(
        ["docker-compose", "-f", str(DOCKER_COMPOSE), "down"],
        capture_output=False,
    )


def start_server():
    """Start the execution environment."""
    print("Starting execution environment...")
    result = subprocess.run(
        ["docker-compose", "-f", str(DOCKER_COMPOSE), "up", "-d"],
        capture_output=False,
    )
    if result.returncode != 0:
        print("ERROR: Failed to start server")
        sys.exit(1)


def wait_for_ready(timeout: float = 120.0, interval: float = 2.0) -> bool:
    """Wait for server to be ready."""
    print(f"Waiting for server to be ready (timeout: {timeout}s)...")
    start = time.time()

    while time.time() - start < timeout:
        health = check_health()
        if health:
            if health.get("status") == "ok" and health.get("corpus_loaded"):
                return True
            print(f"  Server responding but not ready: {health}")
        else:
            elapsed = int(time.time() - start)
            print(f"  Waiting... ({elapsed}s)", end="\r")
        time.sleep(interval)

    return False


def main():
    # Always stop first, then start fresh
    stop_server()
    start_server()

    # Wait for ready
    if wait_for_ready():
        health = check_health()
        print(f"\nServer ready!")
        print(f"  Status: {health.get('status')}")
        print(f"  Corpus loaded: {health.get('corpus_loaded')}")
        print(f"  Pages: {health.get('pages_count')}")
        print(f"\nEndpoint: http://localhost:8080")
    else:
        print("\nERROR: Server did not become ready in time")
        sys.exit(1)


if __name__ == "__main__":
    main()
