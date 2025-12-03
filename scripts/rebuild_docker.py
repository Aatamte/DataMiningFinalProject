#!/usr/bin/env python3
"""Rebuild and restart the Docker execution environment."""

import subprocess
import sys


def main():
    compose_file = "docker/docker-compose.yml"

    print("Stopping existing containers...")
    subprocess.run(
        ["docker-compose", "-f", compose_file, "down"],
        capture_output=True,
    )

    print("Building Docker image...")
    result = subprocess.run(
        ["docker-compose", "-f", compose_file, "build", "--no-cache"],
    )

    if result.returncode != 0:
        print("Build failed!")
        sys.exit(1)

    print("\nStarting container...")
    result = subprocess.run(
        ["docker-compose", "-f", compose_file, "up", "-d"],
    )

    if result.returncode != 0:
        print("Failed to start container!")
        sys.exit(1)

    print("\nDocker environment rebuilt and started!")
    print("View logs with: docker-compose -f docker/docker-compose.yml logs -f")


if __name__ == "__main__":
    main()
