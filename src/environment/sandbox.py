"""HTTP client for the execution environment server.

This module provides a client to interact with the Docker-based
execution environment server for safe code execution.
"""

import asyncio
import subprocess
import time
from typing import Any

import httpx


class SandboxError(Exception):
    """Error during sandbox execution."""
    pass


class SandboxClient:
    """Client for the execution environment server."""

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        timeout: float = 30.0,
    ):
        """Initialize the sandbox client.

        Args:
            base_url: URL of the execution environment server
            timeout: Default timeout for code execution
        """
        self.base_url = base_url
        self.timeout = timeout
        self._client = httpx.AsyncClient(base_url=base_url, timeout=60.0)

    async def execute(self, code: str, timeout: float | None = None) -> dict[str, Any]:
        """Execute Python code in the sandbox.

        Args:
            code: Python code to execute
            timeout: Execution timeout (uses default if not specified)

        Returns:
            Dict with:
                - output: stdout from the code
                - error: Any error message (None if success)
                - tool_calls: List of tool calls made
        """
        if timeout is None:
            timeout = self.timeout

        try:
            response = await self._client.post(
                "/execute",
                json={"code": code, "timeout": timeout},
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            raise SandboxError(f"Server error: {e.response.status_code} - {e.response.text}")
        except httpx.RequestError as e:
            raise SandboxError(f"Connection error: {e}")

    async def health(self) -> dict[str, Any]:
        """Check server health.

        Returns:
            Dict with status, corpus_loaded, pages_count
        """
        try:
            response = await self._client.get("/health")
            response.raise_for_status()
            return response.json()
        except httpx.RequestError as e:
            raise SandboxError(f"Health check failed: {e}")

    async def wait_for_ready(self, timeout: float = 120.0, interval: float = 2.0) -> bool:
        """Wait for the server to be ready.

        Args:
            timeout: Maximum time to wait
            interval: Time between checks

        Returns:
            True if server is ready, raises SandboxError if timeout
        """
        start = time.time()
        while time.time() - start < timeout:
            try:
                health = await self.health()
                if health.get("status") == "ok" and health.get("corpus_loaded"):
                    return True
            except SandboxError:
                pass
            await asyncio.sleep(interval)

        raise SandboxError(f"Server not ready after {timeout}s")

    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


def start_server(docker_compose_path: str = "docker/docker-compose.yml") -> subprocess.Popen:
    """Start the execution environment server using docker-compose.

    Args:
        docker_compose_path: Path to docker-compose.yml

    Returns:
        Popen object for the docker-compose process
    """
    proc = subprocess.Popen(
        ["docker-compose", "-f", docker_compose_path, "up", "-d"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    proc.wait()

    if proc.returncode != 0:
        _, stderr = proc.communicate()
        raise SandboxError(f"Failed to start server: {stderr.decode()}")

    return proc


def stop_server(docker_compose_path: str = "docker/docker-compose.yml") -> None:
    """Stop the execution environment server.

    Args:
        docker_compose_path: Path to docker-compose.yml
    """
    subprocess.run(
        ["docker-compose", "-f", docker_compose_path, "down"],
        capture_output=True,
    )


def is_server_running(base_url: str = "http://localhost:8080") -> bool:
    """Check if the server is running.

    Args:
        base_url: URL of the execution environment server

    Returns:
        True if server is responding, False otherwise
    """
    try:
        response = httpx.get(f"{base_url}/health", timeout=5.0)
        return response.status_code == 200
    except httpx.RequestError:
        return False


# Convenience function for simple usage
async def execute_code(
    code: str,
    timeout: float = 30.0,
    base_url: str = "http://localhost:8080",
) -> dict[str, Any]:
    """Execute code in the sandbox (convenience function).

    Args:
        code: Python code to execute
        timeout: Execution timeout
        base_url: URL of the execution environment server

    Returns:
        Dict with output, error, and tool_calls
    """
    async with SandboxClient(base_url=base_url, timeout=timeout) as client:
        return await client.execute(code)
