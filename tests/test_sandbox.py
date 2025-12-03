"""Tests for the execution environment sandbox.

These tests require the execution environment server to be running.
Run with: uv run pytest tests/test_sandbox.py -v

Setup:
    uv run python scripts/setup_environment.py
    docker-compose -f docker/docker-compose.yml up -d
"""

import pytest
import pytest_asyncio
from src.environment.sandbox import SandboxClient, SandboxError, is_server_running


def check_server_running() -> None:
    """Check if the execution environment server is running."""
    if not is_server_running():
        pytest.fail(
            "Execution environment server is not running.\n"
            "Start it with: docker-compose -f docker/docker-compose.yml up -d"
        )


@pytest.fixture(scope="session", autouse=True)
def ensure_server():
    """Ensure server is running before any tests."""
    check_server_running()


@pytest_asyncio.fixture
async def client():
    """Create a sandbox client for testing."""
    async with SandboxClient() as c:
        yield c


pytestmark = pytest.mark.asyncio


# --- Basic execution tests ---

class TestBasicExecution:
    """Test basic code execution."""

    async def test_simple_print(self, client):
        """Test that print statements work."""
        result = await client.execute('print("Hello, sandbox!")')
        assert result["error"] is None
        assert "Hello, sandbox!" in result["output"]

    async def test_simple_math(self, client):
        """Test basic arithmetic."""
        code = '''
result = 2 + 2
print(f"2 + 2 = {result}")
'''
        result = await client.execute(code)
        assert result["error"] is None
        assert "2 + 2 = 4" in result["output"]

    async def test_multiline_code(self, client):
        """Test multiline code execution."""
        code = '''
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

result = factorial(5)
print(f"5! = {result}")
'''
        result = await client.execute(code)
        assert result["error"] is None
        assert "5! = 120" in result["output"]

    async def test_list_comprehension(self, client):
        """Test list comprehensions work."""
        code = '''
squares = [x**2 for x in range(5)]
print(squares)
'''
        result = await client.execute(code)
        assert result["error"] is None
        assert "[0, 1, 4, 9, 16]" in result["output"]


# --- Tool calling tests ---

class TestToolCalls:
    """Test tool calling from sandbox."""

    async def test_search_pages(self, client):
        """Test calling search_pages tool."""
        code = '''
results = search_pages("basketball")
print(f"Found {len(results)} results")
for r in results[:3]:
    print(f"  - {r['title']}")
'''
        result = await client.execute(code)
        assert result["error"] is None
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["tool"] == "search_pages"

    async def test_view_sections(self, client):
        """Test calling view_sections tool."""
        code = '''
# First search for a page
results = search_pages("python")
if results:
    page_id = results[0]["page_id"]
    sections = view_sections(page_id)
    print(f"Found {len(sections)} sections")
    for s in sections[:3]:
        print(f"  - {s['section_name']}")
'''
        result = await client.execute(code)
        assert result["error"] is None
        assert len(result["tool_calls"]) >= 2

    async def test_read_section(self, client):
        """Test calling read_section tool."""
        code = '''
results = search_pages("python")
if results:
    page_id = results[0]["page_id"]
    sections = view_sections(page_id)
    if sections:
        content = read_section(sections[0]["section_id"])
        print(f"Content length: {len(content)} chars")
        print(f"Preview: {content[:100]}...")
'''
        result = await client.execute(code)
        assert result["error"] is None
        assert len(result["tool_calls"]) >= 3

    async def test_multiple_searches(self, client):
        """Test multiple search calls."""
        code = '''
queries = ["python", "javascript", "rust"]
for q in queries:
    results = search_pages(q)
    print(f"{q}: {len(results)} results")
'''
        result = await client.execute(code)
        assert result["error"] is None
        assert len(result["tool_calls"]) == 3


# --- Error handling tests ---

class TestErrorHandling:
    """Test error handling in sandbox."""

    async def test_syntax_error(self, client):
        """Test that syntax errors are reported."""
        code = '''
def broken(
    print("missing parenthesis")
'''
        result = await client.execute(code)
        assert result["error"] is not None
        assert "SyntaxError" in result["error"]

    async def test_runtime_error(self, client):
        """Test that runtime errors are reported."""
        code = 'x = 1 / 0'
        result = await client.execute(code)
        assert result["error"] is not None
        assert "ZeroDivisionError" in result["error"]

    async def test_name_error(self, client):
        """Test undefined variable error."""
        code = 'print(undefined_variable)'
        result = await client.execute(code)
        assert result["error"] is not None
        assert "NameError" in result["error"]

    async def test_invalid_page_id(self, client):
        """Test error when page not found."""
        code = '''
try:
    sections = view_sections("nonexistent_page_12345")
    print("Should not reach here")
except Exception as e:
    print(f"Got error: {e}")
'''
        result = await client.execute(code)
        assert result["error"] is None
        assert "Got error" in result["output"]


# --- Security tests ---

class TestSecurity:
    """Test sandbox security restrictions."""

    async def test_no_file_access(self, client):
        """Test that file access is blocked."""
        code = '''
try:
    open("/etc/passwd", "r")
    print("FAIL: Should not be able to open files")
except NameError:
    print("OK: open is not available")
except Exception as e:
    print(f"OK: Got {type(e).__name__}")
'''
        result = await client.execute(code)
        assert "FAIL" not in result["output"]
        assert "OK" in result["output"]

    async def test_no_import(self, client):
        """Test that import is blocked."""
        code = '''
try:
    import os
    print("FAIL: Should not be able to import")
except NameError:
    print("OK: import is not available")
except Exception as e:
    print(f"OK: Got {type(e).__name__}")
'''
        result = await client.execute(code)
        assert "FAIL" not in result["output"]

    async def test_no_eval(self, client):
        """Test that eval is blocked."""
        code = '''
try:
    eval("1 + 1")
    print("FAIL: Should not be able to eval")
except NameError:
    print("OK: eval is not available")
'''
        result = await client.execute(code)
        assert "FAIL" not in result["output"]
        assert "OK" in result["output"]

    async def test_safe_builtins_available(self, client):
        """Test that safe builtins work."""
        code = '''
# Test various safe operations
print(len([1, 2, 3]))
print(str(42))
print(list(range(3)))
print(sorted([3, 1, 2]))
print(sum([1, 2, 3]))
'''
        result = await client.execute(code)
        assert result["error"] is None
        assert "3" in result["output"]
        assert "42" in result["output"]


# --- Integration tests ---

class TestIntegration:
    """Integration tests for realistic workflows."""

    async def test_search_and_read_workflow(self, client):
        """Test a realistic search-and-read workflow."""
        code = '''
# Search for information
results = search_pages("world war")
print(f"Found {len(results)} pages")

if results:
    # Get first result
    page = results[0]
    print(f"Looking at: {page['title']}")

    # View sections
    sections = view_sections(page["page_id"])
    print(f"Has {len(sections)} sections")

    # Read first section
    if sections:
        content = read_section(sections[0]["section_id"])
        print(f"First section has {len(content)} chars")
'''
        result = await client.execute(code)
        assert result["error"] is None
        assert len(result["tool_calls"]) >= 3

    async def test_conditional_search(self, client):
        """Test code with conditional logic."""
        code = '''
query = "programming language"
results = search_pages(query)

found_python = False
for r in results:
    if "python" in r["title"].lower():
        found_python = True
        print(f"Found Python: {r['title']}")
        break

if not found_python:
    print("Python not in top results")
'''
        result = await client.execute(code)
        assert result["error"] is None


# --- Health check tests ---

class TestHealthCheck:
    """Test health check endpoint."""

    async def test_health_returns_ok(self, client):
        """Test health endpoint returns OK status."""
        health = await client.health()
        assert health["status"] == "ok"

    async def test_health_shows_corpus_loaded(self, client):
        """Test health shows corpus is loaded."""
        health = await client.health()
        assert health["corpus_loaded"] is True
        assert health["pages_count"] > 0
