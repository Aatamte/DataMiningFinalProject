"""Test the Docker sandbox execution."""

import asyncio
from src.environment.sandbox import Sandbox


# Mock tool handlers for testing
async def mock_search_pages(query: str) -> list:
    """Mock search that returns fake results."""
    return [
        {"page_id": "test_page_1", "title": f"Result for: {query}"},
        {"page_id": "test_page_2", "title": "Another result"},
    ]


async def mock_view_sections(page_id: str) -> list:
    """Mock view sections."""
    return [
        {"section_id": f"{page_id}:intro", "section_name": "Introduction"},
        {"section_id": f"{page_id}:history", "section_name": "History"},
    ]


async def mock_read_section(section_id: str) -> str:
    """Mock read section."""
    return f"This is the content of section: {section_id}"


async def test_simple_code():
    """Test simple code execution."""
    print("\n=== Test 1: Simple code ===")

    sandbox = Sandbox(
        tool_handlers={
            "search_pages": mock_search_pages,
            "view_sections": mock_view_sections,
            "read_section": mock_read_section,
        },
        timeout=30.0,
    )

    code = '''
print("Hello from sandbox!")
x = 2 + 2
print(f"2 + 2 = {x}")
answer(str(x))
'''

    result = await sandbox.execute(code)
    print(f"Answer: {result['answer']}")
    print(f"Output: {result['output']}")
    print(f"Error: {result['error']}")
    print(f"Tool calls: {result['tool_calls']}")

    assert result["answer"] == "4", f"Expected '4', got {result['answer']}"
    assert result["error"] is None
    print("✓ Test passed!")


async def test_tool_calls():
    """Test code that calls tools."""
    print("\n=== Test 2: Tool calls ===")

    sandbox = Sandbox(
        tool_handlers={
            "search_pages": mock_search_pages,
            "view_sections": mock_view_sections,
            "read_section": mock_read_section,
        },
        timeout=30.0,
    )

    code = '''
# Search for pages about Python
results = search_pages("Python programming")
print(f"Found {len(results)} results")

# View sections of first result
page_id = results[0]["page_id"]
sections = view_sections(page_id)
print(f"Page has {len(sections)} sections")

# Read first section
section_id = sections[0]["section_id"]
content = read_section(section_id)
print(f"Content preview: {content[:50]}")

answer(f"Found {len(results)} pages with {len(sections)} sections")
'''

    result = await sandbox.execute(code)
    print(f"Answer: {result['answer']}")
    print(f"Output: {result['output']}")
    print(f"Error: {result['error']}")
    print(f"Tool calls: {result['tool_calls']}")

    assert len(result["tool_calls"]) == 3
    assert result["tool_calls"][0]["tool"] == "search_pages"
    assert result["tool_calls"][1]["tool"] == "view_sections"
    assert result["tool_calls"][2]["tool"] == "read_section"
    assert result["error"] is None
    print("✓ Test passed!")


async def test_timeout():
    """Test that infinite loops are killed."""
    print("\n=== Test 3: Timeout ===")

    sandbox = Sandbox(
        tool_handlers={},
        timeout=3.0,  # Short timeout for test
    )

    code = '''
# This should timeout
while True:
    pass
'''

    result = await sandbox.execute(code)
    print(f"Error: {result['error']}")

    assert "timed out" in result["error"].lower()
    print("✓ Test passed!")


async def test_no_network():
    """Test that network access is blocked."""
    print("\n=== Test 4: No network ===")

    sandbox = Sandbox(
        tool_handlers={},
        timeout=10.0,
    )

    code = '''
import urllib.request
try:
    urllib.request.urlopen("http://google.com", timeout=5)
    answer("FAIL: Network should be blocked")
except Exception as e:
    answer(f"OK: {type(e).__name__}")
'''

    result = await sandbox.execute(code)
    print(f"Answer: {result['answer']}")

    assert result["answer"].startswith("OK:")
    print("✓ Test passed!")


async def main():
    """Run all tests."""
    print("Testing Docker sandbox...")
    print("Make sure Docker is running!\n")

    await test_simple_code()
    await test_tool_calls()
    await test_timeout()
    await test_no_network()

    print("\n" + "=" * 40)
    print("All tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
