"""Test script to verify the execution environment works.

This tests the Docker sandbox server by executing code that uses the search tools.

Prerequisites:
    1. Run setup: uv run python scripts/setup_environment.py
    2. Start server: uv run python scripts/run_environment.py
"""

import asyncio
import sys
sys.path.insert(0, ".")

from src.environment import SandboxClient, is_server_running, load_questions


async def test_health():
    """Test server health check."""
    print("=" * 50)
    print("Testing server health...")
    print("=" * 50)

    async with SandboxClient() as client:
        health = await client.health()
        print(f"  Status: {health['status']}")
        print(f"  Corpus loaded: {health['corpus_loaded']}")
        print(f"  Pages count: {health['pages_count']}")

        if not health['corpus_loaded']:
            print("\nWARNING: Corpus not loaded! Run setup_environment.py first.")
            return False

    return True


async def test_search_pages():
    """Test search_pages tool."""
    print("\n" + "=" * 50)
    print("Testing search_pages...")
    print("=" * 50)

    code = '''
results = search_pages("basketball history")
print(f"Found {len(results)} results:")
for r in results[:5]:
    sim = r.get('similarity', 'N/A')
    print(f"  - {r['title']} (sim: {sim})")
# Output first page_id on last line for extraction
print(f"FIRST_PAGE_ID:{results[0]['page_id']}")
'''

    async with SandboxClient() as client:
        result = await client.execute(code)

        if result['error']:
            print(f"  ERROR: {result['error']}")
            return None

        print(result['output'])
        print(f"  Tool calls: {result['tool_calls']}")

        # Extract page_id from output
        for line in result['output'].split('\n'):
            if line.startswith('FIRST_PAGE_ID:'):
                return line.split(':', 1)[1]
    return None


async def test_view_sections(page_id: str):
    """Test view_sections tool."""
    print("\n" + "=" * 50)
    print(f"Testing view_sections on '{page_id}'...")
    print("=" * 50)

    code = f'''
sections = view_sections("{page_id}")
print(f"Found {{len(sections)}} sections:")
for s in sections[:5]:
    print(f"  - {{s['section_name']}} (id: {{s['section_id']}})")
'''

    async with SandboxClient() as client:
        result = await client.execute(code)

        if result['error']:
            print(f"  ERROR: {result['error']}")
            return None

        print(result['output'])
        print(f"  Tool calls: {result['tool_calls']}")

        # Return first section_id for next test
        if result['tool_calls']:
            # Extract a section_id from the output
            return f"{page_id}:full"
    return None


async def test_read_section(section_id: str):
    """Test read_section tool."""
    print("\n" + "=" * 50)
    print(f"Testing read_section on '{section_id}'...")
    print("=" * 50)

    code = f'''
content = read_section("{section_id}")
preview = content[:500] + "..." if len(content) > 500 else content
print(f"Content ({{len(content)}} chars):")
print(preview)
'''

    async with SandboxClient() as client:
        result = await client.execute(code)

        if result['error']:
            print(f"  ERROR: {result['error']}")
            return False

        print(result['output'])
        print(f"  Tool calls: {result['tool_calls']}")

    return True


async def test_sample_questions():
    """Test loading sample questions from the dataset."""
    print("\n" + "=" * 50)
    print("Sample questions from dataset:")
    print("=" * 50)

    try:
        dataset = load_questions()
        for i, item in enumerate(dataset):
            if i >= 3:
                break
            print(f"\nQ{i + 1}: {item.get('question', item.get('prompt', 'N/A'))}")
            print(f"A{i + 1}: {item.get('answer', 'N/A')}")
    except Exception as e:
        print(f"  ERROR loading dataset: {e}")


async def test_sandbox_builtins():
    """Test that all expected builtins and modules are available in sandbox."""
    print("\n" + "=" * 50)
    print("Testing sandbox builtins and modules...")
    print("=" * 50)

    code = '''
# Test pre-imported modules
modules_ok = []
modules_fail = []

# Test re
try:
    match = re.search(r"(\\d+)", "test123")
    assert match.group(1) == "123"
    modules_ok.append("re")
except Exception as e:
    modules_fail.append(f"re: {e}")

# Test json
try:
    data = json.loads('{"key": "value"}')
    assert data["key"] == "value"
    modules_ok.append("json")
except Exception as e:
    modules_fail.append(f"json: {e}")

# Test string
try:
    assert "abc" in string.ascii_lowercase
    modules_ok.append("string")
except Exception as e:
    modules_fail.append(f"string: {e}")

# Test Counter
try:
    c = Counter(["a", "b", "a"])
    assert c["a"] == 2
    modules_ok.append("Counter")
except Exception as e:
    modules_fail.append(f"Counter: {e}")

# Test defaultdict
try:
    d = defaultdict(int)
    d["key"] += 1
    assert d["key"] == 1
    modules_ok.append("defaultdict")
except Exception as e:
    modules_fail.append(f"defaultdict: {e}")

# Test additional builtins
builtins_ok = []
builtins_fail = []

try:
    assert any([False, True, False]) == True
    builtins_ok.append("any")
except Exception as e:
    builtins_fail.append(f"any: {e}")

try:
    assert all([True, True, True]) == True
    builtins_ok.append("all")
except Exception as e:
    builtins_fail.append(f"all: {e}")

try:
    assert ord("A") == 65
    builtins_ok.append("ord")
except Exception as e:
    builtins_fail.append(f"ord: {e}")

try:
    assert chr(65) == "A"
    builtins_ok.append("chr")
except Exception as e:
    builtins_fail.append(f"chr: {e}")

try:
    assert repr("test") == "'test'"
    builtins_ok.append("repr")
except Exception as e:
    builtins_fail.append(f"repr: {e}")

print(f"MODULES_OK:{modules_ok}")
print(f"MODULES_FAIL:{modules_fail}")
print(f"BUILTINS_OK:{builtins_ok}")
print(f"BUILTINS_FAIL:{builtins_fail}")
'''

    async with SandboxClient() as client:
        result = await client.execute(code)

        if result['error']:
            print(f"  ERROR: {result['error']}")
            return False

        print(result['output'])

        # Check for failures
        output = result['output']
        if "MODULES_FAIL:[]" in output and "BUILTINS_FAIL:[]" in output:
            print("  [OK] All modules and builtins available")
            return True
        else:
            print("  [FAIL] Some modules or builtins missing")
            return False


async def test_embedding_quality():
    """Test that embeddings are working correctly.

    These tests will FAIL if:
    - Embeddings are mismatched between indexing and querying
    - Similarity scores are not being computed correctly
    - Search results are essentially random
    """
    print("\n" + "=" * 50)
    print("Testing embedding quality...")
    print("=" * 50)

    failures = []

    async with SandboxClient() as client:
        # Test 1: Get a known title from the corpus and search for it exactly
        # It should be the #1 result with very high similarity
        print("\n  Test 1: Exact title search")
        code = '''
import json
# First, get any page to know a real title
results = search_pages("president")
if results:
    known_title = results[0]['title']
    # Now search for that exact title
    exact_results = search_pages(known_title)
    print(f"KNOWN_TITLE:{known_title}")
    print(f"TOP_RESULT:{exact_results[0]['title']}")
    print(f"TOP_SIMILARITY:{exact_results[0].get('similarity', 0)}")
    print(f"MATCH:{known_title == exact_results[0]['title']}")
'''
        result = await client.execute(code)
        if result['error']:
            failures.append(f"Test 1 error: {result['error']}")
        else:
            output = result['output']
            print(output)

            # Parse results
            lines = {l.split(':')[0]: l.split(':', 1)[1] for l in output.strip().split('\n') if ':' in l}

            if lines.get('MATCH') != 'True':
                failures.append(f"Test 1 FAILED: Exact title search didn't return exact match as #1")

            similarity = float(lines.get('TOP_SIMILARITY', 0))
            if similarity < 0.9:
                failures.append(f"Test 1 FAILED: Exact match similarity too low: {similarity} (expected > 0.9)")
            else:
                print(f"    [OK] Exact title match with similarity {similarity}")

        # Test 2: Search should return results with reasonable similarity spread
        print("\n  Test 2: Similarity score distribution")
        code = '''
results = search_pages("famous historical figure", n=5)
sims = [r.get('similarity', 0) for r in results]
print(f"SIMILARITIES:{sims}")
print(f"MAX_SIM:{max(sims)}")
print(f"MIN_SIM:{min(sims)}")
print(f"HAS_SPREAD:{max(sims) - min(sims) > 0.01}")
for r in results:
    print(f"  {r['title']}: {r.get('similarity')}")
'''
        result = await client.execute(code)
        if result['error']:
            failures.append(f"Test 2 error: {result['error']}")
        else:
            output = result['output']
            print(output)

            lines = {l.split(':')[0]: l.split(':', 1)[1] for l in output.strip().split('\n') if ':' in l and not l.startswith('  ')}

            if lines.get('HAS_SPREAD') != 'True':
                failures.append("Test 2 FAILED: No similarity spread (all results have same score - embeddings may be broken)")
            else:
                print(f"    [OK] Results have similarity spread")

        # Test 3: Semantically similar queries should return overlapping results
        print("\n  Test 3: Semantic consistency")
        code = '''
results1 = search_pages("United States president", n=5)
results2 = search_pages("American presidential leader", n=5)
titles1 = set(r['title'] for r in results1)
titles2 = set(r['title'] for r in results2)
overlap = titles1 & titles2
print(f"QUERY1_RESULTS:{[r['title'] for r in results1]}")
print(f"QUERY2_RESULTS:{[r['title'] for r in results2]}")
print(f"OVERLAP_COUNT:{len(overlap)}")
print(f"OVERLAP:{list(overlap)}")
'''
        result = await client.execute(code)
        if result['error']:
            failures.append(f"Test 3 error: {result['error']}")
        else:
            output = result['output']
            print(output)

            lines = {l.split(':')[0]: l.split(':', 1)[1] for l in output.strip().split('\n') if ':' in l}
            overlap_count = int(lines.get('OVERLAP_COUNT', 0))

            if overlap_count == 0:
                failures.append("Test 3 FAILED: Semantically similar queries have NO overlap - embeddings may be random")
            else:
                print(f"    [OK] Semantic queries have {overlap_count} overlapping results")

    # Report results
    print("\n" + "-" * 50)
    if failures:
        print("EMBEDDING QUALITY TESTS FAILED:")
        for f in failures:
            print(f"  [FAIL] {f}")
        return False
    else:
        print("All embedding quality tests passed!")
        return True


async def main():
    print("Testing execution environment\n")

    # Check if server is running
    if not is_server_running():
        print("ERROR: Server not running!")
        print("Start it with: uv run python scripts/run_environment.py")
        sys.exit(1)

    # Test health
    if not await test_health():
        sys.exit(1)

    # Test tools in sequence
    page_id = await test_search_pages()
    if page_id:
        section_id = await test_view_sections(page_id)
        if section_id:
            await test_read_section(section_id)

    # Test dataset loading
    await test_sample_questions()

    # Test sandbox builtins and modules
    builtins_ok = await test_sandbox_builtins()

    # Test embedding quality (will fail if embeddings are broken)
    embedding_ok = await test_embedding_quality()

    print("\n" + "=" * 50)
    if embedding_ok and builtins_ok:
        print("All tests passed!")
    else:
        print("SOME TESTS FAILED - see above for details")
        sys.exit(1)
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
