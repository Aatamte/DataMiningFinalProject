"""Test script to verify the local environment works."""

import asyncio
import sys
sys.path.insert(0, ".")

from src.environment import load_environment


async def test_tools():
    """Test each tool manually."""
    print("Loading environment...")
    env = load_environment(
        chroma_db_dir="data/.chroma_db",
        embed_model="all-MiniLM-L6-v2",
        embed_device="cuda",
    )

    # Get the tools from the environment
    tools = {t.__name__: t for t in env.tools}

    print("\n" + "=" * 50)
    print("Testing search_pages...")
    print("=" * 50)
    results = await tools["search_pages"]("basketball history")
    print(f"Found {len(results)} results:")
    for r in results[:3]:
        print(f"  - {r['title']} (id: {r['page_id']})")

    if results:
        page_id = results[0]["page_id"]

        print("\n" + "=" * 50)
        print(f"Testing view_sections on '{page_id}'...")
        print("=" * 50)
        sections = await tools["view_sections"](page_id)
        print(f"Found {len(sections)} sections:")
        for s in sections[:5]:
            print(f"  - {s['section_name']} (id: {s['section_id']})")

        if sections:
            section_id = sections[0]["section_id"]

            print("\n" + "=" * 50)
            print(f"Testing read_section on '{section_id}'...")
            print("=" * 50)
            content = await tools["read_section"](section_id)
            print(f"Content preview ({len(content)} chars):")
            print(content[:500] + "..." if len(content) > 500 else content)

    print("\n" + "=" * 50)
    print("All tool tests passed!")
    print("=" * 50)


async def test_sample_question():
    """Test with a sample question from the dataset."""
    print("\nLoading environment for question test...")
    env = load_environment(
        chroma_db_dir="data/.chroma_db",
        embed_model="all-MiniLM-L6-v2",
        embed_device="cuda",
    )

    print("\n" + "=" * 50)
    print("Sample questions from dataset:")
    print("=" * 50)
    for i, item in enumerate(env.dataset):
        if i >= 3:
            break
        print(f"\nQ{i + 1}: {item.get('question', item.get('prompt', 'N/A'))}")
        print(f"A{i + 1}: {item.get('answer', 'N/A')}")


def main():
    print("Testing local wiki-search environment\n")
    asyncio.run(test_tools())
    asyncio.run(test_sample_question())


if __name__ == "__main__":
    main()
