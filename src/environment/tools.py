"""Search tools for the wiki environment."""

import asyncio
import json
from typing import Callable


def normalize_id(text: str) -> str:
    """Normalize text to use as an ID."""
    return text.strip().lower().replace(" ", "_")


def create_search_tools(
    collection,
    page_id_to_title: dict[str, str],
    page_id_to_content: dict[str, str],
    semaphore: asyncio.Semaphore,
) -> list[Callable]:
    """Create the search tools bound to the given data.

    Args:
        collection: ChromaDB collection for semantic search
        page_id_to_title: Mapping of page IDs to titles
        page_id_to_content: Mapping of page IDs to content
        semaphore: Semaphore for limiting concurrent ChromaDB queries

    Returns:
        List of tool functions: [search_pages, view_sections, read_section]
    """

    async def search_pages(query: str) -> list[dict]:
        """Search for top 10 relevant articles using title embedding similarity.

        args:
            query (str): The query to search for.

        returns:
            list[dict]: A list of dicts with page_id and title.

        example:
            "basketball" -> [{"page_id": "basketball", "title": "Basketball"}, ...]
        """
        async with semaphore:
            results = await asyncio.to_thread(
                collection.query, query_texts=[query], n_results=10
            )

        if not results:
            raise ValueError(f"No results found for query: {query}")
        if not results["metadatas"]:
            raise ValueError(f"No results metadata found for query: {query}")

        output = []
        for i in range(len(results["ids"][0])):
            output.append(
                {
                    "page_id": results["ids"][0][i],
                    "title": results["metadatas"][0][i]["title"],
                }
            )
        return output

    async def view_sections(page_id: str) -> list[dict]:
        """View the sections of a page.

        args:
            page_id (str): The ID of the page to view.

        returns:
            list[dict]: A list of dicts with section_id and section_name.

        example:
            "basketball" -> [{"section_id": "basketball:history", "section_name": "History"}, ...]
        """
        if page_id not in page_id_to_content:
            raise ValueError(f"Page not found: {page_id}")

        content = page_id_to_content[page_id]
        sections = []
        lines = content.split("\n")

        for i, line in enumerate(lines):
            if line.startswith("#"):
                section_name = line.lstrip("#").strip()
                section_id = f"{page_id}:{normalize_id(section_name)}"
                sections.append(
                    {
                        "section_id": section_id,
                        "section_name": section_name,
                        "start_line": i,
                    }
                )

        # If no sections found, return the whole page as one section
        if not sections:
            sections.append(
                {
                    "section_id": f"{page_id}:full",
                    "section_name": "Full Page",
                    "start_line": 0,
                }
            )

        return [
            {"section_id": s["section_id"], "section_name": s["section_name"]}
            for s in sections
        ]

    async def read_section(section_id: str) -> str:
        """Read a section of a page.

        args:
            section_id (str): The ID of the section to read.

        returns:
            str: The content of the section.

        example:
            "baseball:finnish_baseball" -> "Finnish baseball is a sport..."
        """
        if ":" not in section_id:
            raise ValueError(
                "Invalid section_id format. Expected: page_id:section_name"
            )

        page_id, section_name_id = section_id.split(":", 1)

        if page_id not in page_id_to_content:
            raise ValueError(f"Page not found: {page_id}")

        content = page_id_to_content[page_id]
        lines = content.split("\n")

        # Special case for "full" section
        if section_name_id == "full":
            return content

        # Find section
        section_start = None
        section_end = None

        for i, line in enumerate(lines):
            if line.startswith("#"):
                current_section = normalize_id(line.lstrip("#").strip())
                if current_section == section_name_id and section_start is None:
                    section_start = i
                elif section_start is not None and section_end is None:
                    section_end = i
                    break

        if section_start is not None:
            if section_end is None:
                section_end = len(lines)
            return "\n".join(lines[section_start:section_end])
        else:
            raise ValueError(f"Section not found: {section_id}")

    return [search_pages, view_sections, read_section]
