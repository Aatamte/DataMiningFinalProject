"""Execution environment server for safe code execution.

This FastAPI server runs inside Docker and provides:
- POST /execute - Execute Python code with access to search tools
- GET /health - Health check endpoint

The server loads the Wikipedia corpus and ChromaDB on startup,
then executes agent-generated code in a restricted environment.
"""

import json
import sys
import traceback
from contextlib import redirect_stdout
from io import StringIO
from threading import Thread
from typing import Any

import chromadb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer


# --- Configuration ---

CORPUS_PATH = "/data/corpus/corpus.json"
CHROMA_PATH = "/data/chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"


# --- Global state (loaded on startup) ---

app = FastAPI(title="Execution Environment")

# Will be initialized on startup
corpus: dict[str, dict] = {}  # page_id -> {title, content}
chroma_collection = None
embed_model = None


# --- Pydantic models ---

class ExecuteRequest(BaseModel):
    code: str
    timeout: float = 30.0


class ExecuteResponse(BaseModel):
    output: str
    error: str | None
    tool_calls: list[dict]


class HealthResponse(BaseModel):
    status: str
    corpus_loaded: bool
    pages_count: int


# --- Tool implementations ---

def search_pages(query: str, n: int = 5) -> list[dict]:
    """Search for relevant Wikipedia pages by title similarity.

    Args:
        query: Search query string
        n: Number of results to return (default 5)

    Returns:
        List of {page_id, title} dicts for top n matches
    """
    global chroma_collection, embed_model

    # Manually embed the query
    query_embedding = embed_model.encode([query]).tolist()

    results = chroma_collection.query(
        query_embeddings=query_embedding,
        n_results=n,
        include=["metadatas", "distances"]
    )

    output = []
    if results and results["ids"] and results["ids"][0]:
        for i, page_id in enumerate(results["ids"][0]):
            title = results["metadatas"][0][i]["title"]
            # ChromaDB returns L2 distance, convert to similarity (1 / (1 + distance))
            distance = results["distances"][0][i] if results.get("distances") else None
            similarity = 1 / (1 + distance) if distance is not None else None
            output.append({
                "page_id": page_id,
                "title": title,
                "similarity": round(similarity, 4) if similarity else None
            })

    return output


def view_sections(page_id: str) -> list[dict]:
    """List sections of a Wikipedia page.

    Args:
        page_id: The page ID to view

    Returns:
        List of {section_id, section_name} dicts
    """
    global corpus

    if page_id not in corpus:
        raise ValueError(f"Page not found: {page_id}")

    content = corpus[page_id]["content"]
    sections = []

    for i, line in enumerate(content.split("\n")):
        if line.startswith("#"):
            section_name = line.lstrip("#").strip()
            section_id = f"{page_id}:{_normalize_id(section_name)}"
            sections.append({
                "section_id": section_id,
                "section_name": section_name,
            })

    # If no sections, return full page as one section
    if not sections:
        sections.append({
            "section_id": f"{page_id}:full",
            "section_name": "Full Page",
        })

    return sections


def read_section(section_id: str) -> str:
    """Read the content of a specific section.

    Args:
        section_id: Section ID in format "page_id:section_name"

    Returns:
        Section content as string
    """
    global corpus

    if ":" not in section_id:
        raise ValueError("Invalid section_id format. Expected: page_id:section_name")

    page_id, section_name_id = section_id.split(":", 1)

    if page_id not in corpus:
        raise ValueError(f"Page not found: {page_id}")

    content = corpus[page_id]["content"]
    lines = content.split("\n")

    # Full page case
    if section_name_id == "full":
        return content

    # Find section boundaries
    section_start = None
    section_end = None

    for i, line in enumerate(lines):
        if line.startswith("#"):
            current_section = _normalize_id(line.lstrip("#").strip())
            if current_section == section_name_id and section_start is None:
                section_start = i
            elif section_start is not None and section_end is None:
                section_end = i
                break

    if section_start is not None:
        if section_end is None:
            section_end = len(lines)
        return "\n".join(lines[section_start:section_end])

    raise ValueError(f"Section not found: {section_id}")


def _normalize_id(text: str) -> str:
    """Normalize text for use as section ID."""
    return text.strip().lower().replace(" ", "_")


# --- Code execution ---

def execute_code(code: str, timeout: float) -> dict[str, Any]:
    """Execute Python code with access to search tools.

    Args:
        code: Python code to execute
        timeout: Maximum execution time in seconds

    Returns:
        Dict with output, error, and tool_calls
    """
    tool_calls = []

    # Wrap tools to track calls
    def tracked_search_pages(query: str, n: int = 5) -> list[dict]:
        result = search_pages(query, n)
        tool_calls.append({"tool": "search_pages", "arg": query, "result_count": len(result)})
        return result

    def tracked_view_sections(page_id: str) -> list[dict]:
        result = view_sections(page_id)
        tool_calls.append({"tool": "view_sections", "arg": page_id, "result_count": len(result)})
        return result

    def tracked_read_section(section_id: str) -> str:
        result = read_section(section_id)
        tool_calls.append({"tool": "read_section", "arg": section_id, "result_length": len(result)})
        return result

    # Restricted globals - only safe builtins + tools
    safe_builtins = {
        "print": print,
        "len": len,
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "list": list,
        "dict": dict,
        "tuple": tuple,
        "set": set,
        "range": range,
        "enumerate": enumerate,
        "zip": zip,
        "map": map,
        "filter": filter,
        "sorted": sorted,
        "reversed": reversed,
        "min": min,
        "max": max,
        "sum": sum,
        "abs": abs,
        "round": round,
        "isinstance": isinstance,
        "type": type,
        "hasattr": hasattr,
        "getattr": getattr,
        "any": any,
        "all": all,
        "ord": ord,
        "chr": chr,
        "repr": repr,
        "iter": iter,
        "next": next,
        "slice": slice,
        "format": format,
        "True": True,
        "False": False,
        "None": None,
        # Exception types (for try/except)
        "Exception": Exception,
        "ValueError": ValueError,
        "TypeError": TypeError,
        "KeyError": KeyError,
        "IndexError": IndexError,
        "NameError": NameError,
        "AttributeError": AttributeError,
        "RuntimeError": RuntimeError,
        "ImportError": ImportError,
        # Allow all imports - Docker provides sandboxing
        "__import__": __import__,
    }

    import re
    import json
    import string
    from collections import Counter, defaultdict

    exec_globals = {
        "__builtins__": safe_builtins,
        "search_pages": tracked_search_pages,
        "view_sections": tracked_view_sections,
        "read_section": tracked_read_section,
        # Pre-imported modules
        "re": re,
        "json": json,
        "string": string,
        "Counter": Counter,
        "defaultdict": defaultdict,
    }

    # Capture stdout
    stdout_capture = StringIO()
    error = None

    # Execute with timeout
    result_container = {"done": False, "error": None}

    def run_code():
        try:
            with redirect_stdout(stdout_capture):
                exec(code, exec_globals)
            result_container["done"] = True
        except Exception as e:
            result_container["error"] = f"{type(e).__name__}: {e}"
            result_container["done"] = True

    thread = Thread(target=run_code)
    thread.start()
    thread.join(timeout=timeout)

    if not result_container["done"]:
        error = f"Execution timed out after {timeout}s"
    elif result_container["error"]:
        error = result_container["error"]

    return {
        "output": stdout_capture.getvalue(),
        "error": error,
        "tool_calls": tool_calls,
    }


# --- API endpoints ---

@app.post("/execute", response_model=ExecuteResponse)
async def execute(request: ExecuteRequest):
    """Execute Python code with access to search tools."""
    try:
        result = execute_code(request.code, request.timeout)
        return ExecuteResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        corpus_loaded=len(corpus) > 0,
        pages_count=len(corpus),
    )


# --- Startup ---

@app.on_event("startup")
async def startup():
    """Load corpus and initialize ChromaDB on startup."""
    global corpus, chroma_collection, embed_model

    print("Loading embedding model...")
    embed_model = SentenceTransformer(EMBED_MODEL)

    print(f"Loading corpus from {CORPUS_PATH}...")
    try:
        with open(CORPUS_PATH, "r") as f:
            corpus_data = json.load(f)

        # Convert to dict keyed by page_id
        for page in corpus_data:
            corpus[page["id"]] = {
                "title": page["title"],
                "content": page["content"],
            }
        print(f"Loaded {len(corpus)} pages")
    except FileNotFoundError:
        print(f"WARNING: Corpus file not found at {CORPUS_PATH}")

    print(f"Connecting to ChromaDB at {CHROMA_PATH}...")
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        # Get collection without embedding function - we'll embed manually
        chroma_collection = client.get_collection(name="wiki_titles")
        print(f"ChromaDB collection loaded with {chroma_collection.count()} entries")
    except Exception as e:
        print(f"WARNING: ChromaDB error: {e}")

    print("Server ready!")


def _get_embedding_function():
    """Create ChromaDB-compatible embedding function."""
    class LocalEmbeddingFunction:
        def name(self) -> str:
            return "local-minilm"

        def __call__(self, input: list[str]) -> list[list[float]]:
            embeddings = embed_model.encode(input)
            return embeddings.tolist()

    return LocalEmbeddingFunction()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
