# Execution Flow

## Overview

The SLM learns to answer trivia questions by writing Python code that searches a Wikipedia corpus.

## Flow Diagram

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│    SLM      │────▶│   Sandbox   │────▶│    Judge    │
│ (generates  │     │ (executes   │     │ (evaluates  │
│  code)      │◀────│  code)      │     │  answer)    │
└─────────────┘     └─────────────┘     └─────────────┘
```

## Step-by-Step

1. **Question** → SLM receives trivia question
2. **Code Generation** → SLM writes Python code using tools
3. **Execution** → Sandbox runs code, returns output
4. **Answer** → SLM produces `<answer>...</answer>` based on output
5. **Evaluation** → Judge compares answer to ground truth → reward

## Available Tools

| Tool | Description |
|------|-------------|
| `search_pages(query)` | Semantic search over Wikipedia titles → `[{page_id, title}, ...]` |
| `view_sections(page_id)` | List sections of a page → `[{section_id, section_name}, ...]` |
| `read_section(section_id)` | Get section content → `str` |

## Key Points

- **SLM generates code** that uses the 3 tools above
- **Sandbox executes code** and returns stdout
- **SLM produces answer** in `<answer>...</answer>` tags (NOT a tool call)
- **Judge evaluates** the answer against ground truth

## Example

**Question:** "What university did Miles Teller attend for his BFA?"

**SLM generates:**
```python
results = search_pages("Miles Teller")
page_id = results[0]["page_id"]
sections = view_sections(page_id)
for s in sections:
    content = read_section(s["section_id"])
    print(content)
```

**Sandbox returns:**
```
Miles Teller attended New York University's Tisch School of the Arts...
```

**SLM outputs:**
```
Based on the search results, Miles Teller attended NYU.

<answer>New York University</answer>
```

**Judge:** Compares "New York University" to ground truth → reward = 1.0
