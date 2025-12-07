## Evidence of Meaningfulness

To verify that our results reflect genuine search capability rather than artifacts, we manually inspected individual trajectories from the evaluation set.

### Successful Trajectory

The following example demonstrates the model composing multiple tools to answer a multi-part question:

**Question:** Which 2023 album did Caroline Polachek release that received critical acclaim and includes the single 'Bunny Is a Rider'?

**Turn 1 — Model generates:**
```python
results = search_pages("Caroline Polachek 2023 album Bunny Is a Rider")
if results:
    page_id = results[0]['page_id']
    sections = view_sections(page_id)
    for section in sections:
        section_id = section['section_id']
        content = read_section(section_id)
        if 'Bunny Is a Rider' in content:
            print(content)
            break
```

**Turn 1 — Environment returns:**
```
### 2021–present: *Desire, I Want to Turn Into You*

On July 14, 2021, Polachek released the single "Bunny Is a Rider"...

On December 5, 2022, Polachek released the single "Welcome to My Island"
and announced her second studio album under her full name, *Desire, I Want
to Turn Into You*, for release on February 14, 2023. The album received
critical acclaim and includes "Bunny Is a Rider", "Billions", "Sunset",
and "Welcome to My Island".
```

**Turn 2 — Model outputs:**
```
<answer>Desire, I Want to Turn Into You</answer>
```

This trajectory shows the model: (1) formulating a targeted semantic query, (2) navigating to the relevant page, (3) iterating through sections with a programmatic filter, and (4) extracting the correct answer from retrieved content. The use of conditional logic (`if 'Bunny Is a Rider' in content`) demonstrates learned behavior beyond simple retrieval.

### Failure Modes

We also examined failure cases to understand limitations:

1. **Retrieval failure:** The query "first Chairman of the Joint Chiefs of Staff United States" returned "CEO" (a redirect page) instead of the relevant Wikipedia article. The embedding-based title search matched "Chief" in "CEO" rather than finding "Chairman of the Joint Chiefs of Staff."

2. **Hallucination under uncertainty:** When retrieval returned irrelevant content twice, the model did not reformulate its query or acknowledge uncertainty. Instead, it hallucinated a plausible-sounding but incorrect answer ("Hap Arnold" was a WWII general but never held this position).

These failures suggest directions for improvement: better embedding models for retrieval, and training the model to express uncertainty rather than guess when search fails.
