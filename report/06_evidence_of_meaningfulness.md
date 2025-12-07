## Evidence of Meaningfulness

To verify that our results reflect genuine search capability rather than artifacts, we manually inspected individual trajectories from the evaluation set.

### Successful Trajectories

**Example 1: Multi-step search with filtering**

**Question:** Which 2023 film features Glenn Howerton in the role of Jim Balsillie?

**Turn 1 — Model generates:**
```python
# Search for films from 2023 featuring Glenn Howerton
results = search_pages("2023 film Glenn Howerton")[:10]

for result in results:
    title = result['title']
    page_id = result['page_id']

    # Check if "Jim Balsillie" is mentioned in the content
    content = read_section(page_id + ":full")
    if "Jim Balsillie" in content:
        print(f"Found potential match: {title}")
        print(content)
        break
```

**Turn 1 — Environment returns:**
```
Found potential match: Glenn Howerton
# Glenn Howerton
...
He received acclaim for his portrayal of Jim Balsillie in the film
*BlackBerry* (2023).
...
```

**Turn 2 — Model outputs:** `<answer>BlackBerry</answer>` *(Reward: 0.99)*

This trajectory demonstrates the model: (1) formulating a targeted query combining actor and year, (2) iterating through results with a programmatic filter for the character name, and (3) extracting the correct film title from the retrieved content.

---

**Example 2: Targeted information extraction**

**Question:** Who performs the cover of Leonard Cohen's 'Who by Fire' used as the opening theme for Bad Sisters?

**Turn 1 — Model generates:**
```python
# Search for the TV show "Bad Sisters" to find information about its opening theme
for result in search_pages("Bad Sisters opening theme")[:5]:
    content = read_section(result['page_id'] + ":full")
    if "Leonard Cohen" in content or "Who by Fire" in content:
        print(f"Found relevant content in {result['title']}:")
        print(content)
        break
```

**Turn 1 — Environment returns:**
```
Found relevant content in Bad Sisters:
# Bad Sisters
...
| opentheme = "Who by Fire" by Leonard Cohen, performed by PJ Harvey
...
```

**Turn 2 — Model outputs:** `<answer>PJ Harvey</answer>` *(Reward: 0.98)*

The model learned to search for the show directly and filter for content mentioning either the song title or original artist—a strategy that efficiently locates the performer information.

---

### Failure Modes

We also examined failure cases to understand limitations:

**1. Immediate surrender without search**

**Question:** Which American grocery chain does Aldi Nord own?

**Turn 1 — Model outputs:** `<answer>Answer not found</answer>` *(Reward: 0.03)*

The model gave up without attempting any search, despite this being a straightforward factual question. This failure mode appeared in approximately 15% of incorrect answers, suggesting the model sometimes lacks confidence to initiate search.

**2. Overly complex code truncated by token limit**

**Question:** What award did Bad Sisters win in 2022 recognizing excellence in storytelling?

**Turn 1 — Model generates complex filtering logic that exceeds the token limit:**
```python
results = search_pages("Bad Sisters award 2022")[:5]
awards_found = []
for r in results:
    content = read_section(r['page_id'] + ":full")
    if "award" in content.lower() or "excellence" in content.lower():
        if "2022" in content.lower():
            idx = content.lower().find("award")
            if idx != -1:
                award_context = content[max(0, idx - 200):idx + 500]
                if "storytelling" in award_context.lower():
                    awards_found.append(award_context)
# [CODE TRUNCATED - exceeded token limit]
```

**Turn 2 — Model outputs:** `<answer>Answer not found</answer>` *(Reward: 0.09)*

The answer (Peabody Award) was present in the retrieved content, but the model's overly elaborate filtering logic caused token truncation before it could process results. Simpler code would have succeeded.

**3. Hallucination under uncertainty**

When retrieval returned irrelevant content, some trajectories showed the model hallucinating plausible-sounding but incorrect answers rather than acknowledging uncertainty. This suggests a direction for improvement: training the model to express uncertainty when search fails.

---

These patterns reveal that failures stem primarily from behavioral issues (giving up prematurely, overcomplicating code) rather than fundamental capability limitations—suggesting that continued training or curriculum adjustments could improve performance.
