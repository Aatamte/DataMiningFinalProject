## Evidence of Meaningfulness

To validate that the model genuinely learns search strategies rather than exploiting superficial patterns, we examine individual trajectories from training runs.

### Success Case Analysis

The following trajectory demonstrates effective search behavior:

**Question:** What award did Zac Efron win in 2021 for hosting a Netflix documentary series?

**Expected Answer:** Daytime Emmy Award

---

**Turn 1 — Model generates:**

```
<python>
# Search for information about Zac Efron hosting a Netflix documentary series in 2021
results = search_pages("Zac Efron Netflix documentary 2021 award")
if results:
    page_id = results[0]['page_id']
    sections = view_sections(page_id)
    for section in sections:
        section_id = section['section_id']
        content = read_section(section_id)
        if "award" in content.lower() and "2021" in content.lower():
            print(content)
            break
else:
    print("No relevant information found.")
</python>
```

**Turn 1 — Environment returns:**

```
<python_output>
# Zac Efron

**Zachary David Alexander Efron** (; born October 18, 1987) is an American actor.
He began acting professionally in the early 2000s and rose to prominence in the
late 2000s for his leading role as Troy Bolton in the High School Musical trilogy
(2006–2008). During this time, he also starred in the musical film Hairspray (2007)
and the comedy film 17 Again (2009). Efron subsequently rose to mainstream prominence
with starring roles in the films New Year's Eve (2011), The Lucky One (2012),
The Paperboy (2012), Neighbors (2014), Dirty Grandpa (2016), Baywatch (2017), and
The Greatest Showman (2017). He played Ted Bundy in Extremely Wicked, Shockingly
Evil and Vile (2019). In 2021, he won a Daytime Emmy Award for the Netflix web
documentary series Down to Earth with Zac Efron (2020–present).
</python_output>
```

**Turn 2 — Model outputs:**

```
<answer>Emmy Award</answer>
```

**Result:** Reward 0.21 (partial credit — answer correct but missing "Daytime" qualifier)

---

The model formulates a targeted query combining the entity, platform, year, and topic. It iterates through sections filtering for relevant keywords ("award" and "2021") before extracting the answer from retrieved text. This represents genuine search reasoning: the model does not guess, it locates evidence first.

### Failure Case Analysis

The following trajectory illustrates compounding failure modes:

**Question:** Who was the first Chairman of the Joint Chiefs of Staff in the United States?

**Expected Answer:** Omar Bradley

---

**Turn 1 — Model generates:**

```
<python>
results = search_pages("first Chairman of the Joint Chiefs of Staff United States")
page_id = results[0]['page_id']
content = read_section(page_id + ":full")
print(content[:1000])
</python>
```

**Turn 1 — Environment returns:**

```
<python_output>
# CEO

*Revision ID: 1146897097 | Timestamp: 2023-03-27T16:27:19Z*

---

#REDIRECT [executive officer](Chief)(Chief executive officer)

[*](Category:Chief operating officers)
</python_output>
```

**Turn 2 — Model generates:**

```
<python>
results = search_pages("first Chairman of the Joint Chiefs of Staff United States")
page_id = results[0]['page_id']
content = read_section(page_id + ":full")
print(content)
</python>
```

**Turn 2 — Environment returns:**

```
<python_output>
# CEO

*Revision ID: 1146897097 | Timestamp: 2023-03-27T16:27:19Z*

---

#REDIRECT [executive officer](Chief)(Chief executive officer)

[*](Category:Chief operating officers)
</python_output>
```

**Turn 3 — Model outputs:**

```
<answer>The first Chairman of the Joint Chiefs of Staff in the United States was General Henry H. "Hap" Arnold.</answer>
```

**Result:** Reward 0.04 (incorrect — correct answer was Omar Bradley)

---

This trajectory reveals two compounding failures:

1. **Retrieval failure:** The query "first Chairman of the Joint Chiefs of Staff United States" returned "CEO" (a redirect page) instead of the relevant Wikipedia article. The embedding-based title search matched "Chief" in "CEO" rather than finding "Chairman of the Joint Chiefs of Staff."

2. **Hallucination under uncertainty:** When retrieval returned irrelevant content twice, the model did not reformulate its query or acknowledge uncertainty. Instead, it hallucinated a plausible-sounding but incorrect answer ("Hap Arnold" was a WWII general but never held this position).

### Failure Taxonomy

Analysis of 102 training episodes reveals distinct failure modes:

| Failure Type | Frequency | Description |
|--------------|-----------|-------------|
| **Wrong answer** | 47% | Model extracts incorrect information or hallucinates when retrieval fails |
| **No answer produced** | 33% | Model exhausts 3 turns without submitting `<answer>` tag |
| **Search retrieval failure** | 15% | Semantic search returns irrelevant pages (e.g., "CEO" for "Chairman") |
| **Partial match** | 5% | Answer semantically correct but judge marks incorrect (e.g., "Emmy Award" vs "Daytime Emmy Award") |

### Limitations

- **Corpus coverage:** Some questions reference information not present in the 2,590-page corpus, making correct answers impossible regardless of search strategy
- **Judge reliability:** The reward formula combines correctness (75%) and approach quality (25%). For the same answer "Emmy Award," rewards ranged from 0.21 to 0.96 across rollouts. This variance stems from inconsistent correctness judgments: when the judge marked "Emmy Award" as correct (matching "Daytime Emmy Award"), reward was ~0.96; when marked incorrect due to the missing qualifier, reward dropped to ~0.21 despite similar approach scores. This highlights sensitivity to judge interpretation of semantic equivalence.
- **Compute constraints:** Training on 10 questions with 6 rollouts each limits generalization conclusions
- **Embedding limitations:** Title-only indexing with all-MiniLM-L6-v2 causes semantic mismatches (e.g., "Chairman" matching "Chief" in "CEO")
