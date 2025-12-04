# CS 6220 Final Report Blueprint

## Document Purpose

This blueprint defines exactly what each section of the final report must accomplish, with explicit grading criteria distinguishing A+ work from mediocre work. Use this as a checklist before writing and a rubric for self-evaluation after.

---

## Section 1: Team Members and GitHub Link

### Purpose
Identification and access to code.

### Required Content
- Full name(s)
- GitHub Classroom repository URL (must be clickable in PDF)

### Length
2-3 lines. No elaboration needed.

### Grading Notes
This section cannot earn or lose points beyond presence/absence. Ensure the link works.

---

## Section 2: Project Overview

### Purpose
The overview serves as an executive summary that hooks the reader and establishes why this work matters. A hiring manager or PhD advisor skimming your report will likely read only this section. It must stand alone as a compelling pitch for your work.

### Required Content

**Paragraph 1: Problem and Motivation**
- What problem exists in the world that you're addressing?
- Why does this problem matter? Who cares?
- What's wrong with existing solutions?

**Paragraph 2: Technical Approach (High-Level)**
- What is your core insight or approach?
- What makes it different from obvious solutions?
- What are the key technical components?

**Paragraph 3: Results and Significance**
- What did you achieve quantitatively?
- What does this demonstrate?
- What's the broader implication?

### Length
0.5-0.75 pages (approximately 400-600 words)

### Grading Criteria

**A+ (Exceptional)**
The reader immediately understands why this problem is interesting and non-trivial. The technical approach is explained with clarity and insight—not just "what" but "why this approach." Results are presented with appropriate confidence (neither underselling nor overclaiming). The section tells a story with narrative momentum. A reader who stopped here would have a complete mental model of the project and want to read more.

Specific markers:
- Opens with a concrete, relatable framing of the problem (not "In this project we...")
- Identifies a genuine tension or limitation in existing approaches
- The technical insight is crystallized into a memorable formulation
- Quantitative results are contextualized (what does 42.68% accuracy mean? Is that good?)
- Acknowledges scope appropriately (this is a course project, not a production system)

**B (Adequate)**
Describes what was done accurately but lacks compelling framing. The reader understands the project but isn't excited about it. Technical approach is described but not motivated. Results are stated but not interpreted.

Specific markers:
- Begins with "In this project, we..." or "This report presents..."
- Lists technical components without explaining why they fit together
- States accuracy number without context for what it means
- Missing the "so what?" factor

**C (Insufficient)**
A disconnected list of things that were done. No narrative structure. Reader cannot understand why this matters or what was actually accomplished. May contain inaccuracies or vague handwaving.

Specific markers:
- Reads like bullet points converted to sentences
- Confuses what was attempted with what was achieved
- No quantitative results or wildly unsupported claims
- Could describe almost any ML project with word substitutions

### For This Project Specifically

The compelling framing: Traditional search returns a ranked list that humans must manually sift through. What if an AI could search like a human researcher—formulating targeted queries, scanning results, drilling into relevant sections, and synthesizing answers? This reframes "train an LLM to use tools" into something conceptually richer.

The technical insight: We combine three capabilities that are individually well-studied but rarely integrated: (1) code generation, (2) tool use with real execution, (3) reinforcement learning from sparse task rewards. The model learns to write code that searches, not just to predict the next token.

The results in context: 42.68% on factoid questions using only 3 turns of search over 10k pages, with a 4B parameter model running locally. Compare to: random guessing (~0%), closed-book LLM (limited by training data), simple retrieval (single query, no iteration). The 19pp gap between Qwen and Gemma demonstrates this isn't trivial—model capability genuinely matters.

---

## Section 3: Input Data

### Purpose
Establish exactly what data the system works with, how it was prepared, and why it's appropriate for the task. The reader should be able to reproduce your data pipeline and understand any limitations stemming from data choices.

### Required Content

**Dataset Descriptions**
For each dataset:
- Source (URL, citation)
- Size (records, storage)
- Schema (fields, types)
- Content description (what does each record represent?)

**Concrete Examples**
- 3-4 actual data samples showing variety (easy/hard, short/long, different topics)
- Formatted readably (not raw JSON dumps)

**Preprocessing Pipeline**
- How was raw data transformed?
- What tools/libraries were used?
- Any filtering or cleaning steps?
- Indexing approach and parameters (for vector DB)

**Data Statistics**
- Distributions that matter (question length, answer length, etc.)
- Train/test split details and rationale

**Appropriateness and Limitations**
- Why is this data suitable for the problem?
- What are known limitations or biases?
- What would better data look like?

### Length
0.5-0.75 pages

### Grading Criteria

**A+ (Exceptional)**
The reader could reconstruct your dataset from this description. Statistics reveal thoughtful analysis of data properties. Examples are well-chosen to illustrate variety and difficulty. Preprocessing decisions are justified, not just described. Limitations are honestly assessed with understanding of how they affect results.

Specific markers:
- Quantitative statistics (mean, std, distribution shapes) where relevant
- Examples chosen strategically to show range, not randomly
- Embedding model choice justified (why all-MiniLM-L6-v2?)
- Clear explanation of what "rare wiki pages" means and implies
- Honest about corpus coverage limitations

**B (Adequate)**
Datasets are identified and described. Some examples shown. Preprocessing mentioned but not fully explained. Limited statistical analysis. Limitations acknowledged superficially.

Specific markers:
- Lists dataset names and sizes but shallow on content
- Examples seem random rather than strategically chosen
- "We used sentence-transformers" without explaining why or which model
- Train/test split mentioned without rationale

**C (Insufficient)**
Minimal dataset description. No concrete examples or poorly formatted ones. No preprocessing details. No limitations discussed. Reader cannot understand what data actually looks like.

Specific markers:
- "We used a dataset from HuggingFace" with no further detail
- Raw JSON pasted without formatting or explanation
- No mention of how text was indexed or embedded
- Claims data is "high quality" without evidence

### For This Project Specifically

**Questions Dataset Analysis**
The 598 questions deserve characterization: What types of questions? (Who/what/when/where/why distribution?) How complex are they? (Single fact vs. multi-hop reasoning?) What's the answer format? (Named entities, numbers, dates, descriptions?)

**Corpus Analysis**
"Rare wiki pages" is a specific curation choice. What makes them "rare"? Are they obscure topics? Less frequently accessed pages? This affects difficulty—searching over rare pages is harder because the model's pretraining saw them less often.

**The Retrieval Challenge**
Show the semantic gap between questions and page titles. Example: "What controversy involved the Fallout 76 Power Armor edition?" must match to a page probably titled "Fallout 76" not "Fallout 76 canvas bag controversy." This gap is the core retrieval challenge.

**Embedding Choice**
all-MiniLM-L6-v2 is small (384 dims) and fast but not state-of-art. Why this over larger models? (Speed during training, good enough for title matching, runs on same GPU as main model.) What's the tradeoff?

---

## Section 4: Problem

### Purpose
Precisely define the technical problem being solved. A reader with ML background should understand exactly what the inputs, outputs, and success criteria are. They should understand the algorithmic approach at a level where they could reimplement it.

### Required Content

**Formal Problem Definition**
- What is the input to the system?
- What is the expected output?
- What constitutes success/failure?

**Environment and Agent Architecture**
- State representation (what does the agent observe?)
- Action space (what can the agent do?)
- Transition dynamics (how does the environment respond?)
- Episode structure (when does it start/end?)

**Tool Interface Specification**
- Each tool: signature, semantics, return format
- How tools interact with the underlying data
- Error handling and edge cases

**Reinforcement Learning Formulation**
- Policy representation (how is π(a|s) parameterized?)
- Reward function (exact formula with justification for each term)
- Training algorithm (REINFORCE, PPO, etc.)
- Gradient computation (how are policy gradients estimated?)
- Baseline and variance reduction techniques

**Key Design Decisions**
- Why this RL algorithm over alternatives?
- Why this reward structure?
- Why these hyperparameters?
- What alternatives were considered?

**Training Infrastructure**
- How is the model fine-tuned? (LoRA details)
- How is code executed safely? (Sandbox)
- How is judging performed? (LLM-as-judge details)

### Length
0.75-1 page

### Grading Criteria

**A+ (Exceptional)**
The problem formulation is precise enough that a competent ML engineer could reimplement the system. RL formulation uses correct terminology and notation. Design decisions are justified with reasoning, not just stated. The section demonstrates deep understanding of why REINFORCE works, what its limitations are, and how design choices address practical challenges.

Specific markers:
- Mathematical notation for policy, reward, gradient (where appropriate)
- Clear distinction between what the agent observes vs. ground truth
- Reward function terms individually justified
- REINFORCE gradient formula stated and explained intuitively
- LoRA explained with understanding (not just "we used LoRA")
- Alternative approaches mentioned with reasons for rejection
- Acknowledgment of credit assignment challenges in multi-turn setting

**B (Adequate)**
Problem is described but not precisely formulated. RL terminology used but perhaps imprecisely. Design decisions stated without justification. Implementation details present but shallow.

Specific markers:
- "The agent generates code or answers" without formal action space definition
- "We use REINFORCE" without explaining the algorithm
- "Reward is based on correctness" without exact formula
- "We use LoRA for efficiency" without explaining what LoRA does

**C (Insufficient)**
Vague problem description. Incorrect or missing RL formulation. No justification for design choices. Reader cannot understand how the system actually works.

Specific markers:
- Conflates problem definition with solution description
- Misuses RL terminology (e.g., confuses reward with loss)
- No mention of how policy gradients are computed
- "We trained the model" without explaining how

### For This Project Specifically

**The State Space**
State = conversation history as token sequence. But this is processed through a chat template—explain that format. The state grows each turn (system prompt + question + assistant response + execution result + ...). Context window limits become relevant.

**The Action Space**
Actions are text generations, but structured: either `<python>...</python>` (execute code) or `<answer>...</answer>` (terminate with answer). The model must learn this format. What happens if it generates neither? (Turn wasted, continues to next turn or episode ends.)

**The Reward Function**
```
R = 0.5 × correct + 0.5 × (approach_score / 100)
```
Why 0.5/0.5 split? This prevents reward hacking—a model that guesses randomly might get some answers right but will have low approach scores. It also rewards good search strategies that fail to find the answer (partial credit for good process).

Why is approach_score on 0-100 scale divided by 100? To normalize to [0, 0.5] range matching correctness term.

Why fixed baseline of 0.5 instead of learned/moving average? Simplicity, stability, works well when rewards are bounded [0, 1]. Could discuss alternatives (mean reward, value function baseline).

**REINFORCE Gradient**
```
∇J(θ) ≈ E[∇log π(a|s) × (R - b)]
```
Explain intuitively: we increase probability of actions that led to above-baseline rewards, decrease probability of those that led to below-baseline rewards. The magnitude of the update is proportional to how far the reward is from baseline.

**Credit Assignment Challenge**
Multi-turn episodes create credit assignment ambiguity. If the final answer is wrong, was it because of bad search queries in turn 1, bad section selection in turn 2, or bad extraction in turn 3? REINFORCE assigns the same reward to all actions in the trajectory. Discuss this limitation.

**LoRA Deep Dive**
Low-Rank Adaptation freezes pretrained weights W and adds trainable decomposition BA where B ∈ R^(d×r), A ∈ R^(r×k), r << min(d,k). This reduces trainable parameters from d×k to r×(d+k). With r=16, d=k=4096 (typical), we go from 16M to 131K params per adapted matrix—~100x reduction. We adapt attention projections (Q, K, V, O) because these are most important for generation behavior.

**Judge Model Details**
Using LLM-as-judge (DeepSeek-R1-8B) introduces its own biases and failure modes. The judge prompt engineering matters: we ask for JSON output with specific schema, we provide the trajectory for approach scoring. What if the judge is wrong? (It becomes label noise in the reward signal.) Why not human evaluation? (Scale and cost.)

---

## Section 5: Evidence of Success

### Purpose
Provide rigorous quantitative evidence that the approach works. This is where you prove your claims with numbers. A skeptical reader should be convinced by the end that the results are real and meaningful.

### Required Content

**Metric Selection and Justification**
- What metrics are you reporting?
- Why are these the right metrics for this task?
- What metrics did you NOT use and why?

**Main Results**
- Primary quantitative results (accuracy, etc.)
- Results table with all relevant conditions
- Statistical context (confidence intervals, variance, significance)

**Model Comparison**
- Different models tested
- Analysis of why performance differs
- What does the comparison reveal?

**Training Dynamics** (if applicable)
- Loss curves over training
- Reward progression
- Learning rate effects
- Signs of convergence or instability

**Hyperparameter Analysis**
- Which hyperparameters were tuned?
- How sensitive is performance to each?
- Final configuration and rationale

**Ablation Studies** (if applicable)
- What happens if you remove component X?
- Which design decisions matter most?

### Length
0.75-1 page

### Grading Criteria

**A+ (Exceptional)**
Results are presented with scientific rigor. Metrics are justified not just stated. Comparisons are meaningful and analyzed, not just tabulated. Statistical uncertainty is quantified. The reader understands not just what happened but why. Training dynamics (if shown) reveal insight about learning process.

Specific markers:
- Confidence intervals or standard errors on accuracy
- Analysis of WHY Qwen outperforms Gemma (not just that it does)
- Discussion of what 42.68% accuracy means in absolute terms
- Hyperparameter sensitivity analysis shows understanding of tradeoffs
- Honest about what was NOT tuned due to compute constraints

**B (Adequate)**
Results are correctly reported. Some comparison between conditions. Limited statistical analysis. Hyperparameters mentioned but not analyzed. Reader knows what happened but not why.

Specific markers:
- Reports accuracy without confidence intervals
- "Qwen is better than Gemma" without explaining why
- "We used learning rate 1e-5" without explaining how it was chosen
- No discussion of variance across runs

**C (Insufficient)**
Results incomplete or potentially incorrect. No meaningful comparison. No statistical context. Hyperparameters seem arbitrary. Results could be cherry-picked.

Specific markers:
- Single number with no context
- Comparing apples to oranges (different test sets, etc.)
- "We tried different values and this worked best" with no details
- No acknowledgment of uncertainty or variance

### For This Project Specifically

**Contextualizing 42.68% Accuracy**
This number means nothing without context. Provide baselines:
- Random guessing: ~0% (open-ended questions)
- Majority class: N/A (all answers unique)
- Closed-book LLM: Unknown, but limited by training data—these are "rare" wiki pages
- Single-query retrieval: What if you just searched once and answered from top result?

Also contextualize by task difficulty: these are factoid questions over 10k pages with 3 turns maximum. A human with the same interface would likely score 90%+. So 42.68% is a meaningful improvement over nothing but far from solved.

**The Qwen vs Gemma Gap**
19.46 percentage points is large. Why? Hypotheses:
1. Qwen has better instruction following (more likely to use correct `<python>` format)
2. Qwen has better code generation (writes working Python more often)
3. Qwen was trained on more tool-use data
4. Random variance (but 478 samples should be enough to detect 19pp gap)

Can you provide evidence for any of these? (e.g., what fraction of Gemma responses had syntax errors?)

**Per-Question Analysis**
478 questions is enough for aggregate statistics but allows for per-question analysis too. What question types are easiest/hardest? Is there a correlation between question length and difficulty? Answer type and difficulty?

**Training Dynamics (if available)**
If you ran RL training:
- Did reward increase over episodes?
- Did loss decrease?
- How many episodes until convergence?
- Any signs of reward hacking or mode collapse?

If only baseline evaluation:
- Be explicit that these are zero-shot results
- Discuss what you'd expect from training based on the RL setup

**Hyperparameter Choices**
For each major hyperparameter, explain:
- What value was used
- What range was considered
- How it was selected (prior work, tuning, compute constraints)
- Expected sensitivity

Example: "We used learning rate 1e-5 following standard practice for LoRA fine-tuning of instruction models (Hu et al., 2022). We did not tune this due to compute constraints, but preliminary experiments with 1e-4 showed training instability."

---

## Section 6: Evidence of Meaningfulness

### Purpose
Convince a skeptical reader that the quantitative results reflect genuine capability, not artifacts, gaming, or luck. This requires qualitative deep-dives and honest failure analysis.

### Required Content

**Detailed Success Case Analysis**
- 2-3 complete episode trajectories for successful cases
- Step-by-step analysis of agent behavior
- What made these succeed?
- Evidence of genuine search strategy vs. lucky guessing

**Systematic Failure Taxonomy**
- Categorize ALL failure modes (not just convenient ones)
- Estimate frequency of each failure type
- Analyze root causes
- Discuss which are addressable vs. fundamental

**Limitations Assessment**
- Data limitations
- Method limitations
- Evaluation limitations
- What claims can and cannot be supported?

**Counterfactual Analysis**
- Could these results be achieved by simpler methods?
- What would random/trivial baselines score?
- Is the model actually searching or memorizing?

### Length
0.75-1 page

### Grading Criteria

**A+ (Exceptional)**
Qualitative analysis is as rigorous as quantitative. Success cases are analyzed critically (not just showcased). Failure taxonomy is comprehensive and quantified. Limitations are honestly assessed with clear implications for interpreting results. Reader trusts the author's judgment because they've seen honest engagement with weaknesses.

Specific markers:
- Success cases include analysis of what could have gone wrong
- Failure categories have estimated frequencies (not just examples)
- Each limitation has concrete implication for what we can/cannot conclude
- Author anticipates skeptical questions and addresses them
- Acknowledges uncertainty where appropriate

**B (Adequate)**
Some qualitative examples shown. Failure modes mentioned but not systematically analyzed. Limitations listed but not deeply explored. Reader sees some evidence but questions remain.

Specific markers:
- Cherry-picked success cases with no failure analysis
- "Sometimes the model fails to find the right page" without quantification
- "A limitation is the small corpus" without explaining implications
- Defensive tone about limitations rather than honest engagement

**C (Insufficient)**
Minimal qualitative analysis. Failures ignored or blamed on external factors. Limitations dismissed or not acknowledged. Reader suspects results may not be meaningful.

Specific markers:
- Only success cases shown
- "Failures are due to bad data" without analysis
- No limitations section or perfunctory one
- Overclaims about what results demonstrate

### For This Project Specifically

**Deep Dive Success Case: Multi-Step Search**
Show the Fallout 76 example in full:
```
Question: What controversy involved the Fallout 76 Power Armor edition's physical item?
Expected: Canvas bag replaced with nylon

Turn 1:
Agent generates: search_pages("Fallout 76 Power Armor edition controversy")
Execution returns: [{"page_id": "fallout_76", "title": "Fallout 76"}, ...]

Turn 2:
Agent generates: view_sections("fallout_76") -> finds "Controversies" section
Agent generates: read_section("fallout_76:controversies")
Execution returns: [Full text about canvas bag controversy]

Turn 3:
Agent generates: <answer>Bethesda advertised a canvas bag but included nylon...</answer>
```

Analyze: The agent formulated a targeted query (not just "Fallout 76"), identified the relevant section by name, and extracted the correct answer. This demonstrates genuine search capability.

But also analyze what could have failed: What if "controversies" section didn't exist? What if the search returned a different Fallout game? The success depended on corpus structure matching agent expectations.

**Failure Taxonomy with Frequencies**
Manually categorize ~50 failures to estimate distribution:

1. **Partial match (judge error)**: ~15%
   - Model answer semantically correct but fails string match
   - Example: "Five" vs "Five years old"
   - Root cause: Judge prompt too strict
   - Addressable: Yes, improve judge

2. **Information not in corpus**: ~25%
   - The answer literally doesn't exist in the 10k pages
   - Example: Question about person whose page isn't included
   - Root cause: Corpus coverage
   - Addressable: Larger corpus

3. **Search failure**: ~30%
   - Wrong pages retrieved, good content never found
   - Example: Query too generic, relevant page buried
   - Root cause: Embedding quality, query formulation
   - Addressable: Better embeddings, RL training

4. **Extraction failure (hallucination)**: ~20%
   - Right content found, wrong answer extracted
   - Example: Multiple facts in passage, picked wrong one
   - Root cause: Reading comprehension, attention
   - Addressable: Maybe with RL, maybe fundamental

5. **Format failure**: ~10%
   - Code syntax error, no answer tag, malformed output
   - Example: Missing closing tag, Python exception
   - Root cause: Instruction following
   - Addressable: Better prompting, RL training

**Limitations Deep Dive**

*Corpus Limitation*
10k "rare" pages is not Wikipedia. The model cannot answer questions about common knowledge (no pages for "United States" or "World War II"). Results don't predict performance on general Wikipedia. This is intentional (forces search over retrieval from memory) but limits generalization claims.

*Question Limitation*
Trivia questions are single-hop factoid retrieval. "What year did X happen?" requires finding one fact. Real research questions require synthesis: "How did X influence Y?" The model hasn't been tested on compositional reasoning.

*Judge Limitation*
LLM-as-judge has known biases: prefers verbose answers, may have knowledge that affects "correctness" judgment independently of the provided ground truth, JSON parsing can fail silently. We trust ~95% of judgments are correct based on spot-checking but haven't validated systematically.

*Compute Limitation*
Without full RL training, we're evaluating zero-shot capability, not learned search strategies. The REINFORCE setup is implemented but results are baseline only. This limits claims about "learning to search."

---

## Section 7: Conclusions

### Purpose
Leave the reader with clear understanding of what was achieved, what was learned, and where this leads. This is the last impression—make it count.

### Required Content

**Achievement Summary**
- What was accomplished (factual, not inflated)
- What technical contributions were made
- What was demonstrated/proven

**Key Insights**
- What was learned that wasn't obvious beforehand?
- What surprised you?
- What would you do differently?

**Future Work**
- Specific, actionable next steps (not generic)
- What would move this from course project to research contribution?
- What resources would be needed?

**Broader Implications** (if applicable)
- What does this suggest for the field?
- What related problems could benefit from this approach?

### Length
0.25-0.5 pages

### Grading Criteria

**A+ (Exceptional)**
Conclusions synthesize rather than summarize. Insights are genuine learnings, not restatements of results. Future work is specific and demonstrates understanding of what would actually advance the work. Reader finishes with clear mental model and curiosity about next steps.

Specific markers:
- Distinguishes what was achieved from what was attempted
- Insights reflect genuine intellectual engagement
- Future work items are specific enough to be actionable
- Connects to broader trends or applications thoughtfully
- Honest about what this is (course project) and what it could become

**B (Adequate)**
Summarizes results adequately. Future work mentioned but generic. Reader finishes with accurate understanding but no particular excitement.

Specific markers:
- Restates results rather than synthesizing
- "We could train longer" type future work
- No genuine insights beyond what was already stated
- Perfunctory treatment

**C (Insufficient)**
Conclusion doesn't match rest of report. Overclaims or misrepresents achievements. Future work is hand-wavy or unrealistic. Reader finishes confused about what was actually accomplished.

Specific markers:
- Claims not supported by evidence section
- "In the future we could use GPT-4" (not a contribution)
- Ignores limitations discussed earlier
- Generic conclusions that could apply to any project

### For This Project Specifically

**Genuine Achievements**
1. Built complete local pipeline: data ingestion, embedding, sandbox execution, RL training loop, evaluation. Zero API costs.
2. Established baseline: 42.68% accuracy provides reference point for future work
3. Demonstrated architecture matters: Qwen vs Gemma gap is a real finding
4. Created reusable infrastructure: Future students/researchers could build on this

**Genuine Insights**
1. Tool-use capability varies dramatically between similarly-sized models—this isn't just about scale
2. Multi-turn code generation for search is viable but challenging—the model does learn to formulate queries
3. The judge is a bottleneck—reward signal quality limits training
4. Partial credit for approach matters—pure binary reward would be too sparse

**Specific Future Work**
1. **Complete RL training**: Run full training for 3+ epochs with 100+ questions. Measure improvement over baseline. Estimate: 8-12 GPU hours.

2. **Improve judge**: Switch from exact match to semantic similarity (embed both answers, compare cosine similarity > threshold). This would fix ~15% of false negatives.

3. **Scale corpus**: Index full Simple Wikipedia (~200k pages) to test generalization. Requires more storage but same code.

4. **Harder questions**: Create multi-hop question set requiring 2+ facts. Test compositional reasoning limits.

5. **Algorithm comparison**: Implement PPO or GRPO for comparison. Hypothesis: PPO with value baseline might help credit assignment.

---

## Formatting and Style Requirements

### Tone
- Academic but accessible
- Confident but not overclaiming
- First person plural ("we") is fine
- Avoid: "very", "really", "extremely", hedging phrases like "we tried to"

### Structure
- Flowing prose paragraphs, not bullet points (except in tables)
- Each paragraph has a topic sentence and supporting details
- Transitions between paragraphs and sections
- Technical terms defined on first use

### Figures and Tables
- Every figure/table referenced in text
- Captions are complete (reader understands without text)
- Axis labels, legends, units included
- Placed near first reference

### Length
- Maximum 5 pages (3 + 2×1 for solo project)
- 11-point font per template
- Don't pad; don't cram

### Common Mistakes to Avoid
1. Starting with "In this project, we..." (boring)
2. Listing technologies without explaining why
3. Showing results without interpreting them
4. Ignoring failures or limitations
5. Generic future work ("train longer", "more data")
6. Mismatch between claims and evidence
7. Bullet points in prose sections
8. Missing statistical context for numbers
9. Figures without captions or references
10. Conclusions that just summarize

---

## Self-Evaluation Checklist

Before submitting, verify:

- [ ] Overview hooks reader and stands alone
- [ ] Data section includes concrete examples
- [ ] Problem section is precise enough to reimplement
- [ ] Results include statistical context
- [ ] Failures analyzed as thoroughly as successes
- [ ] Limitations honestly assessed
- [ ] Conclusions synthesize (not summarize)
- [ ] All figures/tables referenced in text
- [ ] Within page limit
- [ ] No bullet points in prose
- [ ] GitHub link works
- [ ] PDF is properly formatted
