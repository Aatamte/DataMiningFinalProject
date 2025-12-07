Teaching Small Language Models to Search with Python


TEAM MEMBERS AND GITHUB LINK

Aaron Tamte

GitHub Repository: https://github.com/2025-F-CS6220/project-slm-rl-search


PROJECT OVERVIEW

Large language models excel at recalling facts from their training data, but struggle with questions requiring information they haven't memorized. Retrieval-augmented generation (RAG) addresses this by fetching relevant documents, yet typically relies on a single query and cannot adapt when initial results prove insufficient. Recent tool-use frameworks give models more flexibility, but constrain them to rigid interfaces—one tool call at a time, with fixed input/output formats.

This work takes the next step: instead of calling tools, the model writes Python code that executes in a sandbox. Search functions are primitives the model can compose with loops, conditionals, and string operations—transforming the problem from "which tool to call" into "what program to write." The model learns entirely through reinforcement learning, discovering effective search strategies through trial and error with no human demonstrations.

Our trained agent (Qwen3-4B, 4 billion parameters) achieves TODO% accuracy on held-out questions, compared to 11.5% for the untrained baseline—a TODO percentage point improvement from reinforcement learning alone. The agent learns non-trivial behaviors: iterating through search results with programmatic filters, navigating from broad topic pages to specific sections, and adapting its strategy based on intermediate outputs. These results demonstrate that small language models can learn to program effective information-seeking behavior without human demonstrations.


INPUT DATA

This work uses two publicly available datasets from HuggingFace: a Wikipedia corpus for retrieval and a question-answer dataset for training and evaluation. The corpus provides the knowledge base the model must search, while the question-answer pairs define the task and supply ground truth for reward computation.

Search Corpus: The corpus contains 2,590 Wikipedia pages from the rare-wiki-pages dataset (https://huggingface.co/datasets/willcb/rare-wiki-pages). These pages cover niche topics less likely to be memorized during language model pretraining, ensuring the task tests search capability rather than recall. Pages are formatted as markdown with hierarchical section headers, providing structure the model can navigate programmatically.

Example corpus entry:
{
    "id": "caroline_polachek",
    "title": "Caroline Polachek",
    "content": "# Caroline Polachek\n\n**Caroline Polachek** (born 1985) is an American singer...\n\n## Early life\n...\n\n## Career\n\n### Chairlift (2008–2017)\n..."
}

Question-Answer Pairs: The evaluation set comprises 478 factoid question-answer pairs from the wiki-trivia-questions-v4 dataset (https://huggingface.co/datasets/willcb/wiki-trivia-questions-v4). Questions span diverse domains including entertainment, science, geography, and current events, with answers typically consisting of short noun phrases, named entities, or numerical values.

Example question-answer pair:
{
    "question": "Which university did Miles Teller attend for his Bachelor of Fine Arts degree?",
    "answer": "New York University",
    "filename": "Miles Teller.md"
}

Data Split: We partition the dataset deterministically by index, allocating the first 80% (382 questions) for training and reserving the remaining 20% (96 questions) for evaluation. This fixed split ensures reproducibility and prevents evaluation on training examples.


PROBLEM

Given a factoid question and a Wikipedia corpus, generate the correct answer by writing and executing Python code.

The model has no direct access to article content. It must write code that calls search primitives—search_pages(query), view_sections(page_id), read_section(section_id)—to locate and extract the answer. Each episode allows up to 2 turns of code execution before requiring a final answer.

Traditional tool-use frameworks constrain models to one action per turn with fixed input/output schemas. Code generation enables composition—the model can loop through results, filter with conditionals, and chain operations in a single generation. This shifts the problem from "which tool to call" to "what program to write." Additionally, unlike large models that can rely on memorized facts, a 4B parameter model lacks sufficient capacity to store the corpus. It must genuinely learn to search rather than recall, making this a test of learned information-seeking behavior rather than knowledge retrieval.


EVIDENCE OF SUCCESS

Results

Our trained agent achieves TODO% accuracy on held-out questions, compared to 11.5% for the untrained Qwen3-4B baseline—a TODO percentage point improvement from reinforcement learning. We also compare against the smaller Qwen3-1.7B model to demonstrate scaling effects.

Model                   Parameters    Accuracy    Avg Turns
Qwen3-1.7B (untrained)  1.7B          5.2%        1.96
Qwen3-4B (untrained)    4B            11.5%       1.99
Qwen3-4B (trained)      4B            TODO%       TODO

We use accuracy as our primary metric because each question has a single correct answer, making binary correctness the most direct measure of task success.

Method

We optimize the policy using Group Relative Policy Optimization (GRPO), which computes advantages relative to the group mean and standard deviation across rollouts for each question. This provides a more stable learning signal than fixed-baseline REINFORCE when rewards are sparse. The reward function combines correctness (75%) and approach quality (25%), with a judge model (Qwen3-8B-AWQ) evaluating each trajectory.

Configuration

Final configuration: 10 epochs, 300 questions, 20 rollouts per question, LoRA (r=8, alpha=16, dropout=0.05), temperature sampling 0.7-1.15 for trajectory diversity. Full hyperparameters are listed in the Appendix.


EVIDENCE OF MEANINGFULNESS

To verify that our results reflect genuine search capability rather than artifacts, we manually inspected individual trajectories from the evaluation set.

Successful Trajectories

Example 1: Multi-step search with filtering

Question: Which 2023 film features Glenn Howerton in the role of Jim Balsillie?

Turn 1 — Model generates:

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

Turn 1 — Environment returns:

    Found potential match: Glenn Howerton
    # Glenn Howerton
    ...
    He received acclaim for his portrayal of Jim Balsillie in the film
    *BlackBerry* (2023).
    ...

Turn 2 — Model outputs: BlackBerry (Reward: 0.99)

This trajectory demonstrates the model: (1) formulating a targeted query combining actor and year, (2) iterating through results with a programmatic filter for the character name, and (3) extracting the correct film title from the retrieved content.


Example 2: Targeted information extraction

Question: Who performs the cover of Leonard Cohen's 'Who by Fire' used as the opening theme for Bad Sisters?

Turn 1 — Model generates:

    # Search for the TV show "Bad Sisters" to find information about its opening theme
    for result in search_pages("Bad Sisters opening theme")[:5]:
        content = read_section(result['page_id'] + ":full")
        if "Leonard Cohen" in content or "Who by Fire" in content:
            print(f"Found relevant content in {result['title']}:")
            print(content)
            break

Turn 1 — Environment returns:

    Found relevant content in Bad Sisters:
    # Bad Sisters
    ...
    | opentheme = "Who by Fire" by Leonard Cohen, performed by PJ Harvey
    ...

Turn 2 — Model outputs: PJ Harvey (Reward: 0.98)

The model learned to search for the show directly and filter for content mentioning either the song title or original artist—a strategy that efficiently locates the performer information.


Failure Modes

We also examined failure cases to understand limitations:

1. Immediate surrender without search

Question: Which American grocery chain does Aldi Nord own?

Turn 1 — Model outputs: Answer not found (Reward: 0.03)

The model gave up without attempting any search, despite this being a straightforward factual question. This failure mode appeared in approximately 15% of incorrect answers, suggesting the model sometimes lacks confidence to initiate search.

2. Overly complex code truncated by token limit

Question: What award did Bad Sisters win in 2022 recognizing excellence in storytelling?

Turn 1 — Model generates complex filtering logic that exceeds the token limit:

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

Turn 2 — Model outputs: Answer not found (Reward: 0.09)

The answer (Peabody Award) was present in the retrieved content, but the model's overly elaborate filtering logic caused token truncation before it could process results. Simpler code would have succeeded.

3. Hallucination under uncertainty

When retrieval returned irrelevant content, some trajectories showed the model hallucinating plausible-sounding but incorrect answers rather than acknowledging uncertainty. This suggests a direction for improvement: training the model to express uncertainty when search fails.

These patterns reveal that failures stem primarily from behavioral issues (giving up prematurely, overcomplicating code) rather than fundamental capability limitations—suggesting that continued training or curriculum adjustments could improve performance.


CONCLUSIONS

This project demonstrates that a 4-billion parameter model can learn effective information-seeking behavior through reinforcement learning alone, without human demonstrations. By framing search as code generation rather than tool selection, the model learns to compose multi-step strategies—iterating through results, filtering with conditionals, and adapting based on intermediate outputs—achieving TODO% accuracy compared to 11.5% for the untrained baseline. Our failure analysis reveals that most errors stem from behavioral issues (premature surrender, overly complex code) rather than capability limits, suggesting that continued training or curriculum learning could yield further gains. A natural next step is training the model to recognize retrieval failure and reformulate queries, rather than hallucinating answers when initial searches return irrelevant content.


APPENDIX

Hyperparameters

Parameter                Value
Base model               Qwen3-4B-Instruct-2507
LoRA rank                8
LoRA alpha               16
LoRA dropout             0.05
LoRA targets             q_proj, k_proj, v_proj, o_proj
Gradient checkpointing   Enabled
Epochs                   10
Training questions       300
Rollouts per question    20
Max turns                2
Max new tokens           256
Context window           3000
Learning rate            1e-4
Max gradient norm        1.0
Temperature range        0.7 - 1.15
Discount (γ)             0.99
Correctness weight       0.75
Approach weight          0.25
RL algorithm             GRPO
Judge model              Qwen3-8B-AWQ

Training Algorithm (Pseudocode)

TRAIN(questions, model, judge):
    FOR each epoch:
        FOR each (question, answer) in questions:

            // Collect K rollouts with temperature diversity
            rollouts ← []
            FOR k = 1 to K:
                temp ← uniform(temp_min, temp_max)
                episode ← RUN_EPISODE(model, question, temp)
                rollouts.append(episode)

            // Judge all rollouts
            rewards ← [JUDGE(ep, answer) for ep in rollouts]

            // Compute advantages (GRPO with fallbacks)
            advantages ← COMPUTE_ADVANTAGES(rewards)
            IF advantages = skip: CONTINUE

            // Policy gradient update
            UPDATE_POLICY(model, rollouts, advantages, γ)


COMPUTE_ADVANTAGES(rewards):
    μ ← mean(rewards)
    σ ← std(rewards)

    IF σ < 0.01:
        IF μ < 0.1:
            RETURN skip                         // all failures
        ELSE:
            RETURN [r - 0.5 for r in rewards]   // REINFORCE fallback
    ELSE:
        RETURN [(r - μ) / σ for r in rewards]   // GRPO


RUN_EPISODE(model, question, temperature):
    history ← [system_prompt, question]

    FOR turn = 1 to T:
        response ← model.generate(history, temperature)

        IF response contains <answer>:
            RETURN episode with extracted answer

        IF response contains <python>:
            output ← sandbox.execute(extract_code(response))
            history.append(response, output)

    RETURN episode with no answer


JUDGE(episode, expected):
    correct ← LLM_judge.is_correct(episode.answer, expected)
    approach ← LLM_judge.score_approach(episode.trajectory)
    RETURN 0.75 × correct + 0.25 × (approach / 100)
