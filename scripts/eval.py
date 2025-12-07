"""CLI entrypoint for evaluation.

Evaluate a model on the wiki-search task without training.
Config loaded from configs/eval.yaml (no CLI flags).

Both the eval model and judge model are accessed via API endpoints.

Usage:
    uv run python scripts/eval.py
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import httpx
import yaml
from dotenv import load_dotenv

load_dotenv()

# Load config
CONFIG_PATH = Path("configs/eval.yaml")
if not CONFIG_PATH.exists():
    print(f"ERROR: Config file not found: {CONFIG_PATH}")
    sys.exit(1)

with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

from src.environment import SandboxClient


async def check_sandbox() -> tuple[bool, str]:
    """Test sandbox execution with a simple code snippet."""
    try:
        async with SandboxClient() as sandbox:
            result = await sandbox.execute("print(2 + 2)")
            if result.get("error"):
                return False, f"Execution error: {result['error']}"
            if "4" in result.get("output", ""):
                return True, "Sandbox executed test code successfully"
            return False, f"Unexpected output: {result.get('output', '')}"
    except Exception as e:
        return False, str(e)


def check_api_model(base_url: str, model_name: str, label: str, api_key: str | None = None) -> tuple[bool, list[str]]:
    """Check if a model is available at an OpenAI-compatible API endpoint.

    Args:
        base_url: API base URL (e.g., "http://localhost:1234/v1")
        model_name: Expected model name/id
        label: Human-readable label for logging (e.g., "Eval model", "Judge")
        api_key: Optional API key for authentication

    Returns:
        Tuple of (success, list of available model ids)
    """
    try:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        resp = httpx.get(f"{base_url}/models", timeout=30.0, headers=headers)
        if resp.status_code == 200:
            models = resp.json()
            model_ids = [m.get("id", "") for m in models.get("data", [])]
            return True, model_ids
        else:
            print(f"  [FAIL] {label} API returned status {resp.status_code}")
            return False, []
    except httpx.ConnectError:
        print(f"  [FAIL] Cannot connect to {base_url}")
        return False, []
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        return False, []


async def validate_environment() -> bool:
    """Validate that required services are reachable.

    Returns True if all checks pass, False otherwise.
    """
    model_cfg = CONFIG["model"]
    model_url = model_cfg["base_url"]
    model_name = model_cfg["name"]

    judge_cfg = CONFIG["judge"]
    judge_url = judge_cfg["base_url"]
    judge_model = judge_cfg["model"]

    # Get API key for OpenAI endpoints
    openai_api_key = os.getenv("OPENAI_API_KEY")

    print("=" * 60)
    print("VALIDATING ENVIRONMENT")
    print("=" * 60)

    all_ok = True

    # Check eval model API
    print(f"\n[1/3] Checking eval model API: {model_url}")
    model_api_key = openai_api_key if "openai.com" in model_url else None
    ok, model_ids = check_api_model(model_url, model_name, "Eval model", api_key=model_api_key)
    if ok:
        print(f"  [OK] API reachable, available models: {model_ids}")
        # Check if eval model is available
        if model_name in model_ids or any(model_name in m for m in model_ids):
            print(f"  [OK] Eval model '{model_name}' available")
        else:
            print(f"  [FAIL] Eval model '{model_name}' not found in available models")
            print(f"    Available: {model_ids}")
            all_ok = False
    else:
        all_ok = False

    # Check judge API (may be same endpoint)
    print(f"\n[2/3] Checking judge API: {judge_url}")
    if judge_url == model_url:
        # Same endpoint, reuse model list
        print(f"  [OK] Same endpoint as eval model")
        if judge_model in model_ids or any(judge_model in m for m in model_ids):
            print(f"  [OK] Judge model '{judge_model}' available")
        else:
            print(f"  [FAIL] Judge model '{judge_model}' not found in available models")
            print(f"    Available: {model_ids}")
            all_ok = False
    else:
        judge_api_key = openai_api_key if "openai.com" in judge_url else None
        ok, judge_model_ids = check_api_model(judge_url, judge_model, "Judge", api_key=judge_api_key)
        if ok:
            print(f"  [OK] API reachable, available models: {judge_model_ids}")
            if judge_model in judge_model_ids or any(judge_model in m for m in judge_model_ids):
                print(f"  [OK] Judge model '{judge_model}' available")
            else:
                print(f"  [FAIL] Judge model '{judge_model}' not found in available models")
                print(f"    Available: {judge_model_ids}")
                all_ok = False
        else:
            all_ok = False

    # Check sandbox execution
    print(f"\n[3/3] Checking sandbox execution...")
    try:
        sandbox_ok, sandbox_msg = await check_sandbox()
        if sandbox_ok:
            print(f"  [OK] {sandbox_msg}")
        else:
            print(f"  [FAIL] {sandbox_msg}")
            print("    Is the Docker sandbox running? Try: uv run python scripts/run_environment.py")
            all_ok = False
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        all_ok = False

    print("\n" + "=" * 60)
    if all_ok:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED - please fix before evaluating")
    print("=" * 60 + "\n")

    return all_ok


from src.judge import get_local_judge_client
from src.trainer.episode import get_judge_reward
from src.agent import Agent, AgentConfig


def setup_eval_dir(model_name: str) -> Path:
    """Create timestamped eval directory.

    Args:
        model_name: Model being evaluated (used in dir name)

    Returns:
        Path to eval directory
    """
    # Clean model name for directory
    clean_name = model_name.replace("/", "_").replace("\\", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir = Path("evals") / f"eval_{clean_name}_{timestamp}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    return eval_dir


def load_questions_for_eval(subset: str) -> list[dict]:
    """Load evaluation questions from dataset.

    Args:
        subset: Which subset to load - "train", "test", or "all"

    Returns:
        List of {question, answer} dicts
    """
    from src.environment.core import load_questions

    ds = load_questions(subset=subset)
    return [{"question": row["question"], "answer": row["answer"]} for row in ds]




async def eval_subset(
    subset: str,
    questions: list[dict],
    agent: Agent,
    judge_client,
    judge_model: str,
    n_per_q: int,
    results_dir: Path,
    start_from: int = 0,
) -> dict:
    """Evaluate on a single subset.

    Writes one JSON file per question to results_dir.

    Returns:
        Dict with results for this subset
    """
    from src.trainer.episode import JudgeResult

    results = []
    total_correct = 0
    total_runs = 0
    total_turns = 0

    for i, q in enumerate(questions):
        question = q["question"]
        expected = q["answer"]

        print(f"\n  Q{i+1}/{len(questions)}: {question}")
        print(f"     Expected: {expected}")

        q_correct = 0
        q_runs = []

        for run_idx in range(n_per_q):
            # Run episode
            episode = await agent.run(question)
            answer = episode.final_answer

            # Get judge result (skip if no answer)
            if answer is None:
                judge_result = JudgeResult(reward=0.0, correct=False, approach_score=0)
            else:
                # Exact match shortcut - bypass judge if answer matches exactly
                answer_clean = answer.strip().lower()
                expected_clean = expected.strip().lower()
                if answer_clean == expected_clean:
                    judge_result = JudgeResult(reward=1.0, correct=True, approach_score=100)
                else:
                    trajectory = episode.format_for_judge()
                    judge_result = await get_judge_reward(
                        judge_client, judge_model,
                        question, expected, answer,
                        trajectory=trajectory
                    )

            # Record run
            run_result = {
                "run": run_idx + 1,
                "answer": answer or "(no answer)",
                "correct": judge_result.correct,
                "approach_score": judge_result.approach_score,
                "reward": judge_result.reward,
                "num_turns": episode.num_turns,
                "messages": episode.conversation.to_messages(),
            }
            q_runs.append(run_result)

            if judge_result.correct:
                q_correct += 1
                total_correct += 1
            total_runs += 1
            total_turns += episode.num_turns

            # Print run result
            if n_per_q > 1:
                status = "[OK]" if judge_result.correct else "[FAIL]"
                print(f"     Run {run_idx + 1}: {status} | Approach: {judge_result.approach_score}/100")

        # Record question result
        result = {
            "index": i,
            "question": question,
            "expected": expected,
            "runs": q_runs,
            "accuracy": q_correct / n_per_q * 100,
        }
        results.append(result)

        # Save to individual file
        q_file = results_dir / f"q_{i:04d}.json"
        with open(q_file, "w") as f:
            json.dump(result, f, indent=2)

        # Print question summary
        if n_per_q == 1:
            run = q_runs[0]
            display_answer = run["answer"]
            print(f"     Answer: {display_answer}")
            status = "CORRECT" if run["correct"] else "INCORRECT"
            print(f"     Judge: {status} | Approach: {run['approach_score']}/100 | Reward: {run['reward']:.2f}")
        else:
            print(f"     Question accuracy: {q_correct}/{n_per_q} ({result['accuracy']:.0f}%)")

    accuracy = total_correct / total_runs * 100 if total_runs > 0 else 0
    avg_turns = total_turns / total_runs if total_runs > 0 else 0

    return {
        "subset": subset,
        "num_questions": len(questions),
        "total_runs": total_runs,
        "total_correct": total_correct,
        "accuracy": accuracy,
        "avg_turns": avg_turns,
        "results": results,
    }


async def main_async() -> None:
    """Async main function."""
    # Validate environment before starting
    if not await validate_environment():
        sys.exit(1)

    # Extract config values
    model_cfg = CONFIG["model"]
    model_name = model_cfg["name"]
    model_url = model_cfg["base_url"]

    eval_cfg = CONFIG["eval"]
    subset = eval_cfg["subset"]
    n_questions = eval_cfg.get("n_questions", 0)  # 0 = all
    n_samples = eval_cfg["n_samples"]
    start_from = eval_cfg.get("start_from_question", 0)
    max_turns = eval_cfg["max_turns"]
    max_new_tokens = eval_cfg["max_new_tokens"]
    temperature = eval_cfg.get("temperature", 0.7)

    judge_cfg = CONFIG["judge"]
    judge_model = judge_cfg["model"]
    judge_url = judge_cfg["base_url"]

    output_dir = CONFIG["logging"]["output_dir"]

    # Setup eval directory and results subdirectory
    eval_dir = setup_eval_dir(model_name)
    results_dir = eval_dir / "results"
    results_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("SLM-RL EVALUATION (API Mode)")
    print("=" * 60)
    print(f"Config: {CONFIG_PATH}")
    print(f"Model: {model_name}")
    print(f"Model endpoint: {model_url}")
    print(f"Subset: {subset}")
    print(f"N questions: {n_questions if n_questions > 0 else 'all'}")
    print(f"Start from question: {start_from}")
    print(f"Runs per question: {n_samples}")
    print(f"Max turns: {max_turns}")
    print(f"Max new tokens: {max_new_tokens}")
    print(f"Judge: {judge_model}")
    print(f"Judge endpoint: {judge_url}")
    print(f"Output dir: {eval_dir}")
    print("=" * 60)

    # Setup judge
    judge_client = get_local_judge_client(judge_url)

    # Setup sandbox
    sandbox = SandboxClient()

    # Evaluate
    all_results = {}

    async with sandbox:
        # API mode - no local model loading
        agent_config = AgentConfig(
            max_turns=max_turns,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            use_api=True,
            api_url=model_url,
            api_model=model_name,
        )
        agent = Agent(None, None, sandbox, agent_config)  # No model/tokenizer needed

        print(f"\n{'=' * 60}")
        print(f"EVALUATING: {subset.upper()} SET")
        print("=" * 60)

        questions = load_questions_for_eval(subset)
        # Apply start_from first, then limit by n_questions
        if start_from > 0:
            questions = questions[start_from:]
        if n_questions > 0:
            questions = questions[:n_questions]
        print(f"Questions: {len(questions)} (starting from index {start_from})")

        subset_results = await eval_subset(
            subset=subset,
            questions=questions,
            agent=agent,
            judge_client=judge_client,
            judge_model=judge_model,
            n_per_q=n_samples,
            results_dir=results_dir,
            start_from=start_from,
        )
        all_results[subset] = subset_results

        print(f"\n  {subset.upper()} Accuracy: {subset_results['total_correct']}/{subset_results['total_runs']} ({subset_results['accuracy']:.1f}%)")

    # Final Summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    r = all_results[subset]
    print(f"  {subset.upper():6} : {r['total_correct']:3}/{r['total_runs']:3} ({r['accuracy']:5.1f}%) - {r['num_questions']} questions, avg {r['avg_turns']:.1f} turns")

    print("=" * 60)

    # Save summary
    summary_file = eval_dir / "summary.json"
    summary = {
        "config": str(CONFIG_PATH),
        "model": model_name,
        "eval": eval_cfg,
        "judge": judge_cfg,
        "results": all_results,
    }
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_file}")
    print(f"Individual results in: {results_dir}")


def main() -> None:
    """Main entry point."""
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user. Cleaning up...")
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Done.")


if __name__ == "__main__":
    main()
