"""Prepare SFT training data from eval results.

Extracts correct trajectories from eval runs and saves them for SFT training.

Usage:
    uv run python scripts/prepare_sft_data.py
    uv run python scripts/prepare_sft_data.py --model-filter gpt-5-mini
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Prepare SFT data from eval results")
    parser.add_argument("--source", default="evals", help="Source evals directory")
    parser.add_argument("--output", default="sft", help="Output directory for trajectories")
    parser.add_argument("--model-filter", default="gpt-5", help="Only include evals from models containing this string")
    parser.add_argument("--all", action="store_true", help="Include all evals, ignore model filter")
    parser.add_argument("--dry", action="store_true", help="Dry run - just print counts, don't save files")
    args = parser.parse_args()

    source_dir = Path(args.source)
    output_dir = Path(args.output)

    if not source_dir.exists():
        print(f"ERROR: Source directory not found: {source_dir}")
        return

    # Find eval directories
    if args.all:
        eval_dirs = [d for d in source_dir.iterdir() if d.is_dir() and d.name.startswith("eval_")]
        filter_desc = "all"
    else:
        model_filter = args.model_filter.lower()
        eval_dirs = [d for d in source_dir.iterdir() if d.is_dir() and model_filter in d.name.lower()]
        filter_desc = f"'{args.model_filter}'"

    print(f"Found {len(eval_dirs)} eval runs matching {filter_desc}")

    # Track unique questions to avoid duplicates
    seen_questions = set()
    trajectories = []
    source_files = []

    for eval_dir in sorted(eval_dirs):
        results_dir = eval_dir / "results"
        if not results_dir.exists():
            continue

        for json_file in sorted(results_dir.glob("q_*.json")):
            with open(json_file) as f:
                data = json.load(f)

            question = data.get("question", "")
            runs = data.get("runs", [])

            if not runs:
                continue

            # Take first run
            run = runs[0]
            if not run.get("correct", False):
                continue

            # Skip if we've seen this question
            if question in seen_questions:
                continue
            seen_questions.add(question)

            # Extract messages
            messages = run.get("messages", [])
            if not messages:
                continue

            trajectories.append({
                "messages": messages,
                "question": question,
                "answer": run.get("answer"),
                "source": str(json_file),
            })
            source_files.append(str(json_file))

    print(f"\nFound {len(trajectories)} correct trajectories")
    print(f"  Source evals: {len(eval_dirs)}")
    print(f"  Filter: {filter_desc}")

    if args.dry:
        print("\n[DRY RUN] No files written")
        return

    # Save individual trajectory files
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, traj in enumerate(trajectories):
        traj_file = output_dir / f"traj_{i:04d}.json"
        with open(traj_file, "w") as f:
            json.dump({"messages": traj["messages"]}, f, indent=2)

    # Save manifest
    manifest = {
        "count": len(trajectories),
        "filter": filter_desc,
        "source_evals": [str(d) for d in eval_dirs],
        "trajectories": [
            {"file": f"traj_{i:04d}.json", "question": t["question"][:80], "answer": t["answer"]}
            for i, t in enumerate(trajectories)
        ],
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nSaved to: {output_dir}")
    print(f"  - {len(trajectories)} trajectory files")
    print(f"  - manifest.json")


if __name__ == "__main__":
    main()
