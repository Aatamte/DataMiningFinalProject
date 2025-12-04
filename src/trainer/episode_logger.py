"""Episode logging for detailed per-rollout traces."""

import json
from pathlib import Path

from src.agent import EpisodeResult
from src.agent.message import Role
from src.trainer.parsing import parse_python_code


class EpisodeLogger:
    """Logger for writing detailed episode traces to JSONL files.

    Creates one .jsonl file per question/rollout with the full episode trace.
    """

    def __init__(self, run_dir: Path):
        """Initialize the episode logger.

        Args:
            run_dir: The run directory (e.g., runs/train_20251203_160651/)
        """
        self.episodes_dir = run_dir / "episodes"
        self.episodes_dir.mkdir(parents=True, exist_ok=True)

    def _get_episode_path(self, q_idx: int, rollout_idx: int) -> Path:
        """Get the path for an episode file.

        Args:
            q_idx: Question index (1-based)
            rollout_idx: Rollout index (1-based)

        Returns:
            Path to the episode JSONL file
        """
        return self.episodes_dir / f"q{q_idx:03d}_rollout{rollout_idx:02d}.jsonl"

    def log_episode(
        self,
        q_idx: int,
        rollout_idx: int,
        question: str,
        expected: str,
        episode: EpisodeResult,
        reward: float,
    ) -> Path:
        """Log a complete episode to a JSONL file.

        Args:
            q_idx: Question index (1-based)
            rollout_idx: Rollout index (1-based)
            question: The question text
            expected: The expected answer
            episode: The episode result from the agent
            reward: The reward from the judge

        Returns:
            Path to the written file
        """
        filepath = self._get_episode_path(q_idx, rollout_idx)

        with open(filepath, "w", encoding="utf-8") as f:
            # Write question line
            self._write_line(f, {
                "type": "question",
                "question": question,
                "expected": expected,
            })

            # Parse conversation into turns
            turns = self._extract_turns(episode)
            for turn in turns:
                self._write_line(f, turn)

            # Write result line
            self._write_line(f, {
                "type": "result",
                "final_answer": episode.final_answer,
                "reward": reward,
                "num_turns": episode.num_turns,
            })

        return filepath

    def _write_line(self, f, data: dict) -> None:
        """Write a single JSON line to file."""
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

    def _extract_turns(self, episode: EpisodeResult) -> list[dict]:
        """Extract turns from episode conversation.

        Logs all messages to show exactly what the model sees.

        Args:
            episode: The episode result

        Returns:
            List of turn dicts
        """
        turns = []
        turn_num = 0

        for msg in episode.conversation.messages:
            if msg.role == Role.SYSTEM:
                # Skip system prompt (too long)
                continue
            elif msg.role == Role.USER:
                turns.append({
                    "type": "user",
                    "content": msg.content,
                })
            elif msg.role == Role.EXECUTION:
                turns.append({
                    "type": "execution",
                    "content": msg.content,
                })
            elif msg.role == Role.ASSISTANT:
                turn_num += 1
                turn_data = {
                    "type": "assistant",
                    "turn": turn_num,
                    "content": msg.content,
                }
                # Parse code from the response
                code = parse_python_code(msg.content)
                if code:
                    turn_data["code"] = code
                turns.append(turn_data)

        return turns
