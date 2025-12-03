"""LLM request/response logging for observability."""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class LLMLogger:
    """Logger for capturing LLM request/response pairs."""

    def __init__(self, log_dir: str | Path, prefix: str = "llm"):
        """Initialize the LLM logger.

        Args:
            log_dir: Directory for log files
            prefix: Prefix for log file name
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = self.log_dir / f"{prefix}_calls.jsonl"
        self.call_count = 0

    def _write(self, record: dict) -> None:
        """Write a record to the log file."""
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, indent=2, ensure_ascii=False) + "\n\n")

    def log_model_call(
        self,
        model: str,
        messages: list[dict],
        response: str,
        max_new_tokens: int,
        temperature: float,
        latency_ms: float,
        tokens_generated: int | None = None,
        metadata: dict | None = None,
    ) -> str:
        """Log a model generation call.

        Args:
            model: Model name/path
            messages: Chat messages sent to the model
            response: Generated response
            max_new_tokens: Max tokens setting
            temperature: Temperature setting
            latency_ms: Time taken in milliseconds
            tokens_generated: Number of tokens generated (if known)
            metadata: Additional metadata (question, turn, etc.)

        Returns:
            Call ID for reference
        """
        call_id = f"{self.call_count:06d}"
        self.call_count += 1

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "call_id": call_id,
            "type": "model",
            "model": model,
            "request": {
                "messages": messages,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
            },
            "response": {
                "text": response,
                "tokens_generated": tokens_generated,
            },
            "latency_ms": round(latency_ms, 2),
        }

        if metadata:
            record["metadata"] = metadata

        self._write(record)
        return call_id

    def log_judge_call(
        self,
        model: str,
        messages: list[dict],
        response_content: str,
        max_tokens: int,
        temperature: float,
        latency_ms: float,
        parsed_result: dict | None = None,
        usage: dict | None = None,
        metadata: dict | None = None,
    ) -> str:
        """Log a judge model call.

        Args:
            model: Judge model name
            messages: Chat messages sent
            response_content: Raw response content
            max_tokens: Max tokens setting
            temperature: Temperature setting
            latency_ms: Time taken in milliseconds
            parsed_result: Parsed JSON result (if successful)
            usage: Token usage info (if available)
            metadata: Additional metadata

        Returns:
            Call ID for reference
        """
        call_id = f"{self.call_count:06d}"
        self.call_count += 1

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "call_id": call_id,
            "type": "judge",
            "model": model,
            "request": {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            "response": {
                "content": response_content,
                "parsed": parsed_result,
            },
            "latency_ms": round(latency_ms, 2),
        }

        if usage:
            record["response"]["usage"] = usage
        if metadata:
            record["metadata"] = metadata

        self._write(record)
        return call_id

    def save_summary(self, summary: dict) -> Path:
        """Save run summary metadata.

        Args:
            summary: Summary dict to save

        Returns:
            Path to summary file
        """
        summary_file = self.log_dir / "summary.json"
        summary["total_llm_calls"] = self.call_count
        summary["log_file"] = str(self.log_file)

        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        return summary_file


# Global logger instance (set during training setup)
_logger: LLMLogger | None = None


def get_llm_logger() -> LLMLogger | None:
    """Get the global LLM logger instance."""
    return _logger


def set_llm_logger(logger: LLMLogger) -> None:
    """Set the global LLM logger instance."""
    global _logger
    _logger = logger
