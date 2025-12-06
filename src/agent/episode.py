"""Episode result data structure."""

from dataclasses import dataclass

from .message import Message, Role
from .conversation import Conversation


@dataclass
class EpisodeResult:
    """Result of running an agent episode.

    Attributes:
        question: The original question
        conversation: The full conversation history
        final_answer: The extracted final answer (if any)
        num_turns: Number of turns taken
    """

    question: str
    conversation: Conversation
    final_answer: str | None
    num_turns: int

    def trajectory(self, tokenizer=None) -> list[tuple[str, str]]:
        """Return trajectory in format compatible with REINFORCE training.

        Args:
            tokenizer: If provided, uses apply_chat_template for proper formatting.
                       This ensures loss computation matches generation format.

        Returns:
            List of (prompt_so_far, assistant_response) tuples for each turn.
        """
        trajectory = []
        messages_so_far = []

        for msg in self.conversation.messages:
            if msg.role == Role.SYSTEM:
                messages_so_far.append({"role": "system", "content": msg.content})
            elif msg.role == Role.USER:
                messages_so_far.append({"role": "user", "content": msg.content})
            elif msg.role == Role.ASSISTANT:
                # Build prompt using chat template (matches generation format)
                if tokenizer is not None:
                    prompt = tokenizer.apply_chat_template(
                        messages_so_far,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                else:
                    # Fallback to raw text (legacy behavior)
                    prompt = self._raw_prompt(messages_so_far)

                trajectory.append((prompt, msg.content))
                messages_so_far.append({"role": "assistant", "content": msg.content})
            elif msg.role == Role.EXECUTION:
                # Execution results are sent as user messages
                messages_so_far.append({"role": "user", "content": msg.content})

        return trajectory

    def _raw_prompt(self, messages: list[dict]) -> str:
        """Fallback raw prompt building (legacy, no chat template)."""
        parts = []
        for msg in messages:
            if msg["role"] == "system":
                parts.append(msg["content"])
            else:
                parts.append(f"\n\n{msg['content']}")
        parts.append("\n\nAssistant:")
        return "".join(parts)

    @property
    def success(self) -> bool:
        """Whether the episode produced a final answer."""
        return self.final_answer is not None

    def format_for_judge(self) -> str:
        """Format the trajectory for judge evaluation.

        Returns a human-readable string showing the agent's actions and outputs.
        Excludes system prompt to focus on the agent's approach.
        """
        parts = []
        turn = 0

        for msg in self.conversation.messages:
            if msg.role == Role.SYSTEM:
                # Skip system prompt - judge doesn't need it
                continue
            elif msg.role == Role.USER:
                if "Question:" in msg.content:
                    # This is the initial question, skip (already shown separately)
                    continue
                # This is likely the user prompting for more (shouldn't happen in our flow)
                parts.append(f"[USER]: {msg.content}")
            elif msg.role == Role.ASSISTANT:
                turn += 1
                parts.append(f"[TURN {turn} - AGENT]:\n{msg.content}")
            elif msg.role == Role.EXECUTION:
                # Check if this is a system reminder (not actual code output)
                if msg.content.startswith("[FINAL TURN]"):
                    parts.append(f"[SYSTEM REMINDER]:\n{msg.content}")
                else:
                    parts.append(f"[EXECUTION RESULT]:\n{msg.content}")

        return "\n\n".join(parts)
