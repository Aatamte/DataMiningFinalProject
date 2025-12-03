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

    def trajectory(self) -> list[tuple[str, str]]:
        """Return trajectory in format compatible with REINFORCE training.

        Returns:
            List of (prompt_so_far, assistant_response) tuples for each turn.
        """
        trajectory = []
        prompt_so_far = ""

        for msg in self.conversation.messages:
            if msg.role == Role.SYSTEM:
                prompt_so_far = msg.content
            elif msg.role == Role.USER:
                prompt_so_far += f"\n\n{msg.content}\n\nAssistant:"
            elif msg.role == Role.ASSISTANT:
                trajectory.append((prompt_so_far, msg.content))
                prompt_so_far += msg.content
            elif msg.role == Role.EXECUTION:
                prompt_so_far += f"\n\n{msg.content}\n\nAssistant:"

        return trajectory

    @property
    def success(self) -> bool:
        """Whether the episode produced a final answer."""
        return self.final_answer is not None
