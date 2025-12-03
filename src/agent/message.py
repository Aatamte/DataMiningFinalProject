"""Message and Role definitions for agent conversations."""

from dataclasses import dataclass
from enum import Enum


class Role(Enum):
    """Role of a message in a conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    EXECUTION = "execution"  # Code execution output


@dataclass
class Message:
    """A single message in a conversation.

    Attributes:
        role: The role of the message sender
        content: The message content
    """

    role: Role
    content: str

    def __str__(self) -> str:
        return f"[{self.role.value}]: {self.content[:50]}..."
