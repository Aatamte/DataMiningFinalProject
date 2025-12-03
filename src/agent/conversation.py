"""Conversation class for managing message history."""

from dataclasses import dataclass, field
from typing import Iterator

from .message import Message, Role


@dataclass
class Conversation:
    """A conversation consisting of a list of messages.

    Attributes:
        messages: List of messages in the conversation
    """

    messages: list[Message] = field(default_factory=list)

    def add(self, role: Role, content: str) -> "Conversation":
        """Add a message to the conversation.

        Args:
            role: The role of the message sender
            content: The message content

        Returns:
            Self for chaining
        """
        self.messages.append(Message(role, content))
        return self

    def add_message(self, message: Message) -> "Conversation":
        """Add an existing message to the conversation.

        Args:
            message: The message to add

        Returns:
            Self for chaining
        """
        self.messages.append(message)
        return self

    def to_messages(self) -> list[dict[str, str]]:
        """Convert to OpenAI-style message list for chat templates.

        Returns:
            List of {"role": str, "content": str} dicts.
            EXECUTION messages are converted to "user" role with [Output] prefix.
        """
        messages = []

        for msg in self.messages:
            if msg.role == Role.SYSTEM:
                messages.append({"role": "system", "content": msg.content})
            elif msg.role == Role.USER:
                messages.append({"role": "user", "content": msg.content})
            elif msg.role == Role.ASSISTANT:
                messages.append({"role": "assistant", "content": msg.content})
            elif msg.role == Role.EXECUTION:
                # Execution results come from the environment, use "user" role
                messages.append({"role": "user", "content": f"[Output]\n{msg.content}"})

        return messages

    def render(self) -> str:
        """Render the conversation to a raw string (legacy method).

        Note: Prefer to_messages() with apply_chat_template for chat models.
        """
        parts = []

        for msg in self.messages:
            if msg.role == Role.SYSTEM:
                parts.append(msg.content)
            elif msg.role == Role.USER:
                parts.append(f"\n\n{msg.content}")
            elif msg.role == Role.ASSISTANT:
                parts.append(msg.content)
            elif msg.role == Role.EXECUTION:
                parts.append(f"\n\n{msg.content}\n\nAssistant:")

        # If last message is USER, add Assistant: prompt
        if self.messages and self.messages[-1].role == Role.USER:
            parts.append("\n\nAssistant:")

        return "".join(parts)

    def last_message(self, role: Role | None = None) -> Message | None:
        """Get the last message, optionally filtered by role.

        Args:
            role: If provided, get last message with this role

        Returns:
            The last matching message, or None if not found
        """
        for msg in reversed(self.messages):
            if role is None or msg.role == role:
                return msg
        return None

    def copy(self) -> "Conversation":
        """Create a shallow copy of the conversation.

        Returns:
            A new Conversation with copied message list
        """
        return Conversation(messages=list(self.messages))

    def __len__(self) -> int:
        return len(self.messages)

    def __iter__(self) -> Iterator[Message]:
        return iter(self.messages)

    def __getitem__(self, index: int) -> Message:
        return self.messages[index]
