"""Tests for the Agent module: Message, Conversation, and Agent classes."""

import pytest

from src.agent import Message, Role, Conversation, EpisodeResult
from src.prompts import SYSTEM_PROMPT


class TestMessage:
    """Tests for Message class."""

    def test_create_message(self):
        """Test creating a message."""
        msg = Message(Role.USER, "Hello")
        assert msg.role == Role.USER
        assert msg.content == "Hello"

    def test_message_roles(self):
        """Test all message roles."""
        assert Role.SYSTEM.value == "system"
        assert Role.USER.value == "user"
        assert Role.ASSISTANT.value == "assistant"
        assert Role.EXECUTION.value == "execution"

    def test_message_str(self):
        """Test message string representation."""
        msg = Message(Role.ASSISTANT, "This is a long response that gets truncated")
        assert "[assistant]:" in str(msg)


class TestConversation:
    """Tests for Conversation class."""

    def test_empty_conversation(self):
        """Test empty conversation."""
        conv = Conversation()
        assert len(conv) == 0
        assert conv.render() == ""

    def test_add_message(self):
        """Test adding messages."""
        conv = Conversation()
        conv.add(Role.SYSTEM, "System prompt")
        conv.add(Role.USER, "Question: What is 2+2?")

        assert len(conv) == 2
        assert conv[0].role == Role.SYSTEM
        assert conv[1].role == Role.USER

    def test_add_returns_self(self):
        """Test that add() returns self for chaining."""
        conv = Conversation()
        result = conv.add(Role.SYSTEM, "test")
        assert result is conv

    def test_chained_add(self):
        """Test chaining add() calls."""
        conv = Conversation()
        conv.add(Role.SYSTEM, "sys").add(Role.USER, "user").add(Role.ASSISTANT, "asst")
        assert len(conv) == 3

    def test_render_system_user(self):
        """Test rendering system + user messages."""
        conv = Conversation()
        conv.add(Role.SYSTEM, "You are helpful.")
        conv.add(Role.USER, "Question: What is 2+2?")

        rendered = conv.render()
        assert "You are helpful." in rendered
        assert "Question: What is 2+2?" in rendered
        assert rendered.endswith("Assistant:")

    def test_render_with_assistant(self):
        """Test rendering includes assistant response."""
        conv = Conversation()
        conv.add(Role.SYSTEM, "System")
        conv.add(Role.USER, "Question: Test")
        conv.add(Role.ASSISTANT, "<python>print('hello')</python>")

        rendered = conv.render()
        assert "<python>" in rendered
        assert not rendered.endswith("Assistant:")  # No trailing prompt after assistant

    def test_render_with_execution(self):
        """Test rendering includes execution output."""
        conv = Conversation()
        conv.add(Role.SYSTEM, "System")
        conv.add(Role.USER, "Question: Test")
        conv.add(Role.ASSISTANT, "<python>print('hello')</python>")
        conv.add(Role.EXECUTION, "Output:\nhello")

        rendered = conv.render()
        assert "<python>" in rendered
        assert "Output:\nhello" in rendered
        assert rendered.endswith("Assistant:")  # Prompts model after execution

    def test_copy(self):
        """Test copying a conversation."""
        conv = Conversation()
        conv.add(Role.SYSTEM, "System")
        conv.add(Role.USER, "User")

        copy = conv.copy()
        assert len(copy) == 2
        assert copy[0].content == "System"

        # Modifying copy shouldn't affect original
        copy.add(Role.ASSISTANT, "New")
        assert len(conv) == 2
        assert len(copy) == 3

    def test_last_message(self):
        """Test getting last message."""
        conv = Conversation()
        conv.add(Role.SYSTEM, "sys")
        conv.add(Role.USER, "user")
        conv.add(Role.ASSISTANT, "asst")

        assert conv.last_message().content == "asst"
        assert conv.last_message(Role.USER).content == "user"
        assert conv.last_message(Role.SYSTEM).content == "sys"

    def test_iteration(self):
        """Test iterating over conversation."""
        conv = Conversation()
        conv.add(Role.SYSTEM, "1")
        conv.add(Role.USER, "2")
        conv.add(Role.ASSISTANT, "3")

        contents = [m.content for m in conv]
        assert contents == ["1", "2", "3"]


class TestEpisodeResult:
    """Tests for EpisodeResult class."""

    def test_create_episode_result(self):
        """Test creating an episode result."""
        conv = Conversation()
        conv.add(Role.SYSTEM, SYSTEM_PROMPT)
        conv.add(Role.USER, "Question: What is 2+2?")
        conv.add(Role.ASSISTANT, "<answer>4</answer>")

        result = EpisodeResult(
            question="What is 2+2?",
            conversation=conv,
            final_answer="4",
            num_turns=1,
        )

        assert result.question == "What is 2+2?"
        assert result.final_answer == "4"
        assert result.num_turns == 1
        assert result.success is True

    def test_success_property(self):
        """Test success property."""
        conv = Conversation()
        result_success = EpisodeResult("q", conv, "answer", 1)
        result_fail = EpisodeResult("q", conv, None, 3)

        assert result_success.success is True
        assert result_fail.success is False

    def test_trajectory_single_turn(self):
        """Test trajectory for single-turn episode."""
        conv = Conversation()
        conv.add(Role.SYSTEM, "System prompt")
        conv.add(Role.USER, "Question: Test")
        conv.add(Role.ASSISTANT, "<answer>42</answer>")

        result = EpisodeResult("Test", conv, "42", 1)
        trajectory = result.trajectory()

        assert len(trajectory) == 1
        prompt, response = trajectory[0]
        assert "System prompt" in prompt
        assert "Question: Test" in prompt
        assert response == "<answer>42</answer>"

    def test_trajectory_multi_turn(self):
        """Test trajectory for multi-turn episode."""
        conv = Conversation()
        conv.add(Role.SYSTEM, "System")
        conv.add(Role.USER, "Question: Find X")
        conv.add(Role.ASSISTANT, "<python>search('X')</python>")
        conv.add(Role.EXECUTION, "Output:\nFound X = 42")
        conv.add(Role.ASSISTANT, "<answer>42</answer>")

        result = EpisodeResult("Find X", conv, "42", 2)
        trajectory = result.trajectory()

        assert len(trajectory) == 2

        # First turn
        prompt1, response1 = trajectory[0]
        assert "System" in prompt1
        assert "Question: Find X" in prompt1
        assert response1 == "<python>search('X')</python>"

        # Second turn (includes execution output in prompt)
        prompt2, response2 = trajectory[1]
        assert "Output:\nFound X = 42" in prompt2
        assert response2 == "<answer>42</answer>"


class TestConversationFlow:
    """Integration tests showing conversation flow."""

    def test_full_episode_flow(self):
        """Test a complete episode conversation flow."""
        # Initialize with system prompt and question
        conv = Conversation()
        conv.add(Role.SYSTEM, SYSTEM_PROMPT)
        conv.add(Role.USER, "Question: What year was Python created?")

        # Verify initial render
        initial = conv.render()
        assert "search_pages" in initial  # System prompt has tools
        assert "Question: What year was Python created?" in initial
        assert initial.endswith("Assistant:")

        # Model generates code
        conv.add(Role.ASSISTANT, '<python>\nresults = search_pages("Python programming language")\nprint(results)\n</python>')

        # Code executes
        conv.add(Role.EXECUTION, 'Output:\n[{"page_id": "python", "title": "Python"}]')

        # After execution, render should prompt for next response
        after_exec = conv.render()
        assert after_exec.endswith("Assistant:")
        assert "Output:" in after_exec

        # Model provides answer
        conv.add(Role.ASSISTANT, "<answer>1991</answer>")

        # Create episode result
        result = EpisodeResult(
            question="What year was Python created?",
            conversation=conv,
            final_answer="1991",
            num_turns=2,
        )

        # Verify trajectory
        trajectory = result.trajectory()
        assert len(trajectory) == 2
        assert result.success is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
