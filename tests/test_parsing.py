"""Tests for parsing tool calls and answers."""

import pytest

from src.trainer.parsing import parse_tool_call, parse_answer


class TestParseToolCall:
    """Tests for parse_tool_call function."""

    def test_xml_style_format(self):
        """Test XML-style tool call: <tool>func(arg)</tool>"""
        text = 'I will search for that. <tool>search_pages(python programming)</tool>'
        func_name, args = parse_tool_call(text)
        assert func_name == "search_pages"
        assert args == "python programming"

    def test_xml_style_with_quotes(self):
        """Test XML-style with quoted argument."""
        text = '<tool>search_pages("machine learning")</tool>'
        func_name, args = parse_tool_call(text)
        assert func_name == "search_pages"
        assert args == '"machine learning"'

    def test_xml_style_view_sections(self):
        """Test view_sections tool call."""
        text = '<tool>view_sections(12345)</tool>'
        func_name, args = parse_tool_call(text)
        assert func_name == "view_sections"
        assert args == "12345"

    def test_xml_style_read_section(self):
        """Test read_section tool call."""
        text = '<tool>read_section(section_abc)</tool>'
        func_name, args = parse_tool_call(text)
        assert func_name == "read_section"
        assert args == "section_abc"

    def test_function_style_format(self):
        """Test function-style: func("arg")"""
        text = 'Let me search. search_pages("quantum physics")'
        func_name, args = parse_tool_call(text)
        assert func_name == "search_pages"
        assert args == "quantum physics"

    def test_simple_colon_format(self):
        """Test simple format: search_pages: query"""
        text = 'search_pages: history of computers'
        func_name, args = parse_tool_call(text)
        assert func_name == "search_pages"
        assert args == "history of computers"

    def test_simple_format_with_quotes(self):
        """Test simple format with quotes."""
        text = 'search_pages: "artificial intelligence"'
        func_name, args = parse_tool_call(text)
        assert func_name == "search_pages"
        assert args == "artificial intelligence"

    def test_case_insensitive_simple_format(self):
        """Test case insensitivity for simple format."""
        text = 'SEARCH_PAGES: neural networks'
        func_name, args = parse_tool_call(text)
        assert func_name == "SEARCH_PAGES"
        assert args == "neural networks"

    def test_no_tool_call(self):
        """Test when no tool call is present."""
        text = 'This is just a regular response without any tool calls.'
        func_name, args = parse_tool_call(text)
        assert func_name is None
        assert args is None

    def test_empty_string(self):
        """Test empty string input."""
        func_name, args = parse_tool_call("")
        assert func_name is None
        assert args is None

    def test_tool_call_with_surrounding_text(self):
        """Test tool call embedded in longer text."""
        text = '''I need to find information about this topic.
        Let me search for it: <tool>search_pages(world war 2)</tool>
        This should help us find the answer.'''
        func_name, args = parse_tool_call(text)
        assert func_name == "search_pages"
        assert args == "world war 2"


class TestParseAnswer:
    """Tests for parse_answer function."""

    def test_simple_answer(self):
        """Test simple answer extraction."""
        text = '<answer>42</answer>'
        result = parse_answer(text)
        assert result == "42"

    def test_answer_with_text(self):
        """Test answer with surrounding text."""
        text = 'Based on my research, <answer>The capital of France is Paris</answer>'
        result = parse_answer(text)
        assert result == "The capital of France is Paris"

    def test_answer_with_whitespace(self):
        """Test answer with whitespace inside tags."""
        text = '<answer>  spaced answer  </answer>'
        result = parse_answer(text)
        assert result == "spaced answer"

    def test_multiword_answer(self):
        """Test multi-word answer."""
        text = '<answer>Albert Einstein developed the theory of relativity</answer>'
        result = parse_answer(text)
        assert result == "Albert Einstein developed the theory of relativity"

    def test_answer_with_numbers(self):
        """Test answer containing numbers."""
        text = '<answer>1969</answer>'
        result = parse_answer(text)
        assert result == "1969"

    def test_no_answer_tag(self):
        """Test when no answer tag is present."""
        text = 'This response has no answer tag.'
        result = parse_answer(text)
        assert result is None

    def test_malformed_answer_tag(self):
        """Test malformed answer tag (missing closing)."""
        text = '<answer>incomplete'
        result = parse_answer(text)
        assert result is None

    def test_answer_in_longer_response(self):
        """Test answer embedded in a longer model response."""
        text = '''After searching through the documents, I found the information.

        The answer to your question is:
        <answer>Mount Everest is 8,849 meters tall</answer>

        This was updated in 2020 after new measurements.'''
        result = parse_answer(text)
        assert result == "Mount Everest is 8,849 meters tall"

    def test_first_answer_extracted(self):
        """Test that first answer is extracted when multiple present."""
        text = '<answer>first</answer> some text <answer>second</answer>'
        result = parse_answer(text)
        assert result == "first"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
