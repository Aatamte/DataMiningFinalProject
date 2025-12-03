"""Tests for parsing Python code and answers."""

import pytest

from src.trainer.parsing import parse_python_code, parse_answer


class TestParsePythonCode:
    """Tests for parse_python_code function."""

    def test_simple_code(self):
        """Test simple Python code extraction."""
        text = '<python>print("hello")</python>'
        code = parse_python_code(text)
        assert code == 'print("hello")'

    def test_multiline_code(self):
        """Test multiline Python code extraction."""
        text = '''<python>
results = search_pages("iPhone")
print(results)
</python>'''
        code = parse_python_code(text)
        assert 'results = search_pages("iPhone")' in code
        assert 'print(results)' in code

    def test_code_with_surrounding_text(self):
        """Test code embedded in longer text."""
        text = '''I will search for that information.

<python>
results = search_pages("quantum physics")
for r in results:
    print(r["title"])
</python>

This should help us find the answer.'''
        code = parse_python_code(text)
        assert 'search_pages("quantum physics")' in code
        assert 'for r in results:' in code

    def test_no_python_tag(self):
        """Test when no python tag is present."""
        text = 'This is just a regular response without any code.'
        code = parse_python_code(text)
        assert code is None

    def test_empty_string(self):
        """Test empty string input."""
        code = parse_python_code("")
        assert code is None

    def test_malformed_tag(self):
        """Test malformed python tag (missing closing)."""
        text = '<python>print("incomplete")'
        code = parse_python_code(text)
        assert code is None

    def test_code_with_function_calls(self):
        """Test code with tool function calls."""
        text = '''<python>
pages = search_pages("history of computing")
sections = view_sections(pages[0]["page_id"])
content = read_section(sections[0]["section_id"])
print(content[:200])
</python>'''
        code = parse_python_code(text)
        assert 'search_pages' in code
        assert 'view_sections' in code
        assert 'read_section' in code

    def test_whitespace_trimmed(self):
        """Test that whitespace is trimmed from code."""
        text = '<python>   print("test")   </python>'
        code = parse_python_code(text)
        assert code == 'print("test")'


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
