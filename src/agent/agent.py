"""Agent class for running episodes."""

import os
import time
from dataclasses import dataclass

import httpx

from src.prompts import SYSTEM_PROMPT
from src.environment import SandboxClient
from src.trainer.parsing import parse_python_code, parse_answer
from src.llm_logger import get_llm_logger

from .message import Message, Role
from .conversation import Conversation
from .episode import EpisodeResult


@dataclass
class AgentConfig:
    """Configuration for the Agent.

    Attributes:
        max_turns: Maximum number of code execution turns
        max_new_tokens: Maximum tokens to generate per turn
        temperature: Sampling temperature
        max_context: Maximum context length in tokens
        enable_thinking: Enable thinking mode for reasoning models (Qwen3, etc.)
        use_api: Use OpenAI-compatible API instead of local model
        api_url: Base URL for API (when use_api=True)
        api_model: Model name for API (when use_api=True)
        debug: Print full agent traces (prompts, generations, execution)
    """

    max_turns: int = 3
    max_new_tokens: int = 1024
    temperature: float = 0.7
    max_context: int = 1500
    enable_thinking: bool = os.environ.get("TRAIN_ENABLE_THINKING", "false").lower() == "true"
    use_api: bool = os.environ.get("EVAL_USE_API", "false").lower() == "true"
    api_url: str = os.environ.get("EVAL_API_URL", "http://localhost:1234/v1")
    api_model: str = os.environ.get("EVAL_MODEL", "")
    debug: bool = os.environ.get("AGENT_DEBUG", "false").lower() == "true"


class Agent:
    """Agent that runs episodes by generating code and executing it.

    The agent manages the conversation, generates responses, executes code
    in a sandbox, and returns structured episode results.

    Supports two modes:
    - Local model: Pass model/tokenizer, uses torch for inference
    - API mode: Set use_api=True in config, uses OpenAI-compatible API
    """

    def __init__(
        self,
        model,
        tokenizer,
        sandbox: SandboxClient,
        config: AgentConfig | None = None,
    ):
        """Initialize the agent.

        Args:
            model: The language model (can be None if using API mode)
            tokenizer: Tokenizer for the model (can be None if using API mode)
            sandbox: Sandbox client for code execution
            config: Agent configuration (uses defaults if None)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.sandbox = sandbox
        self.config = config or AgentConfig()

        if self.config.use_api:
            self.device = None
            self._http_client = httpx.AsyncClient(timeout=120.0)
        else:
            import torch
            self.device = next(model.parameters()).device
            self._http_client = None

    def _create_conversation(self, question: str) -> Conversation:
        """Create initial conversation with system prompt and question.

        Args:
            question: The question to answer

        Returns:
            Initialized conversation
        """
        conv = Conversation()
        conv.add(Role.SYSTEM, SYSTEM_PROMPT)
        conv.add(Role.USER, f"Question: {question}")
        return conv

    async def step(self, conversation: Conversation, temperature: float | None = None) -> Message:
        """Generate a single response from the model.

        Args:
            conversation: Current conversation state
            temperature: Override temperature (uses config default if None)

        Returns:
            The generated assistant message
        """
        temp = temperature if temperature is not None else self.config.temperature
        messages = conversation.to_messages()

        if self.config.use_api:
            response_text, latency_ms, tokens_generated = await self._step_api(messages, temp)
            model_name = self.config.api_model
        else:
            response_text, latency_ms, tokens_generated = await self._step_local(messages, temp)
            model_name = self.model.config.name_or_path

        # Truncate after </python> to prevent hallucinated outputs
        # Model tends to generate fake "Output:" after code blocks
        if "</python>" in response_text:
            python_end = response_text.find("</python>") + len("</python>")
            response_text = response_text[:python_end]

        # Log the call
        llm_logger = get_llm_logger()
        if llm_logger:
            llm_logger.log_model_call(
                model=model_name,
                messages=messages,
                response=response_text,
                max_new_tokens=self.config.max_new_tokens,
                temperature=temp,
                latency_ms=latency_ms,
                tokens_generated=tokens_generated,
            )

        return Message(Role.ASSISTANT, response_text)

    async def _step_api(self, messages: list[dict], temperature: float) -> tuple[str, float, int]:
        """Generate via OpenAI-compatible API."""
        start_time = time.perf_counter()

        response = await self._http_client.post(
            f"{self.config.api_url}/chat/completions",
            json={
                "model": self.config.api_model,
                "messages": messages,
                "max_tokens": self.config.max_new_tokens,
                "temperature": temperature,
            },
        )
        response.raise_for_status()
        data = response.json()

        latency_ms = (time.perf_counter() - start_time) * 1000
        response_text = data["choices"][0]["message"]["content"]
        tokens_generated = data.get("usage", {}).get("completion_tokens", 0)

        return response_text, latency_ms, tokens_generated

    async def _step_local(self, messages: list[dict], temperature: float) -> tuple[str, float, int]:
        """Generate via local model."""
        import torch

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.config.enable_thinking,
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_context,
        ).to(self.device)

        input_length = inputs["input_ids"].shape[1]

        start_time = time.perf_counter()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        latency_ms = (time.perf_counter() - start_time) * 1000

        generated_tokens = outputs[0][input_length:]
        response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        tokens_generated = outputs.shape[1] - input_length

        # Clean up GPU tensors
        del inputs, outputs, generated_tokens
        torch.cuda.empty_cache()

        return response_text, latency_ms, tokens_generated

    def _truncate_middle(self, text: str, max_chars: int = 6000) -> str:
        """Truncate text by removing the middle, keeping beginning and end.

        Args:
            text: Text to truncate
            max_chars: Maximum characters to keep

        Returns:
            Truncated text with middle replaced by indicator
        """
        if len(text) <= max_chars:
            return text

        # Keep 40% at start, 40% at end, leave room for indicator
        keep_each = int(max_chars * 0.4)
        removed = len(text) - (keep_each * 2)
        return f"{text[:keep_each]}\n\n... [{removed} chars truncated] ...\n\n{text[-keep_each:]}"

    async def execute_code(self, code: str, max_output_chars: int = 6000) -> Message:
        """Execute code in the sandbox.

        Args:
            code: Python code to execute
            max_output_chars: Max chars for output (truncates middle if exceeded)

        Returns:
            Execution result message
        """
        try:
            result = await self.sandbox.execute(code)
            if result.get("error"):
                output = f"Error: {result['error']}"
            else:
                output = result['output']

            # Truncate long outputs (keep beginning and end)
            output = self._truncate_middle(output, max_output_chars)
            content = f"<python_output>\n{output}\n</python_output>"
        except Exception as e:
            content = f"Execution error: {str(e)}"

        return Message(Role.EXECUTION, content)

    def _debug(self, *args, **kwargs):
        """Print debug info if debug mode is enabled."""
        if self.config.debug:
            print(*args, **kwargs)

    async def run(self, question: str, temperature: float | None = None) -> EpisodeResult:
        """Run a full episode to answer a question.

        Args:
            question: The question to answer
            temperature: Override temperature (uses config default if None)

        Returns:
            EpisodeResult with conversation history and answer
        """
        conversation = self._create_conversation(question)
        num_turns = 0

        self._debug(f"\n{'='*60}")
        self._debug(f"[AGENT] Starting episode: {question[:80]}...")
        self._debug(f"{'='*60}")

        for turn in range(self.config.max_turns):
            self._debug(f"\n--- Turn {turn + 1}/{self.config.max_turns} ---")

            # Show what we're sending to model
            if self.config.debug:
                msgs = conversation.to_messages()
                self._debug(f"[PROMPT] {len(msgs)} messages:")
                for m in msgs:
                    role = m['role'].upper()
                    self._debug(f"  [{role}]:\n{m['content']}\n")

            # Generate response
            response = await self.step(conversation, temperature=temperature)
            conversation.add_message(response)
            num_turns += 1

            self._debug(f"\n[MODEL RESPONSE] ({len(response.content)} chars):")
            self._debug(response.content)

            # Check for final answer
            final_answer = parse_answer(response.content)
            if final_answer is not None:
                self._debug(f"\n[ANSWER FOUND]: {final_answer}")
                return EpisodeResult(
                    question=question,
                    conversation=conversation,
                    final_answer=final_answer,
                    num_turns=num_turns,
                )

            # Check for code to execute
            code = parse_python_code(response.content)
            if code:
                self._debug(f"\n[CODE EXTRACTED]:\n{code}")
                execution_result = await self.execute_code(code)
                conversation.add_message(execution_result)
                self._debug(f"\n[EXECUTION RESULT]:\n{execution_result.content}")
            else:
                self._debug("[NO CODE FOUND in response]")
            # If no code and no answer, continue to next turn (or hit max turns)

        # Max turns reached without <answer> tag - no answer
        self._debug(f"\n[MAX TURNS REACHED] No answer found after {num_turns} turns")
        return EpisodeResult(
            question=question,
            conversation=conversation,
            final_answer=None,
            num_turns=num_turns,
        )
