"""Agent class for running episodes."""

import asyncio
import logging
import os
import time
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

from src.prompts import get_system_prompt
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
        top_p: Nucleus sampling threshold (0-1, higher = more diverse)
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
    top_p: float = 0.95  # Nucleus sampling threshold
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
        conv.add(Role.SYSTEM, get_system_prompt(self.config.max_turns))
        conv.add(Role.USER, f"Question: {question}")
        return conv

    async def step(self, conversation: Conversation, temperature: float | None = None, top_p: float | None = None) -> Message:
        """Generate a single response from the model.

        Args:
            conversation: Current conversation state
            temperature: Override temperature (uses config default if None)
            top_p: Override top_p (uses config default if None)

        Returns:
            The generated assistant message
        """
        temp = temperature if temperature is not None else self.config.temperature
        p = top_p if top_p is not None else self.config.top_p
        messages = conversation.to_messages()

        if self.config.use_api:
            response_text, latency_ms, tokens_generated = await self._step_api(messages, temp, p)
            model_name = self.config.api_model
        else:
            response_text, latency_ms, tokens_generated = await self._step_local(messages, temp, p)
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

    async def _step_api(self, messages: list[dict], temperature: float, top_p: float) -> tuple[str, float, int]:
        """Generate via OpenAI-compatible API."""
        start_time = time.perf_counter()

        response = await self._http_client.post(
            f"{self.config.api_url}/chat/completions",
            json={
                "model": self.config.api_model,
                "messages": messages,
                "max_tokens": self.config.max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
            },
        )
        response.raise_for_status()
        data = response.json()

        latency_ms = (time.perf_counter() - start_time) * 1000
        response_text = data["choices"][0]["message"]["content"]
        tokens_generated = data.get("usage", {}).get("completion_tokens", 0)

        return response_text, latency_ms, tokens_generated

    async def _step_local(self, messages: list[dict], temperature: float, top_p: float) -> tuple[str, float, int]:
        """Generate via local model."""
        import torch

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.config.enable_thinking,
        )

        # Check for truncation by comparing full length vs truncated
        full_length = len(self.tokenizer.encode(prompt, add_special_tokens=False))

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_context,
        ).to(self.device)

        input_length = inputs["input_ids"].shape[1]

        if full_length > input_length:
            logger.warning(
                f"Context truncated: {full_length} -> {input_length} tokens "
                f"(lost {full_length - input_length} tokens from beginning)"
            )

        start_time = time.perf_counter()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
        latency_ms = (time.perf_counter() - start_time) * 1000

        generated_tokens = outputs[0][input_length:]
        response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        tokens_generated = outputs.shape[1] - input_length

        # Clean up GPU tensors (skip empty_cache - adds overhead, PyTorch handles automatically)
        del inputs, outputs, generated_tokens

        return response_text, latency_ms, tokens_generated

    async def _step_local_batch(self, messages: list[dict], batch_size: int, temperature: float, top_p: float) -> list[tuple[str, float, int]]:
        """Generate multiple responses in one batched forward pass.

        Args:
            messages: The conversation messages (same for all batch items)
            batch_size: Number of responses to generate
            temperature: Sampling temperature (same for all - diversity comes from sampling)
            top_p: Nucleus sampling threshold (same for all - diversity comes from sampling)

        Returns:
            List of (response_text, latency_ms, tokens_generated) tuples
        """
        import torch

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.config.enable_thinking,
        )

        # Check for truncation
        full_length = len(self.tokenizer.encode(prompt, add_special_tokens=False))

        # Tokenize once, then repeat for batch
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_context,
        ).to(self.device)

        input_length = inputs["input_ids"].shape[1]

        if full_length > input_length:
            logger.warning(
                f"Context truncated: {full_length} -> {input_length} tokens "
                f"(lost {full_length - input_length} tokens from beginning)"
            )

        # Expand to batch size
        inputs["input_ids"] = inputs["input_ids"].expand(batch_size, -1)
        inputs["attention_mask"] = inputs["attention_mask"].expand(batch_size, -1)

        start_time = time.perf_counter()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Decode each response in the batch
        results = []
        for i in range(batch_size):
            generated_tokens = outputs[i][input_length:]
            response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            tokens_generated = len(generated_tokens)
            # Split latency evenly across batch (approximate)
            results.append((response_text, latency_ms / batch_size, tokens_generated))

        # Clean up GPU tensors (skip empty_cache - adds overhead)
        del inputs, outputs

        return results

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

    async def run(self, question: str, temperature: float | None = None, top_p: float | None = None) -> EpisodeResult:
        """Run a full episode to answer a question.

        Args:
            question: The question to answer
            temperature: Override temperature (uses config default if None)
            top_p: Override top_p (uses config default if None)

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

            # On final turn, remind model it MUST provide an answer
            is_final_turn = (turn == self.config.max_turns - 1)
            if is_final_turn and turn > 0:  # Only if not the first turn
                reminder = Message(
                    Role.EXECUTION,
                    "[FINAL TURN] You MUST provide your answer NOW using <answer>your answer</answer>. "
                    "A partial or uncertain answer is better than no answer."
                )
                conversation.add_message(reminder)
                self._debug(f"[INJECTED FINAL TURN REMINDER]")

            # Show what we're sending to model
            if self.config.debug:
                msgs = conversation.to_messages()
                self._debug(f"[PROMPT] {len(msgs)} messages:")
                for m in msgs:
                    role = m['role'].upper()
                    self._debug(f"  [{role}]:\n{m['content']}\n")

            # Generate response
            response = await self.step(conversation, temperature=temperature, top_p=top_p)
            conversation.add_message(response)
            num_turns += 1
            # Yield to event loop - allows background tasks (e.g. judge) to make progress
            await asyncio.sleep(0)

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

    async def run_batch_turn1(self, question: str, num_rollouts: int, temperature: float, top_p: float | None = None) -> list[tuple[Conversation, Message, bool]]:
        """Generate turn 1 for multiple rollouts in a single batched forward pass.

        Args:
            question: The question to answer
            num_rollouts: Number of rollouts to generate
            temperature: Sampling temperature (diversity comes from sampling)
            top_p: Nucleus sampling threshold (uses config default if None)

        Returns:
            List of (conversation, response, is_complete) tuples.
            is_complete=True if the response already contains a final answer.
        """
        p = top_p if top_p is not None else self.config.top_p

        # Create the initial conversation (same for all rollouts)
        base_conversation = self._create_conversation(question)
        messages = base_conversation.to_messages()

        # Batch generate turn 1
        gen_start = time.perf_counter()
        batch_results = await self._step_local_batch(messages, num_rollouts, temperature, p)
        gen_time = time.perf_counter() - gen_start
        logger.warning(f"Batch turn1 LLM generation: {gen_time:.2f}s for {num_rollouts} rollouts")

        # Process each response
        results = []
        llm_logger = get_llm_logger()
        model_name = self.model.config.name_or_path

        for response_text, latency_ms, tokens_generated in batch_results:
            # Truncate after </python> to prevent hallucinated outputs
            if "</python>" in response_text:
                python_end = response_text.find("</python>") + len("</python>")
                response_text = response_text[:python_end]

            # Log the call
            if llm_logger:
                llm_logger.log_model_call(
                    model=model_name,
                    messages=messages,
                    response=response_text,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=temperature,
                    latency_ms=latency_ms,
                    tokens_generated=tokens_generated,
                )

            # Create conversation copy with response
            conversation = self._create_conversation(question)
            response = Message(Role.ASSISTANT, response_text)
            conversation.add_message(response)

            # Check if already complete (has answer)
            final_answer = parse_answer(response_text)
            is_complete = final_answer is not None

            results.append((conversation, response, is_complete))

        return results

    async def continue_from_turn1(
        self,
        question: str,
        conversation: Conversation,
        turn1_response: Message,
        temperature: float,
        top_p: float | None = None,
    ) -> EpisodeResult:
        """Continue a rollout from after turn 1.

        Args:
            question: The original question
            conversation: Conversation with turn 1 response already added
            turn1_response: The turn 1 response message
            temperature: Sampling temperature for remaining turns
            top_p: Nucleus sampling threshold (uses config default if None)

        Returns:
            Complete EpisodeResult
        """
        p = top_p if top_p is not None else self.config.top_p
        num_turns = 1  # Already did turn 1

        # Check if turn 1 already has answer
        final_answer = parse_answer(turn1_response.content)
        if final_answer is not None:
            return EpisodeResult(
                question=question,
                conversation=conversation,
                final_answer=final_answer,
                num_turns=num_turns,
            )

        # Execute code from turn 1 if present
        code = parse_python_code(turn1_response.content)
        if code:
            execution_result = await self.execute_code(code)
            conversation.add_message(execution_result)

        # Continue with remaining turns
        for turn in range(1, self.config.max_turns):
            # On final turn, remind model it MUST provide an answer
            is_final_turn = (turn == self.config.max_turns - 1)
            if is_final_turn:
                reminder = Message(
                    Role.EXECUTION,
                    "[FINAL TURN] You MUST provide your answer NOW using <answer>your answer</answer>. "
                    "A partial or uncertain answer is better than no answer."
                )
                conversation.add_message(reminder)

            response = await self.step(conversation, temperature=temperature, top_p=p)
            conversation.add_message(response)
            num_turns += 1
            # Yield to event loop - allows background tasks (e.g. judge) to make progress
            await asyncio.sleep(0)

            # Check for final answer
            final_answer = parse_answer(response.content)
            if final_answer is not None:
                return EpisodeResult(
                    question=question,
                    conversation=conversation,
                    final_answer=final_answer,
                    num_turns=num_turns,
                )

            # Execute code if present
            code = parse_python_code(response.content)
            if code:
                execution_result = await self.execute_code(code)
                conversation.add_message(execution_result)

        # Max turns reached
        return EpisodeResult(
            question=question,
            conversation=conversation,
            final_answer=None,
            num_turns=num_turns,
        )
