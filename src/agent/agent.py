"""Agent class for running episodes."""

import time
from dataclasses import dataclass

import torch

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
    """

    max_turns: int = 3
    max_new_tokens: int = 200
    temperature: float = 0.7
    max_context: int = 1500


class Agent:
    """Agent that runs episodes by generating code and executing it.

    The agent manages the conversation, generates responses, executes code
    in a sandbox, and returns structured episode results.
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
            model: The language model
            tokenizer: Tokenizer for the model
            sandbox: Sandbox client for code execution
            config: Agent configuration (uses defaults if None)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.sandbox = sandbox
        self.config = config or AgentConfig()
        self.device = next(model.parameters()).device

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

    async def step(self, conversation: Conversation) -> Message:
        """Generate a single response from the model.

        Args:
            conversation: Current conversation state

        Returns:
            The generated assistant message
        """
        # Convert to chat messages and apply template
        messages = conversation.to_messages()
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_context,
        ).to(self.device)

        input_length = inputs["input_ids"].shape[1]

        # Generate
        start_time = time.perf_counter()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=True,
                temperature=self.config.temperature,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Decode only the generated tokens (not the prompt)
        generated_tokens = outputs[0][input_length:]
        response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Truncate after </python> to prevent hallucinated outputs
        # Model tends to generate fake "Output:" after code blocks
        if "</python>" in response_text:
            python_end = response_text.find("</python>") + len("</python>")
            response_text = response_text[:python_end]

        # Log the call
        llm_logger = get_llm_logger()
        if llm_logger:
            tokens_generated = outputs.shape[1] - input_length
            llm_logger.log_model_call(
                model=self.model.config.name_or_path,
                messages=messages,
                response=response_text,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                latency_ms=latency_ms,
                tokens_generated=tokens_generated,
            )

        return Message(Role.ASSISTANT, response_text)

    async def execute_code(self, code: str) -> Message:
        """Execute code in the sandbox.

        Args:
            code: Python code to execute

        Returns:
            Execution result message
        """
        try:
            result = await self.sandbox.execute(code)
            if result.get("error"):
                content = f"Error:\n{result['error']}"
            else:
                content = f"Output:\n{result['output']}"
        except Exception as e:
            content = f"Execution error: {str(e)}"

        return Message(Role.EXECUTION, content)

    async def run(self, question: str) -> EpisodeResult:
        """Run a full episode to answer a question.

        Args:
            question: The question to answer

        Returns:
            EpisodeResult with conversation history and answer
        """
        conversation = self._create_conversation(question)
        num_turns = 0

        for turn in range(self.config.max_turns):
            # Generate response
            response = await self.step(conversation)
            conversation.add_message(response)
            num_turns += 1

            # Check for final answer
            final_answer = parse_answer(response.content)
            if final_answer is not None:
                return EpisodeResult(
                    question=question,
                    conversation=conversation,
                    final_answer=final_answer,
                    num_turns=num_turns,
                )

            # Check for code to execute
            code = parse_python_code(response.content)
            if code:
                execution_result = await self.execute_code(code)
                conversation.add_message(execution_result)
            else:
                # No code or answer - treat response as final answer
                return EpisodeResult(
                    question=question,
                    conversation=conversation,
                    final_answer=response.content.strip(),
                    num_turns=num_turns,
                )

        # Max turns reached
        last_response = conversation.last_message(Role.ASSISTANT)
        final_answer = last_response.content.strip() if last_response else None

        return EpisodeResult(
            question=question,
            conversation=conversation,
            final_answer=final_answer,
            num_turns=num_turns,
        )
