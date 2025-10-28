"""Deep research agent backed by llama.cpp.

This adapter mirrors the behaviour of :class:`SimpleDeepResearchAgent` but uses
the llama.cpp inference engine to run GGUF models (e.g. PokeeResearch-7B converted
for llama.cpp). It relies on the Python bindings shipped in the `llama-cpp-python`
package so that inference can be performed in-process without spawning a
separate server.
"""

from __future__ import annotations

import threading
from typing import Optional

from agent.base_agent import BaseDeepResearchAgent
from agent.chat_template import render_qwen_chat
from logging_utils import setup_colored_logger

logger = setup_colored_logger(__name__)


class LlamaCPPDeepResearchAgent(BaseDeepResearchAgent):
    """Deep research agent that uses llama.cpp for inference."""

    _llama = None
    _model_path = None
    _model_lock = None

    def __init__(
        self,
        model_path: str,
        tool_config_path: str = "config/tool_config/pokee_tool_config.yaml",
        max_turns: int = 10,
        max_tool_response_length: int = 32768,
        n_ctx: int = 32768,
        n_threads: Optional[int] = None,
        n_gpu_layers: int = 0,
        rope_frequency_base: Optional[float] = None,
        rope_frequency_scale: Optional[float] = None,
        max_new_tokens: int = 2048,
        seed: Optional[int] = None,
        verbose: bool = False,
    ):
        """Initialise the llama.cpp backed agent.

        Args:
            model_path: Path to the GGUF model file.
            tool_config_path: YAML file describing available tools.
            max_turns: Maximum number of agent turns allowed.
            max_tool_response_length: Max characters to retain from tool output.
            n_ctx: Context window passed to llama.cpp.
            n_threads: Number of CPU threads to use (defaults to os.cpu_count()).
            n_gpu_layers: Number of transformer layers to offload to GPU.
            rope_frequency_base: Optional RoPE base override for long context.
            rope_frequency_scale: Optional RoPE scale override for long context.
            max_new_tokens: Maximum tokens generated per response.
            seed: Optional random seed for deterministic sampling.
            verbose: Whether llama.cpp should emit verbose logs.
        """
        super().__init__(
            tool_config_path=tool_config_path,
            max_turns=max_turns,
            max_tool_response_length=max_tool_response_length,
        )

        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_gpu_layers = n_gpu_layers
        self.rope_frequency_base = rope_frequency_base
        self.rope_frequency_scale = rope_frequency_scale
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        self.verbose = verbose

        if LlamaCPPDeepResearchAgent._model_lock is None:
            LlamaCPPDeepResearchAgent._model_lock = threading.Lock()

        with LlamaCPPDeepResearchAgent._model_lock:
            if (
                LlamaCPPDeepResearchAgent._llama is None
                or LlamaCPPDeepResearchAgent._model_path != model_path
            ):
                self._load_model()

        self.llama = LlamaCPPDeepResearchAgent._llama
        logger.info("llama.cpp model ready for inference.")

    def _load_model(self):
        """Load GGUF model via llama.cpp Python bindings."""
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise ImportError(
                "llama-cpp-python is required to use LlamaCPPDeepResearchAgent. "
                "Install it with `pip install llama-cpp-python`."
            ) from exc

        logger.info(f"Loading llama.cpp model from {self.model_path}...")
        kwargs = {
            "model_path": self.model_path,
            "n_ctx": self.n_ctx,
            "n_threads": self.n_threads,
            "n_gpu_layers": self.n_gpu_layers,
            "seed": self.seed,
            "verbose": self.verbose,
        }
        if self.rope_frequency_base is not None:
            kwargs["rope_freq_base"] = self.rope_frequency_base
        if self.rope_frequency_scale is not None:
            kwargs["rope_freq_scale"] = self.rope_frequency_scale

        # Remove keys with None values to keep llama.cpp happy.
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        LlamaCPPDeepResearchAgent._llama = Llama(**kwargs)
        LlamaCPPDeepResearchAgent._model_path = self.model_path
        logger.info("llama.cpp model loaded successfully.")

    async def generate(
        self, messages: list[dict], temperature: float = 0.7, top_p: float = 0.9
    ) -> str:
        """Generate a response using llama.cpp."""
        prompt = render_qwen_chat(
            messages,
            tools=self.tool_schemas,
            add_generation_prompt=True,
        )

        stop_tokens = ["<|im_end|>", "<|im_start|>user", "<|im_start|>assistant"]

        result = self.llama(
            prompt=prompt,
            max_tokens=self.max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop_tokens,
        )

        choices = result.get("choices", [])
        if not choices:
            logger.warning("llama.cpp returned no choices.")
            return ""

        text = choices[0].get("text", "")
        return text.strip()
