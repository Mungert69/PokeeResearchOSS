# Copyright 2025 Pokee AI Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



import asyncio
import logging
import os
import re
import threading
from functools import partial
from typing import Optional

from dotenv import load_dotenv
from llama_cpp import Llama
from shared_llama_lock import llama_cpp_lock
from pydantic import BaseModel

from agent.chat_template import render_qwen_chat
from logging_utils import setup_colored_logger

load_dotenv()


logger = setup_colored_logger(__name__, level=logging.INFO)
SUMMARY_GGUF_PATH = os.getenv("SUMMARY_GGUF_PATH")
SUMMARY_N_CTX = int(os.getenv("SUMMARY_N_CTX", "32768"))
SUMMARY_THREADS = os.getenv("SUMMARY_THREADS")
SUMMARY_GPU_LAYERS = int(os.getenv("SUMMARY_GPU_LAYERS", "0"))
SUMMARY_VERBOSE = os.getenv("SUMMARY_VERBOSE", "0").lower() in {"1", "true", "yes"}
SUMMARY_MAX_NEW_TOKENS = int(os.getenv("SUMMARY_MAX_NEW_TOKENS", "512"))
SUMMARY_TEMPERATURE = float(os.getenv("SUMMARY_TEMPERATURE", "0.1"))
SUMMARY_TOP_P = float(os.getenv("SUMMARY_TOP_P", "0.9"))


def _is_valid_url(url: str) -> bool:
    """
    Check if a URL is valid and not a common non-informative link.

    Args:
        url (str): The URL to validate

    Returns:
        bool: True if the URL is valid and informative
    """
    if not url or not isinstance(url, str):
        return False

    url_lower = url.lower()

    # Skip common non-informative link patterns
    skip_patterns = [
        "javascript:",
        "mailto:",
        "#",
        "tel:",
        "data:",
        "blob:",
    ]

    # Skip social media and common non-content domains
    skip_domains = [
        "facebook.com",
        "twitter.com",
        "x.com",  # Twitter's new domain
        "instagram.com",
        "youtube.com",
        "tiktok.com",
        "pinterest.com",
        "snapchat.com",
        "discord.com",
        "telegram.org",
        "whatsapp.com",
        "wechat.com",
        "weibo.com",
        "douyin.com",
        "substack.com",  # Newsletter platform
        "patreon.com",
        "onlyfans.com",
        "twitch.tv",
        "vimeo.com",
        "dailymotion.com",
        "rumble.com",
        "bitchute.com",
    ]

    # Check skip patterns
    if any(pattern in url_lower for pattern in skip_patterns):
        return False

    # Check skip domains (including subdomains)
    for domain in skip_domains:
        if domain in url_lower:
            return False

    # Must start with http/https or be a relative URL starting with /
    if not (url.startswith(("http://", "https://")) or url.startswith("/")):
        return False

    return True


logger = setup_colored_logger(__name__, level=logging.INFO)
_summary_model = None
_summary_model_lock = threading.Lock()


class LlamaCppSummarizer:
    """Summariser backed by llama.cpp running a Qwen3 GGUF checkpoint."""

    def __init__(
        self,
        model_path: str,
        n_ctx: int,
        n_threads: Optional[int],
        n_gpu_layers: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        verbose: bool = False,
    ):
        if not model_path:
            raise ValueError(
                "SUMMARY_GGUF_PATH is not set. Please point it to a Qwen3-4B GGUF file."
            )

        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_gpu_layers = n_gpu_layers
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.verbose = verbose

        logger.info(
            "Loading summary GGUF model from '%s' (ctx=%d, threads=%s, gpu_layers=%d)...",
            self.model_path,
            self.n_ctx,
            self.n_threads if self.n_threads is not None else "auto",
            self.n_gpu_layers,
        )

        self.llama = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            n_gpu_layers=self.n_gpu_layers,
            verbose=self.verbose,
        )
        self.stop_tokens = ["<|im_end|>", "<|im_start|>user", "<|im_start|>assistant"]
        self._generate_lock = threading.Lock()
        logger.info("Summary GGUF model loaded successfully.")

    def summarize(self, question: str, content: str) -> str:
        """Synchronously generate a question-conditioned summary."""
        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {
                "role": "user",
                "content": f"<question>{question}</question>\n<content>{content}</content>",
            },
        ]
        prompt = render_qwen_chat(messages, add_generation_prompt=True)

        with self._generate_lock:
            with llama_cpp_lock():
                output = self.llama(
                    prompt=prompt,
                    max_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    stop=self.stop_tokens,
                )

        choices = output.get("choices", [])
        if not choices:
            return ""

        text = choices[0].get("text", "")
        return text.strip()


def _parse_threads(value: Optional[str]) -> Optional[int]:
    if value is None or value.strip() == "":
        return None
    try:
        threads = int(value)
        return threads if threads > 0 else None
    except ValueError:
        logger.warning("Invalid SUMMARY_THREADS value '%s', using auto.", value)
        return None


def get_summary_model() -> LlamaCppSummarizer:
    global _summary_model
    if _summary_model is None:
        with _summary_model_lock:
            if _summary_model is None:
                _summary_model = LlamaCppSummarizer(
                    model_path=SUMMARY_GGUF_PATH,
                    n_ctx=SUMMARY_N_CTX,
                    n_threads=_parse_threads(SUMMARY_THREADS),
                    n_gpu_layers=SUMMARY_GPU_LAYERS,
                    max_new_tokens=SUMMARY_MAX_NEW_TOKENS,
                    temperature=SUMMARY_TEMPERATURE,
                    top_p=SUMMARY_TOP_P,
                    verbose=SUMMARY_VERBOSE,
                )
    return _summary_model


def extract_retry_delay_from_error(error_str: str) -> Optional[float]:
    """
    Extract retry delay from Gemini API error response.
    Returns the delay in seconds if found, None otherwise.
    """
    try:
        if "RESOURCE_EXHAUSTED" in error_str and "retryDelay" in error_str:
            # check retry delay for gemini models
            # Look for retryDelay pattern in the error message
            retry_delay_match = re.search(
                r"'retryDelay': '(\d+(?:\.\d+)?)s'", error_str
            )
            if retry_delay_match:
                return float(retry_delay_match.group(1))

            # Alternative pattern matching
            retry_delay_match = re.search(
                r'retryDelay["\']?\s*:\s*["\']?(\d+(?:\.\d+)?)s', error_str
            )
            if retry_delay_match:
                return float(retry_delay_match.group(1))

    except Exception as e:
        logger.warning(f"Could not extract retry delay from error: {e}")

    return None


def get_retry_delay(try_cnt: int, error_str: str) -> float:
    """
    Calculate appropriate retry delay based on error type and attempt number.

    First tries to extract explicit retry delay from API error response.
    Falls back to exponential backoff if no delay is specified.

    Args:
        try_cnt: Current attempt number (0-indexed)
        error_str: Error message to check for retry delay hints

    Returns:
        Delay in seconds before next retry (always returns a value)
    """
    # Try to extract API-specified retry delay
    retry_delay = extract_retry_delay_from_error(error_str)
    if retry_delay is None:
        # Fallback to exponential backoff: 1s, 4s, 16s, 64s, etc.
        retry_delay = min(4**try_cnt, 60)  # Cap at 60 seconds
    return retry_delay


SYSTEM_INSTRUCTION = """You are a helpful deep research assistant. I will provide you:
* A complex question that requires a deep research to answer.
* The content of a webpage returned by our web reader.

Your task is to read the webpage content carefully and extract all information that could help answer the question. Provide detailed information including numbers, dates, facts, examples, and explanations when available. Remove the irrelevant parts to reduce noise. Note that there could be no useful information on the webpage.

Important note: Use the same language as the user's main question for the summary. For example, if the question is in Chinese, then the summary should also be in Chinese.

Now think and extract the information that could help answer the question."""

class LLMSummaryResult(BaseModel):
    """Result from LLM summarization attempt."""

    success: bool
    text: str
    error: Optional[str] = None
    recoverable: bool = False  # Whether the error is recoverable by retrying


async def llm_summary(question: str, content: str) -> LLMSummaryResult:
    """
    Generate a question-conditioned summary using the local Qwen model.
    """
    model = get_summary_model()
    loop = asyncio.get_running_loop()

    try:
        text = await loop.run_in_executor(
            None, partial(model.summarize, question=question, content=content)
        )
        if text:
            return LLMSummaryResult(success=True, text=text)
        return LLMSummaryResult(
            success=False,
            text="",
            error="Model returned empty summary",
            recoverable=False,
        )
    except RuntimeError as e:
        logger.error(f"LLM summary OOM/runtime error: {e}")
        return LLMSummaryResult(
            success=False,
            text="",
            error=str(e)[:200],
            recoverable=False,
        )
    except Exception as e:
        logger.error(f"LLM summary failed: {e}", exc_info=True)
        return LLMSummaryResult(
            success=False,
            text="",
            error=str(e)[:200],
            recoverable=False,
        )
