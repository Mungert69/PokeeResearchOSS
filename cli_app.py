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

"""
Deep Research Agent - User Interface
This script provides a simple CLI interface to interact with the trained deep research agent.
"""

from __future__ import annotations

import argparse
import asyncio
import time

try:
    import torch
except ImportError:
    torch = None

from logging_utils import setup_colored_logger

logger = setup_colored_logger("cli_app")


async def interactive_mode_async(
    agent,
    temperature: float,
    top_p: float,
    verbose: bool,
):
    """Async interactive mode loop."""
    while True:
        try:
            question = input("\nYou: ").strip()

            if not question:
                continue

            if question.lower() in ["exit", "quit"]:
                print("\nGoodbye!")
                break

            print("\nAgent: Researching...\n")
            start_time = time.time()
            answer = await agent.run(
                question_raw=question,
                temperature=temperature,
                top_p=top_p,
                verbose=verbose,
            )

            print(f"\nAgent: {answer}\n")
            print("Time taken: {:.2f} seconds".format(time.time() - start_time))
            print("-" * 80)

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            logger.error(f"\nError: {e}")
            if verbose:
                import traceback

                traceback.print_exc()


def interactive_mode(
    serving_mode: str,
    model_path: str,
    tool_config_path: str,
    device: str,
    max_turns: int,
    temperature: float,
    top_p: float,
    verbose: bool,
    vllm_url: str = None,
    llama_n_ctx: int = 32768,
    llama_threads: int | None = None,
    llama_gpu_layers: int = 0,
    llama_rope_base: float | None = None,
    llama_rope_scale: float | None = None,
    llama_max_new_tokens: int = 2048,
    llama_seed: int | None = None,
    llama_verbose: bool = False,
):
    """Run interactive mode."""
    # Create agent based on type
    if serving_mode == "vllm":
        if not vllm_url:
            raise ValueError("VLLM URL must be provided when using VLLM agent")
        from agent.vllm_agent import VLLMDeepResearchAgent

        logger.info(f"Using VLLM agent at {vllm_url}")
        agent = VLLMDeepResearchAgent(
            vllm_url=vllm_url,
            model_name=model_path,
            tool_config_path=tool_config_path,
            max_turns=max_turns,
        )
    elif serving_mode == "llamacpp":
        from agent.llamacpp_agent import LlamaCPPDeepResearchAgent

        logger.info("Using llama.cpp agent")
        agent = LlamaCPPDeepResearchAgent(
            model_path=model_path,
            tool_config_path=tool_config_path,
            max_turns=max_turns,
            n_ctx=llama_n_ctx,
            n_threads=llama_threads,
            n_gpu_layers=llama_gpu_layers,
            rope_frequency_base=llama_rope_base,
            rope_frequency_scale=llama_rope_scale,
            max_new_tokens=llama_max_new_tokens,
            seed=llama_seed,
            verbose=llama_verbose,
        )
    else:
        from agent.simple_agent import SimpleDeepResearchAgent

        logger.info("Using local model agent")
        agent = SimpleDeepResearchAgent(
            model_path=model_path,
            tool_config_path=tool_config_path,
            device=device,
            max_turns=max_turns,
        )

    print("\n" + "=" * 80)
    print("Deep Research Agent - Interactive Mode")
    print(f"Serving Mode: {serving_mode.upper()}")
    print(f"Model: {model_path}")
    print("=" * 80)
    print("Type 'exit' or 'quit' to end the session")
    print("=" * 80 + "\n")

    # Run entire interactive session in single event loop
    asyncio.run(interactive_mode_async(agent, temperature, top_p, verbose))


def single_query_mode(
    question: str,
    serving_mode: str,
    model_path: str,
    tool_config_path: str,
    device: str,
    max_turns: int,
    temperature: float,
    top_p: float,
    verbose: bool,
    vllm_url: str = None,
    llama_n_ctx: int = 32768,
    llama_threads: int | None = None,
    llama_gpu_layers: int = 0,
    llama_rope_base: float | None = None,
    llama_rope_scale: float | None = None,
    llama_max_new_tokens: int = 2048,
    llama_seed: int | None = None,
    llama_verbose: bool = False,
) -> str:
    """Run single query."""
    # Create agent based on type
    if serving_mode == "vllm":
        if not vllm_url:
            raise ValueError("VLLM URL must be provided when using VLLM agent")
        from agent.vllm_agent import VLLMDeepResearchAgent

        logger.info(f"Using VLLM agent at {vllm_url}")
        agent = VLLMDeepResearchAgent(
            vllm_url=vllm_url,
            model_name=model_path,
            tool_config_path=tool_config_path,
            max_turns=max_turns,
        )
    elif serving_mode == "llamacpp":
        from agent.llamacpp_agent import LlamaCPPDeepResearchAgent

        logger.info("Using llama.cpp agent")
        agent = LlamaCPPDeepResearchAgent(
            model_path=model_path,
            tool_config_path=tool_config_path,
            max_turns=max_turns,
            n_ctx=llama_n_ctx,
            n_threads=llama_threads,
            n_gpu_layers=llama_gpu_layers,
            rope_frequency_base=llama_rope_base,
            rope_frequency_scale=llama_rope_scale,
            max_new_tokens=llama_max_new_tokens,
            seed=llama_seed,
            verbose=llama_verbose,
        )
    else:
        from agent.simple_agent import SimpleDeepResearchAgent

        logger.info("Using local model agent")
        agent = SimpleDeepResearchAgent(
            model_path=model_path,
            tool_config_path=tool_config_path,
            device=device,
            max_turns=max_turns,
        )

    start_time = time.time()
    try:
        answer = asyncio.run(
            agent.run(
                question_raw=question,
                temperature=temperature,
                top_p=top_p,
                verbose=verbose,
            )
        )
        print("Time taken: {:.2f} seconds".format(time.time() - start_time))
        return answer
    except Exception as e:
        logger.error(f"\nError: {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        return "Error occurred while processing the query."


def main():
    parser = argparse.ArgumentParser(
        description="Deep Research Agent - User Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode with local model
  python cli_app.py --serving-mode local
  
  # Interactive mode with VLLM
  python cli_app.py --serving-mode vllm --vllm-url http://localhost:9999/v1
  
  # Single query with VLLM
  python cli_app.py --serving-mode vllm --vllm-url http://localhost:9999/v1 --question "What is the capital of France?"
        """,
    )
    parser.add_argument(
        "--serving-mode",
        type=str,
        choices=["local", "vllm", "llamacpp"],
        default="local",
        help=(
            "Serving mode to use: 'local' for transformers, 'vllm' for VLLM server, "
            "'llamacpp' for llama.cpp GGUF models"
        ),
    )
    parser.add_argument(
        "--vllm-url",
        type=str,
        default="http://localhost:9999/v1",
        help="URL of the VLLM server (required when using --serving-mode vllm)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="PokeeAI/pokee_research_7b",
        help="Path to model or HuggingFace model ID",
    )
    parser.add_argument(
        "--tool-config",
        type=str,
        default="config/tool_config/pokee_tool_config.yaml",
        help="Path to tool configuration file",
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Single question to answer (non-interactive mode)",
    )
    default_device = "cpu"
    parser.add_argument(
        "--device",
        type=str,
        default=default_device,
        help="Device to use for the local Transformers backend (default: cpu)",
    )
    parser.add_argument(
        "--max-turns", type=int, default=10, help="Maximum number of agent turns"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature"
    )
    parser.add_argument(
        "--top-p", type=float, default=0.9, help="Nucleus sampling parameter"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    parser.add_argument(
        "--llama-n-ctx",
        type=int,
        default=32768,
        help="Context window size for llama.cpp (default: 32768)",
    )
    parser.add_argument(
        "--llama-threads",
        type=int,
        default=None,
        help="Number of CPU threads for llama.cpp (default: auto)",
    )
    parser.add_argument(
        "--llama-gpu-layers",
        type=int,
        default=0,
        help="Number of layers to offload to GPU for llama.cpp",
    )
    parser.add_argument(
        "--llama-rope-base",
        type=float,
        default=None,
        help="Optional RoPE frequency base override for llama.cpp",
    )
    parser.add_argument(
        "--llama-rope-scale",
        type=float,
        default=None,
        help="Optional RoPE frequency scale override for llama.cpp",
    )
    parser.add_argument(
        "--llama-max-new-tokens",
        type=int,
        default=2048,
        help="Max new tokens to generate with llama.cpp (default: 2048)",
    )
    parser.add_argument(
        "--llama-seed",
        type=int,
        default=None,
        help="Random seed for llama.cpp sampling (default: random)",
    )
    parser.add_argument(
        "--llama-verbose",
        action="store_true",
        help="Enable verbose llama.cpp logging",
    )

    args = parser.parse_args()

    # Validate VLLM URL if using VLLM
    if args.serving_mode == "vllm" and not args.vllm_url:
        parser.error("--vllm-url is required when using --serving-mode vllm")

    if args.question:
        # Single query mode
        answer = single_query_mode(
            question=args.question,
            serving_mode=args.serving_mode,
            model_path=args.model_path,
            tool_config_path=args.tool_config,
            device=args.device,
            max_turns=args.max_turns,
            temperature=args.temperature,
            top_p=args.top_p,
            verbose=args.verbose,
            vllm_url=args.vllm_url,
            llama_n_ctx=args.llama_n_ctx,
            llama_threads=args.llama_threads,
            llama_gpu_layers=args.llama_gpu_layers,
            llama_rope_base=args.llama_rope_base,
            llama_rope_scale=args.llama_rope_scale,
            llama_max_new_tokens=args.llama_max_new_tokens,
            llama_seed=args.llama_seed,
            llama_verbose=args.llama_verbose,
        )
        print(f"\nQuestion: {args.question}")
        print(f"\nAnswer: {answer}\n")
    else:
        # Interactive mode
        interactive_mode(
            serving_mode=args.serving_mode,
            model_path=args.model_path,
            tool_config_path=args.tool_config,
            device=args.device,
            max_turns=args.max_turns,
            temperature=args.temperature,
            top_p=args.top_p,
            verbose=args.verbose,
            vllm_url=args.vllm_url,
            llama_n_ctx=args.llama_n_ctx,
            llama_threads=args.llama_threads,
            llama_gpu_layers=args.llama_gpu_layers,
            llama_rope_base=args.llama_rope_base,
            llama_rope_scale=args.llama_rope_scale,
            llama_max_new_tokens=args.llama_max_new_tokens,
            llama_seed=args.llama_seed,
            llama_verbose=args.llama_verbose,
        )


if __name__ == "__main__":
    main()
