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
import re
from html import unescape
from typing import Any, Dict, List, Tuple
from urllib.parse import urljoin, urldefrag

import aiohttp
from dotenv import load_dotenv
from pydantic import BaseModel, Field

try:
    from bs4 import BeautifulSoup, NavigableString, Tag
except ImportError as exc:  # pragma: no cover - informative error
    raise ImportError(
        "beautifulsoup4 is required for the local HTML reader. "
        "Install it with `pip install beautifulsoup4`."
    ) from exc

from logging_utils import setup_colored_logger
from tool_server.utils import (
    SUMMARY_N_CTX,
    _is_valid_url,
    get_retry_delay,
    llm_summary,
)

load_dotenv()

logger = setup_colored_logger(__name__)


class ReadURLItem(BaseModel):
    """
    A single URL extracted from a webpage with contextual information.

    Attributes:
        url: The complete URL found on the page
        title: Descriptive title explaining what the URL links to
    """

    url: str = Field(description="The full URL found on the page")
    title: str = Field(description="Title or description of the linked content")


class ReadResult(BaseModel):
    """
    Results from reading and analyzing a webpage.

    Attributes:
        success: Whether the read operation completed successfully
        content: Raw text content extracted from the webpage
        summary: LLM-generated summary based on the question, or truncated
                content if LLM summarization fails
        raw_response: Original API response (truncated to 500 chars)
        url_items: Relevant URLs discovered on the page
        metadata: API usage statistics and response metadata
        error: Error message if operation failed, empty string otherwise
    """

    success: bool
    content: str
    summary: str = ""
    raw_response: str = ""
    url_items: List[ReadURLItem] = []
    metadata: Dict[str, Any] = {}
    error: str = ""


class HTMLToMarkdownConverter:
    """Convert HTML content to Markdown with basic structural fidelity."""

    _INLINE_TAGS = {"span", "em", "i", "strong", "b", "a", "code", "kbd", "mark"}
    _SKIP_TAGS = {"script", "style", "noscript", "iframe", "canvas", "svg"}

    def __init__(self, base_url: str):
        self.base_url = base_url

    def convert(self, soup: BeautifulSoup) -> str:
        body = soup.body or soup
        markdown = self._convert_children(body).strip()
        markdown = re.sub(r"\n{3,}", "\n\n", markdown)
        markdown = re.sub(r"[ \t]+\n", "\n", markdown)
        return markdown.strip()

    def _convert_children(self, node: Tag, indent: str = "") -> str:
        parts: List[str] = []
        for child in node.children:
            part = self._convert_node(child, indent)
            if part:
                parts.append(part)
        return "".join(parts)

    def _convert_node(self, node, indent: str = "") -> str:
        if isinstance(node, NavigableString):
            text = unescape(str(node))
            if not text.strip():
                return ""
            collapsed = re.sub(r"\s+", " ", text)
            return collapsed

        if not isinstance(node, Tag):
            return ""

        name = node.name.lower()
        if name in self._SKIP_TAGS:
            return ""

        if name == "br":
            return "\n"

        if name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            level = int(name[1])
            content = self._convert_children(node).strip()
            if not content:
                return ""
            return f"\n{'#' * level} {content}\n\n"

        if name == "p":
            content = self._convert_children(node).strip()
            if not content:
                return ""
            return f"\n{content}\n\n"

        if name in {"strong", "b"}:
            content = self._convert_children(node).strip()
            return f"**{content}**" if content else ""

        if name in {"em", "i"}:
            content = self._convert_children(node).strip()
            return f"*{content}*" if content else ""

        if name == "code":
            text = node.get_text()
            if "\n" in text:
                return f"\n```\n{text.rstrip()}\n```\n"
            return f"`{text.strip()}`"

        if name == "pre":
            text = node.get_text()
            language = ""
            class_attr = node.get("class")
            if isinstance(class_attr, list) and class_attr:
                for item in class_attr:
                    if item.startswith("language-"):
                        language = item.split("language-", 1)[1]
                        break
            return f"\n```{language}\n{text.rstrip()}\n```\n"

        if name == "ul":
            return self._convert_list(node, indent, ordered=False)

        if name == "ol":
            return self._convert_list(node, indent, ordered=True)

        if name == "li":
            content = self._convert_children(node, indent + "  ").strip()
            if not content:
                return ""
            bullet = f"{indent}- {content.replace(chr(10), chr(10) + indent + '  ')}\n"
            return bullet

        if name == "a":
            href = node.get("href")
            text = self._convert_children(node).strip() or href or ""
            if not href:
                return text
            href = urljoin(self.base_url, href)
            return f"[{text}]({href})" if text else href

        if name == "img":
            alt = node.get("alt", "").strip()
            src = node.get("src")
            if not src:
                return ""
            src = urljoin(self.base_url, src)
            return f"![{alt}]({src})"

        if name == "blockquote":
            content = self._convert_children(node).strip()
            if not content:
                return ""
            quoted = "\n".join(
                f"> {line}" if line.strip() else ">"
                for line in content.splitlines()
            )
            return f"\n{quoted}\n\n"

        if name == "table":
            return self._convert_table(node)

        if name in {"thead", "tbody"}:
            return self._convert_children(node, indent)

        return self._convert_children(node, indent)

    def _convert_list(self, node: Tag, indent: str, ordered: bool) -> str:
        lines: List[str] = []
        for idx, li in enumerate(node.find_all("li", recursive=False), start=1):
            marker = f"{indent}{idx}. " if ordered else f"{indent}- "
            content = self._convert_children(li, indent + ("   " if ordered else "  ")).strip()
            if not content:
                continue
            formatted = content.replace("\n", f"\n{indent}{'   ' if ordered else '  '}")
            lines.append(marker + formatted)
        if not lines:
            return ""
        return "\n".join(lines) + "\n"

    def _convert_table(self, node: Tag) -> str:
        rows: List[List[str]] = []
        for tr in node.find_all("tr"):
            row: List[str] = []
            for cell in tr.find_all(["th", "td"]):
                text = cell.get_text(" ", strip=True)
                text = re.sub(r"\s+", " ", text)
                row.append(text)
            if row:
                rows.append(row)

        if not rows:
            return ""

        column_count = max(len(row) for row in rows)
        normalized_rows = [
            row + [""] * (column_count - len(row)) for row in rows
        ]

        header = normalized_rows[0]
        separators = ["---"] * column_count
        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(separators) + " |",
        ]
        for row in normalized_rows[1:]:
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines) + "\n\n"


async def _fetch_html(url: str, timeout: int) -> Tuple[int, str, str, Dict[str, str]]:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
    }

    async with aiohttp.ClientSession(headers=headers) as session:
        async with session.get(
            url,
            timeout=aiohttp.ClientTimeout(total=timeout),
            allow_redirects=True,
        ) as response:
            status = response.status
            final_url = str(response.url)
            text = await response.text(errors="ignore")
            return status, final_url, text, dict(response.headers)


def _extract_links(
    soup: BeautifulSoup, base_url: str
) -> Tuple[List[ReadURLItem], int]:
    url_items: List[ReadURLItem] = []
    seen: set[str] = set()
    total_links = 0

    for anchor in soup.find_all("a"):
        href = anchor.get("href")
        if not href:
            continue
        total_links += 1
        absolute = urljoin(base_url, href.strip())
        absolute, _ = urldefrag(absolute)

        if absolute in seen:
            continue
        if not _is_valid_url(absolute):
            continue

        title = anchor.get_text(" ", strip=True) or absolute
        title = re.sub(r"\s+", " ", title).strip()

        url_items.append(ReadURLItem(url=absolute, title=title[:256]))
        seen.add(absolute)

    return url_items, total_links


def _clean_html(html_text: str) -> BeautifulSoup:
    soup = BeautifulSoup(html_text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup


async def local_read(url: str, timeout: int = 30) -> ReadResult:
    """
    Read and extract content from a webpage using a local HTML-to-Markdown
    converter (no external API calls).
    """
    loop = asyncio.get_running_loop()
    start_time = loop.time()

    try:
        status, final_url, html_text, headers = await _fetch_html(url, timeout)
        execution_time = loop.time() - start_time

        content_type = headers.get("Content-Type", "").lower()
        if "application/pdf" in content_type:
            logger.warning(f"Unsupported content type for summariser: {content_type} ({url})")
            return ReadResult(
                success=False,
                content="",
                url_items=[],
                raw_response=html_text[:500],
                metadata={
                    "source": "local_reader",
                    "url": url,
                    "final_url": final_url,
                    "status": status,
                    "execution_time": execution_time,
                    "content_type": content_type,
                },
                error=f"Unsupported content type: {content_type or 'unknown'}",
            )

        if status != 200 or not html_text.strip():
            logger.warning(
                f"Read failed for '{url}' with HTTP status {status}"
            )
            return ReadResult(
                success=False,
                content="",
                url_items=[],
                raw_response=html_text[:500],
                metadata={
                    "source": "local_reader",
                    "url": url,
                    "status": status,
                    "execution_time": execution_time,
                    "links_found": 0,
                    "relevant_links": 0,
                },
                error=f"HTTP {status}: unable to fetch content",
            )

        soup = _clean_html(html_text)
        base_url = final_url or url
        converter = HTMLToMarkdownConverter(base_url=base_url)
        markdown_content = converter.convert(soup)

        max_chars = max(0, (SUMMARY_N_CTX - 1024) * 4)
        if max_chars and len(markdown_content) > max_chars:
            logger.warning(
                f"Truncating content for '{url}' from {len(markdown_content)} to {max_chars} chars to respect context window"
            )
            markdown_content = markdown_content[:max_chars]

        url_items, total_links = _extract_links(soup, base_url)

        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        elif soup.find("h1"):
            title = soup.find("h1").get_text(" ", strip=True)

        metadata = {
            "source": "local_reader",
            "url": url,
            "final_url": final_url,
            "status": status,
            "title": title,
            "execution_time": execution_time,
            "links_found": total_links,
            "relevant_links": len(url_items),
            "content_length": len(markdown_content),
            "headers": {k: headers.get(k, "") for k in ("Content-Type", "Content-Length")},
        }

        logger.info(
            f"Successfully read '{url}', converted to markdown ({len(markdown_content)} chars)"
        )

        return ReadResult(
            success=True,
            content=markdown_content,
            url_items=url_items,
            raw_response=html_text[:500],
            metadata=metadata,
        )

    except asyncio.TimeoutError:
        execution_time = loop.time() - start_time
        logger.warning(f"Read request for '{url}' timed out after {timeout}s")
        return ReadResult(
            success=False,
            content="",
            url_items=[],
            raw_response="Request timed out",
            metadata={
                "source": "local_reader",
                "url": url,
                "status": 408,
                "execution_time": execution_time,
                "links_found": 0,
                "relevant_links": 0,
            },
            error=f"Request timed out after {timeout}s",
        )

    except aiohttp.ClientError as e:
        execution_time = loop.time() - start_time
        logger.warning(f"Client error during read for '{url}': {str(e)}")
        return ReadResult(
            success=False,
            content="",
            url_items=[],
            raw_response=str(e)[:500],
            metadata={
                "source": "local_reader",
                "url": url,
                "status": 502,
                "execution_time": execution_time,
                "links_found": 0,
                "relevant_links": 0,
            },
            error=f"Client error: {str(e)[:200]}",
        )

    except Exception as e:
        execution_time = loop.time() - start_time
        logger.error(f"Unexpected error during read for '{url}': {str(e)}", exc_info=True)
        return ReadResult(
            success=False,
            content="",
            url_items=[],
            raw_response=str(e)[:500],
            metadata={
                "source": "local_reader",
                "url": url,
                "status": 500,
                "execution_time": execution_time,
                "links_found": 0,
                "relevant_links": 0,
            },
            error=f"Unexpected error: {str(e)[:200]}",
        )


class WebReadAgent:
    """
    Agent for reading web content with LLM summarization and concurrency control.

    This agent reads webpages, extracts content, and generates summaries using
    an LLM. It includes retry logic for recoverable errors and falls back to
    truncated content if LLM summarization fails.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the WebReadAgent.

        Args:
            config: Configuration dictionary with optional keys:
                - max_concurrent_requests: Max concurrent reads (default: 500)
                - max_content_words: Max words to send to LLM (default: 10000)
                - max_summary_words: Max words for fallback summary (default: 2048)
                - max_summary_retries: Max retries for LLM (default: 3)
        """
        self._timeout = config.get("timeout", 30)
        self._semaphore = asyncio.Semaphore(config.get("max_concurrent_requests", 500))
        self.max_content_words = config.get("max_content_words", 10000)
        self.max_summary_words = config.get("max_summary_words", 2048)
        self.max_summary_retries = config.get("max_summary_retries", 3)

    def _truncate_content_to_words(self, content: str, max_words: int) -> str:
        """
        Truncate content to max words, keeping beginning and end.

        Args:
            content: Content to truncate
            max_words: Maximum number of words to keep

        Returns:
            Truncated content with " ... " in the middle if exceeded
        """
        words = content.split()
        if len(words) <= max_words:
            return content

        half_words = max_words // 2
        return " ".join(words[:half_words]) + " ... " + " ".join(words[-half_words:])

    async def read(self, question: str, url: str) -> ReadResult:
        """
        Read a webpage and generate a summary based on the question.

        This method:
        1. Reads webpage content via local HTML fetch + Markdown conversion
        2. Truncates content if too long
        3. Generates LLM summary with up to 3 retries for recoverable errors
        4. Falls back to truncated content if LLM fails

        Args:
            question: Question or context for summarization
            url: URL of the webpage to read

        Returns:
            ReadResult with content and summary (either LLM-generated or
            truncated content as fallback)

        Example:
            >>> agent = WebReadAgent(config={})
            >>> result = await agent.read("What is Python?", "https://python.org")
            >>> if result.success:
            ...     print(result.summary)
        """
        logger.info(f"Reading '{url}' with question: '{question[:100]}...'")

        try:
            async with self._semaphore:
                result = await local_read(url.strip(), timeout=self._timeout)

            if not result.success:
                logger.warning(
                    f"Read failed for '{url}' with status "
                    f"{result.metadata.get('status', 'unknown')}: {result.error}"
                )
                return result

            logger.info(
                f"Read successful for '{url}', content: {len(result.content)} chars"
            )

            # Truncate content if too long
            original_content = result.content
            words = result.content.split()
            if len(words) > self.max_content_words:
                result.content = self._truncate_content_to_words(
                    result.content, self.max_content_words
                )
                logger.info(
                    f"Truncated content from {len(words)} to {self.max_content_words} words"
                )

            # Generate summary with retry logic
            for attempt in range(self.max_summary_retries):
                summary_result = await llm_summary(
                    question=question,
                    content=result.content,
                )

                if summary_result.success:
                    logger.info(
                        f"Summary generated for '{url}' on attempt {attempt + 1}, "
                        f"length: {len(summary_result.text)} chars"
                    )
                    result.summary = summary_result.text
                    return result

                if (
                    summary_result.recoverable
                    and attempt < self.max_summary_retries - 1
                ):
                    retry_delay = get_retry_delay(attempt, summary_result.error or "")
                    logger.warning(
                        f"Recoverable error for '{url}' "
                        f"(attempt {attempt + 1}/{self.max_summary_retries}): "
                        f"{summary_result.error}. Retrying in {retry_delay}s..."
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    logger.warning(
                        f"Summary failed for '{url}': {summary_result.error}"
                    )
                    break

            # Fallback to truncated content
            logger.info(f"Using truncated content as fallback for '{url}'")
            result.summary = self._truncate_content_to_words(
                original_content, self.max_summary_words
            )
            return result

        except Exception as e:
            logger.error(
                f"Unexpected error in WebReadAgent for '{url}': {str(e)}",
                exc_info=True,
            )
            return ReadResult(
                success=False,
                content="",
                url_items=[],
                raw_response="",
                metadata={
                    "source": "local_reader",
                    "url": url,
                    "status": 500,
                    "execution_time": 0.0,
                    "links_found": 0,
                    "relevant_links": 0,
                },
                error=f"WebReadAgent error: {str(e)[:200]}",
            )
