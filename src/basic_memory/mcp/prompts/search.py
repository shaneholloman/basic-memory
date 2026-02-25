"""Search prompts for Basic Memory MCP server.

These prompts help users search and explore their knowledge base.
"""

from textwrap import dedent
from typing import Annotated, Optional

from loguru import logger
from pydantic import Field

from basic_memory.mcp.server import mcp
from basic_memory.mcp.tools.search import search_notes
from basic_memory.schemas.search import SearchResponse


@mcp.prompt(
    name="search_knowledge_base",
    description="Search across all content in basic-memory",
)
async def search_prompt(
    query: str,
    timeframe: Annotated[
        Optional[str],
        Field(description="How far back to search (e.g. '1d', '1 week')"),
    ] = None,
) -> str:
    """Search across all content in basic-memory.

    This prompt helps search for content in the knowledge base and
    provides helpful context about the results.

    Args:
        query: The search text to look for
        timeframe: Optional timeframe to limit results (e.g. '1d', '1 week')

    Returns:
        Formatted search results with context
    """
    logger.info(f"Searching knowledge base, query: {query}, timeframe: {timeframe}")

    # Call the search tool directly — it returns SearchResponse, dict, or error string
    result = await search_notes(query=query, after_date=timeframe)

    # Format the tool output into a prompt with guidance
    if isinstance(result, SearchResponse):
        result_count = len(result.results)
        result_text = _format_search_results(result, query)
    elif isinstance(result, dict):
        # json output format
        results = result.get("results", [])
        result_count = len(results)
        result_text = str(result)
    else:
        # Error string from search tool
        result_count = 0
        result_text = str(result)

    return dedent(f"""
        # Search Results: "{query}"

        This is a memory retrieval session showing search results.

        {result_text}

        ---

        ## Next Steps

        Based on these {result_count} results, you can:

        1. **Read a specific note** - Use `read_note("permalink")` to see full content
        2. **Build context** - Use `build_context("memory://path")` to see relationships
        3. **Refine search** - Use `search_notes("refined query")` to narrow results
        4. **Check recent activity** - Use `recent_activity(timeframe="7d")` for recent changes
    """)


def _format_search_results(result: SearchResponse, query: str) -> str:
    """Format SearchResponse into readable markdown."""
    if not result.results:
        return f"No results found for '{query}'."

    lines = [f"Found {len(result.results)} results:\n"]

    for item in result.results:
        title = item.title or "Untitled"
        permalink = item.permalink or ""
        score = f" (score: {item.score:.2f})" if item.score else ""

        lines.append(f"- **{title}**{score}")
        if permalink:
            lines.append(f"  permalink: {permalink}")
        if item.content:
            # Truncate content snippet
            content = item.content[:200] + "..." if len(item.content) > 200 else item.content
            lines.append(f"  {content}")
        lines.append("")

    if result.has_more:
        lines.append("*More results available. Use page=2 to see next page.*")

    return "\n".join(lines)
