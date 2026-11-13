from __future__ import annotations

from typing import List, Optional

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class TavilySearchInput(BaseModel):
    query: str = Field(..., description="Search query. Keep it short and specific, one idea per query.")
    max_results: Optional[int] = Field(
        default=None,
        description="Maximum number of search results to return."
    )
    search_depth: Optional[str] = Field(
        default=None,
        description="Depth of the search: basic, advanced, fast, or ultra-fast."
    )
    topic: Optional[str] = Field(
        default=None,
        description="Search category: general or news."
    )
    time_range: Optional[str] = Field(
        default=None,
        description="Time range filter: day, week, month, or year."
    )
    include_images: Optional[bool] = Field(
        default=None,
        description="Include a list of query-related images in the response."
    )
    include_image_descriptions: Optional[bool] = Field(
        default=None,
        description="Include image descriptions in the response."
    )
    include_raw_content: Optional[bool] = Field(
        default=None,
        description="Include cleaned and parsed HTML content of each search result."
    )
    include_domains: Optional[List[str]] = Field(
        default=None,
        description="Domains to specifically include in the search results."
    )
    exclude_domains: Optional[List[str]] = Field(
        default=None,
        description="Domains to specifically exclude from search results."
    )
    country: Optional[str] = Field(
        default=None,
        description="Boost search results from a specific country (full name)."
    )
    include_favicon: Optional[bool] = Field(
        default=None,
        description="Include the favicon URL for each result."
    )
    start_date: Optional[str] = Field(
        default=None,
        description="Return results published after this date (YYYY-MM-DD)."
    )
    end_date: Optional[str] = Field(
        default=None,
        description="Return results published before this date (YYYY-MM-DD)."
    )
    exact_match: Optional[bool] = Field(
        default=None,
        description="Only return results containing the exact phrase(s)."
    )
    include_published_date: Optional[bool] = Field(
        default=None,
        description="Include detected publication dates."
    )
    filter_by_published_date: Optional[bool] = Field(
        default=None,
        description="Filter results to the requested date window and exclude records missing a date."
    )


class TavilySearchTool(BaseTool):
    name: str = "tavily_search"
    description: str = (
        "Search the web for current information and return snippets with source URLs. "
        "The tool summarizes supplied arguments in this template; connect it to the Tavily MCP client to execute real searches."
    )
    args_schema = TavilySearchInput

    def _run(self, **kwargs) -> str:
        entries = [f"{key}={value}" for key, value in kwargs.items() if value is not None]
        entries.insert(0, f"query={kwargs.get('query')}")
        return "tavily_search simulated run with arguments:
" + "
".join(entries)
