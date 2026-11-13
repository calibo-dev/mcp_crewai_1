from __future__ import annotations

from typing import List, Optional

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class TavilyExtractInput(BaseModel):
    urls: List[str] = Field(..., description="List of URLs to extract content from.")
    extract_depth: Optional[str] = Field(
        default="basic",
        description="Extraction depth: basic or advanced."
    )
    include_images: Optional[bool] = Field(
        default=False,
        description="Include images from pages."
    )
    format: Optional[str] = Field(
        default="markdown",
        description="Output format: markdown or text."
    )
    include_favicon: Optional[bool] = Field(
        default=False,
        description="Include favicon URLs."
    )
    query: Optional[str] = Field(
        default="",
        description="Query to rerank content chunks by relevance."
    )
    include_usage: Optional[bool] = Field(
        default=None,
        description="Include credit usage."
    )


class TavilyExtractTool(BaseTool):
    name: str = "tavily_extract"
    description: str = (
        "Extract content from the provided URLs. This template version summarizes the request; "
        "connect to the Tavily MCP service to fetch actual content."
    )
    args_schema = TavilyExtractInput

    def _run(self, **kwargs) -> str:
        summary = [f"{key}={value}" for key, value in kwargs.items() if value not in (None, [])]
        return "tavily_extract simulated run with arguments:
" + "
".join(summary)
