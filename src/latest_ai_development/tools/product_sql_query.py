from __future__ import annotations

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class ProductSqlQueryInput(BaseModel):
    sql: str = Field(..., description="Single SQL query to execute.")


class ProductSqlQueryTool(BaseTool):
    name: str = "product_sql_query"
    description: str = (
        "Use this tool to query Databricks product information in DEV_DB.SALES_RAG_DEMO.DIM_PRODUCT. "
        "The tool currently echoes the requested SQL; integrate the MCP connector to execute queries."
    )
    args_schema = ProductSqlQueryInput

    def _run(self, sql: str) -> str:
        cleaned = (sql or "").strip()
        if not cleaned:
            return "product_sql_query invoked with an empty SQL string. Provide a SELECT query to retrieve product data."
        return f"product_sql_query would execute SQL: {cleaned}"
