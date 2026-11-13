from __future__ import annotations

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class CustomerSqlQueryInput(BaseModel):
    sql: str = Field(..., description="Single SQL query to execute.")


class CustomerSqlQueryTool(BaseTool):
    name: str = "customer_sql_query"
    description: str = (
        "Use this tool to query Databricks customer profiles in DEV_DB.SALES_RAG_DEMO.DIM_CUSTOMER. "
        "The tool currently echoes the requested SQL; integrate the MCP connector to execute queries."
    )
    args_schema = CustomerSqlQueryInput

    def _run(self, sql: str) -> str:
        cleaned = (sql or "").strip()
        if not cleaned:
            return "customer_sql_query invoked with an empty SQL string. Provide a SELECT query to retrieve customer information."
        return f"customer_sql_query would execute SQL: {cleaned}"
