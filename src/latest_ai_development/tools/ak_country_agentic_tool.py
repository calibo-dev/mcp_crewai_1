from __future__ import annotations

import re
from typing import Optional, Type

import pandas as pd
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from .snowflake_connection import SnowflakeConnection


SCHEMA_CONTEXT = """Account  : Not provided
Database : Not provided
Schema   : Not provided
Warehouse: Not provided

Static table context:
  No static table schema is provided by the supported tool payload.
  Runtime metadata discovery is enabled. First call the tool without a query to inspect authorized Snowflake databases, schemas, tables, and columns.
  Use the tool purpose and guidelines to choose relevant metadata, then call the tool again with the required SQL query."""

TOOL_PURPOSE = """Enable agents to inspect the metadata of the ak_country table and execute validated read-only queries against the same table to retrieve country and geographic reference information required for downstream analysis and response generation."""

TOOL_GUIDELINES = """Usage guidelines:
- - Use this tool exclusively for the ak_country table.
- Retrieve metadata such as column names and data types when required.
- Use the latest ak_country metadata to support accurate SQL generation.
- Execute only read-only SELECT queries against ak_country.
- Support filtering, sorting, DISTINCT, grouping, counting, and aggregation.
- Validate that queries reference only ak_country and valid columns.
- Use country, ISO code, region, sub-region, intermediate-region, and related code fields as needed.
- Return metadata and query results required for downstream analysis and response generation.

Non-usage guidelines:
- - Do not access any table, view, or data source other than ak_country.
- Do not execute queries that reference other tables.
- Do not perform joins with any other table or data source.
- Do not execute INSERT, UPDATE, DELETE, MERGE, TRUNCATE, CREATE, ALTER, DROP, GRANT, or REVOKE.
- Do not execute queries using columns that do not exist in ak_country.
- Do not retrieve metadata for unrelated tables or schemas.
- Do not use this tool for sales, customer, product, employee, financial, or other unrelated data.
- Do not fabricate or infer values when the requested data is unavailable."""

TOOL_ENV_PREFIX = "AK_COUNTRY_AGENTIC_TOOL"

SNOWFLAKE_CONNECTION_DEFAULTS = {}


class AkCountryAgenticToolInput(BaseModel):
    query: Optional[str] = Field(
        default=None,
        description=(
            "Optional Snowflake SQL query. "
            "Leave empty to discover authorized Snowflake metadata before writing SQL."
        )
    )
    row_limit: Optional[int] = Field(
        default=1000,
        description=(
            "Optional maximum rows to return. When omitted, metadata discovery and SQL "
            "execution are not limited by this tool."
        )
    )
    metadata_scope: Optional[str] = Field(
        default="auto",
        description=(
            "Metadata discovery scope when query is empty: auto, databases, schemas, tables, or columns."
        )
    )
    database: Optional[str] = Field(
        default=None,
        description="Optional Snowflake database name to inspect for schemas, tables, or columns."
    )
    schema: Optional[str] = Field(
        default=None,
        description="Optional Snowflake schema name to inspect for tables or columns."
    )
    table_name: Optional[str] = Field(
        default=None,
        description="Optional table name to inspect columns for."
    )
    search_term: Optional[str] = Field(
        default=None,
        description="Optional case-insensitive term used to filter discovered metadata names."
    )


class AkCountryAgenticTool(BaseTool):
    name: str = "ak_country_agentic_tool"
    description: str = (
        "Use this tool to discover authorized Snowflake metadata and run Snowflake SQL. "
        "When table structure is not known, first call the tool without a query and use "
        "metadata_scope/search arguments to inspect databases, schemas, tables, and columns. "
        "After selecting relevant objects, call the tool again with the required SQL query. "
        "SQL is executed as provided unless row_limit is passed for a SELECT/WITH query. "
        "Snowflake permissions determine what succeeds. "
        "Prefer fully qualified table names in the format DATABASE.SCHEMA.TABLE.\n\n"
        f"Purpose - primary guidance:\n{TOOL_PURPOSE}\n\n"
        f"Usage and non-usage guidelines - primary guidance:\n{TOOL_GUIDELINES}\n\n"
        "Description - secondary context:\n"
        "Provides read-only access to the ak_country table, including metadata retrieval and SQL execution, to return country, ISO code, region, sub-region, intermediate region, and related geographic reference data.\n\n"
        f"Static Context:\n{SCHEMA_CONTEXT}\n\n"
    )
    args_schema: Type[BaseModel] = AkCountryAgenticToolInput

    def _run(
        self,
        query: Optional[str] = None,
        row_limit: Optional[int] = 1000,
        metadata_scope: Optional[str] = "auto",
        database: Optional[str] = None,
        schema: Optional[str] = None,
        table_name: Optional[str] = None,
        search_term: Optional[str] = None,
    ) -> str:
        query = (query or "").strip()

        normalized_row_limit = self._normalize_row_limit(row_limit)

        if not query:
            return self._discover_metadata(
                row_limit=normalized_row_limit,
                metadata_scope=metadata_scope,
                database=database,
                schema=schema,
                table_name=table_name,
                search_term=search_term,
            )

        return self._execute_query(query=query, row_limit=normalized_row_limit)

    def _execute_query(self, query: str, row_limit: Optional[int]) -> str:
        cursor = None

        try:
            conn = SnowflakeConnection.get_connection(
                env_prefix=TOOL_ENV_PREFIX,
                defaults=SNOWFLAKE_CONNECTION_DEFAULTS,
            )
            cursor = conn.cursor()

            if row_limit is not None and self._should_apply_row_limit(query):
                query = f"{query.rstrip(';')} LIMIT {row_limit};"

            cursor.execute(query)
            if cursor.description is None:
                affected = cursor.rowcount
                affected_text = str(affected) if affected is not None and affected >= 0 else "unknown"
                return f"Query executed successfully. Rows affected: {affected_text}."

            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]

            if not rows:
                return "Query executed successfully - 0 rows returned."

            df = pd.DataFrame(rows, columns=columns)
            return (
                f"{len(rows)} row(s) returned:\n\n"
                f"{df.to_string(index=False)}\n\n"
                f"Columns: {', '.join(columns)}"
            )

        except Exception as exc:
            return f"Query error: {type(exc).__name__}: {str(exc)}"

        finally:
            if cursor is not None:
                cursor.close()

    def _discover_metadata(
        self,
        *,
        row_limit: Optional[int],
        metadata_scope: Optional[str],
        database: Optional[str],
        schema: Optional[str],
        table_name: Optional[str],
        search_term: Optional[str],
    ) -> str:
        cursor = None
        database = (database or SNOWFLAKE_CONNECTION_DEFAULTS.get("database") or "").strip()
        schema = (schema or SNOWFLAKE_CONNECTION_DEFAULTS.get("schema") or "").strip()
        table_name = (table_name or "").strip()
        search_term = (search_term or "").strip()
        scope = (metadata_scope or "auto").strip().lower()

        if scope == "auto":
            if table_name:
                scope = "columns"
            elif database and schema:
                scope = "tables"
            elif database:
                scope = "schemas"
            else:
                scope = "databases"

        try:
            conn = SnowflakeConnection.get_connection(
                env_prefix=TOOL_ENV_PREFIX,
                defaults=SNOWFLAKE_CONNECTION_DEFAULTS,
            )
            cursor = conn.cursor()

            if scope == "databases":
                cursor.execute("SHOW DATABASES")
                return self._format_cursor_rows(
                    cursor,
                    row_limit=row_limit,
                    title="Authorized Snowflake databases",
                    search_term=search_term,
                )

            if scope == "schemas":
                if database:
                    sql = (
                        "SELECT CATALOG_NAME AS DATABASE_NAME, SCHEMA_NAME, COMMENT "
                        f"FROM {self._quote_identifier(database)}.INFORMATION_SCHEMA.SCHEMATA "
                    )
                    if search_term:
                        sql += (
                            "WHERE UPPER(SCHEMA_NAME) LIKE "
                            f"{self._sql_like_literal(search_term)} "
                        )
                    sql += "ORDER BY SCHEMA_NAME"
                    if row_limit is not None:
                        sql += f" LIMIT {row_limit}"
                    cursor.execute(sql)
                    return self._format_cursor_rows(
                        cursor,
                        row_limit=row_limit,
                        title=f"Snowflake schemas in {database}",
                    )

                cursor.execute("SHOW SCHEMAS IN ACCOUNT")
                return self._format_cursor_rows(
                    cursor,
                    row_limit=row_limit,
                    title="Authorized Snowflake schemas",
                    search_term=search_term,
                )

            if scope == "tables":
                if database:
                    sql = (
                        "SELECT TABLE_CATALOG AS DATABASE_NAME, TABLE_SCHEMA, TABLE_NAME, TABLE_TYPE, COMMENT "
                        f"FROM {self._quote_identifier(database)}.INFORMATION_SCHEMA.TABLES "
                    )
                    filters = []
                    if schema:
                        filters.append(
                            f"UPPER(TABLE_SCHEMA) = {self._sql_literal(schema.upper())}"
                        )
                    if search_term:
                        filters.append(
                            "(UPPER(TABLE_NAME) LIKE "
                            f"{self._sql_like_literal(search_term)} OR UPPER(COMMENT) LIKE "
                            f"{self._sql_like_literal(search_term)})"
                        )
                    if filters:
                        sql += "WHERE " + " AND ".join(filters) + " "
                    sql += "ORDER BY TABLE_SCHEMA, TABLE_NAME"
                    if row_limit is not None:
                        sql += f" LIMIT {row_limit}"
                    cursor.execute(sql)
                    return self._format_cursor_rows(
                        cursor,
                        row_limit=row_limit,
                        title=f"Snowflake tables in {database}{'.' + schema if schema else ''}",
                    )

                cursor.execute("SHOW TABLES IN ACCOUNT")
                return self._format_cursor_rows(
                    cursor,
                    row_limit=row_limit,
                    title="Authorized Snowflake tables",
                    search_term=search_term,
                )

            if scope == "columns":
                if not database:
                    return (
                        "Column discovery requires a database. "
                        "First discover databases/tables, then call metadata_scope='columns' "
                        "with database, schema, and optional table_name."
                    )

                sql = (
                    "SELECT TABLE_CATALOG AS DATABASE_NAME, TABLE_SCHEMA, TABLE_NAME, "
                    "COLUMN_NAME, DATA_TYPE, IS_NULLABLE, COMMENT "
                    f"FROM {self._quote_identifier(database)}.INFORMATION_SCHEMA.COLUMNS "
                )
                filters = []
                if schema:
                    filters.append(f"UPPER(TABLE_SCHEMA) = {self._sql_literal(schema.upper())}")
                if table_name:
                    filters.append(f"UPPER(TABLE_NAME) = {self._sql_literal(table_name.upper())}")
                if search_term:
                    filters.append(
                        "(UPPER(TABLE_NAME) LIKE "
                        f"{self._sql_like_literal(search_term)} OR UPPER(COLUMN_NAME) LIKE "
                        f"{self._sql_like_literal(search_term)} OR UPPER(COMMENT) LIKE "
                        f"{self._sql_like_literal(search_term)})"
                    )
                if filters:
                    sql += "WHERE " + " AND ".join(filters) + " "
                sql += "ORDER BY TABLE_SCHEMA, TABLE_NAME, ORDINAL_POSITION"
                if row_limit is not None:
                    sql += f" LIMIT {row_limit}"
                cursor.execute(sql)
                return self._format_cursor_rows(
                    cursor,
                    row_limit=row_limit,
                    title=f"Snowflake columns in {database}",
                )

            return "Error: metadata_scope must be one of auto, databases, schemas, tables, or columns."

        except Exception as exc:
            return f"Metadata discovery error: {type(exc).__name__}: {str(exc)}"

        finally:
            if cursor is not None:
                cursor.close()

    def _format_cursor_rows(
        self,
        cursor,
        *,
        row_limit: Optional[int],
        title: str,
        search_term: str = "",
    ) -> str:
        rows = cursor.fetchmany(row_limit) if row_limit is not None else cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]

        if search_term:
            term = search_term.lower()
            rows = [
                row
                for row in rows
                if term in " ".join("" if value is None else str(value) for value in row).lower()
            ]

        if not rows:
            return f"{title}: 0 metadata row(s) found."

        df = pd.DataFrame(rows, columns=columns)
        return (
            f"{title}: {len(rows)} metadata row(s) found:\n\n"
            f"{df.to_string(index=False)}\n\n"
            f"Columns: {', '.join(columns)}"
        )

    def _quote_identifier(self, identifier: str) -> str:
        parts = [part.strip() for part in str(identifier or "").split(".") if part.strip()]
        return ".".join(self._quote_identifier_part(part) for part in parts)

    def _quote_identifier_part(self, identifier: str) -> str:
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", identifier):
            return identifier
        return f'"{identifier.replace(chr(34), chr(34) * 2)}"'

    def _sql_literal(self, value: str) -> str:
        return "'" + str(value or "").replace("'", "''") + "'"

    def _sql_like_literal(self, value: str) -> str:
        return self._sql_literal(f"%{str(value or '').upper()}%")

    def _normalize_row_limit(self, row_limit: Optional[int]) -> Optional[int]:
        if row_limit is None:
            return None
        try:
            return min(max(int(row_limit), 1), 5000)
        except (TypeError, ValueError):
            return None

    def _should_apply_row_limit(self, query: str) -> bool:
        normalized = query.lstrip().upper()
        if not normalized.startswith(("SELECT", "WITH")):
            return False
        return re.search(r"\bLIMIT\s+\d+\b", normalized) is None


class AkCountryAgenticToolTool(AkCountryAgenticTool):
    """Backward-compatible alias expected by runtime wiring."""
    pass
