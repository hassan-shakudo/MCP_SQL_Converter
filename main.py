"""
Vanna AI NL-to-SQL Service for Dremio with OpenAI-compatible API
Connects to Dremio via REST API and provides OpenAI chat completions endpoint
"""

import os
import json
import time
import uuid
import logging
from typing import List, Optional, Dict, Any, AsyncGenerator, Tuple
from contextlib import asynccontextmanager

import httpx
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

DREMIO_HOST = os.getenv(
    "DREMIO_HOST", "dremio-client.hyperplane-dremio.svc.cluster.local"
)
DREMIO_PORT = int(os.getenv("DREMIO_PORT", "9047"))
DREMIO_USERNAME = os.getenv("DREMIO_USERNAME", "admin")
DREMIO_PASSWORD = os.getenv("DREMIO_PASSWORD", "Shakudo312!")
DREMIO_SSL = os.getenv("DREMIO_SSL", "false").lower() == "true"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "300"))

CHROMA_PATH = os.getenv("CHROMA_PATH", "/tmp/chroma_db")

# Cache removed for accuracy - semantic similarity caused wrong results for date variants

TABLE_BASE = 'minio."mcp-reports"."mcp_parquet"'


# =============================================================================
# Dremio Client
# =============================================================================


class DremioClient:
    """Client for interacting with Dremio REST API"""

    def __init__(self):
        self.base_url = (
            f"{'https' if DREMIO_SSL else 'http'}://{DREMIO_HOST}:{DREMIO_PORT}"
        )
        self.token = None
        self.client = httpx.Client(timeout=60.0)

    def login(self) -> bool:
        """Authenticate with Dremio and get token"""
        try:
            response = self.client.post(
                f"{self.base_url}/apiv2/login",
                json={"userName": DREMIO_USERNAME, "password": DREMIO_PASSWORD},
            )
            response.raise_for_status()
            data = response.json()
            self.token = data.get("token")
            logger.info("Successfully authenticated with Dremio")
            return True
        except Exception as e:
            logger.error(f"Failed to authenticate with Dremio: {e}")
            return False

    def _get_headers(self) -> Dict[str, str]:
        """Get headers with auth token"""
        return {
            "Authorization": f"_dremio{self.token}",
            "Content-Type": "application/json",
        }

    def execute_sql(self, sql: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Execute SQL query and return DataFrame with timing metrics.

        Returns:
            Tuple of (DataFrame, timing_dict) where timing_dict contains:
            - submit_ms: Time to submit job to Dremio
            - poll_ms: Time spent polling for completion
            - poll_count: Number of poll iterations
            - fetch_ms: Time to fetch results
            - total_ms: Total execution time
            - job_id: Dremio job ID for debugging
        """
        timing = {
            "submit_ms": 0,
            "poll_ms": 0,
            "poll_count": 0,
            "fetch_ms": 0,
            "total_ms": 0,
            "job_id": None,
        }
        exec_start = time.time()

        for attempt in range(2):
            if not self.token:
                if not self.login():
                    raise Exception("Failed to authenticate with Dremio")
            try:
                # Submit job
                submit_start = time.time()
                response = self.client.post(
                    f"{self.base_url}/api/v3/sql",
                    headers=self._get_headers(),
                    json={"sql": sql},
                )
                response.raise_for_status()
                job_data = response.json()
                job_id = job_data.get("id")
                timing["submit_ms"] = int((time.time() - submit_start) * 1000)
                timing["job_id"] = job_id

                logger.info(
                    f"[DREMIO] Job submitted: {job_id} ({timing['submit_ms']}ms)"
                )

                # Poll for job completion
                poll_start = time.time()
                max_attempts = 60
                poll_count = 0
                for i in range(max_attempts):
                    poll_count = i + 1
                    status_response = self.client.get(
                        f"{self.base_url}/api/v3/job/{job_id}",
                        headers=self._get_headers(),
                    )
                    status_response.raise_for_status()
                    status_data = status_response.json()
                    job_state = status_data.get("jobState")

                    if job_state == "COMPLETED":
                        break
                    elif job_state in ["FAILED", "CANCELED"]:
                        error_msg = status_data.get("errorMessage", "Unknown error")
                        raise Exception(f"Query failed: {error_msg}")

                    time.sleep(0.5)
                else:
                    raise Exception("Query timeout")

                timing["poll_ms"] = int((time.time() - poll_start) * 1000)
                timing["poll_count"] = poll_count
                logger.info(
                    f"[DREMIO] Job completed: {poll_count} polls, {timing['poll_ms']}ms"
                )

                # Get results
                fetch_start = time.time()
                results_response = self.client.get(
                    f"{self.base_url}/api/v3/job/{job_id}/results",
                    headers=self._get_headers(),
                    params={"offset": 0, "limit": 500},
                )
                results_response.raise_for_status()
                results_data = results_response.json()
                timing["fetch_ms"] = int((time.time() - fetch_start) * 1000)

                # Convert to DataFrame
                rows = results_data.get("rows", [])
                timing["total_ms"] = int((time.time() - exec_start) * 1000)

                logger.info(
                    f"[DREMIO] Execution complete: "
                    f"submit={timing['submit_ms']}ms, "
                    f"poll={timing['poll_ms']}ms ({timing['poll_count']} iterations), "
                    f"fetch={timing['fetch_ms']}ms, "
                    f"total={timing['total_ms']}ms, "
                    f"rows={len(rows)}"
                )

                if rows:
                    return pd.DataFrame(rows), timing
                return pd.DataFrame(), timing

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 401 and attempt == 0:
                    logger.warning("Dremio token expired, re-authenticating...")
                    self.token = None
                    continue
                logger.error(f"Error executing SQL: {e}")
                raise
            except Exception as e:
                logger.error(f"Error executing SQL: {e}")
                raise

        raise Exception("Failed to execute SQL after re-authentication")


# =============================================================================
# Vanna AI NL-to-SQL Engine
# =============================================================================


class VannaNLToSQL:
    """Custom NL-to-SQL engine using OpenAI and ChromaDB with SQL caching"""

    def __init__(self, dremio_client: DremioClient):
        self.dremio = dremio_client
        self.openai = OpenAI(api_key=OPENAI_API_KEY)
        self.model = OPENAI_MODEL

        # Initialize ChromaDB for RAG context (DDL, examples, docs)
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.ddl_collection = self.chroma_client.get_or_create_collection("ddl")
        self.sql_collection = self.chroma_client.get_or_create_collection(
            "sql_examples"
        )
        self.doc_collection = self.chroma_client.get_or_create_collection(
            "documentation"
        )

        logger.info("VannaNLToSQL initialized (cache disabled for accuracy)")

        # Train on schema
        self._train_on_schema()

    def _train_on_schema(self):
        """Train on the Dremio table schema"""
        # DDL for the table
        ddl = """
        Data is stored in MinIO as partitioned Parquet files accessed via Dremio using partition-path syntax.
        Each process type is a separate virtual table. Base path: minio."mcp-reports"."mcp_parquet"

        IMPORTANT: There is NO single flat table. Always use the correct proc-specific path below.

        --- Virtual Table 1: proc=revenue ---
        Path: minio."mcp-reports"."mcp_parquet"."proc=revenue"
        Columns:
        - _meta_resort: VARCHAR - Resort name (snowbowl, brian, lee-canyon, purgatory, sipapu, spider, sandia, nordic, willamette, pajarito)
        - _meta_date: VARCHAR - Date (YYYY-MM-DD format)
        - DepartmentTitle: VARCHAR - Department name
        - revenue: DECIMAL - Actual revenue amount

        --- Virtual Table 2: proc=budget ---
        Path: minio."mcp-reports"."mcp_parquet"."proc=budget"
        Columns:
        - _meta_resort: VARCHAR - Resort name
        - _meta_date: VARCHAR - Date (YYYY-MM-DD format)
        - DepartmentTitle: VARCHAR - Department name
        - Amount: DECIMAL - Budgeted dollar amount
        - Type: VARCHAR - Budget category (Payroll, Revenue, Visits)

        --- Virtual Table 3: proc=weather ---
        Path: minio."mcp-reports"."mcp_parquet"."proc=weather"
        Columns:
        - _meta_resort: VARCHAR - Resort name
        - _meta_date: VARCHAR - Date (YYYY-MM-DD format)
        - date: VARCHAR - Date field
        - snow_24hrs: DECIMAL - Snowfall in last 24 hours (inches)
        - base_depth: DECIMAL - Snow base depth (inches)

        --- Virtual Table 4: proc=visits ---
        Path: minio."mcp-reports"."mcp_parquet"."proc=visits"
        Columns:
        - _meta_resort: VARCHAR - Resort name
        - _meta_date: VARCHAR - Date (YYYY-MM-DD format)
        - Location: VARCHAR - Visitor entry point or location
        - Visits: INTEGER - Number of visits

        --- Virtual Table 5: proc=processed_payroll ---
        Path: minio."mcp-reports"."mcp_parquet"."proc=processed_payroll"
        Columns:
        - _meta_resort: VARCHAR - Resort name
        - _meta_date: VARCHAR - Date (YYYY-MM-DD format)
        - departmentTitle: VARCHAR - Department name (note: lowercase 'd')
        - payroll: DECIMAL - Actual processed payroll amount

        --- Join Rules ---
        - Cross-proc queries (e.g. revenue vs budget/payroll) require explicit JOINs on DepartmentTitle AND _meta_date
        - Always alias tables when joining: e.g. ... AS revenue JOIN ... AS payroll
        - Winter season = November 1 to March 31
        - _meta_date is VARCHAR — use CAST(_meta_date AS DATE) for range comparisons
        - Use date_trunc() for week/month bucketing
        - Use current_date for today's date
        - Resort names are kebab case.
        """

        # Add DDL to ChromaDB if not exists
        existing = self.ddl_collection.get(ids=["main_ddl"])
        if not existing["ids"]:
            self.ddl_collection.add(documents=[ddl], ids=["main_ddl"])

        # Add example SQL queries
        examples = [
            {
                "question": "Give me the Budgeted Revenue of Snowbowl for 1st Feb, 2026.",
                "sql": """SELECT "DepartmentTitle", "Amount"
                  FROM minio."mcp-reports"."mcp_parquet"."proc=budget"
                  WHERE "_meta_resort"='snowbowl'
                  AND "_meta_date"='2026-02-01'
                  AND "Type"='Revenue'""",
            },
            {
                "question": "Give me the Budget for the visits of Snowbowl for 1st Feb, 2026.",
                "sql": """SELECT "DepartmentTitle", "Amount"
                  FROM minio."mcp-reports"."mcp_parquet"."proc=budget"
                  WHERE "_meta_resort"='snowbowl'
                  AND "_meta_date"='2026-02-01'
                  AND "Type"='Visits'""",
            },
            {
                "question": "What is the revenue of Snowbowl for the month?",
                "sql": """SELECT "DepartmentTitle", SUM(revenue) AS TotalRevenue
                  FROM minio."mcp-reports"."mcp_parquet"."proc=revenue"
                  WHERE "_meta_resort"='snowbowl'
                  AND "_meta_date" BETWEEN date_trunc('month', current_date) AND current_date
                  GROUP BY "DepartmentTitle" """,
            },
            {
                "question": "Give me the labour budget of Snowbowl for 1st Feb, 2026.",
                "sql": """SELECT "DepartmentTitle", "Amount"
                  FROM minio."mcp-reports"."mcp_parquet"."proc=budget"
                  WHERE "_meta_resort"='snowbowl'
                  AND "_meta_date"='2026-02-01'
                  AND "Type"='Payroll'""",
            },
            {
                "question": "How much snow and base depth did Snowbowl have on 21st Feb, 2026?",
                "sql": """SELECT SUM("snow_24hrs") AS Snow, SUM("base_depth") AS BaseDepth
                  FROM minio."mcp-reports"."mcp_parquet"."proc=weather"
                  WHERE "_meta_resort"='snowbowl'
                  AND "_meta_date"='2026-02-21'""",
            },
            {
                "question": "What were the visits at Snowbowl on 02/21/26?",
                "sql": """SELECT Location, SUM(Visits) AS Visits
                  FROM minio."mcp-reports"."mcp_parquet"."proc=visits"
                  WHERE "_meta_resort"='snowbowl'
                  AND "_meta_date" = '2026-02-21'
                  GROUP BY Location""",
            },
            {
                "question": "Tell me about the labour of Snowbowl on 2026-02-22.",
                "sql": """SELECT "departmentTitle", "payroll"
                  FROM minio."mcp-reports"."mcp_parquet"."proc=processed_payroll"
                  WHERE "_meta_resort"='snowbowl'
                  AND "_meta_date"='2026-02-22'""",
            },
            {
                "question": "Give me the revenue to payroll ratio of Snowbowl for 6th Feb, 2026.",
                "sql": """SELECT revenue.DepartmentTitle,
                          SUM(revenue.revenue) AS TotalRevenue,
                          SUM(processed_payroll.payroll) AS TotalPayroll,
                          (SUM(revenue.revenue) / SUM(processed_payroll.payroll)) AS RevenueToPayrollRatio
                  FROM minio."mcp-reports"."mcp_parquet"."proc=revenue" AS revenue
                  JOIN minio."mcp-reports"."mcp_parquet"."proc=processed_payroll" AS processed_payroll
                      ON revenue.DepartmentTitle = processed_payroll.departmentTitle
                      AND revenue."_meta_date" = processed_payroll."_meta_date"
                  WHERE revenue."_meta_resort"='snowbowl'
                    AND revenue."_meta_date"='2026-02-06'
                    AND processed_payroll."_meta_resort"='snowbowl'
                  GROUP BY revenue.DepartmentTitle""",
            },
            {
                "question": "Provide the labour to revenue ratio of Snowbowl for the Winter season 2025.",
                "sql": """SELECT revenue."_meta_date",
                          SUM(revenue.revenue) AS TotalRevenue,
                          SUM(processed_payroll.payroll) AS TotalPayroll,
                          (SUM(processed_payroll.payroll) / SUM(revenue.revenue)) AS PayrollToRevenueRatio
                  FROM minio."mcp-reports"."mcp_parquet"."proc=revenue" AS revenue
                  JOIN minio."mcp-reports"."mcp_parquet"."proc=processed_payroll" AS processed_payroll
                      ON revenue.DepartmentTitle = processed_payroll.departmentTitle
                      AND revenue."_meta_date" = processed_payroll."_meta_date"
                  WHERE revenue."_meta_resort"='snowbowl'
                    AND revenue."_meta_date" BETWEEN '2025-11-01' AND '2026-03-31'
                    AND processed_payroll."_meta_resort"='snowbowl'
                  GROUP BY revenue."_meta_date"
                  ORDER BY revenue."_meta_date""",
            },
            {
                "question": "Show me the payroll to revenue variance for Snowbowl on 20th Feb, 2026.",
                "sql": """SELECT revenue."DepartmentTitle",
                          SUM(revenue.revenue) AS TotalRevenue,
                          SUM(processed_payroll.payroll) AS TotalPayroll,
                          (SUM(processed_payroll.payroll) - SUM(revenue.revenue)) AS PayrollRevenueVariance
                  FROM minio."mcp-reports"."mcp_parquet"."proc=revenue" AS revenue
                  JOIN minio."mcp-reports"."mcp_parquet"."proc=processed_payroll" AS processed_payroll
                      ON revenue.DepartmentTitle = processed_payroll.departmentTitle
                      AND revenue."_meta_date" = processed_payroll."_meta_date"
                  WHERE revenue."_meta_resort"='snowbowl'
                    AND revenue."_meta_date"='2026-02-20'
                    AND processed_payroll."_meta_resort"='snowbowl'
                  GROUP BY revenue.DepartmentTitle""",
            },
            {
                "question": "Compare the labour to revenue variance for Snowbowl between 15th Feb and 21st Feb, 2026.",
                "sql": """SELECT revenue."_meta_date",
                          SUM(revenue.revenue) AS TotalRevenue,
                          SUM(processed_payroll.payroll) AS TotalPayroll,
                          (SUM(processed_payroll.payroll) - SUM(revenue.revenue)) AS PayrollRevenueVariance
                  FROM minio."mcp-reports"."mcp_parquet"."proc=revenue" AS revenue
                  JOIN minio."mcp-reports"."mcp_parquet"."proc=processed_payroll" AS processed_payroll
                      ON revenue.DepartmentTitle = processed_payroll.departmentTitle
                      AND revenue."_meta_date" = processed_payroll."_meta_date"
                  WHERE revenue."_meta_resort"='snowbowl'
                    AND revenue."_meta_date" BETWEEN '2026-02-15' AND '2026-02-21'
                    AND processed_payroll."_meta_resort"='snowbowl'
                  GROUP BY revenue."_meta_date"
                  ORDER BY revenue."_meta_date""",
            },
            {
                "question": "Compare last week revenue with the current week for Snowbowl.",
                "sql": """WITH current_week AS (
                      SELECT SUM(revenue) AS TotalRevenue
                      FROM minio."mcp-reports"."mcp_parquet"."proc=revenue"
                      WHERE "_meta_resort"='snowbowl'
                        AND "_meta_date" BETWEEN date_trunc('week', current_date) AND current_date
                  ),
                  prior_week AS (
                      SELECT SUM(revenue) AS TotalRevenue
                      FROM minio."mcp-reports"."mcp_parquet"."proc=revenue"
                      WHERE "_meta_resort"='snowbowl'
                        AND "_meta_date" BETWEEN date_trunc('week', current_date) - interval '7 days'
                        AND current_date - interval '7 days'
                  )
                  SELECT current_week.TotalRevenue AS CurrentWeekRevenue,
                         prior_week.TotalRevenue AS PriorWeekRevenue,
                         (current_week.TotalRevenue - prior_week.TotalRevenue) AS RevenueDifference
                  FROM current_week, prior_week""",
            },
            {
                "question": "Show labour to revenue ratio for Snowbowl this month.",
                "sql": """SELECT revenue."_meta_date",
                          SUM(revenue.revenue) AS TotalRevenue,
                          SUM(payroll.Amount) AS TotalPayroll,
                          (SUM(revenue.revenue)/SUM(payroll.Amount)) AS RevenueToPayrollRatio
                  FROM minio."mcp-reports"."mcp_parquet"."proc=revenue" AS revenue
                  JOIN minio."mcp-reports"."mcp_parquet"."proc=budget" AS payroll
                      ON revenue."DepartmentTitle" = payroll."DepartmentTitle"
                      AND revenue."_meta_date" = payroll."_meta_date"
                  WHERE revenue."_meta_resort"='snowbowl'
                    AND revenue."_meta_date" BETWEEN date_trunc('month', current_date) AND current_date
                    AND payroll."_meta_resort"='snowbowl'
                    AND payroll."Type"='Payroll'
                  GROUP BY revenue."_meta_date"
                  ORDER BY revenue."_meta_date""",
            },
            {
                "question": "Provide revenue to Labour ratio for current Winter season.",
                "sql": """SELECT 
                          SUM(revenue.revenue) AS TotalRevenue,
                          SUM(processed_payroll.payroll) AS TotalPayroll,
                          SUM(revenue.revenue) / NULLIF(SUM(processed_payroll.payroll), 0) AS RevenueToPayrollRatio
                  FROM minio."mcp-reports"."mcp_parquet"."proc=revenue" AS revenue
                  JOIN minio."mcp-reports"."mcp_parquet"."proc=processed_payroll" AS processed_payroll
                      ON revenue."DepartmentTitle" = processed_payroll."departmentTitle"
                      AND revenue."_meta_date" = processed_payroll."_meta_date"
                  WHERE revenue."_meta_resort" = 'snowbowl'
                  AND revenue."_meta_date" BETWEEN 
                      CASE 
                          WHEN EXTRACT(MONTH FROM CURRENT_DATE) BETWEEN 1 AND 3 THEN
                              DATE '2025-11-25'
                          WHEN EXTRACT(MONTH FROM CURRENT_DATE) BETWEEN 4 AND 9 THEN
                              DATE '2025-11-25'
                          ELSE
                              DATE '2026-11-25'
                      END
                      AND CURRENT_DATE
                  AND processed_payroll."_meta_resort" = 'snowbowl'""",
            },
            {
                "question": "Provide revenue for current Winter season.",
                "sql": """SELECT 
                          SUM(revenue.revenue) AS TotalRevenue
                  FROM minio."mcp-reports"."mcp_parquet"."proc=revenue" AS revenue
                  WHERE revenue."_meta_resort" = 'snowbowl'
                  AND revenue."_meta_date" BETWEEN 
                      CASE 
                          WHEN EXTRACT(MONTH FROM CURRENT_DATE) BETWEEN 1 AND 3 THEN 
                              DATE '2025-11-25' 
                          WHEN EXTRACT(MONTH FROM CURRENT_DATE) BETWEEN 4 AND 9 THEN 
                              DATE '2025-11-25' 
                          ELSE 
                              DATE '2026-11-25' 
                      END
                      AND CURRENT_DATE""",
            },
        ]

        for i, ex in enumerate(examples):
            ex_id = f"example_{i}"
            existing = self.sql_collection.get(ids=[ex_id])
            if not existing["ids"]:
                self.sql_collection.add(
                    documents=[f"Question: {ex['question']}\nSQL: {ex['sql']}"],
                    metadatas=[{"question": ex["question"], "sql": ex["sql"]}],
                    ids=[ex_id],
                )

        # Add documentation
        doc = """
        This database contains ski resort operational data from MinIO storage.

        Available resorts (always use exact lowercase kebab-case): snowbowl, brian, lee-canyon, purgatory, sipapu, spider, sandia, nordic, willamette, pajarito

        Data types by proc:
        - proc=revenue: Actual daily revenue by department
        - proc=budget: Planned budget amounts (Payroll, Revenue, Visits types)
        - proc=processed_payroll: Actual processed payroll by department (also called "labour")
        - proc=visits: Daily visitor counts by location/entry point
        - proc=weather: Daily snow conditions (snow_24hrs, base_depth)

        Common departments:
        - Lift Operations, Ski Patrol, Tickets, IT Services, Marketing
        - Mountain G&A, General Administration, Executive
        - Ski School, Rentals, Facilities Maintenance
        - Retail, Cafe/Food service

        Date field is _meta_date (VARCHAR, YYYY-MM-DD). Use CAST(_meta_date AS DATE) for range comparisons.
        Resort names are always lowercase kebab-case. Never use UPPER().
        """

        existing = self.doc_collection.get(ids=["main_doc"])
        if not existing["ids"]:
            self.doc_collection.add(documents=[doc], ids=["main_doc"])

        logger.info("Schema training complete")

    def _get_context(self, question: str) -> str:
        """Get relevant context from ChromaDB"""
        context_parts = []

        # Get DDL
        ddl_results = self.ddl_collection.query(query_texts=[question], n_results=1)
        if ddl_results["documents"]:
            context_parts.append("## Table Schema\n" + ddl_results["documents"][0][0])

        # Get similar SQL examples (reduced from 3 to 1 for faster responses)
        sql_results = self.sql_collection.query(query_texts=[question], n_results=1)
        if sql_results["documents"]:
            context_parts.append("## Similar SQL Examples")
            for doc in sql_results["documents"][0]:
                context_parts.append(doc)

        # Get documentation
        doc_results = self.doc_collection.query(query_texts=[question], n_results=1)
        if doc_results["documents"]:
            context_parts.append("## Documentation\n" + doc_results["documents"][0][0])

        return "\n\n".join(context_parts)

    def generate_sql(self, question: str) -> str:
        """Generate SQL from natural language question"""
        context = self._get_context(question)

        system_prompt = """You are an expert SQL analyst for Dremio. Generate SQL queries based on user questions.

IMPORTANT RULES:
1. Data is stored as partitioned Parquet files. Each process type is a SEPARATE virtual table — never query a single flat table.
2. Always use the correct proc-specific path:
   - Actual revenue:      minio."mcp-reports"."mcp_parquet"."proc=revenue"
   - Budget (planned):    minio."mcp-reports"."mcp_parquet"."proc=budget"
   - Weather:             minio."mcp-reports"."mcp_parquet"."proc=weather"
   - Visits/ticketing:    minio."mcp-reports"."mcp_parquet"."proc=visits"
   - Processed payroll:   minio."mcp-reports"."mcp_parquet"."proc=processed_payroll"
3. Resort names are lowercase kebab-case (e.g. 'snowbowl', 'lee-canyon', 'purgatory'). Never use UPPER().
4. The date field is _meta_date (VARCHAR, YYYY-MM-DD). Use CAST(_meta_date AS DATE) for range comparisons.
5. For cross-proc queries, JOIN on DepartmentTitle AND _meta_date, and always alias each table.
6. "Labour" and "payroll" refer to the same concept — use proc=processed_payroll for actuals, proc=budget with Type='Payroll' for budget.
7. Winter season = November 1 to March 31.
8. Use date_trunc() for week/month bucketing. Use current_date for today.
9. Return ONLY the SQL query, no explanations or markdown fences.
10. Use proper Dremio SQL syntax."""

        user_prompt = f"""Context:
{context}

Question: {question}

Generate the SQL query:"""

        response = self.openai.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=OPENAI_MAX_TOKENS,
        )

        sql = response.choices[0].message.content.strip()

        # Clean up SQL (remove markdown code blocks if present)
        if sql.startswith("```"):
            sql = sql.split("\n", 1)[1] if "\n" in sql else sql[3:]
        if sql.endswith("```"):
            sql = sql[:-3]
        sql = sql.strip()

        return sql

    def ask(self, question: str) -> Dict[str, Any]:
        """
        Process question: generate SQL, execute, and return results.

        Args:
            question: Natural language question

        Returns:
            Dictionary with question, sql, results, result_text, row_count, and detailed timing
        """
        start_time = time.time()
        generation_time_ms = 0
        dremio_timing = {}

        try:
            # Step 1: Generate SQL via LLM
            generation_start = time.time()
            sql = self.generate_sql(question)
            generation_time_ms = int((time.time() - generation_start) * 1000)
            logger.info(f"Generated SQL in {generation_time_ms}ms: {sql[:100]}...")

            # Step 2: Execute SQL on Dremio
            df, dremio_timing = self.dremio.execute_sql(sql)

            # Format results
            if df.empty:
                result_text = "No results found."
            else:
                result_text = df.to_markdown(index=False)

            total_time_ms = int((time.time() - start_time) * 1000)

            # Detailed timing summary
            logger.info(
                f"[QUERY COMPLETE] "
                f"Total: {total_time_ms}ms | "
                f"LLM generation: {generation_time_ms}ms | "
                f"Dremio: {dremio_timing.get('total_ms', 0)}ms "
                f"(submit={dremio_timing.get('submit_ms', 0)}ms, "
                f"poll={dremio_timing.get('poll_ms', 0)}ms/{dremio_timing.get('poll_count', 0)} iterations, "
                f"fetch={dremio_timing.get('fetch_ms', 0)}ms) | "
                f"Rows: {len(df)}"
            )

            return {
                "question": question,
                "sql": sql,
                "results": df.to_dict(orient="records"),
                "result_text": result_text,
                "row_count": len(df),
                "timing": {
                    "total_ms": total_time_ms,
                    "llm_generation_ms": generation_time_ms,
                    "dremio_total_ms": dremio_timing.get("total_ms", 0),
                    "dremio_submit_ms": dremio_timing.get("submit_ms", 0),
                    "dremio_poll_ms": dremio_timing.get("poll_ms", 0),
                    "dremio_poll_count": dremio_timing.get("poll_count", 0),
                    "dremio_fetch_ms": dremio_timing.get("fetch_ms", 0),
                    "dremio_job_id": dremio_timing.get("job_id"),
                },
            }

        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return {
                "question": question,
                "error": str(e),
            }


# =============================================================================
# OpenAI-Compatible API Models
# =============================================================================


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "vanna-dremio"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: UsageInfo


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str


class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


# =============================================================================
# FastAPI Application
# =============================================================================

# Global instances
dremio_client: Optional[DremioClient] = None
vanna_engine: Optional[VannaNLToSQL] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup"""
    global dremio_client, vanna_engine

    logger.info("Starting Vanna Dremio NL-to-SQL Service...")

    # Initialize Dremio client
    dremio_client = DremioClient()
    if not dremio_client.login():
        logger.warning("Failed to connect to Dremio - will retry on first request")

    # Initialize Vanna engine
    vanna_engine = VannaNLToSQL(dremio_client)

    logger.info("Service initialized successfully")
    yield

    logger.info("Shutting down...")


app = FastAPI(
    title="Vanna Dremio NL-to-SQL Service",
    description="Natural language to SQL translation for Dremio with OpenAI-compatible API",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# API Endpoints
# =============================================================================


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible)"""
    return ModelsResponse(
        data=[
            ModelInfo(id="vanna-dremio", created=int(time.time()), owned_by="shakudo"),
            ModelInfo(id="vanna-nl2sql", created=int(time.time()), owned_by="shakudo"),
        ]
    )


@app.get("/v1/models/{model_id}")
async def get_model(model_id: str):
    """Get model info (OpenAI-compatible)"""
    return ModelInfo(id=model_id, created=int(time.time()), owned_by="shakudo")


async def generate_streaming_response(
    question: str, model: str, completion_id: str
) -> AsyncGenerator[str, None]:
    """Generate streaming response in OpenAI format"""
    created = int(time.time())

    # Initial chunk with role
    initial_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": ""},
                "finish_reason": None,
            }
        ],
    }
    yield f"data: {json.dumps(initial_chunk)}\n\n"

    try:
        # Process the question
        result = vanna_engine.ask(question)

        if "error" in result:
            response_text = f"I encountered an error: {result['error']}"
        else:
            # Format nice response
            response_text = (
                f"I've translated your question into SQL and executed it.\n\n"
            )
            response_text += f"**SQL Query:**\n```sql\n{result['sql']}\n```\n\n"
            response_text += f"**Results ({result['row_count']} rows):**\n"
            response_text += result["result_text"]

        # Stream the response in chunks
        chunk_size = 20
        for i in range(0, len(response_text), chunk_size):
            chunk_text = response_text[i : i + chunk_size]
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": chunk_text},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

    except Exception as e:
        error_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": f"\n\nError: {str(e)}"},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"

    # Final chunk
    final_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint"""

    # Get the last user message
    user_messages = [m for m in request.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message provided")

    question = user_messages[-1].content
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

    if request.stream:
        return StreamingResponse(
            generate_streaming_response(question, request.model, completion_id),
            media_type="text/event-stream",
        )

    # Non-streaming response
    try:
        result = vanna_engine.ask(question)

        if "error" in result:
            response_text = f"I encountered an error: {result['error']}"
        else:
            response_text = (
                f"I've translated your question into SQL and executed it.\n\n"
            )
            response_text += f"**SQL Query:**\n```sql\n{result['sql']}\n```\n\n"
            response_text += f"**Results ({result['row_count']} rows):**\n"
            response_text += result["result_text"]

        return ChatCompletionResponse(
            id=completion_id,
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response_text),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(
                prompt_tokens=len(question.split()),
                completion_tokens=len(response_text.split()),
                total_tokens=len(question.split()) + len(response_text.split()),
            ),
        )

    except Exception as e:
        logger.error(f"Error in chat completion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
async def direct_query(request: dict):
    """Direct query endpoint for testing"""
    question = request.get("question")
    if not question:
        raise HTTPException(status_code=400, detail="Question required")

    result = vanna_engine.ask(question)
    return result


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8787"))
    uvicorn.run(app, host="0.0.0.0", port=port)
