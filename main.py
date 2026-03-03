"""
Vanna AI NL-to-SQL Service for Dremio with OpenAI-compatible API
Connects to Dremio via REST API and provides OpenAI chat completions endpoint
"""

import os
import json
import time
import uuid
import hashlib
import logging
from typing import List, Optional, Dict, Any, AsyncGenerator, Tuple
from contextlib import asynccontextmanager

import httpx
import pandas as pd
import sqlparse
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
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2")

CHROMA_PATH = os.getenv("CHROMA_PATH", "/tmp/chroma_db")

# SQL Cache Configuration
CACHE_SIMILARITY_THRESHOLD = float(
    os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.05")
)  # cosine distance < 0.05 = similarity > 0.95
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"

TABLE_BASE = 'minio."mcp-reports"."mcp_parquet"'


# =============================================================================
# SQL Cache Utilities
# =============================================================================


def normalize_sql(sql: str) -> str:
    """
    Normalize SQL for consistent hashing and deduplication.

    Handles:
    - Whitespace normalization
    - Keyword case normalization (uppercase)
    - Comment removal
    - Consistent formatting

    Args:
        sql: Raw SQL string

    Returns:
        Normalized SQL string suitable for hashing
    """
    if not sql or not sql.strip():
        return ""

    try:
        # Use sqlparse for robust SQL normalization
        normalized = sqlparse.format(
            sql,
            keyword_case="upper",  # SELECT, FROM, WHERE -> uppercase
            identifier_case=None,  # Keep identifier case (table/column names)
            strip_comments=True,  # Remove -- and /* */ comments
            strip_whitespace=True,  # Remove extra whitespace
            reindent=False,  # Don't change indentation structure
        )
        # Additional whitespace normalization: collapse all whitespace to single spaces
        return " ".join(normalized.split())
    except Exception as e:
        # Fallback: basic normalization if sqlparse fails
        logger.warning(f"SQL normalization fallback due to: {e}")
        return " ".join(sql.upper().split())


def hash_string(s: str) -> str:
    """
    Generate a SHA-256 hash of a string.

    Args:
        s: Input string

    Returns:
        Hexadecimal hash string
    """
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def question_hash(question: str) -> str:
    """
    Generate hash for exact question matching (Level 1 cache).

    Args:
        question: User's question

    Returns:
        Hash of the lowercased, whitespace-normalized question
    """
    # Normalize: lowercase and collapse whitespace for exact matching
    normalized = " ".join(question.lower().split())
    return hash_string(normalized)


def sql_hash(sql: str) -> str:
    """
    Generate hash for SQL deduplication.

    Args:
        sql: SQL query

    Returns:
        Hash of the normalized SQL
    """
    return hash_string(normalize_sql(sql))


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

    def execute_sql(self, sql: str) -> pd.DataFrame:
        """Execute SQL query and return DataFrame, with automatic token refresh on 401"""
        for attempt in range(2):
            if not self.token:
                if not self.login():
                    raise Exception("Failed to authenticate with Dremio")
            try:
                # Submit job
                response = self.client.post(
                    f"{self.base_url}/api/v3/sql",
                    headers=self._get_headers(),
                    json={"sql": sql},
                )
                response.raise_for_status()
                job_data = response.json()
                job_id = job_data.get("id")

                # Poll for job completion
                max_attempts = 60
                for _ in range(max_attempts):
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

                # Get results
                results_response = self.client.get(
                    f"{self.base_url}/api/v3/job/{job_id}/results",
                    headers=self._get_headers(),
                    params={"offset": 0, "limit": 500},
                )
                results_response.raise_for_status()
                results_data = results_response.json()

                # Convert to DataFrame
                rows = results_data.get("rows", [])
                if rows:
                    return pd.DataFrame(rows)
                return pd.DataFrame()

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
        self.cache_enabled = CACHE_ENABLED
        self.cache_similarity_threshold = CACHE_SIMILARITY_THRESHOLD

        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.ddl_collection = self.chroma_client.get_or_create_collection("ddl")
        self.sql_collection = self.chroma_client.get_or_create_collection(
            "sql_examples"
        )
        self.doc_collection = self.chroma_client.get_or_create_collection(
            "documentation"
        )

        # Initialize SQL cache collection
        # Uses ChromaDB's default embedding function for semantic similarity
        self.query_cache = self.chroma_client.get_or_create_collection(
            name="query_cache",
            metadata={"hnsw:space": "cosine"},  # cosine distance for semantic matching
        )

        logger.info(
            f"SQL Cache initialized (enabled={self.cache_enabled}, threshold={self.cache_similarity_threshold})"
        )
        cache_count = self.query_cache.count()
        if cache_count > 0:
            logger.info(f"Loaded {cache_count} cached queries from persistent storage")

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

    # =========================================================================
    # SQL Cache Methods
    # =========================================================================

    def _cache_lookup(self, question: str) -> Tuple[Optional[str], str]:
        """
        Look up question in SQL cache using two-level strategy:

        Level 1 (L1): Exact match on question hash (fastest)
        Level 3 (L3): Semantic similarity using embeddings (catches paraphrases)

        Args:
            question: User's natural language question

        Returns:
            Tuple of (cached_sql, cache_hit_type) where:
            - cached_sql is the SQL string if found, None otherwise
            - cache_hit_type is "exact", "semantic", or "miss"
        """
        if not self.cache_enabled:
            logger.debug("Cache disabled, skipping lookup")
            return None, "disabled"

        try:
            # Level 1: Exact question hash match (fastest path)
            q_hash = question_hash(question)
            exact_results = self.query_cache.get(
                where={"question_hash": q_hash}, include=["metadatas"]
            )

            if exact_results and exact_results.get("ids"):
                cached_sql = exact_results["metadatas"][0].get("sql")
                if cached_sql:
                    logger.info(f"[CACHE HIT - EXACT] Question hash: {q_hash[:16]}...")
                    return cached_sql, "exact"

            # Level 3: Semantic similarity search
            # ChromaDB returns cosine distance (0 = identical, 2 = opposite)
            # threshold 0.05 means similarity > 0.95
            semantic_results = self.query_cache.query(
                query_texts=[question],
                n_results=1,
                include=["metadatas", "distances", "documents"],
            )

            if (
                semantic_results
                and semantic_results.get("ids")
                and semantic_results["ids"][0]
                and semantic_results.get("distances")
                and semantic_results["distances"][0]
            ):
                distance = semantic_results["distances"][0][0]

                if distance < self.cache_similarity_threshold:
                    cached_sql = semantic_results["metadatas"][0][0].get("sql")
                    cached_question = (
                        semantic_results["documents"][0][0]
                        if semantic_results.get("documents")
                        else "unknown"
                    )

                    if cached_sql:
                        similarity = 1 - distance  # Convert distance to similarity
                        logger.info(
                            f"[CACHE HIT - SEMANTIC] Distance: {distance:.4f}, "
                            f"Similarity: {similarity:.2%}, "
                            f"Matched question: '{cached_question[:50]}...'"
                        )
                        return cached_sql, "semantic"
                else:
                    logger.debug(
                        f"[CACHE MISS - SEMANTIC] Best distance: {distance:.4f} "
                        f"(threshold: {self.cache_similarity_threshold})"
                    )

            logger.info(
                f"[CACHE MISS] No match found for question: '{question[:50]}...'"
            )
            return None, "miss"

        except Exception as e:
            logger.error(f"Cache lookup error: {e}")
            return None, "error"

    def _cache_store(self, question: str, sql: str) -> bool:
        """
        Store question-SQL mapping in cache with deduplication.

        Deduplication: If the normalized SQL already exists in cache from a
        different question, we skip storing to avoid redundancy.

        Args:
            question: User's natural language question
            sql: Generated SQL query

        Returns:
            True if stored successfully, False if deduplicated or error
        """
        if not self.cache_enabled:
            return False

        if not sql or not sql.strip():
            logger.warning("Attempted to cache empty SQL, skipping")
            return False

        try:
            q_hash = question_hash(question)
            s_hash = sql_hash(sql)

            # Check for SQL deduplication: same SQL from different question
            existing_sql = self.query_cache.get(
                where={"sql_hash": s_hash}, include=["metadatas", "documents"]
            )

            if existing_sql and existing_sql.get("ids"):
                existing_question = (
                    existing_sql["documents"][0]
                    if existing_sql.get("documents")
                    else "unknown"
                )
                logger.info(
                    f"[CACHE DEDUP] SQL already cached from different question. "
                    f"Existing: '{existing_question[:50]}...', "
                    f"New: '{question[:50]}...'"
                )
                return False

            # Check if this exact question already cached (shouldn't happen, but safety check)
            existing_question_entry = self.query_cache.get(
                where={"question_hash": q_hash}, include=["metadatas"]
            )

            if existing_question_entry and existing_question_entry.get("ids"):
                logger.debug(f"Question already cached, skipping: {q_hash[:16]}...")
                return False

            # Generate unique ID for this cache entry
            cache_id = f"cache_{q_hash[:16]}_{int(time.time())}"

            # Store in cache
            self.query_cache.add(
                documents=[question],  # Question text for embedding/semantic search
                metadatas=[
                    {
                        "question_hash": q_hash,
                        "sql": sql,
                        "sql_hash": s_hash,
                        "created_at": int(time.time()),
                        "question_preview": question[:100],  # For logging/debugging
                    }
                ],
                ids=[cache_id],
            )

            logger.info(
                f"[CACHE STORE] New entry cached. "
                f"ID: {cache_id}, "
                f"Question: '{question[:50]}...', "
                f"SQL hash: {s_hash[:16]}..."
            )
            return True

        except Exception as e:
            logger.error(f"Cache store error: {e}")
            return False

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics for monitoring.

        Returns:
            Dictionary with cache statistics
        """
        try:
            count = self.query_cache.count()
            return {
                "enabled": self.cache_enabled,
                "entry_count": count,
                "similarity_threshold": self.cache_similarity_threshold,
                "chroma_path": CHROMA_PATH,
            }
        except Exception as e:
            return {"error": str(e)}

    def clear_cache(self) -> bool:
        """
        Clear all cached queries. Use with caution.

        Returns:
            True if cleared successfully
        """
        try:
            # Get all IDs and delete them
            all_entries = self.query_cache.get()
            if all_entries and all_entries.get("ids"):
                self.query_cache.delete(ids=all_entries["ids"])
                logger.info(
                    f"[CACHE CLEAR] Deleted {len(all_entries['ids'])} cache entries"
                )
            return True
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return False

    def _get_context(self, question: str) -> str:
        """Get relevant context from ChromaDB"""
        context_parts = []

        # Get DDL
        ddl_results = self.ddl_collection.query(query_texts=[question], n_results=1)
        if ddl_results["documents"]:
            context_parts.append("## Table Schema\n" + ddl_results["documents"][0][0])

        # Get similar SQL examples
        sql_results = self.sql_collection.query(query_texts=[question], n_results=3)
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
        Process question: check cache, generate SQL if needed, execute, and return results.

        Cache Strategy (SQL-only, data always fresh from Dremio):
        1. Check cache for existing SQL (exact match, then semantic similarity)
        2. If cache hit: use cached SQL, execute on Dremio for fresh data
        3. If cache miss: generate SQL via LLM, cache it, execute on Dremio

        Args:
            question: Natural language question

        Returns:
            Dictionary with question, sql, results, result_text, row_count, and cache_hit
        """
        start_time = time.time()
        cache_hit_type = "miss"
        sql = None

        try:
            # Step 1: Check cache for SQL
            cached_sql, cache_hit_type = self._cache_lookup(question)

            if cached_sql:
                sql = cached_sql
                logger.info(f"Using cached SQL ({cache_hit_type} hit)")
            else:
                # Step 2: Generate SQL via LLM (cache miss)
                generation_start = time.time()
                sql = self.generate_sql(question)
                generation_time = time.time() - generation_start
                logger.info(f"Generated SQL in {generation_time:.2f}s: {sql[:100]}...")

                # Step 3: Cache the newly generated SQL
                self._cache_store(question, sql)

            # Step 4: Execute SQL on Dremio (always, for fresh data)
            execution_start = time.time()
            df = self.dremio.execute_sql(sql)
            execution_time = time.time() - execution_start
            logger.info(
                f"Dremio execution completed in {execution_time:.2f}s, {len(df)} rows returned"
            )

            # Format results
            if df.empty:
                result_text = "No results found."
            else:
                result_text = df.to_markdown(index=False)

            total_time = time.time() - start_time
            logger.info(
                f"[QUERY COMPLETE] Total: {total_time:.2f}s, "
                f"Cache: {cache_hit_type}, "
                f"Rows: {len(df)}"
            )

            return {
                "question": question,
                "sql": sql,
                "results": df.to_dict(orient="records"),
                "result_text": result_text,
                "row_count": len(df),
                "cache_hit": cache_hit_type
                if cache_hit_type in ["exact", "semantic"]
                else None,
                "timing": {
                    "total_seconds": round(total_time, 2),
                    "cache_hit_type": cache_hit_type,
                },
            }

        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return {
                "question": question,
                "error": str(e),
                "cache_hit": cache_hit_type
                if cache_hit_type in ["exact", "semantic"]
                else None,
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


@app.get("/cache/stats")
async def cache_stats():
    """Get SQL cache statistics"""
    if vanna_engine:
        return vanna_engine.get_cache_stats()
    return {"error": "Engine not initialized"}


@app.delete("/cache/clear")
async def cache_clear():
    """Clear the SQL cache (use with caution)"""
    if vanna_engine:
        success = vanna_engine.clear_cache()
        return {
            "success": success,
            "message": "Cache cleared" if success else "Failed to clear cache",
        }
    return {"error": "Engine not initialized"}


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
