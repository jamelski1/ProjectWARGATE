"""
CVE Vector Store for ProjectWARGATE
====================================

Fetches CVE data from the NVD (National Vulnerability Database) API,
embeds it into a ChromaDB vector store, and provides a query interface
for the cyberintel retriever tool.

Usage:
    # Ingest CVEs (run once or periodically to refresh):
    python cve_vectorstore.py --ingest --days 120

    # Query from Python:
    from cve_vectorstore import cve_query
    results = cve_query("remote code execution in Apache")
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from textwrap import dedent

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CVE_DB_DIR = Path(__file__).parent / "cve_db"

# ---------------------------------------------------------------------------
# NVD API helpers
# ---------------------------------------------------------------------------
NVD_API_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
NVD_PAGE_SIZE = 200  # max allowed by the NVD API v2
NVD_REQUEST_DELAY = 6  # seconds between requests (NVD rate-limits unauthenticated to ~5 req/30s)


def _nvd_headers() -> dict[str, str]:
    """Return request headers, including an API key if available."""
    headers = {"Accept": "application/json"}
    api_key = os.environ.get("NVD_API_KEY")
    if api_key:
        headers["apiKey"] = api_key
    return headers


def fetch_cves(
    days_back: int = 120,
    keyword: str | None = None,
    max_results: int | None = None,
) -> list[dict]:
    """
    Fetch CVEs from the NVD API v2.

    Args:
        days_back: How many days of history to pull.
        keyword: Optional keyword filter (e.g. "apache").
        max_results: Cap on total CVEs fetched (None = all matching).

    Returns:
        List of raw CVE item dicts from the NVD response.
    """
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days_back)
    params: dict[str, str | int] = {
        "pubStartDate": start.strftime("%Y-%m-%dT%H:%M:%S.000"),
        "pubEndDate": end.strftime("%Y-%m-%dT%H:%M:%S.000"),
        "resultsPerPage": NVD_PAGE_SIZE,
        "startIndex": 0,
    }
    if keyword:
        params["keywordSearch"] = keyword

    all_items: list[dict] = []
    headers = _nvd_headers()
    has_api_key = "apiKey" in headers
    delay = 0.6 if has_api_key else NVD_REQUEST_DELAY

    while True:
        logger.info("Fetching NVD page startIndex=%s ...", params["startIndex"])
        resp = requests.get(NVD_API_URL, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        items = data.get("vulnerabilities", [])
        all_items.extend(items)
        logger.info(
            "  Got %d items (total so far: %d / %d)",
            len(items),
            len(all_items),
            data.get("totalResults", "?"),
        )

        if max_results and len(all_items) >= max_results:
            all_items = all_items[:max_results]
            break

        total = data.get("totalResults", 0)
        if params["startIndex"] + NVD_PAGE_SIZE >= total:
            break

        params["startIndex"] += NVD_PAGE_SIZE
        time.sleep(delay)

    return all_items


# ---------------------------------------------------------------------------
# Document preparation
# ---------------------------------------------------------------------------

def _format_cve_document(item: dict) -> tuple[str, dict]:
    """
    Convert a raw NVD CVE item into (text, metadata) suitable for embedding.

    Returns:
        (document_text, metadata_dict)
    """
    cve = item.get("cve", {})
    cve_id = cve.get("id", "UNKNOWN")

    # Description — prefer English
    descriptions = cve.get("descriptions", [])
    desc = next(
        (d["value"] for d in descriptions if d.get("lang") == "en"),
        descriptions[0]["value"] if descriptions else "No description available.",
    )

    # CVSS — try 3.1, then 3.0, then 2.0
    metrics = cve.get("metrics", {})
    severity = "UNKNOWN"
    base_score = None
    vector_string = ""

    for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
        metric_list = metrics.get(key, [])
        if metric_list:
            cvss = metric_list[0].get("cvssData", {})
            base_score = cvss.get("baseScore")
            severity = cvss.get("baseSeverity", metric_list[0].get("baseSeverity", "UNKNOWN"))
            vector_string = cvss.get("vectorString", "")
            break

    # Weaknesses (CWE IDs)
    cwes: list[str] = []
    for w in cve.get("weaknesses", []):
        for d in w.get("description", []):
            val = d.get("value", "")
            if val.startswith("CWE-"):
                cwes.append(val)

    # Affected products (CPE matches)
    products: list[str] = []
    for config in cve.get("configurations", []):
        for node in config.get("nodes", []):
            for match in node.get("cpeMatch", []):
                criteria = match.get("criteria", "")
                # CPE 2.3 format: cpe:2.3:a:vendor:product:version:...
                parts = criteria.split(":")
                if len(parts) >= 5:
                    vendor = parts[3]
                    product = parts[4]
                    version = parts[5] if len(parts) > 5 and parts[5] != "*" else ""
                    entry = f"{vendor}/{product}"
                    if version:
                        entry += f" {version}"
                    if entry not in products:
                        products.append(entry)

    # References
    refs = [r.get("url", "") for r in cve.get("references", [])[:5]]

    # Published / modified dates
    published = cve.get("published", "")[:10]
    modified = cve.get("lastModified", "")[:10]

    # Build the document text
    score_str = f"{base_score}" if base_score is not None else "N/A"
    doc = dedent(f"""\
        CVE ID: {cve_id}
        Published: {published}
        Last Modified: {modified}
        Severity: {severity} (CVSS {score_str})
        CVSS Vector: {vector_string}
        CWE: {', '.join(cwes) if cwes else 'N/A'}
        Affected Products: {', '.join(products[:10]) if products else 'N/A'}

        Description:
        {desc}

        References:
        {chr(10).join(refs) if refs else 'None'}
    """).strip()

    metadata = {
        "cve_id": cve_id,
        "published": published,
        "severity": severity.upper() if severity else "UNKNOWN",
        "base_score": float(base_score) if base_score is not None else 0.0,
        "cwes": ", ".join(cwes),
        "products": ", ".join(products[:10]),
    }

    return doc, metadata


# ---------------------------------------------------------------------------
# ChromaDB vector store
# ---------------------------------------------------------------------------

def _get_embedding_function():
    """Return an embedding function. Uses OpenAI by default."""
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model="text-embedding-3-small")


def ingest_cves(
    days_back: int = 120,
    keyword: str | None = None,
    max_results: int | None = None,
) -> int:
    """
    Fetch CVEs from NVD and ingest them into the ChromaDB vector store.

    Args:
        days_back: Days of CVE history to pull.
        keyword: Optional keyword filter.
        max_results: Cap on number of CVEs.

    Returns:
        Number of documents ingested.
    """
    from langchain_community.vectorstores import Chroma

    logger.info("Fetching CVEs from NVD (last %d days) ...", days_back)
    raw_items = fetch_cves(days_back=days_back, keyword=keyword, max_results=max_results)
    if not raw_items:
        logger.warning("No CVEs fetched.")
        return 0

    docs: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []

    for item in raw_items:
        text, meta = _format_cve_document(item)
        docs.append(text)
        metadatas.append(meta)
        ids.append(meta["cve_id"])

    logger.info("Ingesting %d CVEs into ChromaDB at %s ...", len(docs), CVE_DB_DIR)
    CVE_DB_DIR.mkdir(parents=True, exist_ok=True)

    embedding_fn = _get_embedding_function()

    # Upsert in batches to avoid hitting API limits
    batch_size = 100
    vectorstore = None
    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i : i + batch_size]
        batch_meta = metadatas[i : i + batch_size]
        batch_ids = ids[i : i + batch_size]

        if vectorstore is None:
            vectorstore = Chroma.from_texts(
                texts=batch_docs,
                metadatas=batch_meta,
                ids=batch_ids,
                embedding=embedding_fn,
                persist_directory=str(CVE_DB_DIR),
                collection_name="cve_collection",
            )
        else:
            vectorstore.add_texts(
                texts=batch_docs,
                metadatas=batch_meta,
                ids=batch_ids,
            )
        logger.info("  Ingested batch %d-%d", i, i + len(batch_docs))

    logger.info("Ingestion complete: %d CVEs stored.", len(docs))
    return len(docs)


def _load_vectorstore():
    """Load the persisted ChromaDB vector store (returns None if it doesn't exist)."""
    if not CVE_DB_DIR.exists():
        return None
    from langchain_community.vectorstores import Chroma

    return Chroma(
        persist_directory=str(CVE_DB_DIR),
        embedding_function=_get_embedding_function(),
        collection_name="cve_collection",
    )


def cve_query(query: str, k: int = 5) -> str:
    """
    Query the CVE vector store and return formatted results.

    This is the function that plugs into wargate.py's cyberintel_retriever.

    Args:
        query: Natural-language search query.
        k: Number of results to return.

    Returns:
        Formatted string of matching CVE records, or a fallback message
        if the vector store hasn't been built yet.
    """
    store = _load_vectorstore()
    if store is None:
        return (
            "[CVE_VECTORSTORE] No CVE database found. "
            "Run `python cve_vectorstore.py --ingest` to build it."
        )

    results = store.similarity_search_with_relevance_scores(query, k=k)
    if not results:
        return f"[CVE_VECTORSTORE] No CVEs matched query: '{query}'"

    sections: list[str] = []
    for doc, score in results:
        sections.append(
            f"--- (relevance: {score:.3f}) ---\n{doc.page_content}"
        )

    header = f"[CVE DATABASE] Top {len(results)} results for: '{query}'\n"
    return header + "\n\n".join(sections)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="CVE Vector Store for ProjectWARGATE")
    parser.add_argument("--ingest", action="store_true", help="Fetch CVEs from NVD and ingest into ChromaDB")
    parser.add_argument("--days", type=int, default=120, help="Days of CVE history to pull (default: 120)")
    parser.add_argument("--keyword", type=str, default=None, help="Optional keyword filter for NVD search")
    parser.add_argument("--max", type=int, default=None, help="Maximum number of CVEs to fetch")
    parser.add_argument("--query", type=str, default=None, help="Run a test query against the vector store")
    parser.add_argument("--results", type=int, default=5, help="Number of results for --query (default: 5)")

    args = parser.parse_args()

    if args.ingest:
        count = ingest_cves(
            days_back=args.days,
            keyword=args.keyword,
            max_results=args.max,
        )
        print(f"\nIngested {count} CVEs into {CVE_DB_DIR}")

    if args.query:
        print(cve_query(args.query, k=args.results))

    if not args.ingest and not args.query:
        parser.print_help()


if __name__ == "__main__":
    main()
