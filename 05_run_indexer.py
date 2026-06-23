"""
05_run_indexer.py
=================
Per-submission pipeline step.  Call once per Ref_Id when new documents arrive.

Steps:
  1. Fetch documents (email body + attachments) using 01_fetch_from_blob.py.
  2. Upload them to a staging prefix inside the same 'aidata' container
     under  aidata/indexed/<Ref_Id>/<doc_id>   with rich blob metadata.
  3. Trigger the Azure AI Search indexer to pull and enrich those files.
  4. Poll until indexer run completes and report status.

Why a staging prefix?
  The indexer datasource points at  aidata/  .  We upload processed files
  to a sub-prefix so the indexer can be scoped with a virtual path query.
  Alternatively you can point the indexer directly at files/ and input/ —
  but having a staging area lets you control exactly what gets re-indexed.
"""

from __future__ import annotations
import io
import logging
import time
from typing import Optional

from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient, ContentSettings
from azure.search.documents.indexes import SearchIndexerClient
from azure.search.documents.indexes.models import SearchIndexerDataContainer

from fetch_from_blob import fetch_documents_for_ref_id, DocumentRecord
from config import (
    AZURE_STORAGE_CONNECTION_STRING,
    AZURE_SEARCH_ENDPOINT,
    AZURE_SEARCH_ADMIN_KEY,
    BLOB_CONTAINER_NAME,
    DATASOURCE_NAME,
    INDEXER_NAME,
)

log = logging.getLogger(__name__)

STAGING_PREFIX = "indexed"   # aidata/indexed/<Ref_Id>/<doc_id>
POLL_INTERVAL  = 10          # seconds between status checks
MAX_WAIT_SECS  = 600         # 10 minutes max wait


# ── Upload ────────────────────────────────────────────────────────────────────

def upload_documents_to_blob(records: list[DocumentRecord]) -> list[str]:
    """
    Uploads DocumentRecord content bytes to the staging prefix in Blob Storage.
    Sets custom blob metadata so the indexer can map them to index fields.
    Returns list of blob paths uploaded.
    """
    svc = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
    container = svc.get_container_client(BLOB_CONTAINER_NAME)
    uploaded: list[str] = []

    for rec in records:
        blob_path = f"{STAGING_PREFIX}/{rec.ref_id}/{rec.doc_id}/{rec.file_name}"

        # Metadata values must be strings and ASCII-safe
        meta = {
            "ref_id":        rec.ref_id,
            "doctype":       rec.doctype,
            "email_from":    _safe_meta(rec.extra_meta.get("from", "")),
            "email_to":      _safe_meta(rec.extra_meta.get("to", "")),
            "email_cc":      _safe_meta(rec.extra_meta.get("cc", "")),
            "email_subject": _safe_meta(rec.extra_meta.get("subject", "")),
            "email_date":    _safe_meta(rec.extra_meta.get("date", "")),
        }

        blob_client = container.get_blob_client(blob_path)
        blob_client.upload_blob(
            data           = io.BytesIO(rec.content),
            overwrite      = True,
            content_settings=ContentSettings(content_type=rec.mime_type),
            metadata       = meta,
        )
        log.info("Uploaded → %s  [%d bytes]", blob_path, len(rec.content))
        uploaded.append(blob_path)

    return uploaded


def _safe_meta(value: str, max_len: int = 256) -> str:
    """Truncate and ASCII-encode metadata values for Blob metadata restrictions."""
    return value.encode("ascii", errors="replace").decode("ascii")[:max_len]


# ── Scope the indexer datasource to this Ref_Id ───────────────────────────────

def _scope_datasource_to_ref(
    indexer_client: SearchIndexerClient,
    ref_id: str,
) -> None:
    """
    Updates the datasource container query so this indexer run only
    processes files for the given Ref_Id.
    """
    ds = indexer_client.get_data_source_connection(DATASOURCE_NAME)
    ds.container = SearchIndexerDataContainer(
        name  = BLOB_CONTAINER_NAME,
        query = f"{STAGING_PREFIX}/{ref_id}/",
    )
    indexer_client.create_or_update_data_source_connection(ds)
    log.info("Datasource scoped to prefix: %s/%s/", STAGING_PREFIX, ref_id)


# ── Run indexer ───────────────────────────────────────────────────────────────

def run_indexer_for_ref(ref_id: str) -> dict:
    """
    Triggers an indexer run scoped to ref_id.
    Returns the final indexer status dict.
    """
    credential     = AzureKeyCredential(AZURE_SEARCH_ADMIN_KEY)
    indexer_client = SearchIndexerClient(
        endpoint   = AZURE_SEARCH_ENDPOINT,
        credential = credential,
    )

    # Update datasource scope
    _scope_datasource_to_ref(indexer_client, ref_id)

    # Reset indexer so it re-processes all docs in the scoped prefix
    indexer_client.reset_indexer(INDEXER_NAME)
    log.info("Indexer reset for Ref_Id=%s", ref_id)

    # Trigger run
    indexer_client.run_indexer(INDEXER_NAME)
    log.info("Indexer run triggered for Ref_Id=%s", ref_id)

    # Poll for completion
    return _poll_indexer_status(indexer_client)


def _poll_indexer_status(indexer_client: SearchIndexerClient) -> dict:
    elapsed = 0
    while elapsed < MAX_WAIT_SECS:
        status = indexer_client.get_indexer_status(INDEXER_NAME)
        last   = status.last_result

        if last is not None:
            state = last.status
            log.info(
                "Indexer status: %s | docs succeeded: %s | failed: %s",
                state,
                last.item_count,
                last.failed_item_count,
            )
            if state in ("success", "transientFailure", "persistentFailure", "reset"):
                return {
                    "status":        state,
                    "item_count":    last.item_count,
                    "failed_count":  last.failed_item_count,
                    "errors":        [str(e) for e in (last.errors or [])],
                    "warnings":      [str(w) for w in (last.warnings or [])],
                }
        else:
            log.info("Waiting for indexer to start…")

        time.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL

    log.warning("Indexer did not complete within %d seconds.", MAX_WAIT_SECS)
    return {"status": "timeout"}


# ── Main orchestration ────────────────────────────────────────────────────────

def index_ref_id(ref_id: str) -> dict:
    """
    Full pipeline for one Ref_Id:
      fetch → upload with metadata → run indexer → return status
    """
    log.info("── Starting indexing pipeline for Ref_Id=%s ──", ref_id)

    # Step 1: Fetch from source blob locations
    records = fetch_documents_for_ref_id(ref_id)
    if not records:
        log.error("No documents found for Ref_Id=%s — aborting.", ref_id)
        return {"status": "no_documents"}

    # Step 2: Upload to staging prefix with metadata
    uploaded_paths = upload_documents_to_blob(records)
    log.info("Uploaded %d files to staging.", len(uploaded_paths))

    # Step 3: Run indexer scoped to this Ref_Id
    result = run_indexer_for_ref(ref_id)
    result["ref_id"]   = ref_id
    result["uploaded"] = len(uploaded_paths)
    log.info("Pipeline complete for Ref_Id=%s → %s", ref_id, result)
    return result


if __name__ == "__main__":
    import sys, json
    logging.basicConfig(level=logging.INFO)
    ref = sys.argv[1] if len(sys.argv) > 1 else "REF001"
    result = index_ref_id(ref)
    print(json.dumps(result, indent=2))
