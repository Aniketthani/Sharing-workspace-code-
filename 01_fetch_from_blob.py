"""
01_fetch_from_blob.py
=====================
Fetches .eml files from   aidata/files/*.eml
and attachment files from aidata/input/<Ref_Id>/*

For each Ref_Id it:
  1. Downloads the .eml that matches (same base name or the only .eml, depending
     on your naming convention — see NOTE below).
  2. Parses the .eml → plain-text email body  →  saves as <base>.txt in memory.
  3. Downloads every file inside  aidata/input/<Ref_Id>/.
  4. Returns a list of DocumentRecord dicts ready for the indexer.

NOTE on EML ↔ Ref_Id matching
──────────────────────────────
The code assumes the .eml file name (without extension) equals Ref_Id.
Example:  aidata/files/REF001.eml  ←→  aidata/input/REF001/*
Adjust `_find_eml_for_ref` if your naming convention differs.
"""

from __future__ import annotations
import email
import io
import logging
from dataclasses import dataclass, field
from typing import Optional

from azure.storage.blob import BlobServiceClient, ContainerClient
from 00_config import (
    AZURE_STORAGE_CONNECTION_STRING,
    BLOB_CONTAINER_NAME,
    EML_FOLDER,
    ATTACHMENTS_ROOT_FOLDER,
)

log = logging.getLogger(__name__)


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class DocumentRecord:
    """Represents a single file to be indexed."""
    ref_id:    str
    doc_id:    str          # unique ID for the index document
    file_name: str
    blob_path: str          # full blob path (for audit trail)
    doctype:   str          # "email_content" | "attachment"
    content:   bytes        # raw file bytes (txt for email body, original for attachments)
    mime_type: str          = "application/octet-stream"
    extra_meta: dict        = field(default_factory=dict)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_container_client() -> ContainerClient:
    svc = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
    return svc.get_container_client(BLOB_CONTAINER_NAME)


def _list_blobs_with_prefix(container: ContainerClient, prefix: str) -> list[str]:
    return [b.name for b in container.list_blobs(name_starts_with=prefix)]


def _download_blob(container: ContainerClient, blob_path: str) -> bytes:
    blob_client = container.get_blob_client(blob_path)
    stream = io.BytesIO()
    blob_client.download_blob().readinto(stream)
    return stream.getvalue()


def _eml_to_text(raw_bytes: bytes) -> tuple[str, dict]:
    """
    Parse an .eml file and extract:
      - combined plain-text body (falling back to HTML→text strip if needed)
      - headers dict: From, To, Cc, Subject, Date
    Returns (text_body, headers_dict).
    """
    msg = email.message_from_bytes(raw_bytes)

    headers = {
        "from":    msg.get("From", ""),
        "to":      msg.get("To", ""),
        "cc":      msg.get("Cc", ""),
        "subject": msg.get("Subject", ""),
        "date":    msg.get("Date", ""),
    }

    body_parts: list[str] = []

    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            disp = str(part.get("Content-Disposition", ""))
            if "attachment" in disp:
                continue
            if ct == "text/plain":
                charset = part.get_content_charset() or "utf-8"
                body_parts.append(part.get_payload(decode=True).decode(charset, errors="replace"))
            elif ct == "text/html" and not body_parts:
                # fallback: strip tags crudely
                import re
                raw_html = part.get_payload(decode=True).decode(
                    part.get_content_charset() or "utf-8", errors="replace"
                )
                body_parts.append(re.sub(r"<[^>]+>", " ", raw_html))
    else:
        charset = msg.get_content_charset() or "utf-8"
        body_parts.append(msg.get_payload(decode=True).decode(charset, errors="replace"))

    full_body = "\n".join(body_parts)

    # Prepend headers so they are searchable too
    header_text = (
        f"From: {headers['from']}\n"
        f"To: {headers['to']}\n"
        f"Cc: {headers['cc']}\n"
        f"Subject: {headers['subject']}\n"
        f"Date: {headers['date']}\n"
        f"{'─' * 60}\n"
    )
    return header_text + full_body, headers


def _find_eml_for_ref(container: ContainerClient, ref_id: str) -> Optional[str]:
    """
    Returns the blob path of the .eml file for this Ref_Id, or None.
    Convention:  aidata/files/<Ref_Id>.eml
    """
    candidate = f"{EML_FOLDER}/{ref_id}.eml"
    blobs = _list_blobs_with_prefix(container, candidate)
    if blobs:
        return blobs[0]

    # Fallback: list all .eml files and let caller handle it
    all_emls = [b for b in _list_blobs_with_prefix(container, EML_FOLDER + "/")
                if b.endswith(".eml")]
    if len(all_emls) == 1:
        log.warning(
            "No exact match for %s.eml; using only .eml found: %s",
            ref_id, all_emls[0]
        )
        return all_emls[0]

    log.error("Cannot determine .eml for Ref_Id=%s from %s", ref_id, all_emls)
    return None


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_documents_for_ref_id(ref_id: str) -> list[DocumentRecord]:
    """
    Main entry point.  Call this with a Ref_Id string; returns all
    DocumentRecord objects (email body + attachments) ready for indexing.
    """
    container = _get_container_client()
    records: list[DocumentRecord] = []

    # ── 1. Email body ─────────────────────────────────────────────────────────
    eml_path = _find_eml_for_ref(container, ref_id)
    if eml_path is None:
        log.error("Skipping Ref_Id=%s — no .eml found.", ref_id)
        return records

    log.info("Downloading EML: %s", eml_path)
    raw_eml = _download_blob(container, eml_path)
    text_body, headers = _eml_to_text(raw_eml)
    email_txt_bytes = text_body.encode("utf-8")

    records.append(DocumentRecord(
        ref_id    = ref_id,
        doc_id    = f"{ref_id}__email_body",
        file_name = f"{ref_id}_email_body.txt",
        blob_path = eml_path,
        doctype   = "email_content",
        content   = email_txt_bytes,
        mime_type = "text/plain",
        extra_meta = headers,
    ))

    # ── 2. Attachments from  aidata/input/<Ref_Id>/ ───────────────────────────
    att_prefix = f"{ATTACHMENTS_ROOT_FOLDER}/{ref_id}/"
    att_blobs  = _list_blobs_with_prefix(container, att_prefix)

    if not att_blobs:
        log.warning("No attachments found under %s", att_prefix)

    for blob_path in att_blobs:
        file_name = blob_path.split("/")[-1]
        if not file_name:           # skip "folder" placeholder blobs
            continue

        log.info("Downloading attachment: %s", blob_path)
        raw_bytes = _download_blob(container, blob_path)

        records.append(DocumentRecord(
            ref_id    = ref_id,
            doc_id    = f"{ref_id}__att__{file_name}",
            file_name = file_name,
            blob_path = blob_path,
            doctype   = "attachment",
            content   = raw_bytes,
            mime_type = _guess_mime(file_name),
            extra_meta = {},
        ))

    log.info(
        "Ref_Id=%s → %d document records (%d attachments)",
        ref_id, len(records), len(records) - 1
    )
    return records


def _guess_mime(file_name: str) -> str:
    import mimetypes
    mt, _ = mimetypes.guess_type(file_name)
    return mt or "application/octet-stream"


# ── CLI helper ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, json, logging
    logging.basicConfig(level=logging.INFO)
    ref = sys.argv[1] if len(sys.argv) > 1 else "REF001"
    docs = fetch_documents_for_ref_id(ref)
    for d in docs:
        print(json.dumps({
            "doc_id":    d.doc_id,
            "file_name": d.file_name,
            "doctype":   d.doctype,
            "bytes":     len(d.content),
            "blob_path": d.blob_path,
        }, indent=2))
