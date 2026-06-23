"""
07_query_adverse_keywords.py
============================
Queries the XRAY AKS Knowledge Base using the native Azure AI Search
KnowledgeBaseRetrievalClient (agentic retrieval — 2026-05-01-preview SDK).

How it works end-to-end
────────────────────────
  1.  Build a structured system prompt that lists all 31 risk categories
      and their keywords.  This drives the KB's internal query planner to
      decompose the request into per-category sub-queries.

  2.  Pass the user message with Ref_Id as a metadata filter inside
      SearchIndexKnowledgeSourceParams.filter  (OData: ref_id eq '<id>').

  3.  KnowledgeBaseRetrievalClient.retrieve() fires hybrid BM25 + vector
      sub-queries in parallel, runs the semantic reranker, and returns
      raw grounding chunks  (output_mode = "rawDocuments" in the KB).

  4.  In-process Aho-Corasick pass confirms exact keyword presence in
      each returned chunk.

  5.  Results are structured as:
        category → keywords_matched → evidence (snippet, file, scores)

Output format
─────────────
  {
    "ref_id": "REF001",
    "total_chunks_scanned": 18,
    "categories_flagged": 4,
    "report": [
      {
        "category": "Heavy liquor exposure",
        "keywords_found": ["full bar", "bottle service"],
        "match_count": 3,
        "evidence": [
          {
            "keyword": "full bar",
            "evidence_snippet": "…operates a **FULL BAR** with…",
            "file_name": "REF001_email_body.txt",
            "doctype": "email_content",
            "ref_id": "REF001",
            "reranker_score": 0.89
          }
        ]
      }
    ]
  }
"""

from __future__ import annotations
import json
import logging
import re
from pathlib import Path
from typing import Any

import ahocorasick           # pip install pyahocorasick

from azure.core.credentials import AzureKeyCredential
from azure.search.documents.knowledgebases import KnowledgeBaseRetrievalClient
from azure.search.documents.knowledgebases.models import (
    KnowledgeBaseMessage,
    KnowledgeBaseMessageTextContent,
    KnowledgeBaseRetrievalRequest,
    SearchIndexKnowledgeSourceParams,
)

from config import (
    AZURE_SEARCH_ENDPOINT,
    AZURE_SEARCH_ADMIN_KEY,
    KNOWLEDGE_BASE_NAME,
    KNOWLEDGE_SOURCE_NAME,
    TOP_K_RESULTS,
)

log = logging.getLogger(__name__)

# ── Keyword taxonomy ──────────────────────────────────────────────────────────
_KEYWORDS_PATH = Path(__file__).parent / "business_operation_keywords.json"
with _KEYWORDS_PATH.open() as _f:
    KEYWORD_TAXONOMY: dict[str, list[str]] = json.load(_f)


# ── Aho-Corasick automaton (built once at import time) ────────────────────────
def _build_automaton(
    taxonomy: dict[str, list[str]],
) -> tuple[ahocorasick.Automaton, dict[str, str]]:
    A = ahocorasick.Automaton()
    kw_to_cat: dict[str, str] = {}
    idx = 0
    for category, keywords in taxonomy.items():
        for kw in keywords:
            needle = kw.lower()
            if needle not in A:
                A.add_word(needle, (idx, needle))
                idx += 1
            kw_to_cat[needle] = category
    A.make_automaton()
    return A, kw_to_cat


AUTOMATON, KW_TO_CAT = _build_automaton(KEYWORD_TAXONOMY)


# ── Prompt builders ───────────────────────────────────────────────────────────

def _build_system_prompt() -> str:
    """
    System / assistant-turn prompt injected into the KB retrieve call.

    This does two things:
      (a) Tells the internal LLM query planner what domain it is working in.
      (b) Lists every category + keyword so the planner generates targeted
          sub-queries for each category rather than one omnibus query.

    The result is that Azure AI Search fires parallel hybrid sub-queries
    (BM25 + vector) — one per category cluster — and merges the top results
    before returning them to us.
    """
    category_lines = []
    for cat, kws in KEYWORD_TAXONOMY.items():
        category_lines.append(f"  [{cat}]: {', '.join(kws)}")
    categories_block = "\n".join(category_lines)

    return (
        "You are an insurance underwriting risk analyst assistant. "
        "Your task is to identify adverse risk signals in an insurance submission "
        "by scanning for the following risk categories and their associated keywords.\n\n"
        "Risk categories and keywords to search for:\n"
        f"{categories_block}\n\n"
        "For each category, generate a targeted search query to find evidence of "
        "those keywords in the submission documents. "
        "Return the raw grounding passages — do not summarise. "
        "Each result must retain its source metadata (file_name, doctype, ref_id)."
    )


def _build_user_message(ref_id: str) -> str:
    """
    User-turn message.  States the task clearly so the query planner knows
    what to do, and names the Ref_Id for context (the hard filter is applied
    via SearchIndexKnowledgeSourceParams.filter below).
    """
    return (
        f"Scan all documents for submission Ref_Id '{ref_id}' and identify every "
        f"adverse risk keyword present across all 31 risk categories. "
        f"Return every passage that contains any of the listed keywords. "
        f"Include the document file name and type in the citation for each passage."
    )


# ── KB retrieve call ──────────────────────────────────────────────────────────

def retrieve_from_knowledge_base(
    ref_id: str,
    top_k: int = TOP_K_RESULTS,
) -> list[dict]:
    """
    Calls KnowledgeBaseRetrievalClient.retrieve() with:
      • system prompt  — domain context + full keyword taxonomy
      • user message   — task instruction naming the Ref_Id
      • OData filter   — ref_id eq '<ref_id>'  (hard scopes to this submission)
      • kind           — "searchIndex"          (our index-backed knowledge source)
      • maxOutputDocuments — top_k             (cap on grounding chunks returned)
      • includeActivity    — True              (for debugging sub-query activity)

    Returns a list of raw chunk dicts extracted from the KB response.
    Each dict contains at minimum:
      content_text, file_name, doctype, ref_id, blob_path, reranker_score
    """
    kb_client = KnowledgeBaseRetrievalClient(
        endpoint            = AZURE_SEARCH_ENDPOINT,
        knowledge_base_name = KNOWLEDGE_BASE_NAME,
        credential          = AzureKeyCredential(AZURE_SEARCH_ADMIN_KEY),
    )

    request = KnowledgeBaseRetrievalRequest(
        messages = [
            # assistant turn = system-level instructions to the query planner
            KnowledgeBaseMessage(
                role    = "assistant",
                content = [
                    KnowledgeBaseMessageTextContent(text=_build_system_prompt())
                ],
            ),
            # user turn = the actual task
            KnowledgeBaseMessage(
                role    = "user",
                content = [
                    KnowledgeBaseMessageTextContent(text=_build_user_message(ref_id))
                ],
            ),
        ],

        knowledge_source_params = [
            SearchIndexKnowledgeSourceParams(
                knowledge_source_name = KNOWLEDGE_SOURCE_NAME,
                kind                  = "searchIndex",

                # ── Hard metadata filter: only chunks for this Ref_Id ─────────
                # This is the OData filter applied at index query time, before
                # the semantic reranker.  Ensures zero cross-submission leakage.
                filter = f"ref_id eq '{ref_id}'",

                # ── Hybrid retrieval settings ─────────────────────────────────
                # The KB query planner generates sub-queries; each sub-query
                # runs BM25 (keyword) + vector (semantic) and blends scores
                # (RRF fusion) before passing to the semantic reranker.
                # These settings control the reranker and candidate pool.
                top                    = top_k,
                reranker_threshold     = 0.4,     # lower than default to cast wider net
                max_docs_for_reranker  = 100,     # more candidates → better recall
            )
        ],

        # Cap the total response payload (prevents runaway costs on large submissions)
        max_output_documents = top_k,
        max_output_size      = 128_000,   # ~100k tokens — fits gpt-4o context

        # Include activity log so we can inspect which sub-queries the planner
        # generated (useful for debugging category coverage)
        include_activity = True,
    )

    result = kb_client.retrieve(request)

    # ── Parse the three-pronged response ─────────────────────────────────────
    # result.response  — list of KnowledgeBaseMessage (the synthesised answer
    #                    or raw docs, depending on output_mode)
    # result.activity  — list of activity steps (sub-queries, index calls, etc.)
    # result.references — grounding documents (the actual index chunks)

    # Log query-planning activity for observability
    if hasattr(result, "activity") and result.activity:
        for step in result.activity:
            log.debug("KB activity step: %s", step)

    # Extract grounding chunks from references
    chunks: list[dict] = []
    if hasattr(result, "references") and result.references:
        for ref in result.references:
            chunk = _parse_reference(ref)
            if chunk:
                chunks.append(chunk)

    # Fallback: if references are empty, try extracting from response messages
    # (happens when output_mode is set to rawDocuments on some SDK versions)
    if not chunks and hasattr(result, "response") and result.response:
        for msg in result.response:
            if hasattr(msg, "content") and msg.content:
                for content_item in msg.content:
                    if hasattr(content_item, "text") and content_item.text:
                        # Each content item may be a JSON array of doc chunks
                        try:
                            parsed = json.loads(content_item.text)
                            if isinstance(parsed, list):
                                chunks.extend(parsed)
                            elif isinstance(parsed, dict):
                                chunks.append(parsed)
                        except json.JSONDecodeError:
                            # Plain text chunk — wrap it
                            chunks.append({"content_text": content_item.text, "ref_id": ref_id})

    log.info("KB retrieve for Ref_Id=%s returned %d chunks.", ref_id, len(chunks))
    return chunks


def _parse_reference(ref: Any) -> dict | None:
    """
    Converts a KnowledgeBaseReference (or dict) into a flat chunk dict
    compatible with the Aho-Corasick post-processing step.
    """
    try:
        # SDK model — access via attributes
        if hasattr(ref, "as_dict"):
            d = ref.as_dict()
        elif isinstance(ref, dict):
            d = ref
        else:
            return None

        # Normalise field names (SDK may return camelCase or snake_case)
        return {
            "content_text":   d.get("content_text") or d.get("contentText") or d.get("content") or "",
            "file_name":      d.get("file_name")    or d.get("fileName")    or d.get("title", ""),
            "doctype":        d.get("doctype", ""),
            "ref_id":         d.get("ref_id")       or d.get("refId", ""),
            "blob_path":      d.get("blob_path")    or d.get("blobPath", ""),
            "email_subject":  d.get("email_subject") or d.get("emailSubject", ""),
            "email_date":     d.get("email_date")   or d.get("emailDate", ""),
            "reranker_score": float(d.get("@search.rerankerScore") or d.get("rerankerScore") or 0.0),
            "search_score":   float(d.get("@search.score") or d.get("searchScore") or 0.0),
        }
    except Exception as exc:
        log.warning("Could not parse reference: %s — %s", ref, exc)
        return None


# ── Aho-Corasick exact match scan ─────────────────────────────────────────────

def _scan_chunk(text: str) -> list[tuple[str, str]]:
    """Returns list of (keyword, category) pairs found in text."""
    text_lower = text.lower()
    found: dict[str, str] = {}
    for _, (_, keyword) in AUTOMATON.iter(text_lower):
        found[keyword] = KW_TO_CAT[keyword]
    return list(found.items())


def _evidence_snippet(text: str, keyword: str, context: int = 200) -> str:
    """Returns a short passage centred on the keyword with the keyword uppercased."""
    idx = text.lower().find(keyword.lower())
    if idx == -1:
        return text[:context]
    start = max(0, idx - context // 2)
    end   = min(len(text), idx + len(keyword) + context // 2)
    snippet = text[start:end]
    snippet = re.sub(re.escape(keyword), f"**{keyword.upper()}**", snippet, flags=re.IGNORECASE)
    return f"…{snippet}…"


# ── Report builder ────────────────────────────────────────────────────────────

def build_adverse_report(ref_id: str, chunks: list[dict]) -> list[dict]:
    """
    Scans each returned chunk with Aho-Corasick and organises findings by
    risk category.
    """
    # category → keyword → [evidence_items]
    category_map: dict[str, dict[str, list[dict]]] = {}

    for chunk in chunks:
        text = chunk.get("content_text", "")
        if not text.strip():
            continue

        matches = _scan_chunk(text)
        if not matches:
            continue

        for keyword, category in matches:
            cat_bucket = category_map.setdefault(category, {})
            kw_bucket  = cat_bucket.setdefault(keyword, [])

            kw_bucket.append({
                "keyword":          keyword,
                "evidence_snippet": _evidence_snippet(text, keyword),
                "file_name":        chunk.get("file_name", ""),
                "doctype":          chunk.get("doctype", ""),
                "ref_id":           chunk.get("ref_id", ref_id),
                "blob_path":        chunk.get("blob_path", ""),
                "email_subject":    chunk.get("email_subject", ""),
                "email_date":       chunk.get("email_date", ""),
                "reranker_score":   round(chunk.get("reranker_score", 0.0), 4),
                "search_score":     round(chunk.get("search_score", 0.0), 4),
            })

    # Flatten and deduplicate
    report: list[dict] = []
    for category, kw_dict in sorted(category_map.items()):
        keywords_found = sorted(kw_dict.keys())
        evidence: list[dict] = []
        seen: set[tuple] = set()

        for kw in keywords_found:
            for ev in kw_dict[kw]:
                dedup_key = (ev["file_name"], ev["evidence_snippet"][:80])
                if dedup_key not in seen:
                    seen.add(dedup_key)
                    evidence.append(ev)
                if len(evidence) >= 5:   # max 5 evidence items per category
                    break
            if len(evidence) >= 5:
                break

        report.append({
            "category":       category,
            "keywords_found": keywords_found,
            "match_count":    sum(len(v) for v in kw_dict.values()),
            "evidence":       evidence,
        })

    report.sort(key=lambda x: x["match_count"], reverse=True)
    return report


# ── Public API ────────────────────────────────────────────────────────────────

def run_adverse_keyword_scan(
    ref_id: str,
    top_k: int = TOP_K_RESULTS,
    output_json_path: str | None = None,
) -> dict:
    """
    Full pipeline:
      KB retrieve (hybrid BM25 + vector + semantic reranker, filtered by ref_id)
        → Aho-Corasick exact-match scan
        → structured report

    Args:
        ref_id:           Submission identifier — used as metadata filter.
        top_k:            Max chunks to retrieve from the KB.
        output_json_path: If set, writes the JSON report to this path.

    Returns dict with keys:
        ref_id, total_chunks_scanned, categories_flagged, report
    """
    log.info("Starting adverse keyword scan for Ref_Id=%s via Knowledge Base.", ref_id)

    chunks = retrieve_from_knowledge_base(ref_id=ref_id, top_k=top_k)
    report = build_adverse_report(ref_id, chunks)

    output = {
        "ref_id":               ref_id,
        "total_chunks_scanned": len(chunks),
        "categories_flagged":   len(report),
        "report":               report,
    }

    if output_json_path:
        Path(output_json_path).write_text(json.dumps(output, indent=2, ensure_ascii=False))
        log.info("Report saved to %s", output_json_path)

    return output


def print_report(output: dict) -> None:
    """Prints the adverse keyword report to stdout in a readable format."""
    sep = "═" * 72
    print(f"\n{sep}")
    print("  XRAY AKS — Adverse Keyword Scan Report (Agentic Retrieval)")
    print(f"  Ref_Id   : {output['ref_id']}")
    print(f"  Chunks   : {output['total_chunks_scanned']} retrieved from Knowledge Base")
    print(f"  Flags    : {output['categories_flagged']} risk categories with confirmed matches")
    print(f"{sep}\n")

    for entry in output["report"]:
        print(f"▶  {entry['category']}   [{entry['match_count']} hit(s)]")
        print(f"   Keywords matched : {', '.join(entry['keywords_found'])}")
        for ev in entry["evidence"][:2]:
            print(f"   ┌─ File     : {ev['file_name']}  ({ev['doctype']})")
            print(f"   │  Scores   : reranker={ev['reranker_score']}  bm25+vector={ev['search_score']}")
            print(f"   │  Evidence : {ev['evidence_snippet']}")
            if ev.get("email_subject"):
                print(f"   │  Subject  : {ev['email_subject']}")
            print(f"   └─ Ref_Id   : {ev['ref_id']}")
        print()


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    ref = sys.argv[1] if len(sys.argv) > 1 else "REF001"
    result = run_adverse_keyword_scan(
        ref_id           = ref,
        output_json_path = f"{ref}_aks_report.json",
    )
    print_report(result)
