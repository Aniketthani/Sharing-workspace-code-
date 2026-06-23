# XRAY AKS — Azure AI Search Pipeline
## Architecture & Module Guide

---

## Folder structure

```
xray_aks_azure_search/
├── 00_config.py                          ← Central env config + constants
├── 01_fetch_from_blob.py                 ← Fetch EML + attachments from Blob Storage
├── 02_create_index.py                    ← ONE-TIME: index schema
├── 03_create_skillset_and_datasource.py  ← ONE-TIME: OCR/embed skillset + blob datasource
├── 04_create_indexer.py                  ← ONE-TIME: indexer with field mappings
├── 05_run_indexer.py                     ← Per-submission: upload staging + run indexer
├── 06_create_knowledge_base.py           ← ONE-TIME: Knowledge Source + Knowledge Base
├── 07_query_adverse_keywords.py          ← Per-query: KB agentic retrieve + scan
├── 08_orchestrator.py                    ← CLI entry point
├── business_operation_keywords.json      ← 31-category adverse keyword taxonomy
├── requirements.txt
└── .env.template
```

---

## Architecture

```
Azure Blob Storage  (aidata/)
  ├── files/<Ref_Id>.eml           ← source email
  └── input/<Ref_Id>/*             ← attachments
          │
          ▼  01_fetch_from_blob.py
  DocumentRecord list (in memory)
          │
          ▼  05_run_indexer.py
  aidata/indexed/<Ref_Id>/         ← staging prefix with blob metadata
    metadata: ref_id, doctype, email headers
          │
          ▼  Azure AI Search Indexer (04)
  Skillset (03):
    OCR → Merge → Split (2 000 char chunks) → Azure OpenAI Embedding
          │
          ▼
  Index: xray-aks-submissions-v1  (02)
    Fields: ref_id ✦filterable, doctype ✦filterable,
            content_text ✦BM25, content_vector ✦HNSW 1536-dim,
            email_from/to/subject/date, file_name, blob_path
          │
          │   ONE-TIME SETUP (06)
          ▼
  Knowledge Source: xray-aks-submission-ks
    └── wraps the index above
    └── sets: vector_fields, semantic_config, default_top, reranker_threshold
          │
          ▼
  Knowledge Base: xray-aks-kb
    └── references knowledge source
    └── attaches Azure OpenAI gpt-4o for query planning
    └── retrieval_instructions = insurance underwriting domain context
    └── output_mode = rawDocuments  (we own the post-processing)
          │
          │   PER-QUERY (07)
          ▼
  KnowledgeBaseRetrievalClient.retrieve()
    system prompt  = 31 categories + all keywords → query planner
    user message   = task + Ref_Id name
    OData filter   = ref_id eq '<Ref_Id>'   ← hard scoped to submission
    hybrid search  = BM25 + vector (RRF fusion) per sub-query
    semantic reranker (Azure) re-orders merged results
          │
          ▼
  Aho-Corasick exact-match scan on returned chunks
          │
          ▼
  Report: category → keywords_matched → evidence (snippet, file, scores)
```

---

## Object hierarchy in Azure AI Search

```
index  ──────────────────────► Knowledge Source  (xray-aks-submission-ks)
(xray-aks-submissions-v1)              │
                                       │
                               Knowledge Base  (xray-aks-kb)
                                 • gpt-4o for query planning
                                 • retrieval_instructions
                                 • output_mode = rawDocuments
                                       │
                               KnowledgeBaseRetrievalClient
                                 • messages (system + user)
                                 • SearchIndexKnowledgeSourceParams
                                     ├─ filter: ref_id eq '<Ref_Id>'
                                     ├─ top, reranker_threshold
                                     └─ kind: "searchIndex"
```

---

## Execution order

### One-time setup

```bash
python 08_orchestrator.py --setup
```

Runs in sequence:
1. `02_create_index.py`                   — index schema (BM25 + HNSW + semantic)
2. `03_create_skillset_and_datasource.py` — OCR, merge, split, embed skillset + blob datasource
3. `04_create_indexer.py`                 — indexer with field + output mappings
4. `06_create_knowledge_base.py`          — Knowledge Source wrapping the index, then Knowledge Base with LLM

All steps are **idempotent** — safe to re-run.

### Per-submission indexing

```bash
python 08_orchestrator.py --index REF001
```

### Adverse keyword scan

```bash
python 08_orchestrator.py --scan REF001
```

### Index + scan in one shot

```bash
python 08_orchestrator.py --all REF001
```

---

## What the KB retrieve call does differently from raw SearchClient

| Aspect | Raw SearchClient (old 07) | KnowledgeBaseRetrievalClient (new 07) |
|--------|--------------------------|---------------------------------------|
| Query decomposition | Single query | LLM query planner fans out per-category sub-queries |
| BM25 + vector | Manual VectorizedQuery | Automatic hybrid per sub-query |
| Semantic reranker | Explicit QueryType.SEMANTIC | Automatic via knowledge source config |
| ref_id filter | filter= on SearchClient | filter= inside SearchIndexKnowledgeSourceParams |
| Domain context | Embedded in query text | System prompt to the query planner (retrieval_instructions) |
| Result format | Raw search results | Grounding chunks via result.references |

---

## Scan output format

```json
{
  "ref_id": "REF001",
  "total_chunks_scanned": 22,
  "categories_flagged": 5,
  "report": [
    {
      "category": "Heavy liquor exposure",
      "keywords_found": ["full bar", "bottle service"],
      "match_count": 4,
      "evidence": [
        {
          "keyword": "full bar",
          "evidence_snippet": "…operates a **FULL BAR** with high-proof spirits…",
          "file_name": "REF001_email_body.txt",
          "doctype": "email_content",
          "ref_id": "REF001",
          "email_subject": "RE: Fwd: New submission — Riverside Tavern",
          "reranker_score": 0.912,
          "search_score": 3.41
        }
      ]
    }
  ]
}
```

---

## Suggestions

| Gap | Recommendation |
|-----|----------------|
| Negation detection | Before reporting a match, check a ±10-word window for negation ("no full bar", "does not operate"). |
| Blob trigger auto-indexing | Azure Function on blob creation at `aidata/files/*.eml` → calls `index_ref_id()`. |
| Managed Identity | Replace API-key auth with `DefaultAzureCredential` + RBAC for production. |
| Schema versioning | Index name includes `v1`. To evolve schema: create `v2`, backfill, update knowledge source, swap. |
| Streamlit integration | Pass `ref_id` from sidebar → `run_adverse_keyword_scan()` → render report table. |
