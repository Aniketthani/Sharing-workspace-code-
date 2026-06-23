"""
06_create_knowledge_base.py
============================
ONE-TIME SETUP — creates:

  1. A Knowledge Source  — a top-level Azure AI Search object that wraps the
     existing xray-aks-submissions-v1 index and tells the agentic retrieval
     engine how to query it (hybrid BM25 + vector, semantic reranker).

  2. A Knowledge Base    — a top-level Azure AI Search object that:
       • references the Knowledge Source above
       • attaches an Azure OpenAI LLM (gpt-4o) for query planning and
         answer synthesis
       • holds default retrieval instructions scoped to insurance underwriting

Architecture fit
─────────────────
  index  ──►  Knowledge Source (xray-aks-submission-ks)
                     │
                     ▼
              Knowledge Base  (xray-aks-kb)
                     │
                     ▼
          KnowledgeBaseRetrievalClient.retrieve()   ← module 07

SDK requirement: pip install --pre azure-search-documents
API version:     2026-05-01-preview  (preview package auto-uses this)
"""

from __future__ import annotations
import logging

from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    AzureOpenAIVectorizerParameters,
    KnowledgeBase,
    KnowledgeBaseAzureOpenAIModel,
    KnowledgeSourceReference,
    SearchIndexKnowledgeSource,               # "bring your own index" knowledge source
    SearchIndexKnowledgeSourceIndexedDocumentParameters,
)
from azure.search.documents.knowledgebases.models import (
    KnowledgeRetrievalLowReasoningEffort,
)

from config import (
    AZURE_SEARCH_ENDPOINT,
    AZURE_SEARCH_ADMIN_KEY,
    INDEX_NAME,
    SEMANTIC_CONFIG_NAME,
    KNOWLEDGE_BASE_NAME,
    KNOWLEDGE_SOURCE_NAME,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_KEY,
    CHAT_DEPLOYMENT_NAME,
    EMBEDDING_DEPLOYMENT_NAME,
    TOP_K_RESULTS,
)

log = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_index_client() -> SearchIndexClient:
    return SearchIndexClient(
        endpoint   = AZURE_SEARCH_ENDPOINT,
        credential = AzureKeyCredential(AZURE_SEARCH_ADMIN_KEY),
    )


# ── 1. Knowledge Source ───────────────────────────────────────────────────────

def create_knowledge_source_once(client: SearchIndexClient) -> None:
    """
    Creates a SearchIndex knowledge source that points at the XRAY AKS index.

    The SearchIndexKnowledgeSource is the "bring your own index" variant —
    it wraps an index you already manage (populated by the skillset-based
    indexer pipeline in modules 02–05) and exposes it to the agentic
    retrieval engine.

    Key settings:
      • vectorFields          — the HNSW vector field used for dense retrieval
      • semanticConfiguration — enables the semantic reranker pass
      • defaultTop            — number of chunks returned per sub-query
      • defaultRerankerThreshold — minimum semantic score to include a chunk
    """
    existing_ks = [ks.name for ks in client.list_knowledge_sources()]
    if KNOWLEDGE_SOURCE_NAME in existing_ks:
        log.info("Knowledge source '%s' already exists — skipping.", KNOWLEDGE_SOURCE_NAME)
        return

    knowledge_source = SearchIndexKnowledgeSource(
        name        = KNOWLEDGE_SOURCE_NAME,
        description = "XRAY AKS — insurance submission index (email body + attachments)",
        # Points at the index built by modules 02-05
        index_parameters = SearchIndexKnowledgeSourceIndexedDocumentParameters(
            index_name                  = INDEX_NAME,
            semantic_configuration_name = SEMANTIC_CONFIG_NAME,
            vector_fields               = ["content_vector"],
            # default retrieval behaviour (overridable per-query in retrieve())
            default_top                         = TOP_K_RESULTS,
            default_reranker_threshold          = 0.5,
            default_max_docs_for_reranker       = 50,
            query_language                      = "en-us",
        ),
    )

    client.create_or_update_knowledge_source(knowledge_source)
    log.info("Knowledge source '%s' created.", KNOWLEDGE_SOURCE_NAME)


# ── 2. Knowledge Base ─────────────────────────────────────────────────────────

def create_knowledge_base_once(client: SearchIndexClient) -> None:
    """
    Creates the Knowledge Base that ties together:
      • the XRAY knowledge source (retrieval target)
      • an Azure OpenAI LLM (for query planning and optional answer synthesis)
      • domain-specific retrieval instructions

    retrieval_instructions tells the LLM how to decompose queries —
    crucial for multi-category adverse keyword scans.

    output_mode is set to "rawDocuments" (not "answerSynthesis") because
    module 07 does its own structured post-processing; we want the raw
    grounding chunks, not a narrated answer.
    """
    existing_kb = [kb.name for kb in client.list_knowledge_bases()]
    if KNOWLEDGE_BASE_NAME in existing_kb:
        log.info("Knowledge base '%s' already exists — skipping.", KNOWLEDGE_BASE_NAME)
        return

    aoai_params = AzureOpenAIVectorizerParameters(
        resource_url    = AZURE_OPENAI_ENDPOINT,
        api_key         = AZURE_OPENAI_KEY,
        deployment_name = CHAT_DEPLOYMENT_NAME,
        model_name      = CHAT_DEPLOYMENT_NAME,   # e.g. "gpt-4o"
    )

    knowledge_base = KnowledgeBase(
        name        = KNOWLEDGE_BASE_NAME,
        description = "XRAY AKS agentic retrieval knowledge base over insurance submissions",

        # Domain instructions — injected into the LLM's query-planning step.
        # This shapes how the internal planner decomposes user intents into
        # sub-queries against the index.
        retrieval_instructions = (
            "This knowledge base contains insurance submission documents including email "
            "correspondence and attachments (PDFs, DOCX, spreadsheets). "
            "When scanning for adverse risk keywords, generate sub-queries for each "
            "risk category mentioned by the user. Prefer passages that contain explicit "
            "mentions of operational practices, physical conditions, personnel behaviour, "
            "or business attributes. "
            "Each document is tagged with ref_id (submission identifier) and doctype "
            "('email_content' or 'attachment'). Always respect any ref_id filter "
            "provided in the query parameters."
        ),

        # We want raw grounding chunks so module 07 can run its own
        # Aho-Corasick scan and build a structured report.
        # Switch to 'answerSynthesis' if you ever want the KB to narrate findings.
        output_mode = "rawDocuments",

        # Which knowledge sources to query
        knowledge_sources = [
            KnowledgeSourceReference(name = KNOWLEDGE_SOURCE_NAME),
        ],

        # LLM for query planning (decomposes the multi-category user query
        # into parallel sub-queries, one per risk category cluster)
        models = [
            KnowledgeBaseAzureOpenAIModel(azure_open_ai_parameters = aoai_params),
        ],

        # "low" reasoning effort = fast parallel query fan-out, no deep thinking.
        # Use "medium" if you want the planner to reason about which categories
        # are most relevant before fanning out queries.
        retrieval_reasoning_effort = KnowledgeRetrievalLowReasoningEffort(),
    )

    client.create_or_update_knowledge_base(knowledge_base)
    log.info("Knowledge base '%s' created.", KNOWLEDGE_BASE_NAME)


# ── Entry point ───────────────────────────────────────────────────────────────

def setup_knowledge_base() -> None:
    """Creates knowledge source and knowledge base in the correct order."""
    client = _get_index_client()
    create_knowledge_source_once(client)
    create_knowledge_base_once(client)
    log.info(
        "Knowledge base setup complete. "
        "Use KnowledgeBaseRetrievalClient('%s') to query.",
        KNOWLEDGE_BASE_NAME,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    setup_knowledge_base()
