"""
02_create_index.py
==================
ONE-TIME SETUP — creates the Azure AI Search index.
Run this script once.  If the index already exists the script exits safely.

Index schema:
  - id             (key)
  - ref_id         (filterable, facetable)
  - doctype        (filterable)  — "email_content" | "attachment"
  - file_name      (retrievable)
  - blob_path      (retrievable)
  - content_text   (searchable)  — plain text extracted from file
  - content_vector (searchable)  — 1536-dim embedding
  - email_from / email_to / email_subject / email_date  (retrievable, filterable)
  - page_number    (for chunked docs)
  - chunk_id       (for chunked docs)
"""

import logging
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField,
    SemanticSearch,
)
from config import (
    AZURE_SEARCH_ENDPOINT,
    AZURE_SEARCH_ADMIN_KEY,
    INDEX_NAME,
    EMBEDDING_DIMENSIONS,
    VECTOR_PROFILE_NAME,
    VECTOR_ALGO_NAME,
    SEMANTIC_CONFIG_NAME,
)

log = logging.getLogger(__name__)


def build_index_definition() -> SearchIndex:
    fields = [
        # ── Key ──────────────────────────────────────────────────────────────
        SimpleField(
            name="id",
            type=SearchFieldDataType.String,
            key=True,
            filterable=True,
        ),

        # ── Submission metadata ───────────────────────────────────────────────
        SimpleField(
            name="ref_id",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
            retrievable=True,
        ),
        SimpleField(
            name="doctype",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
            retrievable=True,
        ),
        SimpleField(
            name="file_name",
            type=SearchFieldDataType.String,
            retrievable=True,
        ),
        SimpleField(
            name="blob_path",
            type=SearchFieldDataType.String,
            retrievable=True,
        ),

        # ── Email-specific headers (populated for email_content docs) ─────────
        SimpleField(name="email_from",    type=SearchFieldDataType.String, retrievable=True, filterable=True),
        SimpleField(name="email_to",      type=SearchFieldDataType.String, retrievable=True),
        SimpleField(name="email_cc",      type=SearchFieldDataType.String, retrievable=True),
        SimpleField(name="email_subject", type=SearchFieldDataType.String, retrievable=True, filterable=True),
        SimpleField(name="email_date",    type=SearchFieldDataType.String, retrievable=True, filterable=True),

        # ── Chunking fields ───────────────────────────────────────────────────
        SimpleField(name="chunk_id",    type=SearchFieldDataType.Int32,  retrievable=True, filterable=True),
        SimpleField(name="page_number", type=SearchFieldDataType.Int32,  retrievable=True, filterable=True),

        # ── Primary searchable content ────────────────────────────────────────
        SearchableField(
            name="content_text",
            type=SearchFieldDataType.String,
            retrievable=True,
            analyzer_name="en.lucene",      # BM25 over English text
        ),

        # ── Vector field ──────────────────────────────────────────────────────
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            retrievable=False,              # saves storage; vectors are not needed in results
            vector_search_dimensions=EMBEDDING_DIMENSIONS,
            vector_search_profile_name=VECTOR_PROFILE_NAME,
        ),
    ]

    # ── Vector search config ──────────────────────────────────────────────────
    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name=VECTOR_ALGO_NAME,
                parameters={
                    "m": 4,
                    "efConstruction": 400,
                    "efSearch": 500,
                    "metric": "cosine",
                },
            )
        ],
        profiles=[
            VectorSearchProfile(
                name=VECTOR_PROFILE_NAME,
                algorithm_configuration_name=VECTOR_ALGO_NAME,
            )
        ],
    )

    # ── Semantic search config ────────────────────────────────────────────────
    semantic_search = SemanticSearch(
        configurations=[
            SemanticConfiguration(
                name=SEMANTIC_CONFIG_NAME,
                prioritized_fields=SemanticPrioritizedFields(
                    content_fields=[SemanticField(field_name="content_text")],
                    keywords_fields=[
                        SemanticField(field_name="email_subject"),
                        SemanticField(field_name="doctype"),
                    ],
                    title_field=SemanticField(field_name="file_name"),
                ),
            )
        ]
    )

    return SearchIndex(
        name=INDEX_NAME,
        fields=fields,
        vector_search=vector_search,
        semantic_search=semantic_search,
    )


def create_index_once() -> None:
    credential    = AzureKeyCredential(AZURE_SEARCH_ADMIN_KEY)
    index_client  = SearchIndexClient(endpoint=AZURE_SEARCH_ENDPOINT, credential=credential)

    existing = [idx.name for idx in index_client.list_index_names()]
    if INDEX_NAME in existing:
        log.info("Index '%s' already exists — skipping creation.", INDEX_NAME)
        return

    index_def = build_index_definition()
    created   = index_client.create_index(index_def)
    log.info("Index '%s' created successfully.", created.name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    create_index_once()
