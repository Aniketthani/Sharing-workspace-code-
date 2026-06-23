"""
04_create_indexer.py
====================
ONE-TIME SETUP — creates the Azure AI Search indexer.

The indexer wires together:
  datasource  →  skillset  →  index

Field mappings translate the enriched document tree produced by the
skillset into the flat index schema we defined in 02_create_index.py.

Output field mappings (from skill outputs → index fields):
  /document/pages/*          → content_text
  /document/pages/*/embedding→ content_vector

We push ref_id and doctype via native metadata fields on the blob
(see 05_run_indexer.py where we set blob metadata before upload).
"""

import logging
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexerClient
from azure.search.documents.indexes.models import (
    SearchIndexer,
    FieldMapping,
    FieldMappingFunction,
    IndexingParameters,
    IndexingParametersConfiguration,
    OutputFieldMapping,
)
from config import (
    AZURE_SEARCH_ENDPOINT,
    AZURE_SEARCH_ADMIN_KEY,
    INDEX_NAME,
    DATASOURCE_NAME,
    SKILLSET_NAME,
    INDEXER_NAME,
)

log = logging.getLogger(__name__)


def _build_indexer() -> SearchIndexer:
    # ── Native field mappings: blob metadata → index fields ───────────────────
    # Azure Blob indexer exposes metadata via  metadata_storage_* keys.
    # Custom blob metadata (set in 05_run_indexer.py) is accessible as
    # metadata_<key_name>.
    field_mappings = [
        FieldMapping(
            source_field_name = "metadata_storage_path",
            target_field_name = "id",
            mapping_function  = FieldMappingFunction(name="base64Encode"),
        ),
        FieldMapping(
            source_field_name = "metadata_storage_name",
            target_field_name = "file_name",
        ),
        FieldMapping(
            source_field_name = "metadata_storage_path",
            target_field_name = "blob_path",
        ),
        # Custom blob metadata we set per-upload (see 05_run_indexer.py)
        FieldMapping(
            source_field_name = "metadata_ref_id",
            target_field_name = "ref_id",
        ),
        FieldMapping(
            source_field_name = "metadata_doctype",
            target_field_name = "doctype",
        ),
        FieldMapping(
            source_field_name = "metadata_email_from",
            target_field_name = "email_from",
        ),
        FieldMapping(
            source_field_name = "metadata_email_to",
            target_field_name = "email_to",
        ),
        FieldMapping(
            source_field_name = "metadata_email_cc",
            target_field_name = "email_cc",
        ),
        FieldMapping(
            source_field_name = "metadata_email_subject",
            target_field_name = "email_subject",
        ),
        FieldMapping(
            source_field_name = "metadata_email_date",
            target_field_name = "email_date",
        ),
    ]

    # ── Output field mappings: skillset outputs → index fields ────────────────
    output_field_mappings = [
        # Each page chunk becomes the content_text field.
        # The indexer expands /document/pages/* into one index doc per chunk.
        OutputFieldMapping(
            source_field_name = "/document/pages/*",
            target_field_name = "content_text",
        ),
        OutputFieldMapping(
            source_field_name = "/document/pages/*/embedding",
            target_field_name = "content_vector",
        ),
        # chunk_id and page_number can be mapped from the array index
        # using a custom mapping function if needed; for now we leave them
        # managed by the indexer's internal expansion logic.
    ]

    # ── Indexer parameters ────────────────────────────────────────────────────
    params = IndexingParameters(
        configuration=IndexingParametersConfiguration(
            parsing_mode                = "default",      # auto-detect: pdf, docx, txt, etc.
            image_action                = "generateNormalizedImages",  # enables OCR skill
            allow_skillset_to_read_file_data=True,
            execution_environment       = "standard",
        ),
        batch_size=16,
        max_failed_items=-1,          # -1 = don't stop on failures
        max_failed_items_per_batch=-1,
    )

    return SearchIndexer(
        name            = INDEXER_NAME,
        description     = "XRAY AKS — pull from blob, enrich, write to index",
        data_source_name= DATASOURCE_NAME,
        skillset_name   = SKILLSET_NAME,
        target_index_name=INDEX_NAME,
        field_mappings  = field_mappings,
        output_field_mappings=output_field_mappings,
        parameters      = params,
        # schedule=IndexingSchedule(interval=timedelta(hours=1))  # uncomment to run on schedule
    )


def create_indexer_once() -> None:
    credential     = AzureKeyCredential(AZURE_SEARCH_ADMIN_KEY)
    indexer_client = SearchIndexerClient(
        endpoint   = AZURE_SEARCH_ENDPOINT,
        credential = credential,
    )

    existing = [ix.name for ix in indexer_client.get_indexers()]
    if INDEXER_NAME in existing:
        log.info("Indexer '%s' already exists — skipping.", INDEXER_NAME)
        return

    indexer = _build_indexer()
    indexer_client.create_indexer(indexer)
    log.info("Indexer '%s' created.", INDEXER_NAME)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    create_indexer_once()
