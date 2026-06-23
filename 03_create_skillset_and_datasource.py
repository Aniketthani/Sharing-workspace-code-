"""
03_create_skillset_and_datasource.py
=====================================
ONE-TIME SETUP — creates:
  1. The Blob Storage data source pointing at the 'aidata' container.
  2. The enrichment skillset with:
       • OCR skill          – extracts text from scanned PDFs / images
       • Merge skill        – merges OCR output + raw text into one field
       • Split skill        – chunks long documents
       • Text Embedding     – produces vectors via Azure OpenAI
       • Entity Recognition – optional; flags names / orgs in text
       • Key Phrase         – optional; surface key phrases per chunk

Run once.  Safe to re-run — existing resources are skipped.
"""

import logging
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexerClient
from azure.search.documents.indexes.models import (
    SearchIndexerDataSourceConnection,
    SearchIndexerDataContainer,
    SearchIndexerDataSourceType,
    SearchIndexerSkillset,
    OcrSkill,
    MergeSkill,
    SplitSkill,
    AzureOpenAIEmbeddingSkill,
    EntityRecognitionSkill,
    KeyPhraseExtractionSkill,
    InputFieldMappingEntry,
    OutputFieldMappingEntry,
    CognitiveServicesAccountKey,
)
from config import (
    AZURE_SEARCH_ENDPOINT,
    AZURE_SEARCH_ADMIN_KEY,
    AZURE_STORAGE_CONNECTION_STRING,
    BLOB_CONTAINER_NAME,
    DATASOURCE_NAME,
    SKILLSET_NAME,
    AZURE_AI_SERVICES_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_KEY,
    EMBEDDING_DEPLOYMENT_NAME,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DIMENSIONS,
    MAX_CHUNK_CHARS,
    CHUNK_OVERLAP_CHARS,
)

log = logging.getLogger(__name__)


# ── 1. Data Source ────────────────────────────────────────────────────────────

def create_datasource_once(client: SearchIndexerClient) -> None:
    existing = [ds.name for ds in client.get_data_source_connections()]
    if DATASOURCE_NAME in existing:
        log.info("Data source '%s' already exists — skipping.", DATASOURCE_NAME)
        return

    datasource = SearchIndexerDataSourceConnection(
        name             = DATASOURCE_NAME,
        type             = SearchIndexerDataSourceType.AZURE_BLOB,
        connection_string= AZURE_STORAGE_CONNECTION_STRING,
        container        = SearchIndexerDataContainer(
            name  = BLOB_CONTAINER_NAME,
            # No query here — we pass the blob paths explicitly through the
            # push-mode indexer in 05_run_indexer.py. The datasource is still
            # required for skillset / indexer association.
        ),
        description      = "XRAY AKS — aidata blob container",
    )
    client.create_data_source_connection(datasource)
    log.info("Data source '%s' created.", DATASOURCE_NAME)


# ── 2. Skillset ───────────────────────────────────────────────────────────────

def _build_skillset() -> SearchIndexerSkillset:
    """
    Skill pipeline (executed in order by Azure AI Search):

      /document/content
          │
          ▼
      [OCR]  → /document/ocr_text           (text from images / scanned PDFs)
          │
          ▼
      [Merge] → /document/merged_text        (ocr_text + original content_text)
          │
          ▼
      [Split] → /document/pages/*            (chunks of ~2000 chars)
          │
          ▼
      [Embedding] → /document/pages/*/embedding   (1536-dim vector per chunk)
          │
      [EntityRecognition] → /document/pages/*/entities
      [KeyPhrase]         → /document/pages/*/keyphrases
    """

    # ── OCR ──────────────────────────────────────────────────────────────────
    ocr_skill = OcrSkill(
        name        = "ocr-skill",
        description = "Extract text from images and scanned documents",
        context     = "/document/normalized_images/*",
        inputs      = [InputFieldMappingEntry(name="image", source="/document/normalized_images/*")],
        outputs     = [OutputFieldMappingEntry(name="text",  target_name="ocr_text")],
        should_detect_orientation=True,
        default_language_code="en",
    )

    # ── Merge (raw text + OCR text) ──────────────────────────────────────────
    merge_skill = MergeSkill(
        name        = "merge-skill",
        description = "Merge original text with OCR output",
        context     = "/document",
        inputs      = [
            InputFieldMappingEntry(name="text",        source="/document/content"),
            InputFieldMappingEntry(name="itemsToInsert", source="/document/normalized_images/*/ocr_text"),
            InputFieldMappingEntry(name="offsets",     source="/document/normalized_images/*/contentOffset"),
        ],
        outputs     = [OutputFieldMappingEntry(name="mergedText", target_name="merged_text")],
    )

    # ── Split into chunks ─────────────────────────────────────────────────────
    split_skill = SplitSkill(
        name              = "split-skill",
        description       = "Chunk merged text for embedding",
        context           = "/document",
        text_split_mode   = "pages",
        maximum_page_length= MAX_CHUNK_CHARS,
        page_overlap_length= CHUNK_OVERLAP_CHARS,
        inputs            = [InputFieldMappingEntry(name="text", source="/document/merged_text")],
        outputs           = [OutputFieldMappingEntry(name="textItems", target_name="pages")],
        default_language_code="en",
    )

    # ── Azure OpenAI Embedding ────────────────────────────────────────────────
    embedding_skill = AzureOpenAIEmbeddingSkill(
        name               = "embedding-skill",
        description        = "Generate text embeddings using Azure OpenAI",
        context            = "/document/pages/*",
        resource_uri       = AZURE_OPENAI_ENDPOINT,
        api_key            = AZURE_OPENAI_KEY,
        deployment_id      = EMBEDDING_DEPLOYMENT_NAME,
        model_name         = EMBEDDING_MODEL_NAME,
        dimensions         = EMBEDDING_DIMENSIONS,
        inputs             = [InputFieldMappingEntry(name="text", source="/document/pages/*")],
        outputs            = [OutputFieldMappingEntry(name="embedding", target_name="embedding")],
    )

    # ── Entity Recognition (optional — surfaces org names, people, locations) ─
    entity_skill = EntityRecognitionSkill(
        name        = "entity-skill",
        description = "Recognize entities: Person, Organization, Location",
        context     = "/document/pages/*",
        categories  = ["Person", "Organization", "Location"],
        default_language_code="en",
        inputs      = [InputFieldMappingEntry(name="text", source="/document/pages/*")],
        outputs     = [OutputFieldMappingEntry(name="namedEntities", target_name="entities")],
    )

    # ── Key Phrase Extraction (optional — summary phrases per chunk) ──────────
    kp_skill = KeyPhraseExtractionSkill(
        name        = "keyphrases-skill",
        description = "Extract key phrases from each chunk",
        context     = "/document/pages/*",
        default_language_code="en",
        inputs      = [InputFieldMappingEntry(name="text", source="/document/pages/*")],
        outputs     = [OutputFieldMappingEntry(name="keyPhrases", target_name="keyphrases")],
    )

    return SearchIndexerSkillset(
        name        = SKILLSET_NAME,
        description = "XRAY AKS enrichment pipeline: OCR → Merge → Split → Embed",
        skills      = [ocr_skill, merge_skill, split_skill, embedding_skill, entity_skill, kp_skill],
        cognitive_services_account=CognitiveServicesAccountKey(key=AZURE_AI_SERVICES_KEY),
    )


def create_skillset_once(client: SearchIndexerClient) -> None:
    existing = [ss.name for ss in client.get_skillsets()]
    if SKILLSET_NAME in existing:
        log.info("Skillset '%s' already exists — skipping.", SKILLSET_NAME)
        return

    skillset = _build_skillset()
    client.create_skillset(skillset)
    log.info("Skillset '%s' created.", SKILLSET_NAME)


# ── Entry point ───────────────────────────────────────────────────────────────

def setup_once() -> None:
    credential = AzureKeyCredential(AZURE_SEARCH_ADMIN_KEY)
    indexer_client = SearchIndexerClient(
        endpoint   = AZURE_SEARCH_ENDPOINT,
        credential = credential,
    )
    create_datasource_once(indexer_client)
    create_skillset_once(indexer_client)
    log.info("One-time setup complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    setup_once()
