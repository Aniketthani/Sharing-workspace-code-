"""
00_config.py
============
Central configuration for the XRAY AKS Azure AI Search pipeline.
All environment variables and shared constants live here.
Load this module first before running any other module.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Azure Storage ─────────────────────────────────────────────────────────────
AZURE_STORAGE_CONNECTION_STRING = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
BLOB_CONTAINER_NAME             = "aidata"
EML_FOLDER                      = "files"           # aidata/files/*.eml
ATTACHMENTS_ROOT_FOLDER         = "input"           # aidata/input/<Ref_Id>/

# ── Azure AI Search ───────────────────────────────────────────────────────────
AZURE_SEARCH_ENDPOINT   = os.environ["AZURE_SEARCH_ENDPOINT"]
AZURE_SEARCH_ADMIN_KEY  = os.environ["AZURE_SEARCH_ADMIN_KEY"]
AZURE_SEARCH_QUERY_KEY  = os.environ["AZURE_SEARCH_QUERY_KEY"]

# Index / indexing resources
INDEX_NAME      = "xray-aks-submissions-v1"
DATASOURCE_NAME = "xray-blob-datasource"
SKILLSET_NAME   = "xray-enrichment-skillset"
INDEXER_NAME    = "xray-submission-indexer"

# Agentic retrieval resources  ← these are the two NEW objects
KNOWLEDGE_SOURCE_NAME = "xray-aks-submission-ks"   # wraps INDEX_NAME
KNOWLEDGE_BASE_NAME   = "xray-aks-kb"              # references knowledge source

# ── Azure OpenAI ──────────────────────────────────────────────────────────────
AZURE_OPENAI_ENDPOINT        = os.environ["AZURE_OPENAI_ENDPOINT"]
AZURE_OPENAI_KEY             = os.environ["AZURE_OPENAI_KEY"]
EMBEDDING_DEPLOYMENT_NAME    = "text-embedding-3-small"
EMBEDDING_MODEL_NAME         = "text-embedding-3-small"
EMBEDDING_DIMENSIONS         = 1536
CHAT_DEPLOYMENT_NAME         = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o")

# ── Azure AI Services (for OCR / NLP skills) ──────────────────────────────────
AZURE_AI_SERVICES_KEY = os.environ["AZURE_AI_SERVICES_KEY"]

# ── Search behaviour ──────────────────────────────────────────────────────────
SEMANTIC_CONFIG_NAME  = "xray-semantic-config"
VECTOR_PROFILE_NAME   = "xray-hnsw-profile"
VECTOR_ALGO_NAME      = "xray-hnsw-algo"
TOP_K_RESULTS         = 20     # chunks per KB retrieve call (covers 31 categories)

# ── Chunking ──────────────────────────────────────────────────────────────────
MAX_CHUNK_CHARS       = 2000
CHUNK_OVERLAP_CHARS   = 200
