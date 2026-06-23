"""
08_orchestrator.py
==================
Master CLI runner for the complete XRAY AKS Azure AI Search pipeline.

Commands
─────────
  python 08_orchestrator.py --setup          One-time Azure resource creation
  python 08_orchestrator.py --index REF001   Fetch → upload → run indexer
  python 08_orchestrator.py --scan  REF001   Adverse keyword scan via KB
  python 08_orchestrator.py --all   REF001   Index + scan in one shot

One-time setup order:
  1. create_index.py                  → index schema
  2. create_skillset_and_datasource.py→ OCR/embed skillset + blob datasource
  3. create_indexer.py                → indexer with field mappings
  4. create_knowledge_base.py         → knowledge source + knowledge base (NEW)
"""

import argparse
import json
import logging
import sys

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers= [logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("orchestrator")


def run_setup() -> None:
    log.info("════ ONE-TIME SETUP ════")

    from create_index import create_index_once
    create_index_once()

    from create_skillset_and_datasource import setup_once
    setup_once()

    from create_indexer import create_indexer_once
    create_indexer_once()

    # Creates Knowledge Source (wraps the index) + Knowledge Base (attaches LLM)
    from create_knowledge_base import setup_knowledge_base
    setup_knowledge_base()

    log.info("One-time setup complete. All Azure Search resources are ready.")


def run_index(ref_id: str) -> dict:
    log.info("════ INDEXING Ref_Id=%s ════", ref_id)
    from run_indexer import index_ref_id
    result = index_ref_id(ref_id)
    log.info("Indexing result: %s", json.dumps(result))
    return result


def run_scan(ref_id: str) -> dict:
    log.info("════ ADVERSE KEYWORD SCAN Ref_Id=%s ════", ref_id)
    from query_adverse_keywords import run_adverse_keyword_scan, print_report
    result = run_adverse_keyword_scan(
        ref_id           = ref_id,
        output_json_path = f"{ref_id}_aks_report.json",
    )
    print_report(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="XRAY AKS Azure AI Search Pipeline")
    parser.add_argument("--setup", action="store_true",
                        help="Run one-time Azure Search resource setup")
    parser.add_argument("--index", metavar="REF_ID",
                        help="Fetch and index documents for a Ref_Id")
    parser.add_argument("--scan",  metavar="REF_ID",
                        help="Run adverse keyword scan for a Ref_Id")
    parser.add_argument("--all",   metavar="REF_ID",
                        help="Index and scan a Ref_Id in one shot")
    args = parser.parse_args()

    if args.setup:
        run_setup()
    if args.index:
        run_index(args.index)
    if args.scan:
        run_scan(args.scan)
    if args.all:
        run_index(args.all)
        run_scan(args.all)
    if not any([args.setup, args.index, args.scan, args.all]):
        parser.print_help()


if __name__ == "__main__":
    main()
