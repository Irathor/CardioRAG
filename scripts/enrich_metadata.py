"""Phase 20 fix #3: demonstrate real CrossRef metadata enrichment against
the actual corpus, showing exactly which title/author/year values were
wrong before and what CrossRef corrected them to. Read-only - does not
modify the persisted index; enrichment happens at ingestion time in
whatever pipeline chooses to call enrich_metadata_from_crossref().
"""

import logging
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from cardiorag.ingestion.crossref import enrich_metadata_from_crossref
from cardiorag.ingestion.pdf_loader import load_corpus

logging.basicConfig(level=logging.WARNING)


def main() -> None:
    result = load_corpus(Path("data/corpus"))

    for doc in result.documents:
        before = doc.metadata
        if not before.doi:
            print(f"{before.filename}: no DOI extracted, cannot enrich - skipping\n")
            continue

        after = enrich_metadata_from_crossref(before)
        time.sleep(1.0)  # be polite to CrossRef's public API

        print(f"=== {before.filename} (DOI: {before.doi}) ===")
        title_changed = before.title != after.title
        authors_changed = before.authors != after.authors
        year_changed = before.publication_year != after.publication_year

        print(f"  title:   {'CHANGED' if title_changed else 'same'}")
        if title_changed:
            print(f"    before: {before.title!r}")
            print(f"    after:  {after.title!r}")
        print(f"  authors: {'CHANGED' if authors_changed else 'same'} ({len(before.authors)} -> {len(after.authors)})")
        if authors_changed:
            print(f"    before: {before.authors}")
            print(f"    after:  {after.authors}")
        print(f"  year:    {'CHANGED' if year_changed else 'same'} ({before.publication_year} -> {after.publication_year})")
        print()


if __name__ == "__main__":
    main()
