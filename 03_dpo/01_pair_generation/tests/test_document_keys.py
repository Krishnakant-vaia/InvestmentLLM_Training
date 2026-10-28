"""The document pipeline must not pick up the tear sheets that live beside its documents."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
import prep_corpus  # noqa: E402


def test_tear_sheets_are_not_documents_of_the_document_pipeline():
    p = config.DOCS_PREFIX
    keys = [p + "CIBIL Credit Scoring/a.md", p + "Private Markets Terminology/b.md", p + "tear sheets/01_Ledgerly_Technologies_Tear_Sheet.md",
            p + "Credit Rating Replication/c.txt"]
    assert prep_corpus.document_keys(keys) == [p + "CIBIL Credit Scoring/a.md", p + "Private Markets Terminology/b.md"]


def test_cims_are_not_documents_of_the_document_pipeline():
    p = config.DOCS_PREFIX
    assert prep_corpus.document_keys([p + "cims/Alderwood_CIM.md", p + "CIBIL Credit Scoring/a.md"]) == [p + "CIBIL Credit Scoring/a.md"]
