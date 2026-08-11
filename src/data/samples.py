"""Shared sample documents and query used across similarity backends."""

from __future__ import annotations

DOCUMENTS: list[str] = [
    "Bugs introduced by the intern had to be squashed by the lead developer.",
    "Bugs found by the quality assurance engineer were difficult to debug.",
    "Bugs are common throughout the warm summer months, according to the entomologist.",
    "Bugs, in particular spiders, are extensively studied by arachnologists.",
]

QUERY: str = (
    "Who is responsible for a coding project and fixing others' mistakes?"
)

# Expected top hit for the QUERY under typical semantic embeddings.
EXPECTED_TOP_DOCUMENT: str = DOCUMENTS[0]
