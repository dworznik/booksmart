# Hand-written stub: pymupdf4llm ships no py.typed marker.
# Covers only the surface this project uses.
from pathlib import Path

import pymupdf

def to_markdown(doc: str | Path | pymupdf.Document, **kwargs: object) -> str: ...
