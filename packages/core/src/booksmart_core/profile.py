"""Book profile generation: an LLM-produced summary of what a book covers.

The prompt is versioned; bump PROFILE_PROMPT_VERSION whenever its wording
changes so stored profiles record exactly what produced them.
"""

from booksmart_core.models import Book, Chapter

PROFILE_PROMPT_VERSION = "1"

PROFILE_SYSTEM_PROMPT = (
    "You are a librarian building a knowledge repository of technical books. "
    "Given a book's registration metadata, human-supplied hints, and detected "
    "chapter structure, write a concise profile of what the book covers: its "
    "subject matter, scope, intended audience, and the main themes suggested "
    "by its structure. Write plain prose, no headings or lists."
)

_HINT_FIELDS = (
    ("Primary topic", "primary_topic"),
    ("Language", "language"),
    ("Framework", "framework"),
    ("Methodology", "methodology"),
    ("Notes", "notes"),
    ("Trust level", "trust_level"),
    ("Intended use", "intended_use"),
)


def build_profile_prompt(book: Book, chapters: list[Chapter]) -> str:
    lines = [f"Title: {book.title}", f"Author: {book.author}"]
    if book.edition:
        lines.append(f"Edition: {book.edition}")
    if book.publication_year:
        lines.append(f"Publication year: {book.publication_year}")
    if book.isbn:
        lines.append(f"ISBN: {book.isbn}")

    hints = [(label, getattr(book, field)) for label, field in _HINT_FIELDS]
    if any(value for _, value in hints):
        lines.append("")
        lines.append("Hints supplied by the human curator:")
        lines.extend(f"- {label}: {value}" for label, value in hints if value)

    if chapters:
        lines.append("")
        lines.append("Detected structure:")
        for chapter in chapters:
            lines.append(f"- {chapter.title}")
            lines.extend(f"  - {section.title}" for section in chapter.sections)

    lines.append("")
    lines.append("Write the book profile.")
    return "\n".join(lines)
