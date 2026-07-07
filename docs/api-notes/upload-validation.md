# Upload validation & dedup (removed)

The registration endpoint did three things worth keeping. Source:
[`app/routers/books.py`](https://github.com/dworznik/booksmart/blob/589857c/app/routers/books.py).

## 1. Format validation (extension + magic bytes)

Suffix must be `.pdf` or `.epub`, *and* the leading bytes must match — a `.pdf`
of HTML is rejected `415`. EPUB is a ZIP container, so its magic is the ZIP
local-file-header signature.

```python
SUPPORTED_FORMATS = {
    ".pdf": SupportedFormat("pdf", b"%PDF"),
    ".epub": SupportedFormat("epub", b"PK\x03\x04"),
}

def _validated_format(file: UploadFile) -> str:
    suffix = Path(file.filename or "").suffix.lower()
    supported = SUPPORTED_FORMATS.get(suffix)
    if supported is None:
        raise HTTPException(415, f"Unsupported file type {suffix or '(none)'}; expected .pdf or .epub")
    header = file.file.read(len(supported.magic))
    file.file.seek(0)
    if not header.startswith(supported.magic):
        raise HTTPException(415, f"File content does not look like a {supported.name.upper()} file")
    return supported.name
```

`booksmart-core` does **not** validate format on the way in — `store_book` /
`BookStorage.save_original` trust the caller. A consumer (the API, the CLI's
`add`) should port this check; the CLI reads a local path (validate the file it
was handed) rather than a multipart stream. `BookStorage.hash_stream` /
`save_original` and the `Book` model are the core primitives to build on.

## 2. Byte-identical dedup

Registration hashed the content (SHA-256, `file_hash`) and rejected a duplicate
with `409` carrying the existing id — dedup is by *content*, whatever the
metadata says:

```python
file_hash = hash_stream(file.file)
existing = db.scalars(select(Book).where(Book.file_hash == file_hash)).first()
if existing is not None:
    raise HTTPException(409, {"message": "...already registered", "existing_book_id": str(existing.id)})
```

`hash_stream` is still in `booksmart_core.storage`; `Book.file_hash` still holds
the SHA-256. Re-run the `select(...).where(Book.file_hash == ...)` in the consumer.
The store-then-rollback guard (delete the stored original if the DB insert fails)
is worth keeping too — see `register_book`'s `try/except` in the linked source.

## 3. Metadata updates (`BookUpdate`)

[`app/schemas.py`](https://github.com/dworznik/booksmart/blob/589857c/app/schemas.py)
— `PATCH /books/{id}` was a partial update with real rules, all pinned by
(removed) tests:

- `extra="forbid"` → file fields (`checksum`, `file_hash`, `original_filename`,
  `file_format`, `storage_path`, `uploaded_at`) are rejected `422`, never editable.
- Absent fields untouched; explicit `null` clears an optional field.
- `title`/`author` may not be set to `null` (`422`), but any other field may.
- Books stay editable in any state, forever.

The behavior is pure schema logic over `booksmart_core.models.Book`; a consumer
regenerates `BookUpdate` unchanged.
