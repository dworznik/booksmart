# Which EPUB reader can a published core depend on? (issue #87) — findings

> Research notes for [#87](https://github.com/dworznik/booksmart/issues/87), a
> sub-issue of the wayfinder map [#85](https://github.com/dworznik/booksmart/issues/85).
> If the EPUB path stops going through MuPDF and reads the container directly,
> something has to unzip the OCF container, walk the OPF spine, and pull text out
> of XHTML without flattening code listings. `booksmart-core` ships to PyPI, so
> that "something" lands in every consumer's environment — the CLI's
> `~/.booksmart` install and any server consumer alike. This establishes which
> library that should be, and what it costs.
>
> **Frozen at [`c853a4e`](https://github.com/dworznik/booksmart/tree/c853a4e)**
> (2026-08-07), the state of the code this research was gathered against.
> `packages/core/pyproject.toml` was at `booksmart-core` 0.3.1 with the
> dependency list quoted in §6; read the file for what it declares now (see
> [`docs/agents/domain.md`](../agents/domain.md#living-docs-vs-frozen-docs)).
> Every measurement below was run on 2026-08-07 in a throwaway venv on CPython
> 3.12.13 against `selectolax` 0.4.11, `lxml` 6.1.1, `beautifulsoup4` 4.15.0,
> `html5lib` 1.1 and `ebooklib` 0.20 — nothing was added to the repo's
> `pyproject.toml` or `uv.lock`. The corpus runs used the 11 pinned EPUBs in
> `~/booksmart-bench/sources` (the private assets checkout).

## Summary

- **Recommendation: standard library only** — `zipfile` + `xml.etree.ElementTree`
  for the OCF container and the OPF package document, and
  `html.parser.HTMLParser` for the content documents. Zero new PyPI
  dependencies. Full reasoning in §8.
- **Whitespace fidelity does not discriminate.** Measured, not read: five of the
  seven parser/accessor combinations return a `<pre>` node's text
  **byte-identically**, including leading indentation, tabs, an internal blank
  line and trailing spaces (§2). The stdlib is among them. `xml.etree` and
  `html.parser` both scored `EXACT`. The only deviation is that the two
  HTML5-tree-building candidates (`selectolax`, `html5lib`) drop the single
  newline immediately after `<pre>` — which is the HTML5 spec's behaviour, not a
  bug, and is the newline we would strip anyway.
- **The property that actually discriminates is line-break *reconstruction*, and
  no candidate wins it.** On real corpus markup, every candidate's default text
  accessor fuses code lines: `lxml.html.text_content()`,
  `selectolax .text()` and `bs4 .get_text()` all return
  `'m_pDispatcher->register(listener);...m_nMargins++;'` for a
  `<p class="programlisting">` whose lines are separated by `<br/>` (§3). Once
  the *selection* is right, all four — including the stdlib — return the same
  correct string. The extractor's own rules do the work; the library is
  interchangeable.
- **`ebooklib` is AGPL-3.0-or-later.** Confirmed from the installed
  `LICENSE.txt` and every source-file header. That is disqualifying for an
  MIT-licensed library published on PyPI, before any technical argument. It also
  depends on `lxml` *and* `six`, and its `get_content()` does **not** return the
  spine document's bytes — it round-trips through lxml and rewrites the XML
  declaration and doctype (§5).
- **Strict XML parsing of content documents fails on the real corpus.**
  `tdd-by-example` has **25 of its 123 spine documents** raise
  `undefined entity &nbsp;` under `xml.etree`, and the same under `lxml.etree`
  (§4). Neither fetches the external DTD, so a valid XHTML 1.1 doctype does not
  rescue them. Worse, `lxml.etree(recover=True)` *silently deletes* the
  entity — `'a&nbsp;b&mdash;c'` comes back as `'abc'`. Content documents must go
  through an HTML parser; the container and OPF, which are tool-written XML, are
  safe under `xml.etree` and parsed cleanly on all 11 books.
- **The stdlib route works on all 11 pinned EPUBs.** A 19-line
  `spine_documents()` reads container → OPF → manifest → spine with zero
  failures, and surfaces the code the current MuPDF path misses: `<pre>` counts
  of 944 (`effective-typescript-2e`) and 1,077 (`fp-in-scala-2e`) against the
  **0 fences / 499pp** and near-zero the map records for pymupdf4llm (§4).
- **Core has no HTML or XML parser in its tree today, transitively or
  otherwise** (§6) — so the "cheapest answer is already installed" hypothesis in
  the ticket is false for every third-party candidate, but true for the stdlib.
- **Cost of the alternatives, for a library on PyPI:** `lxml` adds ~5.2 MB
  per wheel and 12 MB installed; `selectolax` ~2.4 MB and 14 MB installed, and
  **caps Python at `<3.15`** — an upper bound `booksmart-core` would inherit and
  propagate to every consumer (§7).

Every claim below is cited to a primary source: a measurement script run on
2026-08-07 (quoted output verbatim), PyPI's JSON API for the package under
discussion, an installed package's own source or `LICENSE`, or the W3C/WHATWG
specification. Where something could not be verified against a primary source it
says so.

---

## 1. The candidates, head to head

| candidate | what it actually is | text accessor | selection | new runtime deps | licence |
| --- | --- | --- | --- | --- | --- |
| `zipfile` + `xml.etree` + `html.parser` | stdlib. `html.parser` is a pure-Python **tokenizer**, not a tree builder | `handle_data` callback | write your own | **none** | PSF (stdlib) |
| `selectolax` | Cython bindings over **Lexbor** (default) or **Modest**, both C HTML5 engines | `.text(deep=, separator=, strip=)` | CSS (`.css`, `.css_first`) | 0 (compiled ext) | MIT |
| `lxml` | Cython bindings over **libxml2/libxslt**. `lxml.etree` = XML parser, `lxml.html` = libxml2's HTML parser | `.text_content()`, `.itertext()` | XPath built in; **CSS needs the separate `cssselect` package** | 0 (compiled ext) | BSD-3-Clause |
| `BeautifulSoup` | pure-Python **tree wrapper**; delegates parsing to a backend — `html.parser` (stdlib), `lxml`, `lxml-xml`, or `html5lib` | `.get_text(separator=)` | CSS via `soupsieve` | `soupsieve`, `typing-extensions` (+ backend) | MIT |
| `ebooklib` | EPUB-specific reader/writer over `lxml` | `.get_content()`, `.get_body_content()` | none of its own | `lxml`, `six` | **AGPL-3.0-or-later** |

`BeautifulSoup` is not a parser. Naming it as a candidate means naming a backend,
and the backend is what determines both the whitespace behaviour (§2) and the
dependency (`bs4` + `soupsieve` + `typing-extensions` alone, or that plus `lxml`,
or that plus `html5lib` + `six` + `webencodings`). Over the stdlib backend, `bs4`
buys an API and costs 3× the parse time (§7) — it adds no parsing capability the
stdlib does not already have, because it *is* the stdlib underneath.

`lxml`'s CSS gap is measured, not assumed: calling `.cssselect()` on a fresh
`lxml` 6.1.1 install raises
`ImportError: cssselect does not seem to be installed`. Class-based selection
against `lxml` alone means XPath like
`//td[contains(concat(' ',normalize-space(@class),' '),' codeline ')]`, or a
third dependency.

---

## 2. Whitespace fidelity — measured

The ticket calls this the decisive property, so it was measured rather than read.
The fixture is an XHTML document whose `<pre>` carries every hazard at once, its
bytes fixed by explicit escapes:

```python
PRE_BODY = (
    "\n"                     # newline right after <pre>
    "def outer():\n"
    "    if flag:\n"
    "\t\tvalue = 1\n"        # tab indentation
    "\n"                     # internal blank line
    "    return value   \n"  # trailing spaces on a code line
    "  "                     # trailing indentation before </pre>
)
```

The assertion is byte-identity against `PRE_BODY`. Verbatim output:

```
## stdlib xml.etree.ElementTree (.itertext)   -> EXACT
   pre    : '\ndef outer():\n    if flag:\n\t\tvalue = 1\n\n    return value   \n  '
## stdlib html.parser.HTMLParser (handle_data)   -> EXACT
   pre    : '\ndef outer():\n    if flag:\n\t\tvalue = 1\n\n    return value   \n  '
## lxml.etree XML parser (.itertext)   -> EXACT
   pre    : '\ndef outer():\n    if flag:\n\t\tvalue = 1\n\n    return value   \n  '
## lxml.html HTML parser (.text_content)   -> EXACT
   pre    : '\ndef outer():\n    if flag:\n\t\tvalue = 1\n\n    return value   \n  '
## selectolax LexborHTMLParser (.text(deep=True))   -> EXACT minus leading \n
   pre    : 'def outer():\n    if flag:\n\t\tvalue = 1\n\n    return value   \n  '
## selectolax HTMLParser/Modest (.text(deep=True))   -> EXACT minus leading \n
   pre    : 'def outer():\n    if flag:\n\t\tvalue = 1\n\n    return value   \n  '
## BeautifulSoup(features='html.parser').get_text()   -> EXACT
## BeautifulSoup(features='lxml').get_text()   -> EXACT
## BeautifulSoup(features='lxml-xml').get_text()   -> EXACT
## BeautifulSoup(features='html5lib').get_text()   -> EXACT minus leading \n
```

**Nothing normalises.** Not one candidate collapsed runs of spaces, converted
tabs, dropped the blank line, or trimmed the trailing whitespace. The worry the
ticket raises — "this is exactly where HTML parsers differ quietly" — is not
borne out for `<pre>` content.

The one deviation is a spec requirement, not a defect. `selectolax` (both
backends) and `html5lib` drop the newline immediately following `<pre>`. This is
the HTML5 tree-construction rule for `pre`/`listing`/`textarea`; the installed
`html5lib` 1.1 implements it at `html5parser.py:986-992`, under the comment
`# want to drop leading newlines`:

```python
if (data.startswith("\n") and
        self.tree.openElements[-1].name in ("pre", "listing", "textarea") and
        not self.tree.openElements[-1].hasContent()):
    data = data[1:]
```

`lxml.html` and `html.parser` do not apply it — `html.parser` because it is a
tokenizer with no tree-construction stage at all. Either behaviour is fine here:
that newline is a typesetting artefact and the extractor strips it before
fencing.

The XML-side result has a normative basis. [XML 1.0 §2.10](https://www.w3.org/TR/xml/)
requires that "An XML processor MUST always pass all characters in a document
that are not markup through to the application", with §2.11 permitting only
end-of-line normalisation of `#xD #xA` and lone `#xD` to `#xA`. So `xml.etree`
and `lxml.etree` returning `EXACT` is guaranteed, not incidental — which matters
because the OPF and container are parsed as XML.

### The `<div class="programlisting">` / one-element-per-line convention

The ticket asks how each candidate reconstructs line breaks when a publisher sets
one `<p>` per code line. With the `<p>`s selected individually and joined on
`\n`, every candidate returned
`'def outer():\n    if flag:\n        return 1'` — identical, and with the
leading indentation intact.

The failure appears when the *container* is flattened instead. Asking for the
wrapper's text with a separator does not produce one line per code line; it
produces one segment per **text node**:

```
## selectolax lexbor div.text(deep=True, separator='\n') on the listing
   '\n\ndef outer():\n\n\n    if flag:\n\n\n        return 1\n\n'
## BeautifulSoup div.get_text('\n') on the listing
   '\n\ndef outer():\n\n\n    if flag:\n\n\n        return 1\n\n'
```

Bit-for-bit the same wrong answer from a C HTML5 engine and from pure Python.
The separator argument is not a line-reconstruction feature.

---

## 3. Line-break reconstruction on real corpus markup

Two pinned books carry **zero** `<pre>` elements and set code some other way.
Both fragments below are lifted from the pinned files.

**A — `working-effectively-with-legacy-code`**, `OEBPS/html/pre03.html`, verbatim:
`<p class="programlisting">m_pDispatcher-&gt;register(listener);<br/>...<br/>m_nMargins++;</p>`
(361 such paragraphs in the book).

**B — `pragmatic-programmer-2e`**, `OEBPS/f_0027.xhtml`: a
`<table class="processedcode">` with one `<tr>` per code line, each row a
`<td class="codeinfo">` gutter plus a `<td class="codeline">` — 163 listing
tables, 1,254 code lines.

| candidate / accessor | A (`<br/>`) | B (`<tr>`) |
| --- | --- | --- |
| stdlib `html.parser`, naive concat | LINES FUSED | wrong |
| stdlib `html.parser` + `<br>`/block rule | **RECONSTRUCTED** | wrong |
| `lxml.html .text_content()` | LINES FUSED | wrong |
| `selectolax` `.text(deep=True)` | LINES FUSED | wrong |
| `selectolax` `.text(separator='\n')` | **RECONSTRUCTED** | wrong |
| `bs4 .get_text()` | LINES FUSED | wrong |
| `bs4 .get_text('\n')` | **RECONSTRUCTED** | wrong |

Verbatim, the fused result — the same string from all three third-party
libraries and from the stdlib:

```
   A (<br/>)  LINES FUSED    'm_pDispatcher->register(listener);...m_nMargins++;'
```

Three code lines silently welded into one. This — not `<pre>` normalisation — is
the whitespace hazard that actually exists in the corpus, and **every candidate
has it by default**.

Case B needs real selection, because flattening the table drags the gutter cell's
content into every line. Selecting `td.codeline`:

```
B with the right selection (td.codeline, one line per cell):
  selectolax css('td.codeline')      RECONSTRUCTED  'def print_balance(account)\n  printf "Debits: %10.2f", account.debits'
  lxml cssselect('td.codeline')      RECONSTRUCTED  'def print_balance(account)\n  printf "Debits: %10.2f", account.debits'
  bs4 select('td.codeline')          RECONSTRUCTED  'def print_balance(account)\n  printf "Debits: %10.2f", account.debits'
  stdlib html.parser (class-aware)   RECONSTRUCTED  'def print_balance(account)\n  printf "Debits: %10.2f", account.debits'
```

**Four for four, byte-identical.** The extractor's rules — which tags are code
containers, which class attributes mark them, where a line ends — are the whole
job. The library underneath contributes nothing that distinguishes it.

---

## 4. The stdlib route against the real corpus

`spine_documents()` — container → OPF → manifest → spine, honouring
`linear="no"` and resolving hrefs relative to the OPF's directory — is **19 lines
of code** using only `zipfile`, `xml.etree.ElementTree` and `posixpath`. The
paths it depends on are normative: [EPUB 3.3](https://www.w3.org/TR/epub-33/)
fixes the container at `META-INF/container.xml`, has its `rootfile/@full-path`
name the package document, and defines the spine's `itemref` sequence as the
default reading order.

Run over all 11 pinned EPUBs, counting with `html.parser` and separately
attempting a strict `xml.etree` parse of each content document:

| book | spine docs | `<pre>` | non-blank `<pre>` lines | `<code>` | code-ish class | strict XML |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| aposd-2e | 32 | 0 | 0 | 0 | 0 | OK |
| code-complete-2e | 728 | 0 | 0 | 10 | 0 | OK |
| effective-python-3e | 41 | 1374 | 11571 | 6225 | 1196 | OK |
| effective-typescript-2e | 21 | 944 | 5945 | 33879 | 0 | OK |
| fp-in-scala-2e | 26 | 1077 | 5008 | 8022 | 528 | OK |
| poodr-2e | 199 | 0 | 0 | 2220 | 159 | OK |
| pragmatic-programmer-2e | 84 | 0 | 0 | 1 | 37 | OK |
| python-distilled | 31 | 926 | 5354 | 4574 | 752 | OK |
| refactoring-2e | 37 | 737 | 5658 | 470 | 708 | OK |
| tdd-by-example | 123 | 311 | 1489 | 0 | 0 | **25/123 fail (undefined entity &nbsp;)** |
| working-effectively-with-legacy-code | 40 | 0 | 0 | 1581 | 405 | OK |

Three things fall out.

**The spine reader never failed.** 11/11, including the 728-document
calibre-converted `code-complete-2e`. No EPUB library was needed to find the
reading order.

**Strict XML is not viable for content documents.** `tdd-by-example` fails on 25
documents with `undefined entity &nbsp;`. A valid doctype does not save it —
neither `xml.etree` nor `lxml.etree` fetches the external DTD subset:

```
XHTML 1.1 DOCTYPE + &nbsp;&mdash; (external DTD not fetched)
  xml.etree  : ParseError: undefined entity &nbsp;: line 3, column 53
  lxml.etree default                : XMLSyntaxError: Entity 'nbsp' not defined, line 3, column 60
  lxml.etree resolve_entities=False : 'a&nbsp;b&mdash;c'
  lxml.etree recover=True           : 'abc'
  html.parser                       : 'a\xa0b—c'
  selectolax/lexbor                 : 'a\xa0b—c'
```

`recover=True` is the trap: it does not fail, it **silently deletes the
character**. `resolve_entities=False` leaves the raw entity text in the output.
Only the HTML parsers get it right, and the stdlib is one of them —
`html.entities.html5` carries all 2,231 HTML5 named references, and
`HTMLParser(convert_charrefs=True)` resolves them.

**The recovery thesis is measurable.** The map records `effective-typescript-2e`
producing **0 fences over 499 pages** and `pragmatic-programmer-2e` **1 over
468** through the current pymupdf4llm path. Reading the containers directly finds
944 `<pre>` elements holding 5,945 non-blank code lines in the first, and 163
`table.processedcode` listings holding 1,254 `td.codeline` cells in the second.
The code was never missing; it was thrown away in the PDF-shaped rendering.

Two decline cases are worth recording for the extractor's rules, since neither is
solved by any library: **`aposd-2e`** has fully obfuscated class names
(`p.class_s6k2`, `span.class_s2c1`) with no semantic hook whatsoever, and
**`code-complete-2e`** has 728 calibre-split documents (`dummy_split_448.html`)
with only 10 `<code>` elements in the entire book.

---

## 5. Does an EPUB-specific library earn a dependency?

No, on three independent grounds.

**Licence — disqualifying on its own.** `ebooklib` 0.20's installed
`LICENSE.txt` opens `GNU AFFERO GENERAL PUBLIC LICENSE, Version 3`, its PyPI
classifier is
`License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)`,
and every source file (`ebooklib/epub.py:5-14`) repeats it. `booksmart-core` is
`license = "MIT"`. AGPL is not MIT-compatible in the direction that matters: an
MIT-licensed library that requires an AGPL library at runtime pushes AGPL §13's
network-use source-disclosure obligation onto every consumer, including any
server consumer. This is a legal fact about the published artefact, independent
of code quality.

**It does not remove the parsing problem.** `ebooklib` gives you the container
and the spine; it gives you nothing for the XHTML. Its own answer is `lxml`,
which it declares as a hard dependency alongside `six` (
`Requires-Dist: ['lxml', 'six', ...]`, read from the installed metadata). Taking
`ebooklib` means taking `lxml` **and** `six` **and** the AGPL, to replace the
19 lines in §4.

**Its content accessor is lossy in an unexpected direction.** Against a
hand-built EPUB whose chapter bytes are known exactly:

```
get_content() type: bytes
get_content() byte-identical to source: False
  source : b'<?xml version="1.0" encoding="utf-8"?>\n<html xmlns="http://www.w3.org/1999/xhtml"><head>...
  ebooklib: b'<?xml version=\'1.0\' encoding=\'utf-8\'?>\n<!DOCTYPE html>\n<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www...
```

`get_content()` is a **re-serialisation** through lxml, not the archive member.
It rewrote the XML declaration's quoting, injected a `<!DOCTYPE html>` and added
an `xmlns:epub` declaration the source never had. The `<pre>` bytes did survive
in this instance, but a reader that rewrites its input is the wrong foundation
for an extractor whose whole premise is byte-level code fidelity — and it means
any strict-XML failure mode of §4 is inherited, silently, at read time.

By contrast, `zipfile.ZipFile.read()` returned the chapter
**byte-identical to source: True**.

The division of labour is therefore: OPF/spine is 19 stdlib lines, and the hard
part — deciding what is code — is booksmart's own rules either way.

---

## 6. What core already depends on

`packages/core/pyproject.toml` at `c853a4e` (`booksmart-core` 0.3.1,
`requires-python = ">=3.12"`, `license = "MIT"`):

```toml
dependencies = [
    "alembic>=1.18.5",
    "anthropic>=0.116.0",
    "openai>=2.44.0",
    "psycopg[binary]>=3.3.4",
    "pydantic>=2.0",
    "pymupdf4llm>=1.28.0",
    "qdrant-client>=1.18.0,<1.19.0",
    "sqlalchemy>=2.0",
]
```

`uv tree --package booksmart-core` resolves 67 packages. Searching the whole
tree and `uv.lock` for `lxml`, `beautifulsoup4`, `soupsieve`, `html5lib`,
`selectolax` or `ebooklib` returns **nothing** — not as a direct dependency, not
transitively, not in the lock. The nearest thing to a parser in the tree is
`pyyaml` (via `pymupdf-layout`).

So the ticket's cheapest-answer hypothesis — "if a suitable parser is already in
the tree transitively, use it" — resolves in the negative for every third-party
candidate. Each one is a genuinely new dependency. It resolves in the
*affirmative* only for the standard library, which is already in every
consumer's environment by construction.

Core's Python floor is **3.12**, and CI runs `ubuntu-latest` only
(`.github/workflows/ci.yml` — three jobs, all `runs-on: ubuntu-latest`). There
is no macOS or Windows job, so a wheel gap on those platforms would not be
caught here; it would be caught by a user.

---

## 7. Wheels, maintenance, licence

Wheel data read from `https://pypi.org/pypi/<name>/json` on 2026-08-07, filtered
to the `cp312` tag (core's floor).

| platform | `selectolax` 0.4.11 | `lxml` 6.1.1 | `beautifulsoup4` 4.15.0 | `ebooklib` 0.20 | stdlib |
| --- | --- | --- | --- | --- | --- |
| manylinux x86_64 | yes, 2.43 MB | yes, 5.09–5.24 MB | pure `py3-none-any` | pure `py3-none-any` | n/a |
| manylinux aarch64 | yes, 2.38 MB | yes, 4.93–5.01 MB | " | " | n/a |
| musllinux x86_64 / aarch64 | yes, 2.45 / 2.39 MB | yes, 5.26 / 5.05 MB | " | " | n/a |
| macOS x86_64 | yes, 2.25 MB | yes, 4.62 MB | " | " | n/a |
| macOS arm64 | yes, 2.30 MB (dedicated) | via `universal2`, 8.57 MB | " | " | n/a |
| Windows amd64 / arm64 | yes, 1.88 / 1.82 MB | yes, 4.00 / 3.66 MB | " | " | n/a |
| linux i686, ppc64le, armv7l, riscv64 | **no — compiler required** | yes (5.35 / 5.63 / 4.69 / 5.24 MB) | " | " | n/a |
| PyPy | **no** | yes (`pp311`) | " | " | n/a |

Both compiled candidates cover every platform a booksmart consumer plausibly
runs on. `lxml`'s coverage is broader — it is one of the most thoroughly
wheel-built packages on PyPI — and `selectolax` would require a C toolchain on
the exotic architectures and on PyPy. Installed footprint: `lxml` 12 MB,
`selectolax` 14 MB, `bs4` 816 KB + `soupsieve` 324 KB, `ebooklib` 208 KB (plus
its `lxml`), stdlib 0.

| | last release | releases in 2025 / 2026 | total | licence | MIT-compatible? |
| --- | --- | --- | --- | --- | --- |
| `selectolax` | 2026-07-15 (0.4.11) | 13 / 5 | 50 since 2018 | MIT (`license_expression`) | yes |
| `lxml` | 2026-05-18 (6.1.1 stable); 7.0.0a3 on 2026-06-17 | 6 / 7 | 127 since 2007 | BSD-3-Clause | yes |
| `beautifulsoup4` | 2026-06-07 (4.15.0) | 11 / 1 | 54 since 2013 | MIT | yes |
| `soupsieve` | 2026-08-07 (2.9.2) | 3 / 6 | 55 since 2018 | MIT | yes |
| `ebooklib` | 2025-10-26 (0.20) | 2 / 0 | **8 since 2013** | **AGPL-3.0-or-later** | **no** |

All four non-`ebooklib` candidates are actively maintained and permissively
licensed; maintenance is not a discriminator either. `ebooklib` is the outlier on
both axes — eight releases in twelve years, with a 2019→2022→2025 gap, and the
wrong licence.

**One version constraint is worth singling out.** `selectolax` 0.4.11 declares
`requires_python: '<3.15,>=3.9'` — an **upper** bound. A published library that
depends on it inherits that ceiling: on the day CPython 3.15 ships,
`booksmart-core` becomes uninstallable on it until `selectolax` cuts a release,
and every consumer's resolver enforces that. `lxml` (`>=3.8`),
`beautifulsoup4` (`>=3.7.0`) and `soupsieve` (`>=3.10`) declare no ceiling. For
an application this is an annoyance; for a library on PyPI it is a liability you
hand to strangers.

### Speed, the one honest argument against the stdlib

`html.parser` is pure Python. Measured over whole pinned EPUBs (spine read +
full text extraction of every content document):

| book | XHTML | `zipfile`+`xml.etree` spine | stdlib `html.parser` | `lxml.html` | `selectolax` | `bs4`+`html.parser` | `bs4`+`lxml` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| code-complete-2e (728 docs) | 5.52 MB | 436 ms | 7,755 ms | 1,594 ms | 1,413 ms | 24,152 ms | 17,165 ms |
| effective-typescript-2e (21 docs) | 1.97 MB | 88 ms | 3,692 ms | 734 ms | 970 ms | 12,628 ms | 8,704 ms |
| pragmatic-programmer-2e (84 docs) | 1.05 MB | 57 ms | 1,402 ms | 346 ms | 199 ms | 3,532 ms | 3,601 ms |

The stdlib is ~4–5× slower than the compiled engines. In absolute terms the
worst case in the corpus is **7.8 seconds, once, at ingestion**, against a
pipeline that already spends minutes in LLM calls per book and, per the map,
~240× that in `marker`. It is not a bottleneck.

The genuinely slow option is `BeautifulSoup`, at 3× the stdlib tokenizer and up
to 24 seconds a book — its Python tree-building dominates regardless of backend,
so `bs4`+`lxml` is slower than raw `html.parser`. That rules `bs4` out on
performance as well as on dependency count.

---

## 8. Recommendation

**Use the standard library only: `zipfile` + `xml.etree.ElementTree` for the OCF
container and the OPF package document, and `html.parser.HTMLParser` for the
content documents. Add no dependency to `packages/core/pyproject.toml`.**

The whitespace-fidelity measurement is what settles it. The stdlib returns a
`<pre>` node's text byte-identically:

```
## stdlib html.parser.HTMLParser (handle_data)   -> EXACT
   pre    : '\ndef outer():\n    if flag:\n\t\tvalue = 1\n\n    return value   \n  '
## stdlib xml.etree.ElementTree (.itertext)   -> EXACT
   pre    : '\ndef outer():\n    if flag:\n\t\tvalue = 1\n\n    return value   \n  '
```

— leading indentation, tabs, the internal blank line and the trailing spaces all
intact, identical to `lxml.html` and `lxml.etree`, and strictly *more* faithful
than `selectolax`, which drops the newline after `<pre>` per the HTML5 spec. The
property the ticket named as decisive turns out not to separate the candidates at
all, and where it moves at all it moves in the stdlib's favour. On the property
that *does* bite — reconstructing line breaks across `<br/>` and one-element-per-line
listings — all four candidates fail identically by default and all four succeed
identically once the extractor's own selection rules are right (§3). There is no
capability here to buy.

**The install-cost consequence, stated plainly.** `booksmart-core` is on PyPI. A
dependency added here is not a private choice: it is resolved, downloaded and
installed in every consumer's environment forever, and its constraints become
booksmart's constraints. Concretely, that would mean shipping a 5.2 MB compiled
wheel and 12 MB on disk for `lxml`, or 2.4 MB and 14 MB for `selectolax` — plus,
in `selectolax`'s case, propagating a `python <3.15` **upper bound** into every
consumer's resolver, so that `booksmart-core` stops installing on the day
CPython 3.15 ships and stays broken until an upstream release. It would also mean
inheriting that project's release cadence, security surface, and wheel coverage
on any platform booksmart's `ubuntu-latest`-only CI does not test (§6). None of
that is catastrophic; all of it is unnecessary, because the measurement shows the
paid option does not parse the corpus any better than the free one.

`ebooklib` is out on licence before anything else: it is AGPL-3.0-or-later, and
an MIT library cannot pull it into a consumer's runtime without pushing AGPL §13
obligations onto that consumer (§5).

### Consequences for the extractor's design

1. **Two parsers, deliberately.** `xml.etree` for `META-INF/container.xml` and
   the OPF — tool-written XML, clean on 11/11 books, and XML §2.10 guarantees
   the whitespace. `html.parser` for content documents, because strict XML dies
   on 25 of `tdd-by-example`'s 123 files (§4) and every `lxml.etree` workaround
   either corrupts (`recover=True` → `'abc'`) or under-decodes
   (`resolve_entities=False`).
2. **`convert_charrefs=True`.** The stdlib carries all 2,231 HTML5 named
   references in `html.entities.html5`; this is what makes the HTML route
   immune to §4's entity failures.
3. **Line breaks come from element boundaries, not from text.** `<br/>` and the
   close of a block-level element each end a code line. This must be explicit —
   the measurement shows no library does it for you, and `separator=` arguments
   emit one segment per *text node*, not per line (§2).
4. **Selection is class-driven and per-book-family.** The corpus needs at least
   `pre`, `p.programlisting` + `<br/>`, and `table.processedcode` /
   `td.codeline`. Writing this against `html.parser`'s `handle_starttag(tag,
   attrs)` is direct; it is the same rule set any library would need.
5. **Two documented decline cases.** `aposd-2e` (obfuscated class names, no
   semantic hook) and `code-complete-2e` (728 calibre-split documents, 10
   `<code>` elements total) offer the extractor nothing to key on. These belong
   with the map's open "what the router does when detection declines" question —
   they are decline cases for the *EPUB* path, and no library choice changes
   that.
6. **The MuPDF segfault dissolves for EPUB.** The map notes one corpus EPUB
   SIGSEGVs MuPDF, uncatchable by `ParserChain`. A stdlib container reader never
   enters MuPDF, so subprocess isolation would only be needed for PDFs.

### Not verified against a primary source

- The WHATWG HTML tree-construction clause for `pre`/`listing` could not be
  fetched verbatim (the parsing spec page exceeds the fetch limit). The rule is
  instead evidenced by the installed `html5lib` 1.1 implementation quoted in §2
  and by the measured behaviour of two independent HTML5 engines. The XML-side
  guarantee *was* verified verbatim against W3C XML 1.0 §2.10/§2.11.
- Wheel coverage is reported for the current release of each package only, not
  across history. A future release could drop a platform.
- The corpus figures cover the 11 pinned EPUBs. `<pre>`/`<code>`/class counts are
  a proxy for how much code exists, not a measure of extraction quality; that
  comparison belongs in `packages/core/tests/parser_eval.py` once an extractor
  exists.
