# What code markup do the corpus EPUBs actually use? (issue #86) — findings

> Research notes for [#86](https://github.com/dworznik/booksmart/issues/86),
> under the wayfinder map [#85](https://github.com/dworznik/booksmart/issues/85).
> The incoming spec for a direct-container EPUB extractor asserts that `<pre>`
> and `<pre><code>` carry code, with an O'Reilly/Manning quirk of
> `<div class="programlisting|codelisting|code">` wrapping styled lines. Nobody
> had checked. This establishes, per book, which elements and class names
> actually carry block code, whether the whitespace inside them survives a
> direct XHTML read, how inline code is marked, and whether the OPF spine is a
> trustworthy reading order.
>
> **Frozen at [`c853a4e`](https://github.com/dworznik/booksmart/tree/c853a4e)**
> (2026-08-07), the state of the corpus and the code this was gathered against.
> Every count below is a measurement over the **11 pinned EPUBs** in the private
> assets checkout at `sources/` — `sources/attic/` (unpinned twins) excluded.
> The books are copyrighted; what follows describes their markup, and quotes
> only the few characters of code needed to show a structure. Re-derive the
> numbers by unzipping the containers and parsing them with stdlib `zipfile` +
> `re`/`html.parser`; no repo dependency was added or needed (see
> [`docs/agents/domain.md`](../agents/domain.md#living-docs-vs-frozen-docs)).

## Summary

- **`<pre>` alone is not enough. It covers 6 of 11 books.** The other five put
  block code in `<p class=…>`, `<tt class=…>`, or — in one case — an HTML
  **`<table>`**. Across the corpus that is 5,369 `<pre>` blocks against 1,333
  non-`<pre>` blocks: 80% of the blocks, but only 55% of the books.
- **The spec's asserted quirk does not exist here.** Not one of the 11 books
  uses `<div class="programlisting">`, `codelisting`, or `code` as a code
  wrapper. The real `programlisting` name does appear, but on `<pre>`
  (fp-in-scala-2e), on `<p>` (working-effectively-with-legacy-code), and as a
  `data-type` attribute value (effective-typescript-2e) — never on a `<div>`.
- **🚩 Exactly one book's code indentation exists only in CSS: `aposd-2e`.**
  All 464 of its code lines are separate `<p>` elements whose text begins flush
  left; the indent level is encoded solely in the `text-indent` and
  `margin-left` percentages of 28 obfuscated, build-specific class names. A
  direct XHTML read of *A Philosophy of Software Design* returns structurally
  correct, **completely unindented** code. Every other book carries its
  indentation in the character stream, as literal spaces, NBSP, or both.
- **Line breaks come three different ways** and only one of them is a newline:
  literal `\n` (6 books), `<br/>` (3 books), or one element per line — `<p>` in
  aposd-2e, `<td class="codeline">` in pragmatic-programmer-2e (2 books). A
  reader that only honours `\n` flattens **four** books into single lines
  (code-complete-2e, working-effectively-with-legacy-code, aposd-2e,
  pragmatic-programmer-2e); poodr-2e escapes only by accident, because its
  source happens to put a literal newline after every `<br/>`.
- **The element alone never distinguishes inline from block code.** In
  effective-typescript-2e, **30,591 of 33,879 `<code>` elements are Pygments
  token spans *inside* a `<pre>`** — `<code>` there usually means "one keyword",
  not "a code span". effective-python-3e wraps each `<pre>` in a `<code>`;
  code-complete-2e uses the same `tt.calibre41` for both. Ancestry, not tag
  name, is the discriminator.
- **The OPF spine is a reliable reading order in all 11 books** — no
  `linear="no"`, no duplicate documents, and the NCX `playOrder` is sorted and
  spine-consistent in 10 of 11 (poodr-2e's single inversion is front matter in
  the ToC, not a content reorder). The NCX is needed for *nothing* here, and in
  aposd-2e it points at a `part0000.xhtml` that is not even in the manifest.
- **The spine is over-inclusive, though.** Four Pearson books ship
  print-fidelity **screenshots** of every listing as extra spine documents:
  207 documents holding 3,110 `<img alt="Image">` and ~15 KB of text between
  them. In poodr-2e that is **174 of 199 spine documents**. A spine walk that
  does not drop them spends most of its documents on nothing.
- **The two ~0-fence books fail for opposite reasons.**
  `effective-typescript-2e` has the *best* markup in the corpus — 944
  `<pre data-type="programlisting">`, 880 of them carrying an explicit
  `data-code-language` — and today's chain still yields 0 fences, so its problem
  is entirely on the parser side. `pragmatic-programmer-2e` has **no `<pre>` at
  all**: its code is 163 `<table class="processedcode">` with one `<tr>` per
  line. That is a markup problem, and no `<pre>`-only extractor will ever see it.
- **Adjacent finding for the map's fake-heading note.** #85 records that the
  corpus is safe from `#`-comment fake headings "by luck". Measured: 16 of
  pragmatic-programmer-2e's 1,129 non-blank code lines are ATX-shaped (`# …`), and
  it is one of the two books that is unfenced today. python-distilled has 228
  such lines and effective-python-3e 82, but both fence cleanly. The luck is
  thinner than assumed.

Every claim below is a count over the pinned `.epub` files themselves — the
primary source — or a quotation from a book's own stylesheet. Where a rule was
inferred rather than measured, it says so.

## Per-book table

Counts are over the whole spine of each pinned file. "Blocks" is logical code
blocks; for the two one-element-per-line books it is the count of consecutive
runs / wrapper tables, with the raw line count beside it.

| book | block code carried by (verbatim) | blocks / lines | inline code | line breaks | indentation | direct read? |
| --- | --- | --- | --- | --- | --- | --- |
| `aposd-2e` | `<p class="class_sc">`, `class_sju`, `class_sjw`, `class_smr`, `class_stw`, … — **28** classes, all `font-family: lucidasanstypewriter` | 58 / 464 | `<span class="class_s2c1">` (+4 more) | one `<p>` per line | **CSS `text-indent` / `margin-left` only** | 🚩 **no** — code comes out flush left |
| `code-complete-2e` | `<tt class="calibre41">` inside `div.calibre27 > div.calibre3`; `.calibre41 { font-family: monospace }` | 549 / 4,496 | same `tt.calibre41` when inside a `<p>` (6 of 549); `code.calibre41` (10) | `<br class="calibre24"/>` (445/549) | NBSP + space mix (323/549 lines start NBSP) | yes, with `<br>`→`\n`, NBSP→space |
| `effective-python-3e` | `<pre class="pre"><code>` | 1,374 / 13,523 | `<code>` (4,851 outside `<pre>`) | literal `\n` | literal spaces (900/1,374) | yes |
| `effective-typescript-2e` | `<pre data-type="programlisting" data-code-language="ts">` | 944 / 6,184 | `<code>` (3,288 outside `<pre>`) | literal `\n` | literal spaces (602/944) | yes — cleanest in the corpus |
| `fp-in-scala-2e` | `<pre class="programlisting">` | 1,077 / 5,522 | `<code class="fm-code-in-text">` (8,022) | literal `\n` | literal spaces (738/1,077) | yes, after dropping per-line `<a id="pgfId-…">` and `span.fm-combinumeral` callouts (193 blocks) |
| `poodr-2e` | `<div class="boxa"><p class="pre-ex">` | 158 / 4,817 | `<code class="calibre15">` (2,212) | `<br class="calibre2"/>` + `\n` (158/158) | NBSP (158/158) | yes, with `<br>`→`\n`, NBSP→space, and stripping `span.pd_mark` line numbers |
| `pragmatic-programmer-2e` | `<table class="processedcode">` → `<tr>` → `<td class="codeinfo">` + `<td class="codeline">` | 163 / 1,254 | `<span class="cf ic">` (339) — styled **`Ubuntu`, proportional** | one `<tr>` per line | literal spaces (617/1,129) | yes, once tables count as code; strip U+200B (688 rows) |
| `python-distilled` | bare `<pre>` (no class) | 926 / 5,985 | `<code>` (4,573) | `&#13;\n` (CRLF) | literal spaces (446/926) | yes, after CR stripping |
| `refactoring-2e` | `<pre class="pre">` | 737 / 5,926 | `<code>` (470) | `&#13;\n` (CRLF) | literal spaces (559/737) | yes, after CR stripping |
| `tdd-by-example` | bare `<pre>` | 311 / 2,147 | `<tt>` (605, **none** inside a `<pre>`) | literal `\n` | literal spaces (267/311) | yes |
| `working-effectively-with-legacy-code` | `<p class="programlisting">` (361), `programlisting1` (29), `programlisting3` (13), `programlisting2` (2) | 405 / 5,059 | `<code>` (1,581, none inside a listing) | `<br/>` (386/405) | NBSP (368/405) | yes, with `<br>`→`\n`, NBSP→space |

## 1. Which elements hold block code

Six books use `<pre>`, and only two of those six spell it `<pre><code>`:

- `<pre class="pre"><code>` — effective-python-3e (1,374; exactly one `<code>`
  per `<pre>`).
- `<pre class="pre">` — refactoring-2e (737), no inner `<code>`.
- `<pre class="programlisting">` — fp-in-scala-2e (1,077).
- `<pre data-type="programlisting">` — effective-typescript-2e (944), all 944
  carrying that attribute; the inner markup is Pygments `<code class="…">`
  tokens, not a single wrapper.
- bare `<pre>` — python-distilled (926) and tdd-by-example (311).

The other five books do not contain a single `<pre>` element:

**pragmatic-programmer-2e** — a table, one row per line:

```html
<table class="processedcode"><tr>
  <td class="codeinfo">​<span class="codeprefix">&#160;</span></td>
  <td class="codeline">  <strong class="kw">if</strong>&#8203; account.fees &lt; 0</td>
</tr>…</table>
```

163 such tables, 1,254 rows. The `codeinfo` gutter column is blank in 1,240 of
them (the remaining 14 hold `-`, `»`, or a line number like `10:`).

**working-effectively-with-legacy-code** — one paragraph per *block*, `<br/>`
per line, NBSP per indent step:

```html
<p class="programlisting">public class CDPlayer<br/>{<br/>&#160;&#160;&#160;&#160;public void addTrackListing(Track track) {<br/>…}</p>
```

Four class variants exist (`programlisting`, `…1`, `…2`, `…3`); the CSS shows
they differ only in `margin-left`, all `font-family: "Courier New"`.

**poodr-2e** — a `<div class="boxa">` wrapping a single `<p class="pre-ex">`,
with `<br class="calibre2"/>` line breaks, NBSP indentation, syntax-colour
spans, **and a baked-in line number per line** (`<span class="pd_mark"> 1</span>`).
Each block averages 30.5 lines. `.pre-ex` is `font-family: Courier New,
monospace; white-space: pre-wrap`.

**code-complete-2e** — `<tt class="calibre41">` with `<br class="calibre24"/>`
breaks, nested in `div.calibre27 > div.calibre3`. The stylesheet's entire
definition of the class is `.calibre41 { font-family: monospace }`; nothing else
in the markup says "code".

**aposd-2e** — see §2. Its 464 code lines are 464 separate `<p>` elements
carrying one of 28 obfuscated class names.

Notably absent: `<div class="programlisting">`, `<div class="codelisting">`, and
`<div class="code">` occur **zero** times across all 11 books. The only
code-adjacent `<div>` classes in the corpus are pragmatic-programmer-2e's
`div.livecodelozenge` (37) — a "run this online" banner above a code table, not
the code — and poodr-2e's `div.boxa` wrapper.

## 2. 🚩 Whitespace: real in ten books, CSS-only in `aposd-2e`

Ten of eleven books carry indentation in the character stream. Measured over
each book's block-code container, with no normalisation:

| carrier | books |
| --- | --- |
| literal newline + literal leading spaces | effective-python-3e, effective-typescript-2e, fp-in-scala-2e, python-distilled\*, refactoring-2e\*, tdd-by-example |
| `<br/>` + NBSP | code-complete-2e, poodr-2e, working-effectively-with-legacy-code |
| one element per line + literal spaces | pragmatic-programmer-2e |
| **one element per line + no whitespace at all** | **aposd-2e** |

\* line endings are `&#13;\n`, i.e. CRLF; the CR is part of the text node.

`white-space: pre` / `pre-wrap` is present in six stylesheets
(effective-python-3e, effective-typescript-2e, fp-in-scala-2e, poodr-2e,
pragmatic-programmer-2e, and refactoring-2e's platform CSS), but in every one of
those the source already contains the whitespace — the rule renders it, it does
not create it. aposd-2e, code-complete-2e, tdd-by-example and
working-effectively-with-legacy-code declare no `white-space` rule at all.
**Dropping the CSS costs nothing in ten books.**

### aposd-2e is the exception, and it is total

Each line is its own `<p>`, and the text of that `<p>` starts at column zero
regardless of the line's real indent. Verbatim, from a listing in the book:

```html
<p class="class_sju"><span class="class_s2c2">public interface Action {</span></p>
<p class="class_sjw"><span class="class_s2c2">public void redo();</span></p>
<p class="class_sjw"><span class="class_s2c2">public void undo();</span></p>
<p class="class_sju"><span class="class_s2c2">}</span></p>
```

The two interface members are indented in the book. Nothing in the character
stream says so. The whole signal is in `stylesheet.css`:

```css
.class_sju { display: block; font-family: lucidasanstypewriter; text-indent: 4.375%;  margin: 0 0 0 4.688%; }
.class_sjw { display: block; font-family: lucidasanstypewriter; text-indent: 8.750%;  margin: 0 0 0 4.688%; }
.class_smr { display: block; font-family: lucidasanstypewriter; text-indent: 13.125%; margin: 0 0 0 4.688%; }
```

Reconstruction is possible but is a geometry problem, not a text problem:

- There are **28 block-level monospace classes** and 5 inline ones. The names
  (`class_sju`, `class_s6k2`, `class_s14h`, …) are Calibre-conversion artifacts
  — they will not match any other book, and there is no guarantee they survive a
  re-conversion of this one. They cannot go in a hand-written table.
- Indentation is spread across **two** properties. One family varies
  `text-indent` off a 4.375%-per-level ladder against a fixed `margin-left:
  4.688%`; another family (`class_sc` → `class_stw`, both `text-indent: 0`)
  varies `margin-left` instead. The nine distinct positive `text-indent` values
  observed are 0.766, 3.062, 4.375, 5.469, 7.656, 8.75, 13.125, 15.313, 17.5 —
  approximately multiples of 1.09375% per character at `1em`, scaled by the
  class's own `font-size` (several are `0.75em`). Turning a percentage into a
  space count therefore needs the font size *and* a rounding rule, and the
  inference has not been validated line-by-line against the print edition.
- Negative `text-indent` (e.g. `class_s14h`, `-3.297%`) marks a *hanging*
  indent — a wrapped continuation of the previous line, not a new one. Seven of
  the 28 classes are these. Reflowing them back is a further step.

The 39 NBSPs that do appear in aposd-2e code are all *interior* column
alignment (`STATUS_OK&#160;…= 0,`), never leading. There is no fallback signal.

**Consequence for the extractor:** aposd-2e needs a CSS-resolving path, or its
code will be fenced correctly and indented wrongly — which is worse than not
fencing it, because it looks right. This is the one book where "read the XHTML"
is not sufficient.

## 3. Inline `<code>`: the element never decides

Three distinct failure modes, none of which a tag-name rule survives:

**`<code>` is mostly *inside* code blocks.** In effective-typescript-2e, 30,591
of 33,879 `<code>` elements sit inside a `<pre>` as Pygments highlight tokens
(`code.p` 9,915, `code.nx` 7,792, `code.o` 4,583, `code.kr` 1,880, `code.c1`
1,174…). Treating `<code>` as an inline code span there produces thirty thousand
one-token spans. effective-python-3e has the milder version: 1,374 of its 6,225
`<code>` elements are the per-`<pre>` wrapper. python-distilled and
tdd-by-example are clean (1 and 0 respectively inside `<pre>`).

**The same class is both.** code-complete-2e's `tt.calibre41` is block code when
its parent is a `<div>` (543 of 549) and inline code when its parent is a `<p>`
(6). Only ancestry separates them.

**Some books have no `<code>` element at all.** aposd-2e marks inline code with
`<span class="class_s2c1">` (337 uses; `lucidasanstypewriter` at `0.83333em`),
plus four rarer sibling classes. pragmatic-programmer-2e uses
`<span class="cf ic">` (339 uses) — and its CSS sets that in
`font-family: "Ubuntu"`, a **proportional** face, while the block code next to
it is `UbuntuMono`. That is the EPUB echo of the #83 font finding: a publisher
that marks code typographically without marking it monospace.

Where `<code>`/`<tt>` *is* inline-only — poodr-2e, refactoring-2e,
fp-in-scala-2e, working-effectively-with-legacy-code, tdd-by-example — the
median span is 5–11 characters and the longest is under 80, so a length
heuristic would also work; but the reliable rule everywhere is **"a `<code>`
with a block-code ancestor is not an inline span"**.

## 4. Spine order: reliable everywhere, but over-inclusive

Checked against each OPF and its NCX:

- **No book has a non-linear spine item** (`linear="no"`: 0 across all 11) and
  **no book repeats a document** in the spine.
- **`playOrder` in the NCX is monotonically sorted in all 11 books**, and the
  documents it visits appear in non-decreasing spine order in 10 of 11.
  poodr-2e is the sole `False`, and inspection shows one front-matter inversion
  — its ToC lists `pref00.xhtml` (spine index 2) after `ded01.xhtml` and
  `toc.xhtml` (indices 6, 7). Chapters 1–9 are strictly in order. Nothing about
  reading order changes.
- **Two books mix EPUB 2 and 3 navigation.** effective-python-3e (`version="3.0"`,
  `nav.xhtml`) and effective-typescript-2e (`version="3.0"`, `toc01.html`)
  declare a nav document *and* an NCX; the other nine are `version="2.0"` with
  an NCX only. Neither nav document is needed for ordering.
- **NCX targets can dangle.** aposd-2e's NCX points at `OEBPS/part0000.xhtml`,
  which is in neither the manifest nor the spine. Anything that resolves NCX
  targets to documents must tolerate a miss.

So: **use the spine, ignore the NCX for ordering.** The real hazard is the
opposite of a missing document — it is documents the spine contains that carry
nothing:

| book | `*_images*` spine documents | `<img>` inside them | text inside them |
| --- | --- | --- | --- |
| `poodr-2e` | 174 (of 199 spine docs) | 173 | 3,460 chars |
| `effective-python-3e` | 14 | 1,378 | 693 chars |
| `python-distilled` | 10 | 846 | 5,883 chars |
| `refactoring-2e` | 9 | 713 | 4,965 chars |

These are Pearson's "print-fidelity code image" back matter — every listing in
the book re-shipped as a screenshot, one `<div class="image-p"><img
alt="Image"/></div>` apiece, reached from the `<p class="codelink">Click here to
view code image</p>` link that precedes each real listing in the text. **None of
them is referenced by the NCX**, which is the cheapest way to spot them; the
`_images` filename convention is the second cheapest. In poodr-2e, skipping them
takes the spine from 199 documents to 25.

The same `p.codelink` line is worth noting on its own: it appears 1,196 times in
effective-python-3e, 752 in python-distilled, 708 in refactoring-2e, and 158 in
poodr-2e, always immediately before a code block. It is both a **free positive
signal** for "a code block starts here" and, if not dropped, a sentence of
boilerplate injected into every listing's context.

## 5. The two ~0-fence books

**`effective-typescript-2e` (0 fences / 499pp today) is not a markup problem.**
Its container is the best-marked in the corpus:

```html
<pre data-code-language="ts" data-type="programlisting"><code class="kd">function</code> <code class="nx">greet</code>…
  <code class="nx">console</code><code class="p">.</code><code class="nx">log</code>…
<code class="p">}</code></pre>
```

944 `<pre>`, **100% of them carrying `data-type="programlisting"`**, and 880
carrying an explicit `data-code-language` (`ts` 819, `js` 44, `json` 11, `html`
5, `javascript` 1) — a free, publisher-supplied language tag for the fence
info string, which #85 lists as out of scope to infer. Line breaks are literal
`\n`; 602 of 944 blocks have literal leading spaces. A direct container read
reproduces these blocks exactly. The 0-fence result therefore comes entirely
from what the current chain does to the file, not from what the file contains.

**`pragmatic-programmer-2e` (1 fence / 468pp today) is a markup problem.** It
contains **zero** `<pre>` elements. Every listing is
`<table class="processedcode">` with one `<tr>` per line, split across a
`td.codeinfo` gutter and a `td.codeline` payload (163 tables, 1,254 rows, 7.7
lines per block). Any extractor keyed on `<pre>` sees nothing; any extractor
that flattens tables cell-by-cell sees the gutter interleaved with the code.
Three quirks that will bite:

- **U+200B (zero-width space) in 688 of 1,129 lines**, emitted at every
  syntax-highlight boundary (`&#8203;<strong class="kw">def</strong>&#8203;`).
  It must be stripped before the text is fenced, hashed, or matched.
- The `codeinfo` gutter is a single NBSP in 1,240 of 1,254 rows, and a real
  marker (`-`, `»`, `1:`) in 14. Dropping the column wholesale loses almost
  nothing; keeping it prepends a space to almost every line.
- Indentation *is* real (literal spaces, 617 of 1,129 lines), so once the rows
  are joined with `\n` the block is faithful.

The books also differ in what an unfenced listing costs downstream. 16 of
pragmatic-programmer-2e's code lines begin with `# ` and would parse as ATX
headings — small, but non-zero, and this is one of the two books that is
unfenced today. effective-typescript-2e has none.

## Verdict: is a class-convention table needed, and how big?

**Yes, and it is small — about five entries — but one of them cannot be a table
entry at all.** `<pre>` handles 6 of 11 books and 80% of the corpus's code
blocks with no configuration. The remaining five books need four literal
conventions: `p[class^="programlisting"]` (working-effectively-with-legacy-code,
405 blocks), `p.pre-ex` inside `div.boxa` (poodr-2e, 158),
`table.processedcode` → `td.codeline` (pragmatic-programmer-2e, 163), and
`tt.calibre41` when its parent is not a `<p>` (code-complete-2e, 549). Each is
one selector, each was stable across every occurrence in its book, and together
they cover every non-`<pre>` block in the corpus except one book's. That book is
`aposd-2e`, whose 28 carrier classes are Calibre-generated names with no meaning
outside this exact file — putting them in a table would be pinning a build
artifact — and whose indentation is not in the text at all. It needs a different
mechanism: resolve the stylesheet, treat any `display: block` rule whose
`font-family` is monospace-ish as a code line, group consecutive lines into a
block, and derive the indent from `text-indent`/`margin-left`. That mechanism
would also subsume `code-complete-2e` (`.calibre41 { font-family: monospace }`)
and would degrade gracefully on the five publishers whose class names *are*
meaningful. So the honest shape is **a four-entry convention table plus one
CSS-resolved fallback**, not a table alone — and the fallback is the part that
needs designing, because it is the only path that can get aposd-2e's
indentation right.

Two smaller rules earn their place beside it, on measured evidence rather than
convention: **skip spine documents the NCX never references and whose bodies are
`<img>`-only** (207 documents and 3,110 screenshots across four books, 174 of
them in poodr-2e alone), and **normalise the character stream before fencing** —
`<br/>` → `\n`, NBSP → space, drop `\r`, drop U+200B, drop `span.pd_mark` line
numbers, drop `a[id^="pgfId"]` anchors, drop `span.fm-combinumeral` callouts.
Skipping either leaves correct-looking fences with wrong contents, which is the
failure mode hardest to notice downstream.
