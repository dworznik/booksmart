"""Guards on the generated fixture documents.

A fixture that does not contain what it claims makes every test built on it a
measurement of nothing — and the gate's loss metric is *entirely* built on one of
these: `first_pages` is how a document with a known, deliberate hole is made.
"""

from pathlib import Path

import pymupdf

from . import documents


class TestProbePdf:
    def test_it_carries_a_heading_a_listing_and_furniture(self, tmp_path: Path) -> None:
        path = documents.build_probe_pdf(tmp_path / "probe.pdf", pages=2)

        text = "\n".join(page.get_text() for page in pymupdf.open(path))

        assert "Extracting Text" in text  # the heading
        assert "def extract(path):" in text  # the listing, line-broken
        assert "P R O B E" in text  # letter-spaced running header
        assert "Prose before the listing" in text


class TestFirstPages:
    def test_it_truncates_and_keeps_the_text(self, tmp_path: Path) -> None:
        source = documents.build_probe_pdf(tmp_path / "whole.pdf", pages=5)

        sliced = documents.first_pages(source, 2, tmp_path / "slice.pdf")

        assert documents.page_count(sliced) == 2
        doc = pymupdf.open(sliced)
        text = doc[0].get_text()
        doc.close()
        assert "Extracting Text" in text

    def test_asking_for_more_pages_than_exist_is_the_whole_document(
        self, tmp_path: Path
    ) -> None:
        source = documents.build_probe_pdf(tmp_path / "whole.pdf", pages=3)

        sliced = documents.first_pages(source, 999, tmp_path / "slice.pdf")

        assert documents.page_count(sliced) == 3

    def test_a_slice_can_start_mid_document(self, tmp_path: Path) -> None:
        """Books keep their code deep in the text; a slice from page one of a
        corpus book held two code lines in twenty pages, which quietly turned a
        code-handling comparison into a front-matter comparison."""
        source = documents.build_probe_pdf(tmp_path / "whole.pdf", pages=6)

        sliced = documents.first_pages(source, 2, tmp_path / "slice.pdf", start=3)

        assert documents.page_count(sliced) == 2
        doc = pymupdf.open(sliced)
        text = doc[0].get_text()
        doc.close()
        assert "Chapter 4" in text  # 0-based start=3 is the fourth page

    def test_an_epub_can_be_sliced_too(self, tmp_path: Path) -> None:
        """select() is PDF-only, so an EPUB slice goes through a whole-document
        conversion first. Without this, the first EPUB anyone pointed a slice at
        would crash inside the helper rather than produce a document."""
        import zipfile

        source = tmp_path / "book.epub"
        with zipfile.ZipFile(source, "w") as archive:
            archive.writestr("mimetype", "application/epub+zip", zipfile.ZIP_STORED)
            archive.writestr(
                "META-INF/container.xml",
                '<?xml version="1.0"?><container version="1.0" '
                'xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles>'
                '<rootfile full-path="content.opf" '
                'media-type="application/oebps-package+xml"/></rootfiles></container>',
            )
            archive.writestr(
                "content.opf",
                '<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" '
                'version="3.0" unique-identifier="id"><metadata '
                'xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>T</dc:title>'
                '<dc:identifier id="id">x</dc:identifier><dc:language>en</dc:language>'
                "</metadata><manifest><item id=\"c1\" href=\"ch1.xhtml\" "
                'media-type="application/xhtml+xml"/></manifest>'
                '<spine><itemref idref="c1"/></spine></package>',
            )
            archive.writestr(
                "ch1.xhtml",
                '<?xml version="1.0"?><html xmlns="http://www.w3.org/1999/xhtml"><body>'
                "<h1>Sliceable</h1>" + "<p>prose to fill pages. </p>" * 400 + "</body></html>",
            )

        sliced = documents.first_pages(source, 2, tmp_path / "slice.epub")

        assert sliced.suffix == ".pdf"  # rendered form; select is PDF-only
        assert documents.page_count(sliced) == 2
        doc = pymupdf.open(sliced)
        text = doc[0].get_text()
        doc.close()
        assert "Sliceable" in text
