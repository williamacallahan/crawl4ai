"""Regression tests for the PDF ``process_batch`` page-number race.

Bug: ``NaivePDFProcessorStrategy.process_batch`` (the live path used by
``PDFContentScrapingStrategy.scrap``) ran pages concurrently in a
``ThreadPoolExecutor`` but every worker wrote a single shared instance
attribute ``self.current_page_number`` and then ``_process_page`` re-read it
*after* ``page.extract_text(...)`` had released the GIL. Concurrent pages
therefore read a stomped page number, so:

* ``clean_pdf_text`` / ``clean_pdf_text_to_html`` got the wrong page context
  (page 1's author-emphasis heuristic lost its ``<strong>`` markup).
* ``img_filename = f"page_{self.current_page_number}_img_{img_count}"``
  collided across pages, clobbering extracted image files on disk.

Fix: the page number is threaded through as an explicit parameter
(``process_page_safely`` -> ``_process_page(page, image_dir, page_num + 1)`` ->
``_extract_images(page, image_dir, page_number)``) instead of using shared
instance state.

Run:
    PYTHONPATH="$PWD:$PWD/deploy/docker" .venv/bin/python -m pytest -xvs \
        tests/test_pdf_process_batch_concurrency.py
"""

import time
from pathlib import Path
from typing import List

import pytest

pypdf = pytest.importorskip("pypdf")

from crawl4ai.processors.pdf.processor import NaivePDFProcessorStrategy

# ---------------------------------------------------------------------------
# Fake pypdf primitives
# ---------------------------------------------------------------------------


class _FakeImage:
    """A minimal image XObject that ``_extract_images`` can consume."""

    def __init__(self, width: int = 2, height: int = 2):
        self._w = width
        self._h = height

    def get_object(self):
        return self

    def get(self, key, default=None):
        return {
            "/Subtype": "/Image",
            "/Width": self._w,
            "/Height": self._h,
            "/ColorSpace": "/DeviceRGB",
            "/BitsPerComponent": 8,
            "/Filter": ["/FlateDecode"],
            "/DecodeParms": {},
        }.get(key, default)

    def get_data(self):
        # Raw RGB buffer (already-decoded bytes, as pypdf would return).
        return b"\x00" * (self._w * self._h * 3)


class _FakeXObjects:
    def __init__(self, images: List[_FakeImage]):
        self._imgs = images

    def get_object(self):
        return self

    def __iter__(self):
        return iter(range(len(self._imgs)))

    def __getitem__(self, idx):
        return self._imgs[idx]


class _FakeResources:
    def __init__(self, xobjects: _FakeXObjects):
        self._xobjects = xobjects

    def get_object(self):
        return self

    def __contains__(self, key):
        return key == "/XObject"

    def __getitem__(self, key):
        return self._xobjects


class _FakePage:
    """A pypdf-like page whose ``extract_text`` sleeps to deterministically
    widen the ``process_batch`` race window so a regression is caught on every
    run rather than probabilistically."""

    def __init__(self, text: str, delay: float, image=None):
        self._text = text
        self._delay = delay
        self._image = image

    def extract_text(self, visitor_text=None):
        # pypdf calls visitor_text(text, cm, tm, font_dict, font_size)
        # tm is a 6-tuple; the real visitor indexes tm[4] (x) and tm[5] (y).
        visitor_text(self._text, None, (0, 0, 0, 0, 0, 0), None, None)
        # Simulate the long-running PDF I/O that releases the GIL; this is the
        # exact window in which the buggy code let other threads overwrite the
        # shared page number.
        time.sleep(self._delay)

    def __contains__(self, key):
        # No /Annots -> no links extracted.
        return False

    def get(self, key, default=None):
        if key == "/Resources":
            if self._image is None:
                return None
            return _FakeResources(_FakeXObjects([self._image]))
        return default


def _make_fake_reader(pages_text, delay, with_images=False):
    class _FakeReader:
        def __init__(self, *args, **kwargs):
            self.metadata = {}
            self.is_encrypted = False
            self.pages = [
                _FakePage(
                    txt,
                    delay,
                    image=(_FakeImage() if with_images else None),
                )
                for txt in pages_text
            ]

    return _FakeReader


def _patch_pdf_reader(monkeypatch, pages_text, delay, with_images=False):
    """Replace ``pypdf.PdfReader`` with a fake reader factory.

    ``process_batch`` does ``from pypdf import PdfReader`` at call time, so
    patching the ``pypdf.PdfReader`` attribute is sufficient.
    """
    fake_reader = _make_fake_reader(pages_text, delay, with_images=with_images)
    import pypdf

    monkeypatch.setattr(pypdf, "PdfReader", fake_reader)
    return fake_reader


def _dummy_pdf(tmp_path: Path) -> Path:
    p = tmp_path / "dummy.pdf"
    p.write_bytes(b"%PDF-1.4 dummy")
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_process_batch_text_uses_each_page_own_number(monkeypatch, tmp_path):
    """With a widened race window, every page must clean its text with its own
    page number: page 1 keeps ``<strong>`` author bold; other pages do not."""
    pages_text = ["John Smith, and Jane Doe"] * 4
    _patch_pdf_reader(monkeypatch, pages_text, delay=0.2, with_images=False)

    processor = NaivePDFProcessorStrategy(
        extract_images=False, save_images_locally=False, batch_size=4
    )
    result = processor.process_batch(_dummy_pdf(tmp_path))

    assert [p.page_number for p in result.pages] == [1, 2, 3, 4]

    # Page 1 must get the author-emphasis heuristic (needs page_number == 1).
    assert "<strong>" in result.pages[0].html
    assert "**" in result.pages[0].markdown

    # All other pages must NOT get the page-1 author heuristic.
    for i in (1, 2, 3):
        assert "<strong>" not in result.pages[i].html, (
            f"page {i + 1} unexpectedly got page-1 author heuristic; "
            f"clean_pdf_text_to_html was called with the wrong page number"
        )
        assert "**" not in result.pages[i].markdown


def test_process_batch_image_filenames_unique_no_clobber(monkeypatch, tmp_path):
    """With image extraction + local save, each page must write its own image
    file; filenames must not collide and clobber each other."""
    pages_text = [f"page content {i}" for i in range(4)]
    _patch_pdf_reader(monkeypatch, pages_text, delay=0.2, with_images=True)

    image_dir = tmp_path / "imgs"
    processor = NaivePDFProcessorStrategy(
        extract_images=True,
        save_images_locally=True,
        image_save_dir=image_dir,
        batch_size=4,
    )
    result = processor.process_batch(_dummy_pdf(tmp_path))

    # Each page produced exactly one image, with a per-page filename.
    assert len(result.pages) == 4
    for i, page in enumerate(result.pages):
        assert len(page.images) == 1
        img_path = Path(page.images[0]["path"])
        assert img_path.name == f"page_{i + 1}_img_1.png"
        assert img_path.exists()

    # Exactly 4 distinct files on disk (pre-fix: only 1 due to collision).
    files = sorted(image_dir.glob("page_*_img_1.png"))
    assert len(files) == 4, [f.name for f in files]
    assert {f.name for f in files} == {
        "page_1_img_1.png",
        "page_2_img_1.png",
        "page_3_img_1.png",
        "page_4_img_1.png",
    }


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-xvs"]))
