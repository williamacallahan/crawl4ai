"""Regression tests for the PDF ``_extract_images`` raw-data fallback.

Bug: ``NaivePDFProcessorStrategy._extract_images`` (in
``crawl4ai/processors/pdf/processor.py``) has a "raw data fallback" branch
that fires when an image XObject is not handled by one of the four
recognized filter branches. The fallback writes the raw bytes to a ``.bin``
file when ``save_images_locally=True`` and inlines them as base64 when
``save_images_locally=False``.

Pre-fix, the local-save sub-branch wrote the ``.bin`` file but never
appended a corresponding entry to ``PDFPage.images`` -- silently dropping
the image from the results and leaving an orphan ``.bin`` on disk when
``image_save_dir`` was supplied. The in-memory sub-branch appended
correctly; only the local-save sub-branch omitted the append. The success
branch and every other code path in the method append, so the local-save
fallback was a missing ``images.append(...)`` rather than intentional
behaviour. The branch had no test coverage, which is why the bug (present
since the original PDF commit ``f8fd9d9``) went undetected.

These tests pin the contract that BOTH fallback sub-branches append an
entry to ``PDFPage.images`` so a future refactor cannot silently drop one
of them again.

Run:
    PYTHONPATH="$PWD:$PWD/deploy/docker" .venv/bin/python -m pytest -xvs \
        tests/test_pdf_extract_images_fallback.py
"""

import pytest

pypdf = pytest.importorskip("pypdf")

from crawl4ai.processors.pdf.processor import NaivePDFProcessorStrategy


# ---------------------------------------------------------------------------
# Fake pypdf primitives: an image XObject whose filter is NOT one of the four
# recognized ones, so ``_extract_images`` skips every filter branch and falls
# through to the raw-data ``else`` fallback. pypdf-version-independent.
# ---------------------------------------------------------------------------


class _FallbackImage:
    """An image XObject with an unrecognized ``/Filter`` (e.g. ``/LZWDecode``)
    so ``_extract_images`` routes into the raw-data fallback."""

    def __init__(self, raw=b"\x01\x02\x03\x04", width=2, height=2):
        self._w = width
        self._h = height
        self._raw = raw

    def get_object(self):
        return self

    def get(self, key, default=None):
        return {
            "/Subtype": "/Image",
            "/Width": self._w,
            "/Height": self._h,
            "/ColorSpace": "/DeviceRGB",
            "/BitsPerComponent": 8,
            "/Filter": ["/LZWDecode"],
        }.get(key, default)

    def get_data(self):
        return self._raw


class _XObjects:
    def __init__(self, images):
        self._imgs = images

    def get_object(self):
        return self

    def __iter__(self):
        return iter(range(len(self._imgs)))

    def __getitem__(self, i):
        return self._imgs[i]


class _Resources:
    def __init__(self, xobjects):
        self._x = xobjects

    def get_object(self):
        return self

    def __contains__(self, key):
        return key == "/XObject"

    def __getitem__(self, key):
        return self._x


class _Page:
    def __init__(self, image):
        self._image = image

    def extract_text(self, visitor_text=None):
        visitor_text("placeholder", None, (0, 0, 0, 0, 0, 0), None, None)

    def __contains__(self, key):
        return False

    def get(self, key, default=None):
        if key == "/Resources" and self._image is not None:
            return _Resources(_XObjects([self._image]))
        return default


def _patch_reader(monkeypatch, image):
    """Replace ``pypdf.PdfReader`` with a fake reader whose single page
    carries ``image``. ``process`` does ``from pypdf import PdfReader`` at
    call time, so patching the ``pypdf.PdfReader`` attribute is sufficient."""

    class _Reader:
        def __init__(self, *args, **kwargs):
            self.metadata = {}
            self.is_encrypted = False
            self.pages = [_Page(image)]

    monkeypatch.setattr(pypdf, "PdfReader", _Reader)


def _dummy_pdf(tmp_path):
    p = tmp_path / "dummy.pdf"
    p.write_bytes(b"%PDF-1.4 dummy")
    return p


def test_fallback_local_save_appends_bin_entry_with_path(monkeypatch, tmp_path):
    """LOCAL-SAVE FALLBACK (the bug): when ``save_images_locally=True`` and an
    image routes into the raw-data fallback, ``PDFPage.images`` must contain
    an entry with ``format="bin"`` and a ``path`` pointing at an existing
    ``.bin`` file holding the raw bytes. Pre-fix, ``images`` was empty and the
    ``.bin`` file was an unreferenced orphan on disk."""
    _patch_reader(monkeypatch, _FallbackImage(raw=b"\x01\x02\x03\x04"))
    image_dir = tmp_path / "imgs"

    processor = NaivePDFProcessorStrategy(
        extract_images=True,
        save_images_locally=True,
        image_save_dir=image_dir,
    )
    result = processor.process(_dummy_pdf(tmp_path))

    page = result.pages[0]
    assert len(page.images) == 1, (
        "fallback image was dropped from PDFPage.images (local-save mode)"
    )
    entry = page.images[0]
    assert entry["format"] == "bin"
    assert entry["width"] == 2
    assert entry["height"] == 2
    assert entry["color_space"] == "/DeviceRGB"
    assert entry["bits_per_component"] == 8
    assert "path" in entry, "local-save fallback must report a 'path' key"
    assert "data" not in entry, "local-save fallback must not report a 'data' key"

    bin_path = entry["path"]
    from pathlib import Path
    bin_path = Path(bin_path)
    assert bin_path.suffix == ".bin"
    assert bin_path.name == "page_1_img_1.bin"
    assert bin_path.exists(), f".bin file missing at {bin_path}"
    assert bin_path.read_bytes() == b"\x01\x02\x03\x04"


def test_fallback_in_memory_appends_bin_entry_with_data(monkeypatch, tmp_path):
    """IN-MEMORY FALLBACK (symmetry guard): the in-memory sub-branch already
    appended pre-fix; it must keep doing so and stay symmetric with the
    local-save sub-branch (``data`` key instead of ``path``)."""
    import base64
    _patch_reader(monkeypatch, _FallbackImage(raw=b"\x01\x02\x03\x04"))

    processor = NaivePDFProcessorStrategy(
        extract_images=True,
        save_images_locally=False,
    )
    result = processor.process(_dummy_pdf(tmp_path))

    page = result.pages[0]
    assert len(page.images) == 1
    entry = page.images[0]
    assert entry["format"] == "bin"
    assert entry["width"] == 2
    assert entry["height"] == 2
    assert entry["color_space"] == "/DeviceRGB"
    assert entry["bits_per_component"] == 8
    assert "data" in entry
    assert "path" not in entry
    assert base64.b64decode(entry["data"]) == b"\x01\x02\x03\x04"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-xvs"]))
