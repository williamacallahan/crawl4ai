"""Regression tests for PNG predictor stride math + predictor double-decode
in PDF image extraction.

Bug 1 -- stride math (``crawl4ai/processors/pdf/utils.py::apply_png_predictor``):
the function computed the predictor row stride as
``width * ceil(bits * color_channels / 8)``. The PNG/PDF spec (PDF 1.7 sec
7.4.4.3 / PNG spec sec 9) defines the stride as
``ceil(width * bits * color_channels / 8)``. These only coincide when
``bits * color_channels`` is a multiple of 8; for 1/2/4-bit images the buggy
formula was ``8 / bits`` times too large, which either rejected valid
predictor rows with a spurious ``ValueError("Invalid scanline structure")``
(dropping the image) or, when ``len(data) % wrong_scanline_length == 0`` by
coincidence, silently decoded misaligned rows into garbage bytes.

Bug 2 -- double-decode (``crawl4ai/processors/pdf/processor.py::_extract_images``):
the FlateDecode branch called ``apply_png_predictor(data, width, bits,
colors)`` on the bytes returned by ``xobj.get_data()``. In pypdf >= 2.x
(including the pinned 6.4.1), ``EncodedStreamObject.get_data`` ->
``decode_stream_data`` -> ``FlateDecode.decode`` -> ``_decode_png_prediction``
already applies the PNG predictor using the spec-correct ``/Columns``-based
row length and strips the filter bytes. Calling ``apply_png_predictor`` on
that output re-decodes already-decoded data, dropping every
``/Predictor >= 10`` image (byte-aligned ones included) into the raw ``bin``
fallback. Fixing the stride math alone does not restore predictor image
extraction -- the redundant call must also be removed.

Fix:
  1. ``apply_png_predictor`` now uses ``stride = (width * bits *
     color_channels + 7) // 8`` (the spec-correct packed row length). The
     existing ``bytes_per_pixel`` (``ceil(bits * color_channels / 8)``) is
     kept for the per-byte Sub/Average/Paeth lookahead distance -- exactly
     what the PNG spec uses it for.
  2. ``_extract_images`` no longer calls ``apply_png_predictor`` on bytes
     from ``xobj.get_data()``; the raw post-predictor bytes go straight to
     ``PIL.Image.frombytes``. ``apply_png_predictor`` is retained in
     ``utils.py`` for callers that feed it raw pre-decode stream bytes.

Run:
    PYTHONPATH="$PWD:$PWD/deploy/docker" .venv/bin/python -m pytest -xvs \
        tests/test_pdf_png_predictor_decoding.py
"""

import base64
import io
import random
from pathlib import Path

import pytest

pypdf = pytest.importorskip("pypdf")
from pypdf import filters as _pypdf_filters  # noqa: E402

from crawl4ai.processors.pdf.processor import NaivePDFProcessorStrategy  # noqa: E402
from crawl4ai.processors.pdf.utils import apply_png_predictor  # noqa: E402


# Spec helpers ----------------------------------------------------------------

def _spec_stride(width, bits, channels):
    """PNG/PDF-spec packed row stride: ceil(width*bits*channels/8) bytes."""
    return (width * bits * channels + 7) // 8


def _row(filter_byte, row_bytes):
    """Build one PNG predictor scanline: 1 filter byte + row_bytes."""
    return bytes([filter_byte]) + bytes(row_bytes)


def _ref_decode(data, width, bits, channels):
    """Authoritative reference via pypdf's ``_decode_png_prediction`` invoked
    with the spec-correct rowlength. Independent of the function under test
    -- exercises pypdf's own row math and per-byte filter arithmetic."""
    rowlength = _spec_stride(width, bits, channels) + 1
    return _pypdf_filters.FlateDecode._decode_png_prediction(
        bytes(data), width, rowlength
    )


def _paeth(a, b, c):
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    elif pb <= pc:
        return b
    return c


# 1. Stride math: bug-report evidence cases now decode correctly. -----------

def test_stride_4bit_grayscale_decodes_without_valueerror():
    """(4, 1, 3, 2): spec row_bytes=2. Pre-fix stride=3 rejected valid data
    with ``Invalid scanline structure``. Post-fix both None rows decode."""
    data = _row(0, [0xE4, 0xC0]) + _row(0, [0x93, 0x80])
    out = apply_png_predictor(data, width=3, bits=4, color_channels=1)
    assert out == b"\xe4\xc0\x93\x80"
    assert out == _ref_decode(data, 3, 4, 1)


def test_stride_1bit_grayscale_decodes_without_valueerror():
    """(1, 1, 10, 3): spec row_bytes=2. Pre-fix stride=10 rejected valid
    data. Post-fix all three None rows decode."""
    data = _row(0, [0xFF, 0xC0]) * 3
    out = apply_png_predictor(data, width=10, bits=1, color_channels=1)
    assert out == b"\xff\xc0" * 3
    assert out == _ref_decode(data, 10, 1, 1)


def test_stride_2bit_grayscale_does_not_silently_corrupt():
    """(2, 1, 5, 2): spec row_bytes=2. Pre-fix stride=5 made
    ``len(data) % scanline`` pass by coincidence and the function returned
    the silently-corrupt 5-byte ``e4c002afc0`` (ground truth: 4 bytes
    ``e4c09380``). Post-fix returns the correct four packed bytes."""
    data = _row(0, [0xE4, 0xC0]) + _row(0, [0x93, 0x80])
    out = apply_png_predictor(data, width=5, bits=2, color_channels=1)
    assert out == b"\xe4\xc0\x93\x80"
    assert out == _ref_decode(data, 5, 2, 1)


def test_stride_4bit_with_up_filter_decodes():
    """Sub-byte depth + Up filter: row 2 encoded as Up(prev) must decode
    to the original pre-image, not be rejected by an inflated stride."""
    # Row1 None([0xE4,0xC0]) -> [0xE4,0xC0]; Row2 Up([0x13,0x80]) ->
    # [0x13+0xE4, 0x80+0xC0] = [0xF7, 0x40].
    data = _row(0, [0xE4, 0xC0]) + _row(2, [0x13, 0x80])
    out = apply_png_predictor(data, width=3, bits=4, color_channels=1)
    assert out == b"\xe4\xc0\xf7\x40"
    assert out == _ref_decode(data, 3, 4, 1)


# 2. Filter coverage at byte-aligned depth: no-regression control cases. ----
# bits*channels is a multiple of 8 so the buggy stride formula coincided
# with the spec; all five PNG filters must still round-trip.

# 8-bit RGB, width=4, two rows -> one 24-byte image. `build` returns the
# (data, expected_decoded) pair for each filter type.
_BITS, _CH, _W = 8, 3, 4
_Row1 = list(range(12))
_Row2 = list(range(12, 24))


def _build_none():
    return _row(0, _Row1) + _row(0, _Row2), bytes(_Row1 + _Row2)


def _build_sub():
    bpp = 3
    f = list(_Row1[:bpp])
    for j in range(bpp, 12):
        f.append((_Row1[j] - _Row1[j - bpp]) % 256)
    return _row(1, f), bytes(_Row1)


def _build_up():
    f = [(b - a) % 256 for a, b in zip(_Row1, _Row2)]
    return _row(0, _Row1) + _row(2, f), bytes(_Row1 + _Row2)


def _build_average():
    bpp = 3
    raw = [200, 100, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140]
    f = list(raw[:bpp])  # first bpp bytes have left == 0 -> avg == 0
    for j in range(bpp, len(raw)):
        f.append((raw[j] - ((raw[j - bpp] + 0) // 2)) % 256)
    return _row(3, f), bytes(raw)


def _build_paeth():
    bpp = 3
    f = []
    for j in range(12):
        left = _Row2[j - bpp] if j >= bpp else 0
        up = _Row1[j]
        up_left = _Row1[j - bpp] if j >= bpp else 0
        f.append((_Row2[j] - _paeth(left, up, up_left)) % 256)
    return _row(0, _Row1) + _row(4, f), bytes(_Row1 + _Row2)


@pytest.mark.parametrize("builder", [
    _build_none, _build_sub, _build_up, _build_average, _build_paeth,
])
def test_filter_byte_aligned_round_trips(builder):
    data, expected = builder()
    out = apply_png_predictor(data, _W, _BITS, _CH)
    assert out == expected
    assert out == _ref_decode(data, _W, _BITS, _CH)


# 3. Cross-check vs pypdf's _decode_png_prediction across configs. ---------

def test_apply_png_predictor_matches_pypdf_reference_across_configs():
    rng = random.Random(0xC0FFEE)
    configs = [
        (1, 1, 10, 3), (1, 1, 7, 5),
        (2, 1, 5, 2), (2, 1, 9, 4),
        (4, 1, 3, 2), (4, 1, 5, 4),
        (4, 2, 3, 2),
        (8, 1, 4, 2), (8, 1, 7, 3),
        (8, 3, 4, 2), (8, 3, 5, 4),
    ]
    for bits, ch, w, h in configs:
        stride = _spec_stride(w, bits, ch)
        # filter=0 on every row keeps this a focused property test of the
        # stride path without re-implementing all five filters.
        rows = [rng.randbytes(stride) for _ in range(h)]
        data = b"".join(_row(0, r) for r in rows)
        out = apply_png_predictor(data, w, bits, ch)
        ref = _ref_decode(data, w, bits, ch)
        assert out == ref, (
            f"(bits={bits}, ch={ch}, w={w}, h={h}): got {out.hex()}, "
            f"ref {ref.hex()}"
        )


def test_apply_png_predictor_still_rejects_non_multiple_lengths():
    """Validation isn't being removed, just relaxed to the correct stride:
    a length that's not a multiple of the spec scanline must still raise."""
    # width=3, bits=4, ch=1 -> spec stride=2, scanline=3; 4 is not a multiple.
    with pytest.raises(ValueError, match="Invalid scanline structure"):
        apply_png_predictor(b"\x00\x01\x02\x03", 3, 4, 1)


def test_apply_png_predictor_empty_input_returns_empty():
    """An empty buffer decodes to empty bytes (0 % scanline == 0)."""
    assert apply_png_predictor(b"", 3, 4, 1) == b""


# 4. End-to-end through ``NaivePDFProcessorStrategy.process``. -------------
# Fake pypdf primitives mirror pypdf's ``get_data()`` contract: they return
# the *post*-predictor raw image bytes (what pypdf hands back after
# FlateDecode.decode -> _decode_png_prediction). This is the exact case
# where pre-fix ``_extract_images`` double-decoded.

class _PredictorImage:
    """RGB 8-bit XObject with ``/Predictor`` on ``/FlateDecode`` whose
    ``get_data()`` returns the already-decoded raw RGB buffer (pypdf's
    ``EncodedStreamObject.get_data`` -> ``decode_stream_data`` ->
    ``FlateDecode.decode`` -> ``_decode_png_prediction`` contract)."""

    def __init__(self, width=2, height=2, predictor=12):
        self._w = width
        self._h = height
        self._predictor = predictor
        self._raw = bytes((i * 7 + j * 11 + k * 3) % 256
                          for i in range(height)
                          for j in range(width)
                          for k in range(3))

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
            "/DecodeParms": {
                "/Predictor": self._predictor,
                "/Columns": self._w,
                "/Colors": 3,
                "/BitsPerComponent": 8,
            },
        }.get(key, default)

    def get_data(self):
        return self._raw


class _PlainFlateImage:
    """Same XObject shape as ``_PredictorImage`` but without ``/Predictor``.
    Control case: already extracted as ``png`` pre-fix and must keep doing
    so post-fix (no regression on the happy path)."""

    def __init__(self, width=2, height=2):
        self._w = width
        self._h = height
        self._raw = bytes((i * 7 + j * 11 + k * 3) % 256
                          for i in range(height)
                          for j in range(width)
                          for k in range(3))

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
        self._xobjects = xobjects

    def get_object(self):
        return self

    def __contains__(self, key):
        return key == "/XObject"

    def __getitem__(self, key):
        return self._xobjects


class _Page:
    def __init__(self, image):
        self._image = image

    def extract_text(self, visitor_text=None):
        # pypdf calls visitor_text(text, cm, tm, font_dict, font_size).
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
    call time, so patching ``pypdf.PdfReader`` is sufficient."""

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


def _png_bytes(image_data_b64):
    from PIL import Image
    return Image.open(io.BytesIO(base64.b64decode(image_data_b64))).tobytes()


@pytest.mark.parametrize("predictor", [10, 12, 15])
def test_predictor_image_extracts_as_png_not_bin(predictor, monkeypatch, tmp_path):
    """Pre-fix: ``_extract_images`` called ``apply_png_predictor`` on the
    already-decoded raw RGB bytes returned by ``get_data()``. With
    width=2, bits=8, channels=3 the spec scanline is 7 and the raw length
    12 is not a multiple of 7 -- ``Invalid scanline structure`` was raised
    and the image demoted to the ``bin`` fallback. Post-fix the raw bytes
    go straight to PIL, the image is extracted as ``png`` and the original
    pixels survive a PNG round-trip. Parametrized across PNG predictor
    values 10 (None), 12 (Up) and 15 (Paeth) -- all decode identically
    here because ``get_data()`` models the post-predictor bytes."""
    img = _PredictorImage(width=2, height=2, predictor=predictor)
    _patch_reader(monkeypatch, img)

    proc = NaivePDFProcessorStrategy(
        extract_images=True, save_images_locally=False
    )
    result = proc.process(_dummy_pdf(tmp_path))

    page = result.pages[0]
    assert len(page.images) == 1, "predictor image was dropped from results"
    entry = page.images[0]
    assert entry["format"] == "png", (
        f"predictor {predictor} image demoted to fallback "
        f"format={entry['format']!r}; the double-decode defect is back"
    )
    assert entry["width"] == 2
    assert entry["height"] == 2
    assert entry["color_space"] == "/DeviceRGB"
    assert entry["bits_per_component"] == 8
    assert "data" in entry
    assert "path" not in entry
    assert _png_bytes(entry["data"]) == img._raw


def test_predictor_12_image_local_save_writes_png(monkeypatch, tmp_path):
    """Local-save variant of the predictor-extraction fix: a ``.png`` file
    is written instead of a ``.bin`` fallback, and it re-decodes to the
    original pixels."""
    from PIL import Image

    img = _PredictorImage(width=2, height=2, predictor=12)
    _patch_reader(monkeypatch, img)
    image_dir = tmp_path / "imgs"

    proc = NaivePDFProcessorStrategy(
        extract_images=True,
        save_images_locally=True,
        image_save_dir=image_dir,
    )
    result = proc.process(_dummy_pdf(tmp_path))

    entry = result.pages[0].images[0]
    assert entry["format"] == "png"
    assert "path" in entry
    assert "data" not in entry
    png_path = Path(entry["path"])
    assert png_path.suffix == ".png"
    assert png_path.name == "page_1_img_1.png"
    assert png_path.exists(), f".png file missing at {png_path}"
    assert Image.open(png_path).tobytes() == img._raw


def test_plain_flate_no_predictor_still_extracts_as_png(monkeypatch, tmp_path):
    """Control: a ``/FlateDecode`` XObject *without* ``/Predictor`` already
    extracted as ``png`` pre-fix and must keep doing so post-fix (no
    regression on the happy path)."""
    img = _PlainFlateImage(width=2, height=2)
    _patch_reader(monkeypatch, img)

    proc = NaivePDFProcessorStrategy(
        extract_images=True, save_images_locally=False
    )
    result = proc.process(_dummy_pdf(tmp_path))

    entry = result.pages[0].images[0]
    assert entry["format"] == "png"
    assert _png_bytes(entry["data"]) == img._raw


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-xvs"]))
