"""Regression tests for the PDF ``process_batch`` temp-directory cross-call race.

Bug: ``NaivePDFProcessorStrategy.process_batch`` (the live path used by
``PDFContentScrapingStrategy.scrap``) stored the per-call temp image
directory in the shared **instance** attribute ``self._temp_dir`` and
re-read that same instance attribute in its ``finally`` to
``shutil.rmtree`` it. ``PDFContentScrapingStrategy`` constructs a single
inner ``NaivePDFProcessorStrategy`` (``self.pdf_processor``) and
``AsyncWebCrawler.arun_many`` runs many ``scrap``/``ascrap`` calls
concurrently on that one instance (via ``asyncio.to_thread``), so multiple
OS threads call ``self.pdf_processor.process_batch(...)`` at the same
time. Because they all read/write the same ``self._temp_dir`` field, a
later call overwrites the earlier call's value, and the earlier call's
``finally`` then ``rmtree``s the *later* call's still-in-use directory.
Two failure modes result:

1. **Permanent temp-dir leak** — the earlier call's own ``pdf_images_*``
   directory is never removed (its ``finally`` deleted the wrong dir).
2. **Cross-delete / broken extraction** — the later call's temp dir is
   deleted out from under it while it is still inside ``_process_page``;
   its subsequent ``_extract_images`` writes fail and its pages return
   with ``images == []`` despite the PDF carrying image XObjects.

Both failures are silent: the wrong-dir ``rmtree`` raises
``FileNotFoundError`` that is caught-and-logged, and ``_extract_images``
failures are caught-and-logged too, so ``CrawlResult.success`` stays
``True`` and the caller has no signal that images were lost or a
directory leaked.

Fix: capture the per-call temp directory in a **local** variable and
clean up only that local in ``finally`` (both ``process_batch`` and the
``process`` sibling).

These tests drive two concurrent ``process_batch`` calls on the SAME
``NaivePDFProcessorStrategy(extract_images=True, save_images_locally=True,
image_save_dir=None)`` instance, coordinated with ``threading.Event``s so
the interleaving is identical on every run (deterministic reproduction
of the cross-delete ordering). The fake-``pypdf`` harness mirrors
``tests/test_pdf_process_batch_concurrency.py``; the difference is that
the race here is *between* two ``process_batch`` calls rather than within
one call's page workers, so coordination is by ``threading.Event``
between the two call-driving threads rather than by ``time.sleep``.

Run:
    PYTHONPATH="$PWD:$PWD/deploy/docker" .venv/bin/python -m pytest -xvs \
        tests/test_pdf_process_batch_temp_dir_race.py
"""

import threading
from pathlib import Path
from typing import List

import pytest

pypdf = pytest.importorskip("pypdf")

from crawl4ai.processors.pdf.processor import NaivePDFProcessorStrategy

# ---------------------------------------------------------------------------
# Fake pypdf primitives (same shape as tests/test_pdf_process_batch_concurrency.py)
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


class _CoordinatedPage:
    """A pypdf-like page whose ``extract_text`` blocks on a ``threading.Event``.

    This parks the call's page worker inside ``_process_page`` (after the
    reader is open and the temp dir is set, but before ``_extract_images``
    runs) so a second ``process_batch`` call on the same processor instance
    can overwrite ``self._temp_dir`` and the first call's ``finally`` can
    then target the second call's still-in-use directory.
    """

    def __init__(self, text: str, blocked_event: threading.Event,
                 release_event: threading.Event, image=None):
        self._text = text
        self._blocked_event = blocked_event
        self._release_event = release_event
        self._image = image

    def extract_text(self, visitor_text=None):
        # pypdf calls visitor_text(text, cm, tm, font_dict, font_size); tm is a
        # 6-tuple and the real visitor indexes tm[4] (x) and tm[5] (y).
        visitor_text(self._text, None, (0, 0, 0, 0, 0, 0), None, None)
        # Signal that this page is now parked inside _process_page, then wait
        # for the test to release us. This widens the cross-call race window
        # deterministically (the same role time.sleep plays in the within-call
        # concurrency test, but crossing process_batch calls).
        self._blocked_event.set()
        self._release_event.wait()

    def __contains__(self, key):
        # No /Annots -> no links extracted.
        return False

    def get(self, key, default=None):
        if key == "/Resources":
            if self._image is None:
                return None
            return _FakeResources(_FakeXObjects([self._image]))
        return default


def _make_fake_reader_factory(pages_for_path):
    """Return a fake ``PdfReader`` class that dispatches pages by file path.

    ``process_batch`` opens one reader in the main thread (for metadata and
    page count) and one reader per page in worker threads (via
    ``process_page_safely``); all of them open the SAME ``pdf_path`` for a
    given call, so dispatching on ``file.name`` returns the same page set
    for every reader that belongs to one call.
    """

    class _FakeReader:
        def __init__(self, file=None, *args, **kwargs):
            name = getattr(file, "name", None)
            self.metadata = {}
            self.is_encrypted = False
            self.pages = pages_for_path.get(name, [])

    return _FakeReader


def _patch_pdf_reader(monkeypatch, pages_for_path):
    """Replace ``pypdf.PdfReader`` with the dispatch-by-path fake reader."""
    fake_reader = _make_fake_reader_factory(pages_for_path)
    import pypdf
    monkeypatch.setattr(pypdf, "PdfReader", fake_reader)
    return fake_reader


def _dummy_pdf(tmp_path: Path, name: str) -> Path:
    p = tmp_path / name
    p.write_bytes(b"%PDF-1.4 dummy")
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_process_batch_concurrent_calls_do_not_cross_delete_temp_dirs(monkeypatch, tmp_path):
    """Two concurrent ``process_batch`` calls on the same processor instance
    must each clean up only their own temp directory and must not interfere
    with each other's image extraction.

    Pre-fix (buggy ``self._temp_dir`` shared-attribute pattern), this
    reproduces BOTH failure modes deterministically:
      * the earlier call (A) permanently leaks its own ``pdf_images_*``
        directory (A's ``finally`` ``rmtree``s B's dir, not A's);
      * the later call (B) has its temp dir deleted out from under it
        while B is still inside ``_process_page``, so B's
        ``_extract_images`` writes fail and B's pages come back with
        ``images == []`` despite each page carrying an image XObject.

    Post-fix (per-call local ``_temp_dir``), both calls clean up only their
    own directory and every page from both calls returns its single image.
    """
    pdf_a = _dummy_pdf(tmp_path, "a.pdf")
    pdf_b = _dummy_pdf(tmp_path, "b.pdf")

    # Coordination events. A parks first; B parks after overwriting
    # self._temp_dir; A is then released so A's finally runs while B is still
    # parked; B is then released so B's _extract_images runs against its own
    # (post-fix: intact) temp dir.
    a_blocked = threading.Event()
    a_release = threading.Event()
    b_blocked = threading.Event()
    b_release = threading.Event()

    page_a = _CoordinatedPage("page A", a_blocked, a_release, image=_FakeImage())
    page_b = _CoordinatedPage("page B", b_blocked, b_release, image=_FakeImage())

    pages_for_path = {str(pdf_a): [page_a], str(pdf_b): [page_b]}
    _patch_pdf_reader(monkeypatch, pages_for_path)

    # Capture every temp directory mkdtemp creates so we can assert on
    # cleanup. Recording in creation order gives us [dirA, dirB].
    created_dirs = []
    import tempfile as _tempfile_mod
    real_mkdtemp = _tempfile_mod.mkdtemp

    def _recording_mkdtemp(*args, **kwargs):
        d = real_mkdtemp(*args, **kwargs)
        created_dirs.append(d)
        return d

    monkeypatch.setattr(_tempfile_mod, "mkdtemp", _recording_mkdtemp)

    processor = NaivePDFProcessorStrategy(
        extract_images=True,
        save_images_locally=True,
        image_save_dir=None,  # <-- the racy branch
        batch_size=1,
    )

    results = {}

    def run_a():
        results["a"] = processor.process_batch(pdf_a)

    def run_b():
        results["b"] = processor.process_batch(pdf_b)

    thread_a = threading.Thread(target=run_a)
    thread_b = threading.Thread(target=run_b)

    # 1. Start A. A sets its temp dir, submits its page, the page worker
    #    enters _process_page and parks inside extract_text.
    thread_a.start()
    assert a_blocked.wait(timeout=10), "A's page never parked in extract_text"

    # 2. Start B on the SAME instance. B overwrites the shared attribute with
    #    its own temp dir (pre-fix) and B's page parks inside extract_text.
    thread_b.start()
    assert b_blocked.wait(timeout=10), "B's page never parked in extract_text"

    # 3. Release A. A's page finishes, A's _extract_images writes A's image to
    #    A's temp dir, A's finally runs rmtree(self._temp_dir). Pre-fix this
    #    targets B's dir (cross-delete); post-fix it targets A's own dir.
    a_release.set()
    thread_a.join(timeout=10)
    assert not thread_a.is_alive(), "A did not finish"

    # 4. Release B. B's page finishes, B's _extract_images writes B's image to
    #    B's temp dir. Pre-fix B's dir was deleted by A's finally, so this
    #    write fails and B's page comes back imageless; post-fix it succeeds.
    b_release.set()
    thread_b.join(timeout=10)
    assert not thread_b.is_alive(), "B did not finish"

    result_a = results["a"]
    result_b = results["b"]

    # --- Guarantee 1: every page from both calls carries its single image.
    # Pre-fix: A's pages have images (written to the leaked dirA before A's
    # finally deleted dirB), but B's pages have images == [] (writes to the
    # already-deleted dirB failed silently).
    assert len(result_a.pages) == 1, "A should have produced exactly one page"
    assert len(result_b.pages) == 1, "B should have produced exactly one page"
    assert len(result_a.pages[0].images) == 1, (
        "A lost its extracted image to the temp-dir race; pre-fix B's finally "
        "or a cross-delete removed A's directory before A's image could be saved"
    )
    assert len(result_b.pages[0].images) == 1, (
        "B lost its extracted image to the temp-dir cross-delete: A's finally "
        "removed B's in-use temp directory before B's _extract_images could write"
    )

    # --- Guarantee 2: every returned image entry proves its save succeeded.
    # ``_extract_images`` only appends an entry when ``img.save(final_path)``
    # ran without raising; on failure it catches the exception and returns an
    # empty list (the cross-delete victim's path). Pre-fix B's ``images`` is
    # empty because its temp dir was gone before the save; post-fix both
    # entries are present, with the expected ``png`` format and ``path`` key.
    # (The path itself is intentionally transient: ``image_save_dir=None``
    # uses a per-call temp dir that is always cleaned up in ``finally``; a
    # caller that needs persistent image files passes ``image_save_dir``.)
    for page in (*result_a.pages, *result_b.pages):
        assert len(page.images) == 1
        entry = page.images[0]
        assert entry["format"] == "png"
        assert "path" in entry
        assert entry["path"].endswith("page_1_img_1.png")

    # --- Guarantee 3: no temp directory leaks. Pre-fix dirA is leaked
    # permanently (A's finally deleted dirB instead).
    assert len(created_dirs) == 2, (
        f"expected exactly two per-call temp dirs, got {created_dirs!r}"
    )
    leaked = [d for d in created_dirs if Path(d).exists()]
    assert not leaked, (
        f"temp image directory leaked (never cleaned up by its owning call): "
        f"{leaked}"
    )

    # --- Guarantee 4: both calls return success with a populated pages list
    # (no silent failure). This pins the contract that, post-fix, the caller
    # never observes a silently-empty result from this race.
    assert result_a.pages[0].page_number == 1
    assert result_b.pages[0].page_number == 1


def test_process_batch_with_image_save_dir_unaffected(monkeypatch, tmp_path):
    """The ``image_save_dir`` branch (caller-supplied dir) never touches the
    per-call temp dir, so concurrent calls sharing an instance must each
    write into their own caller-supplied directory and leave the directory
    intact after the call (no rmtree of a caller-owned dir). This is the
    no-regression guard for the documented common path (the docs example
    supplies an explicit ``image_save_dir``).
    """
    pdf_a = _dummy_pdf(tmp_path, "a.pdf")
    pdf_b = _dummy_pdf(tmp_path, "b.pdf")

    a_done = threading.Event()
    release = threading.Event()

    page_a = _CoordinatedPage("page A", a_done, release, image=_FakeImage())
    page_b = _CoordinatedPage("page B", threading.Event(), release,
                              image=_FakeImage())

    pages_for_path = {str(pdf_a): [page_a], str(pdf_b): [page_b]}
    _patch_pdf_reader(monkeypatch, pages_for_path)

    image_dir = tmp_path / "imgs"
    processor = NaivePDFProcessorStrategy(
        extract_images=True,
        save_images_locally=True,
        image_save_dir=image_dir,  # <-- caller-owned, never rmtree'd
        batch_size=1,
    )

    results = {}

    def run_a():
        results["a"] = processor.process_batch(pdf_a)

    def run_b():
        results["b"] = processor.process_batch(pdf_b)

    ta = threading.Thread(target=run_a)
    tb = threading.Thread(target=run_b)
    ta.start()
    assert a_done.wait(timeout=10)
    tb.start()
    # Wait for B's page to park (its blocked_event is a throwaway, but we need
    # B to have started its page worker before releasing A so the calls
    # actually overlap).
    tb.join(timeout=0.1)  # give B a moment to enter _process_page
    release.set()
    ta.join(timeout=10)
    tb.join(timeout=10)
    assert not ta.is_alive() and not tb.is_alive()

    # The caller-owned image dir must survive both calls.
    assert image_dir.exists(), "caller-owned image_save_dir was removed"

    # Both calls wrote their image into the shared caller dir (filenames are
    # page-number-namespaced, so no clobber here).
    for r in (results["a"], results["b"]):
        assert len(r.pages) == 1
        assert len(r.pages[0].images) == 1
        assert Path(r.pages[0].images[0]["path"]).exists()


def test_process_non_batch_sibling_also_cleans_up_own_temp_dir(monkeypatch, tmp_path):
    """The non-batch ``process`` sibling shares the identical temp-dir
    pattern; defensively apply the same fix there. This test pins the
    contract that a single ``process`` call cleans up its own temp dir
    (no leak) when ``image_save_dir=None``.

    ``process`` has no production caller today (``scrap`` uses
    ``process_batch``), but it is part of the public ``PDFProcessorStrategy``
    surface, so it must not leak on direct/external use.
    """
    pdf = _dummy_pdf(tmp_path, "x.pdf")

    done = threading.Event()
    release = threading.Event()
    page = _CoordinatedPage("page X", done, release, image=_FakeImage())

    pages_for_path = {str(pdf): [page]}
    _patch_pdf_reader(monkeypatch, pages_for_path)

    created_dirs = []
    import tempfile as _tempfile_mod
    real_mkdtemp = _tempfile_mod.mkdtemp

    def _recording_mkdtemp(*args, **kwargs):
        d = real_mkdtemp(*args, **kwargs)
        created_dirs.append(d)
        return d

    monkeypatch.setattr(_tempfile_mod, "mkdtemp", _recording_mkdtemp)

    processor = NaivePDFProcessorStrategy(
        extract_images=True,
        save_images_locally=True,
        image_save_dir=None,
    )

    out = {}

    def run():
        out["r"] = processor.process(pdf)

    t = threading.Thread(target=run)
    t.start()
    assert done.wait(timeout=10)
    release.set()
    t.join(timeout=10)
    assert not t.is_alive()

    result = out["r"]
    assert len(result.pages) == 1
    # The image entry was appended (the save succeeded before the temp dir
    # was cleaned up). The path is intentionally transient for the
    # image_save_dir=None branch.
    assert len(result.pages[0].images) == 1
    assert result.pages[0].images[0]["format"] == "png"
    assert result.pages[0].images[0]["path"].endswith("page_1_img_1.png")

    assert len(created_dirs) == 1
    assert not Path(created_dirs[0]).exists(), (
        f"process() leaked its temp dir: {created_dirs[0]}"
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-xvs"]))
