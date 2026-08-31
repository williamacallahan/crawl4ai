"""
R4 artifact-store behavioral tests: the server owns the directory, names and
bytes. Writes are O_EXCL|O_NOFOLLOW 0600; retrieval requires a 32-hex id,
refuses symlinks/non-regular files, and enforces a TTL.
"""

import os
import time

import pytest


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Point the artifact store at an isolated temp dir and reload it."""
    monkeypatch.setenv("CRAWL4AI_ARTIFACT_DIR", str(tmp_path / "art"))
    monkeypatch.setenv("CRAWL4AI_MAX_ARTIFACT_BYTES", "1024")
    monkeypatch.setenv("CRAWL4AI_ARTIFACT_QUOTA_BYTES", "4096")
    monkeypatch.setenv("CRAWL4AI_ARTIFACT_TTL_SECONDS", "3600")
    import importlib

    import artifacts
    importlib.reload(artifacts)
    artifacts.init_store()
    yield artifacts
    importlib.reload(artifacts)  # restore defaults for other tests


pytestmark = pytest.mark.posture


class TestWrite:
    def test_write_returns_hex_id_and_meta(self, store):
        meta = store.write_artifact("png", b"\x89PNG data")
        assert len(meta["artifact_id"]) == 32 and all(c in "0123456789abcdef" for c in meta["artifact_id"])
        assert meta["mime"] == "image/png" and meta["size"] == len(b"\x89PNG data")

    def test_dir_is_0700_and_file_0600(self, store):
        meta = store.write_artifact("pdf", b"%PDF-1.4")
        assert oct(os.stat(store.ARTIFACT_DIR).st_mode)[-3:] == "700"
        path, _ = store.resolve_artifact(meta["artifact_id"])
        assert oct(os.stat(path).st_mode)[-3:] == "600"

    def test_oversize_rejected(self, store):
        with pytest.raises(store.ArtifactTooLarge):
            store.write_artifact("png", b"x" * 2048)  # cap is 1024

    def test_quota_enforced(self, store):
        for _ in range(4):
            store.write_artifact("png", b"x" * 1000)
        with pytest.raises(store.QuotaExceeded):
            store.write_artifact("png", b"x" * 1000)  # would exceed 4096


class TestResolve:
    def test_roundtrip(self, store):
        meta = store.write_artifact("png", b"hello")
        path, mime = store.resolve_artifact(meta["artifact_id"])
        assert mime == "image/png"
        with open(path, "rb") as f:
            assert f.read() == b"hello"

    @pytest.mark.parametrize("bad", ["../etc/passwd", "..", "g" * 32, "abc", "/etc/passwd", "a" * 31])
    def test_non_hex_or_traversal_404(self, store, bad):
        with pytest.raises(store.ArtifactNotFound):
            store.resolve_artifact(bad)

    def test_symlink_not_followed(self, store):
        # Plant a symlink named like a valid artifact pointing at a secret.
        secret = os.path.join(store.ARTIFACT_DIR, "secret.txt")
        with open(secret, "wb") as f:
            f.write(b"TOPSECRET")
        fake_id = "a" * 32
        link = os.path.join(store.ARTIFACT_DIR, fake_id + ".png")
        os.symlink(secret, link)
        with pytest.raises(store.ArtifactNotFound):
            store.resolve_artifact(fake_id)  # lstat sees a symlink -> refuse

    def test_ttl_expired_404_and_reaped(self, store):
        meta = store.write_artifact("png", b"old")
        path, _ = store.resolve_artifact(meta["artifact_id"])
        old = time.time() - 7200  # 2h ago, TTL is 1h
        os.utime(path, (old, old))
        with pytest.raises(store.ArtifactNotFound):
            store.resolve_artifact(meta["artifact_id"])
        assert not os.path.exists(path)  # resolve reaped it


class TestWriteIsExclusiveAndNoFollow:
    def test_write_uses_oexcl_nofollow(self, store):
        import inspect
        src = inspect.getsource(store._write_artifact)
        assert "O_EXCL" in src and "O_NOFOLLOW" in src

    def test_quota_check_and_write_are_serialized(self, store, monkeypatch):
        import threading
        import time
        from concurrent.futures import ThreadPoolExecutor

        active = 0
        max_active = 0
        counter_lock = threading.Lock()
        real_dir_size = store._dir_size

        def observed_dir_size():
            nonlocal active, max_active
            with counter_lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.02)
            try:
                return real_dir_size()
            finally:
                with counter_lock:
                    active -= 1

        monkeypatch.setattr(store, "_dir_size", observed_dir_size)
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(store.write_artifact, "png", b"data")
                for _ in range(2)
            ]
            for future in futures:
                future.result()

        assert max_active == 1


class TestJanitor:
    def test_janitor_removes_symlinks_and_expired(self, store):
        meta = store.write_artifact("png", b"keep")
        # expired regular file
        old_meta = store.write_artifact("pdf", b"old")
        op, _ = store.resolve_artifact(old_meta["artifact_id"])
        past = time.time() - 7200
        os.utime(op, (past, past))
        # a planted symlink
        os.symlink("/etc/passwd", os.path.join(store.ARTIFACT_DIR, "deadbeef" * 4 + ".png"))
        reaped = store.janitor()
        assert reaped >= 2
        # the fresh one survives
        store.resolve_artifact(meta["artifact_id"])


class TestWriteFailureCleanup:
    """Regression for orphan-on-write-failure (introduced in 60886d1).

    A failed `f.write()` used to leave a partially-written file on disk with no
    cleanup. Because quota is checked pre-write against the larger requested
    size, a partial orphan leaves total usage <= quota, so the janitor's
    quota-based reaper never runs and TTL reaping takes up to 1 hour. A single
    transient I/O error could block legitimate retries for that whole window.
    The fix wraps the write in try/except OSError and unlinks the orphan.
    """

    @staticmethod
    def _install_failing_fdopen(monkeypatch, *, partial_bytes: int = 0,
                               fail_times: int = 1):
        """Patch `os.fdopen` so the next `fail_times` calls return a file whose
        `write()` raises OSError. If `partial_bytes > 0`, that many leading
        bytes are flushed to disk before the error to mimic a mid-stream
        failure leaving a non-empty partial file. Subsequent calls revert to
        the real `os.fdopen` so retries can succeed."""
        real_fdopen = os.fdopen

        class _FailingWriter:
            def __init__(self, fd):
                self._f = real_fdopen(fd, "wb")
                self._partial = partial_bytes

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                self._f.close()  # flushes any partial bytes to disk
                return False

            def write(self, data):
                if self._partial:
                    self._f.write(data[:self._partial])
                    self._partial = 0
                raise OSError("simulated disk I/O error during write")

        state = {"remaining": fail_times}

        def patched_fdopen(fd, mode):
            if state["remaining"] > 0:
                state["remaining"] -= 1
                return _FailingWriter(fd)
            return real_fdopen(fd, mode)

        monkeypatch.setattr(os, "fdopen", patched_fdopen)

    @pytest.mark.parametrize("partial_bytes", [0, 200])
    def test_failed_write_unlinks_orphan_and_reraises(self, store, monkeypatch, partial_bytes):
        # A failed write must reraise OSError AND leave no orphan behind,
        # whether the failure happens before any bytes land (0) or mid-stream
        # leaving a partial file (200) - the actual bug scenario.
        self._install_failing_fdopen(monkeypatch, partial_bytes=partial_bytes)
        with pytest.raises(OSError):
            store.write_artifact("png", b"x" * 500)
        assert os.listdir(store.ARTIFACT_DIR) == []

    def test_failed_write_preserves_existing_artifacts(self, store, monkeypatch):
        keep_meta = store.write_artifact("png", b"keep-me")
        self._install_failing_fdopen(monkeypatch)
        with pytest.raises(OSError):
            store.write_artifact("png", b"x" * 200)
        # The pre-existing legitimate artifact survives untouched.
        path, mime = store.resolve_artifact(keep_meta["artifact_id"])
        assert mime == "image/png"
        with open(path, "rb") as f:
            assert f.read() == b"keep-me"
        remaining = os.listdir(store.ARTIFACT_DIR)
        assert len(remaining) == 1

    def test_failed_write_does_not_block_subsequent_writes(self, store, monkeypatch):
        # Reproduce the report's impact scenario at the small test quota (4096).
        # 3000 bytes of legitimate artifacts already stored.
        for _ in range(3):
            store.write_artifact("png", b"y" * 1000)
        # A 900-byte write passes quota (3000 + 900 = 3900 <= 4096) but fails
        # mid-stream leaving a 500-byte orphan. fail_times=1 so the retry
        # below uses the real os.fdopen.
        self._install_failing_fdopen(monkeypatch, partial_bytes=500, fail_times=1)
        with pytest.raises(OSError):
            store.write_artifact("png", b"z" * 900)
        # WITHOUT the fix: orphan persists (total = 3500), retry of 900 bytes
        # would be rejected (3500 + 900 = 4400 > 4096) -> QuotaExceeded.
        # WITH the fix: orphan is gone (total = 3000), retry succeeds.
        meta = store.write_artifact("png", b"z" * 900)
        assert meta["size"] == 900
        store.resolve_artifact(meta["artifact_id"])

    def test_unlink_failure_does_not_mask_original_oserror(self, store, monkeypatch):
        # If the cleanup unlink also fails, the outer `raise` must still
        # surface the original write OSError (the inner except OSError: pass
        # must swallow only the unlink error).
        self._install_failing_fdopen(monkeypatch, partial_bytes=100)

        def failing_unlink(path):
            raise OSError("simulated cleanup failure")

        monkeypatch.setattr(os, "unlink", failing_unlink)
        with pytest.raises(OSError, match="simulated disk I/O error during write"):
            store.write_artifact("png", b"x" * 400)
