import os
from unittest.mock import Mock

import pytest

from crawl4ai.async_crawler_strategy import _nofollow_opener


def test_posix_nofollow_flag_is_preserved(monkeypatch):
    open_mock = Mock(return_value=17)
    monkeypatch.setattr(os, "O_NOFOLLOW", 0x20000, raising=False)
    monkeypatch.setattr(os, "open", open_mock)

    assert _nofollow_opener("download.bin", os.O_WRONLY) == 17
    open_mock.assert_called_once_with("download.bin", os.O_WRONLY | 0x20000)


def test_platform_without_nofollow_rejects_existing_symlink(monkeypatch):
    open_mock = Mock()
    monkeypatch.delattr(os, "O_NOFOLLOW", raising=False)
    monkeypatch.setattr(os.path, "islink", lambda _: True)
    monkeypatch.setattr(os, "open", open_mock)

    with pytest.raises(OSError, match="symlink"):
        _nofollow_opener("download.bin", os.O_WRONLY)

    open_mock.assert_not_called()


def test_platform_without_nofollow_opens_regular_file(monkeypatch):
    open_mock = Mock(return_value=23)
    monkeypatch.delattr(os, "O_NOFOLLOW", raising=False)
    monkeypatch.setattr(os.path, "islink", lambda _: False)
    monkeypatch.setattr(os, "open", open_mock)

    assert _nofollow_opener("download.bin", os.O_WRONLY) == 23
    open_mock.assert_called_once_with("download.bin", os.O_WRONLY)
