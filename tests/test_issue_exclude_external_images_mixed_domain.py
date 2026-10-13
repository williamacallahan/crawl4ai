"""Regression tests for ``exclude_external_images`` with mixed-domain ``<img>``.

``exclude_external_images`` is documented (``async_configs.py``) to "exclude all
external images from processing." The image-exclusion gate keeps an ``<img>``
whenever at least one real candidate URL is same-domain (see
``_is_image_external``), so a lazy-loaded same-domain image whose real URL lives
in ``data-src``/``srcset``/``<picture><source>`` is preserved. Before this fix,
a *mixed* image - one carrying both external and same-domain URL candidates -
was kept by design but ``process_image.add_variant`` then emitted *every*
variant (including the external ones) into ``media.images``, and an external
``src`` (which is in ``IMPORTANT_ATTRS``) survived into ``cleaned_html``.

These tests pin the corrected behaviour: for a kept mixed image, the external
URL candidates are stripped from both ``media.images`` and the element's
URL-bearing attributes before ``cleaned_html`` serialization, while the
same-domain variants survive.
"""

import re

import pytest

from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy


@pytest.fixture
def scraper():
    return LXMLWebScrapingStrategy()


def _run(scraper, html, **kwargs):
    return scraper.scrap("https://same-domain.com/page", html, **kwargs)


def _media_srcs(res):
    return [m.src for m in res.media.images]


def _img_count(res):
    return len(re.findall(r"<img", res.cleaned_html))


def _external_in(res, host="external"):
    return host in res.cleaned_html


def test_mixed_srcset_local_src_plus_cdn_in_srcset_no_media_leak(scraper):
    """A same-domain ``src`` with a ``srcset`` listing a CDN URL and a
    same-domain fallback must not emit the CDN URL into ``media.images``."""
    html = (
        "<html><body>"
        '<img src="/local.jpg" '
        'srcset="https://cdn.external.com/photo-1x.jpg 1x, /local/photo-2x.jpg 2x" '
        'alt="Mixed srcset" width="800" height="600">'
        "</body></html>"
    )
    res = _run(scraper, html, exclude_external_images=True)
    srcs = _media_srcs(res)
    assert "https://cdn.external.com/photo-1x.jpg" not in srcs
    assert "/local.jpg" in srcs
    assert "/local/photo-2x.jpg" in srcs


def test_mixed_external_src_local_data_src_no_media_leak(scraper):
    """An external ``src`` with a same-domain ``data-src`` must not emit the
    external URL into ``media.images``; the same-domain variant survives and
    the element survives with a same-domain ``src`` in ``cleaned_html``."""
    html = (
        "<html><body>"
        '<img src="https://external.example.com/a.jpg" data-src="/local/b.jpg" '
        'alt="Mixed ext src local dataSrc" width="800" height="600">'
        "</body></html>"
    )
    res = _run(scraper, html, exclude_external_images=True)
    srcs = _media_srcs(res)
    assert "https://external.example.com/a.jpg" not in srcs
    assert "/local/b.jpg" in srcs
    assert not _external_in(res, "external.example.com")
    assert _img_count(res) == 1
    assert "/local/b.jpg" in res.cleaned_html


def test_reverse_mixed_local_src_external_data_src_no_leak(scraper):
    """A same-domain ``src`` with an external ``data-src`` (the pre-existing
    downstream leak) must not emit the external ``data-src`` into
    ``media.images`` nor keep it in ``cleaned_html``."""
    html = (
        "<html><body>"
        '<img src="/local.jpg" data-src="https://external.example.com/real.jpg" '
        'alt="Reverse mixed" width="800" height="600">'
        "</body></html>"
    )
    res = _run(scraper, html, exclude_external_images=True)
    srcs = _media_srcs(res)
    assert "https://external.example.com/real.jpg" not in srcs
    assert "/local.jpg" in srcs
    assert not _external_in(res, "external.example.com")
    assert _img_count(res) == 1


def test_mixed_data_srcset_external_filtered(scraper):
    """A ``data-srcset`` mixing an external and a same-domain variant keeps
    only the same-domain variant in ``media.images`` and out of
    ``cleaned_html``."""
    html = (
        "<html><body>"
        '<img src="/local.jpg" '
        'data-srcset="https://external.example.com/x.webp 480w, /local/y.webp 960w" '
        'alt="Mixed data-srcset" width="800" height="600">'
        "</body></html>"
    )
    res = _run(scraper, html, exclude_external_images=True)
    srcs = _media_srcs(res)
    assert "https://external.example.com/x.webp" not in srcs
    assert "/local/y.webp" in srcs
    assert not _external_in(res, "external.example.com")


def test_mixed_picture_source_external_filtered_media(scraper):
    """A ``<picture>`` with an external ``<source>`` and a same-domain
    ``<source>`` keeps only the same-domain source URL in ``media.images``."""
    html = (
        "<html><body>"
        "<picture>"
        '<source srcset="https://cdn.external.com/big.webp 1x">'
        '<source srcset="/local/big.webp 1x">'
        '<img src="/local.jpg" alt="Pic mixed" width="800" height="600">'
        "</picture>"
        "</body></html>"
    )
    res = _run(scraper, html, exclude_external_images=True)
    srcs = _media_srcs(res)
    assert "https://cdn.external.com/big.webp" not in srcs
    assert "/local/big.webp" in srcs
    assert "/local.jpg" in srcs


def test_mixed_external_img_src_with_same_domain_picture_source(scraper):
    """An ``<img>`` with an external ``src`` kept alive by a same-domain
    ``<picture><source>`` has its ``src`` rewritten to the same-domain source
    URL so the external URL does not leak into ``cleaned_html``."""
    html = (
        "<html><body>"
        "<picture>"
        '<source srcset="/local/big.webp 1x">'
        '<img src="https://cdn.external.com/a.jpg" alt="Pic ext img" width="800" height="600">'
        "</picture>"
        "</body></html>"
    )
    res = _run(scraper, html, exclude_external_images=True)
    assert "https://cdn.external.com/a.jpg" not in _media_srcs(res)
    assert not _external_in(res, "cdn.external.com")
    assert _img_count(res) == 1
    assert "/local/big.webp" in res.cleaned_html


def test_strip_applies_even_when_image_below_score_threshold(scraper):
    """The ``cleaned_html`` strip runs for every kept image regardless of
    whether ``process_image`` returned variants, so a kept mixed image that
    does not enter ``media.images`` still has its external ``src`` stripped
    from ``cleaned_html``."""
    html = (
        "<html><body>"
        '<img src="https://external.example.com/a.jpg" data-src="/local/b.jpg">'
        "</body></html>"
    )
    res = _run(scraper, html, exclude_external_images=True)
    assert _media_srcs(res) == []
    assert not _external_in(res, "external.example.com")
    assert _img_count(res) == 1
    assert "/local/b.jpg" in res.cleaned_html


def test_strip_external_data_src_under_keep_data_attributes(scraper):
    """When ``keep_data_attributes=True`` an external ``data-src`` would
    otherwise survive into ``cleaned_html``; it must be stripped there too
    while the same-domain ``src`` is preserved."""
    html = (
        "<html><body>"
        '<img src="/local.jpg" data-src="https://external.example.com/real.jpg" '
        'alt="Reverse mixed keep data" width="800" height="600">'
        "</body></html>"
    )
    res = _run(scraper, html, exclude_external_images=True, keep_data_attributes=True)
    assert "https://external.example.com/real.jpg" not in _media_srcs(res)
    assert "/local.jpg" in _media_srcs(res)
    assert "external.example.com" not in res.cleaned_html
    assert "/local.jpg" in res.cleaned_html


def test_pure_external_image_still_dropped(scraper):
    """A single-candidate pure-external ``<img>`` is still dropped at the gate
    (no leak in either surface)."""
    html = (
        "<html><body>"
        '<img src="https://external.com/image.jpg" alt="Pure external" width="800" height="600">'
        "</body></html>"
    )
    res = _run(scraper, html, exclude_external_images=True)
    assert _media_srcs(res) == []
    assert not _external_in(res, "external.com")
    assert _img_count(res) == 0


def test_flag_off_keeps_external_variants(scraper):
    """With ``exclude_external_images=False`` the fix is inert: every variant
    (including external) is emitted and the external ``src`` survives in
    ``cleaned_html``."""
    html = (
        "<html><body>"
        '<img src="https://external.example.com/a.jpg" data-src="/local/b.jpg" '
        'alt="Flag off mixed" width="800" height="600">'
        "</body></html>"
    )
    res = _run(scraper, html, exclude_external_images=False)
    srcs = _media_srcs(res)
    assert "https://external.example.com/a.jpg" in srcs
    assert "/local/b.jpg" in srcs
    assert "external.example.com" in res.cleaned_html
