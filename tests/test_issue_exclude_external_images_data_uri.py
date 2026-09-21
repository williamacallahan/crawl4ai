"""Regression tests for ``exclude_external_images`` when ``<img src>`` is a
``data:`` placeholder.

The flag is documented to drop images hosted on other domains while keeping
same-domain images. The exclusion loop previously consulted only ``src`` and
``is_external_url`` treats every ``data:`` URI as external, so a lazy-loaded
``<img`` whose real URL lives in ``data-src``/``srcset``/``<picture><source>``
was removed entirely - discarding a same-domain real URL alongside the
placeholder. These tests pin the corrected behaviour, which mirrors the
attribute surface read by ``process_image``.
"""

import pytest
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy


PH = "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"
SVG = "data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'/%3E"
B64 = "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"


@pytest.fixture
def scraper():
    return LXMLWebScrapingStrategy()


def _run(scraper, html, **kwargs):
    return scraper.scrap("https://same-domain.com/page", html, **kwargs)


def _media_srcs(res):
    return [m.src for m in res.media.images]


def _img_count(res):
    import re

    return len(re.findall(r"<img", res.cleaned_html))


def _alt_list(res):
    import re

    return re.findall(r'alt="([^"]*)"', res.cleaned_html)


ALL_VARIANTS_HTML = f"""<!DOCTYPE html><html><body>
  <img src="{PH}" data-src="/local/real-photo.jpg" alt="Lazy real" width="800" height="600">
  <img src="{PH}" data-src="https://external.com/real.jpg" alt="Lazy ext" width="800" height="600">
  <img src="{SVG}" alt="Inline SVG" width="200" height="200">
  <img src="{B64}" alt="Inline base64" width="200" height="200">
  <img src="https://external.com/direct.jpg" alt="External direct" width="200" height="200">
  <img src="https://same-domain.com/photo.jpg" alt="Same domain direct" width="200" height="200">
</body></html>"""


def test_exclude_external_images_keeps_same_domain_lazy_load(scraper):
    """The reported defect: a same-domain image referenced via ``data-src``
    on an ``<img`` whose ``src`` is a ``data:`` placeholder is recovered into
    ``media.images`` instead of being discarded."""
    res = _run(scraper, ALL_VARIANTS_HTML, exclude_external_images=True)
    srcs = _media_srcs(res)
    assert "/local/real-photo.jpg" in srcs
    assert "https://same-domain.com/photo.jpg" in srcs


def test_exclude_external_images_still_excludes_external_lazy_load(scraper):
    """Edge case the naive ``not src.startswith("data:")`` guard regresses: an
    external real URL carried in ``data-src`` must still be excluded."""
    res = _run(scraper, ALL_VARIANTS_HTML, exclude_external_images=True)
    srcs = _media_srcs(res)
    assert "https://external.com/real.jpg" not in srcs
    assert "https://external.com/direct.jpg" not in srcs


def test_exclude_external_images_keeps_inline_data_uri_in_cleaned_html(scraper):
    """A ``data:``-only ``<img`` (inline content, no real URL) must survive in
    ``cleaned_html`` - consistent with how ``add_variant`` skips ``data:`` URIs
    from ``media.images`` instead of treating them as external resources."""
    res = _run(scraper, ALL_VARIANTS_HTML, exclude_external_images=True)
    alts = _alt_list(res)
    assert "Inline SVG" in alts
    assert "Inline base64" in alts


def test_exclude_external_images_srcset_lazy_load_same_domain(scraper):
    """Same-domain real URL carried in ``srcset`` (with a ``data:`` placeholder
    for ``src``) is recovered into ``media.images`` and the element survives."""
    html = (
        f'<html><body><img src="{PH}" '
        'srcset="/local/x.jpg 1x, /local/x@2x.jpg 2x" '
        'alt="L" width="800" height="600"></body></html>'
    )
    res = _run(scraper, html, exclude_external_images=True)
    srcs = _media_srcs(res)
    assert "/local/x.jpg" in srcs
    assert "/local/x@2x.jpg" in srcs
    assert _img_count(res) == 1


def test_exclude_external_images_srcset_lazy_load_all_external(scraper):
    """When every real candidate (``srcset`` here) is external, the element is
    removed and none of its URLs surface in ``media.images``."""
    html = (
        f'<html><body><img src="{PH}" '
        'srcset="https://external.com/x.jpg 1x, https://external.com/x@2x.jpg 2x" '
        'alt="L" width="800" height="600"></body></html>'
    )
    res = _run(scraper, html, exclude_external_images=True)
    assert _media_srcs(res) == []
    assert _img_count(res) == 0


def test_exclude_external_images_data_srcset_lazy_load(scraper):
    """``data-srcset`` is consulted alongside ``srcset``."""
    html = (
        f'<html><body><img src="{PH}" '
        'data-srcset="/local/y.webp 480w, /local/y@2x.webp 960w" '
        'alt="L" width="800" height="600"></body></html>'
    )
    res = _run(scraper, html, exclude_external_images=True)
    srcs = _media_srcs(res)
    assert "/local/y.webp" in srcs
    assert "/local/y@2x.webp" in srcs
    assert _img_count(res) == 1


def test_exclude_external_images_picture_source_same_domain(scraper):
    """Same-domain URL carried in ``<picture><source>`` is recovered into
    ``media.images`` and the ``<img`` survives."""
    html = f"""<html><body>
<picture>
  <source srcset="/local/big.webp 1x">
  <img src="{PH}" alt="L" width="800" height="600">
</picture></body></html>"""
    res = _run(scraper, html, exclude_external_images=True)
    assert "/local/big.webp" in _media_srcs(res)
    assert _img_count(res) >= 1


def test_exclude_external_images_picture_source_all_external(scraper):
    """All-external ``<picture><source>`` + ``data:``-placeholder ``<img`` is
    excluded - neither the source URL nor the placeholder leaks into media."""
    html = f"""<html><body>
<picture>
  <source srcset="https://external.com/big.webp 1x">
  <img src="{PH}" alt="L" width="800" height="600">
</picture></body></html>"""
    res = _run(scraper, html, exclude_external_images=True)
    assert _media_srcs(res) == []
    assert _img_count(res) == 0


def test_exclude_external_images_direct_external_removed(scraper):
    """Regression guard for the existing 0.7.0 release test: a direct
    external ``src`` is still dropped from ``cleaned_html``."""
    html = (
        "<html><body>"
        '<img src="/local-image.jpg" alt="Local">'
        '<img src="https://external.com/image.jpg" alt="External">'
        "</body></html>"
    )
    res = _run(scraper, html, exclude_external_images=True)
    assert "external.com" not in res.cleaned_html
    assert "/local-image.jpg" in res.cleaned_html


def test_exclude_external_images_direct_same_domain_kept(scraper):
    """Regression guard for the same-domain direct path."""
    html = (
        "<html><body>"
        '<img src="/local-image.jpg" alt="Local">'
        '<img src="https://same-domain.com/image.jpg" alt="Same">'
        "</body></html>"
    )
    res = _run(scraper, html, exclude_external_images=True)
    assert "/local-image.jpg" in res.cleaned_html
    assert "https://same-domain.com/image.jpg" in res.cleaned_html
