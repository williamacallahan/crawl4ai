"""Regression tests for the Docker-transport round-trip of ``DefaultTableExtraction``.

``to_serializable_dict`` (crawl4ai/async_configs.py) walks
``inspect.signature(cls.__init__).parameters`` to decide which instance
attributes to emit on the wire. When ``DefaultTableExtraction.__init__`` was
``def __init__(self, **kwargs)`` the serializer saw no named parameters and
emitted ``{"type": "DefaultTableExtraction", "params": {}}`` regardless of
what the caller set, silently dropping ``min_rows`` / ``min_cols`` on the
SDK -> Docker-server wire path (``CrawlerRunConfig.dump()`` ->
``CrawlerRunConfig.load(..., provenance=Provenance.UNTRUSTED)``). The server
then reconstructed ``DefaultTableExtraction()`` with default
``min_rows=0`` / ``min_cols=0``, disabling the user's minimum-size table
filter and returning sub-minimum-size tables the user asked to exclude.

The fix promotes ``table_score_threshold`` / ``min_rows`` / ``min_cols`` to
explicit named ``__init__`` parameters (keeping their defaults 7/0/0) while
leaving ``verbose`` / ``logger`` in ``**kwargs`` so the parent
``TableExtractionStrategy.__init__`` still receives them.

These tests guard the serializer contract and the ``verbose``-propagation
caveat against future regressions. They are pure-python (no browser, no
network) and use ``lxml.html.fromstring``-parsed fixtures plus the public
``CrawlerRunConfig.dump`` / ``CrawlerRunConfig.load`` round-trip.
"""

import copy
import inspect

import crawl4ai
from lxml import html

from crawl4ai.async_configs import (
    CrawlerRunConfig,
    Provenance,
    UNTRUSTED_ALLOWED_TYPES,
    to_serializable_dict,
)
from crawl4ai.table_extraction import DefaultTableExtraction

UNTRUSTED = Provenance.UNTRUSTED


# A small (1 data row) data table that ``min_rows`` must drop, and a large
# (8 data row) data table that ``min_rows`` must keep. Both pass
# ``is_data_table`` at the default threshold, so the only thing distinguishing
# them in the bug was the lost ``min_rows`` filter.
_SMALL_TABLE = (
    "<table><thead><tr><th>A</th><th>B</th></tr></thead>"
    "<tbody><tr><td>1</td><td>2</td></tr></tbody></table>"
)
_BIG_TABLE = (
    "<table><thead><tr><th>A</th><th>B</th><th>C</th><th>D</th></tr></thead>"
    "<tbody>"
    + "".join(
        f"<tr><td>{i}</td><td>{i}</td><td>{i}</td><td>{i}</td></tr>"
        for i in range(8)
    )
    + "</tbody></table>"
)


def _body_html():
    return "<html><body>" + _SMALL_TABLE + _BIG_TABLE + "</body></html>"


def _extract_tables_with_config(config: CrawlerRunConfig):
    """Mirror the production call site: content_scraping_strategy.py passes
    ``**config.__dict__.copy()`` into ``extract_tables``."""
    root = html.fromstring(_body_html())
    return config.table_extraction.extract_tables(root, **config.__dict__.copy())


def test_dump_envelope_carries_min_rows_and_min_cols():
    """The serializer must emit min_rows/min_cols, not an empty params dict.

    Guards against the original bug (``**kwargs``-only constructor hid the
    fields from ``inspect.signature``) and against any future serializer
    change that stops emitting these user-facing knobs.
    """
    serialized = to_serializable_dict(DefaultTableExtraction(min_rows=3, min_cols=2))

    assert serialized["type"] == "DefaultTableExtraction"
    assert serialized["params"].get("min_rows") == 3
    assert serialized["params"].get("min_cols") == 2


def test_init_signature_exposes_named_params():
    """The constructor must expose the named params the serializer introspects.

    Guards against a future refactor reverting to a ``def __init__(self,
    **kwargs)`` signature, which would silently re-drop ``min_rows`` /
    ``min_cols`` on the wire. ``verbose`` and ``logger`` must stay in
    ``**kwargs`` so the parent ``TableExtractionStrategy.__init__`` resolves
    them (see ``test_verbose_propagates_to_parent_through_kwargs``).
    """
    params = inspect.signature(DefaultTableExtraction.__init__).parameters

    assert "table_score_threshold" in params
    assert "min_rows" in params
    assert "min_cols" in params
    assert params["kwargs"].kind == inspect.Parameter.VAR_KEYWORD
    assert params["table_score_threshold"].default == 7
    assert params["min_rows"].default == 0
    assert params["min_cols"].default == 0
    assert "verbose" not in params
    assert "logger" not in params


def test_no_kwargs_only_init_remains_in_allowlisted_table_strategies():
    """No UNTRUSTED-allowlisted table strategy may have a kwargs-only constructor.

    A ``**kwargs``-only constructor hides user-facing params from the
    serializer (the root cause of this bug). This structural guard would
    have caught the original bug and prevents re-introducing the same
    class of regression for any new UNTRUSTED-allowlisted table strategy.
    """
    for type_name in ("DefaultTableExtraction", "NoTableExtraction"):
        assert type_name in UNTRUSTED_ALLOWED_TYPES
        cls = getattr(crawl4ai, type_name)
        params = [
            p for p in inspect.signature(cls.__init__).parameters.values()
            if p.name != "self"
        ]
        named = [
            p for p in params
            if p.kind not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
        ]
        if type_name == "DefaultTableExtraction":
            assert {"table_score_threshold", "min_rows", "min_cols"} <= {p.name for p in named}
        # NoTableExtraction has no user-facing params and round-trips trivially.


def test_verbose_propagates_to_parent_through_kwargs():
    """``verbose`` must stay in ``**kwargs`` so the parent resolves it.

    The bug-report's caveat: if ``verbose`` were promoted to a named param
    without explicitly forwarding it to ``super().__init__``, the parent's
    ``kwargs.get("verbose", False)`` would resolve to ``False``. This guard
    prevents that subtle regression.
    """
    assert DefaultTableExtraction(verbose=True).verbose is True
    assert DefaultTableExtraction().verbose is False


def test_nested_strategy_roundtrips_through_dump_load_untrusted():
    """``min_rows``/``min_cols`` survive the real SDK -> server transport.

    Mirrors ``Crawl4aiDockerClient._prepare_request`` (``crawler_config.dump()``)
    and the server's ``CrawlerRunConfig.load(..., provenance=UNTRUSTED)``.
    """
    config = CrawlerRunConfig(
        table_extraction=DefaultTableExtraction(min_rows=3, min_cols=2)
    )

    restored = CrawlerRunConfig.load(copy.deepcopy(config.dump()), provenance=UNTRUSTED)

    assert isinstance(restored.table_extraction, DefaultTableExtraction)
    assert restored.table_extraction.min_rows == 3
    assert restored.table_extraction.min_cols == 2


def test_filtered_table_set_is_identical_across_wire_roundtrip():
    """End-to-end: the filtered table set must not diverge across the wire.

    Before the fix: client (direct) returned 1 table while the server (after
    ``dump()``->``load(UNTRUSTED)``) returned 2, because ``min_rows``/``min_cols``
    were dropped. After the fix both ends apply the same filter.
    """
    client_cfg = CrawlerRunConfig(
        table_extraction=DefaultTableExtraction(min_rows=3, min_cols=2)
    )
    client_tables = _extract_tables_with_config(client_cfg)

    server_cfg = CrawlerRunConfig.load(
        copy.deepcopy(client_cfg.dump()), provenance=UNTRUSTED
    )
    server_tables = _extract_tables_with_config(server_cfg)

    assert len(client_tables) == 1
    assert len(server_tables) == 1
    assert len(client_tables) == len(server_tables)
