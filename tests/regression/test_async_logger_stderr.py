"""
Tests for issue #1968: AsyncLogger must write to stderr by default so that
stdout-based transports (e.g. MCP stdio) are not corrupted.
"""

import importlib.util
import io
import sys
from pathlib import Path

from rich.console import Console

# Load async_logger directly without triggering the full crawl4ai __init__
# (which pulls in many optional deps like aiofiles, OpenSSL, playwright …).
_spec = importlib.util.spec_from_file_location(
    "crawl4ai.async_logger",
    Path(__file__).parents[2] / "crawl4ai" / "async_logger.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]

AsyncLogger = _mod.AsyncLogger
AsyncFileLogger = _mod.AsyncFileLogger
LogLevel = _mod.LogLevel
LogColor = _mod.LogColor


class _RecordingConsole:
    def __init__(self):
        self.lines = []

    def print(self, line):
        self.lines.append(line)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _capture_consoles():
    """Return StringIO objects for stdout and stderr plus a no-color Console
    for each, suitable for injecting into AsyncLogger in tests."""
    out = io.StringIO()
    err = io.StringIO()
    stdout_console = Console(file=out, no_color=True, highlight=False)
    stderr_console = Console(file=err, no_color=True, highlight=False)
    return out, err, stdout_console, stderr_console


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAsyncLoggerDefaultsToStderr:
    """The default AsyncLogger must NOT write to stdout."""

    def test_default_console_writes_to_stderr_not_stdout(self, capsys):
        """Logging to the default AsyncLogger must not pollute stdout."""
        logger = AsyncLogger(verbose=True)

        logger.info("hello from logger", tag="TEST")
        logger.error("error message", tag="TEST")
        logger.warning("warning message", tag="TEST")

        captured = capsys.readouterr()
        # stdout must remain pristine (no log lines mixed in)
        assert captured.out == "", (
            "AsyncLogger wrote to stdout — this breaks MCP stdio transport!\n"
            f"stdout content: {captured.out!r}"
        )
        # stderr should contain the output
        assert "hello from logger" in captured.err, (
            "Expected log output on stderr but found nothing.\n"
            f"stderr content: {captured.err!r}"
        )

    def test_default_console_is_stderr(self):
        """The internal console.file should be sys.stderr (or equivalent)."""
        logger = AsyncLogger(verbose=True)
        # Rich Console with stderr=True writes to sys.stderr
        assert logger.console.file is sys.stderr, (
            f"Expected logger.console.file to be sys.stderr, "
            f"got {logger.console.file!r}"
        )


class TestAsyncLoggerCustomConsole:
    """Callers can inject a custom Console (e.g. stdout for non-MCP use)."""

    def test_custom_console_is_respected(self):
        """When a custom Console is passed, it must be used verbatim."""
        buf = io.StringIO()
        custom = Console(file=buf, no_color=True, highlight=False)
        logger = AsyncLogger(verbose=True, console=custom)

        logger.info("custom target", tag="TEST")

        output = buf.getvalue()
        assert "custom target" in output, (
            f"Expected log output in custom console, got: {output!r}"
        )

    def test_stdout_console_can_be_injected(self, capsys):
        """Passing Console(file=sys.stdout) restores legacy behaviour."""
        stdout_console = Console(file=sys.stdout, no_color=True, highlight=False)
        logger = AsyncLogger(verbose=True, console=stdout_console)

        logger.info("going to stdout", tag="TEST")

        captured = capsys.readouterr()
        assert "going to stdout" in captured.out


class TestAsyncLoggerVerboseFalse:
    """verbose=False must suppress console output regardless of the target stream."""

    def test_no_output_when_verbose_false(self, capsys):
        logger = AsyncLogger(verbose=False)
        logger.info("silent message", tag="TEST")
        logger.error("silent error", tag="TEST")

        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""


class TestAsyncLoggerSeverity:
    def test_generic_failures_never_infer_severity_from_text_or_tag(self):
        console = _RecordingConsole()
        logger = AsyncLogger(console=console)

        logger.error_status(
            "https://example.com",
            "Blocked by anti-bot protection: challenge page",
        )
        logger.error_status(
            "https://example.com", "page.goto: net::ERR_CONNECTION_REFUSED"
        )
        logger.url_status("https://example.com", False, 1.0, tag="COMPLETE")
        logger.error_status("https://example.com", "Redis connection refused")

        assert all(line.startswith("[red]") for line in console.lines)
        assert "Blocked by anti-bot protection" in console.lines[0]
        assert "net::ERR_CONNECTION_REFUSED" in console.lines[1]

    def test_file_logger_generic_failures_are_errors(self, tmp_path):
        log_file = tmp_path / "severity.log"
        logger = AsyncFileLogger(str(log_file))

        logger.error_status(
            "https://example.com",
            "Blocked by anti-bot protection: challenge page",
        )
        logger.error_status(
            "https://example.com", "page.goto: net::ERR_CONNECTION_REFUSED"
        )
        logger.url_status("https://example.com", False, 1.0, tag="COMPLETE")

        lines = log_file.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 3
        assert all("[ERROR]" in line for line in lines)

    def test_filtered_messages_are_not_formatted(self):
        logger = AsyncLogger(log_level=LogLevel.WARNING)

        logger.info("{missing}", params={"other": "value"})


class TestAsyncLoggerFileLogging:
    """File logging path should still work after the stderr change."""

    def test_file_logging_still_works(self, tmp_path):
        log_file = tmp_path / "test.log"
        logger = AsyncLogger(log_file=str(log_file), verbose=False)
        logger.info("file log entry", tag="FILE")

        content = log_file.read_text(encoding="utf-8")
        assert "file log entry" in content


class TestMCPScenario:
    """Simulates what MCP stdio transport sees on stdout."""

    def test_stdout_clean_across_all_log_levels(self, capsys):
        """Simulate MCP usage: crawl + log, stdout must only have JSON."""
        logger = AsyncLogger(verbose=True)
        fake_json_response = '{"jsonrpc": "2.0", "result": "ok", "id": 1}'

        # Simulate interleaved logging (as would happen during a crawl)
        logger.info("Crawling started", tag="CRAWL")
        logger.warning("Slow response", tag="CRAWL")
        logger.success("Done", tag="CRAWL")

        # MCP server would write JSON to stdout
        print(fake_json_response)

        captured = capsys.readouterr()

        # stdout must contain ONLY the JSON line
        assert captured.out.strip() == fake_json_response, (
            "Stdout was polluted by logger output!\n"
            f"stdout: {captured.out!r}"
        )
        # All log output should be on stderr
        assert "Crawling started" in captured.err


# ---------------------------------------------------------------------------
# Helpers for bracket-escaping tests
# ---------------------------------------------------------------------------


def _render_markup(lines):
    """Render Rich markup strings through a no-color Console and return the
    concatenated plain-text output. The default AsyncLogger Console applies
    markup parsing, so tests covering literal bracket preservation must
    round-trip through Rich rather than inspect the raw markup string
    (which contains ``\\[`` escapes)."""
    buf = io.StringIO()
    console = Console(file=buf, no_color=True, highlight=False, width=200)
    for line in lines:
        console.print(line)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Tests for Rich bracket escaping (issue: literal "[word]" tokens must survive)
# ---------------------------------------------------------------------------


class TestAsyncLoggerBracketEscaping:
    """Literal bracket tokens like ``[zstd]`` or ``[Errno 2]`` in the message
    template or in param values must render unchanged. Rich parses ``[word]``
    as a markup/style tag, so the logger must escape literal ``[`` as ``\\[``
    (Rich's escape syntax) — NOT as ``[[`` doubling, which strips single-word
    tokens (``[zstd]`` -> ``[]``) and doubles multi-word tokens (``[Errno 2]``
    -> ``[[Errno 2]]``)."""

    def test_template_message_single_word_bracket_preserved(self):
        """A template message (no params) with a single-word bracket token
        must not have the inner word stripped by Rich."""
        console = _RecordingConsole()
        logger = AsyncLogger(console=console)
        logger.error("install httpx[zstd] to enable decoding", tag="FETCH")
        rendered = _render_markup(console.lines)
        assert "httpx[zstd]" in rendered, (
            "Single-word bracket token [zstd] was stripped; rendered: "
            f"{rendered!r}"
        )
        assert "httpx[]" not in rendered, "Bracket word was lost"

    def test_template_message_multi_word_bracket_not_doubled(self):
        """A multi-word bracket token on the no-params path must not have its
        brackets doubled (the old ``[[Errno 2]]`` mangling)."""
        console = _RecordingConsole()
        logger = AsyncLogger(console=console)
        logger.error("[Errno 2] No such file or directory", tag="FETCH")
        rendered = _render_markup(console.lines)
        assert "[Errno 2]" in rendered, (
            f"Multi-word bracket token missing; rendered: {rendered!r}"
        )
        assert "[[Errno 2]]" not in rendered, (
            "Brackets were doubled by broken [[/]] escaping; rendered: "
            f"{rendered!r}"
        )

    def test_error_status_preserves_bracket_in_error_param(self):
        """``error_status`` passes the raw error text as a param with no
        colors/boxes; bracket tokens in that string must survive."""
        console = _RecordingConsole()
        logger = AsyncLogger(console=console)
        logger.error_status(
            "https://example.com/p",
            "Make sure to install httpx using `pip install httpx[zstd]`.",
            tag="FETCH",
        )
        rendered = _render_markup(console.lines)
        assert "httpx[zstd]" in rendered, (
            "Bracket token in error param was stripped; rendered: "
            f"{rendered!r}"
        )

    def test_error_param_closing_tag_does_not_raise(self):
        """A raw closing Rich tag ``[/url]`` in an error string must render
        literally rather than raise ``MarkupError`` (unbalanced tag)."""
        console = _RecordingConsole()
        logger = AsyncLogger(console=console)
        logger.error_status(
            "https://example.com/p",
            "Config error: missing [/url] endpoint",
            tag="FETCH",
        )
        rendered = _render_markup(console.lines)
        assert "[/url]" in rendered, (
            f"Closing tag [/url] not preserved; rendered: {rendered!r}"
        )

    def test_string_param_with_bracket_preserved(self):
        """A string param value containing a bracket token must survive the
        ``.format(**params)`` substitution into the template."""
        console = _RecordingConsole()
        logger = AsyncLogger(console=console)
        logger.error(
            "importing {module}",
            params={"module": "httpx[zstd]"},
            tag="FETCH",
        )
        rendered = _render_markup(console.lines)
        assert "httpx[zstd]" in rendered, (
            "Bracket token in string param was stripped; rendered: "
            f"{rendered!r}"
        )

    def test_url_status_timing_format_spec_still_works(self):
        """Non-string params (floats) must remain unescaped so format specs
        like ``{timing:.2f}`` keep working."""
        console = _RecordingConsole()
        logger = AsyncLogger(console=console)
        logger.url_status("https://example.com/p", True, 1.23333, tag="FETCH")
        rendered = _render_markup(console.lines)
        assert "1.23s" in rendered, (
            f"Float format spec {timing:.2f} broken; rendered: {rendered!r}"
        )

    def test_colored_string_param_with_bracket_preserved(self):
        """When a string param is both colored AND contains brackets, both
        the color markup and the literal brackets must render correctly."""
        console = _RecordingConsole()
        logger = AsyncLogger(console=console)
        logger.info(
            "module is {name}",
            params={"name": "httpx[zstd]"},
            colors={"name": LogColor.CYAN},
        )
        rendered = _render_markup(console.lines)
        assert "httpx[zstd]" in rendered, (
            "Colored bracket param lost its bracket; rendered: "
            f"{rendered!r}"
        )

    def test_file_log_preserves_bracket_in_error_param(self, tmp_path):
        """The file-logging path (Text.from_markup -> plain) must preserve
        bracket tokens from error_status's raw error param."""
        log_file = tmp_path / "brackets.log"
        logger = AsyncLogger(log_file=str(log_file), verbose=False)
        logger.error_status(
            "https://example.com/p",
            "install httpx[zstd] to enable decoding",
            tag="FETCH",
        )
        content = log_file.read_text(encoding="utf-8")
        assert "httpx[zstd]" in content, (
            "Bracket token lost in file output; content: "
            f"{content!r}"
        )
        assert "httpx[]" not in content

    def test_verbose_console_path_does_not_crash_on_brackets(self, capsys):
        """The verbose console-print path (default stderr Console) must not
        raise when the message contains literal bracket tokens."""
        logger = AsyncLogger(verbose=True)
        logger.error("install httpx[zstd] then [Errno 2] and [/url]", tag="FETCH")
        captured = capsys.readouterr()
        assert "httpx[zstd]" in captured.err
        assert "[Errno 2]" in captured.err
        assert "[/url]" in captured.err


def test_raw_backslashes_and_tags_survive_template_and_colored_params():
    text = r"C:\[file] and \[/url]"
    console = _RecordingConsole()
    logger = AsyncLogger(console=console)
    logger.error(text)
    logger.error("{detail}", params={"detail": text}, colors={"detail": LogColor.RED})
    assert _render_markup(console.lines).count(text) == 2


# ---------------------------------------------------------------------------
# Tests for multi-line file-output normalization
# (issue: file sink must preserve one-timestamp-per-record invariant when
# the message template, a param value, or box art contains embedded newlines)
# ---------------------------------------------------------------------------


class TestAsyncLoggerFileNewlineNormalization:
    """The file sink must emit exactly one timestamped physical line per
    ``_log`` call. Multi-line content (template, param values, or box art)
    used to fragment a single record across N physical lines with the
    timestamp scoping only L1, breaking "one record per line" readers."""

    def test_multiline_template_emits_single_timestamped_line(self, tmp_path):
        """A template containing embedded ``\\n`` must produce exactly one
        physical line in the file, prefixed by a single timestamp."""
        log_file = tmp_path / "multiline.log"
        logger = AsyncLogger(log_file=str(log_file), verbose=False)
        logger.error(
            "Database initialization failed:\n{error}\n\nContext:\n{context}\n\nTraceback:\n{traceback}",
            tag="ERROR",
            force_verbose=True,
            params={
                "error": "no such table: crawled_data",
                "context": "  conn = await aiosqlite.connect(self.db_path)\n  await conn.execute('PRAGMA journal_mode = WAL')",
                "traceback": (
                    "Traceback (most recent call last):\n"
                    '  File "async_database.py", line 134, in get_connection\n'
                    "    conn = await aiosqlite.connect(self.db_path)\n"
                    "sqlite3.OperationalError: no such table: crawled_data"
                ),
            },
        )
        lines = log_file.read_text(encoding="utf-8").splitlines()
        # One log call -> exactly one physical line.
        assert len(lines) == 1, (
            f"Expected 1 physical line for 1 multi-line log call, got "
            f"{len(lines)}: {lines!r}"
        )
        # That single line must carry the timestamp prefix.
        assert lines[0].startswith("[20"), f"Missing timestamp prefix: {lines[0]!r}"
        # Every section of the original message must remain readable on the line.
        for fragment in (
            "Database initialization failed:",
            "no such table: crawled_data",
            "Context:",
            "conn = await aiosqlite.connect",
            "Traceback (most recent call last):",
            "sqlite3.OperationalError: no such table: crawled_data",
        ):
            assert fragment in lines[0], (
                f"Fragment {fragment!r} lost from flattened record: {lines[0]!r}"
            )
        # A visible continuation marker must indicate where newlines were.
        assert "⏎" in lines[0], (
            f"Continuation marker missing from flattened record: {lines[0]!r}"
        )

    def test_one_log_call_produces_one_physical_line(self, tmp_path):
        """N log calls must produce exactly N physical lines, regardless of
        how many ``\\n`` characters each call's message contains."""
        log_file = tmp_path / "counts.log"
        logger = AsyncLogger(log_file=str(log_file), verbose=False)
        messages = [
            "single line",
            "two\nlines",
            "three\nline\nmessage",
            "trailing newline\n",
            "leading newline\nmid",
            "\n\n\nblank lines only\n\n\n",
        ]
        for msg in messages:
            logger.error(msg, tag="TEST")

        lines = log_file.read_text(encoding="utf-8").splitlines()
        assert len(lines) == len(messages), (
            f"Expected {len(messages)} physical lines (one per call), got "
            f"{len(lines)}: {lines!r}"
        )
        # Every physical line must carry a timestamp prefix (no orphan
        # continuation lines).
        assert all(line.startswith("[20") for line in lines), (
            f"Found an un-timestamped continuation line: {lines!r}"
        )

    def test_single_line_messages_are_unaffected(self, tmp_path):
        """Messages without ``\\n`` must round-trip unchanged (regression
        guard for the normalization path)."""
        log_file = tmp_path / "single.log"
        logger = AsyncLogger(log_file=str(log_file), verbose=False)
        logger.info("plain message", tag="TEST")
        logger.error("another plain message", tag="TEST")

        lines = log_file.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        assert "plain message" in lines[0]
        assert "another plain message" in lines[1]
        # Single-line messages must NOT receive a continuation marker.
        assert "⏎" not in lines[0], (
            f"Continuation marker injected into single-line message: {lines[0]!r}"
        )
        assert "⏎" not in lines[1], (
            f"Continuation marker injected into single-line message: {lines[1]!r}"
        )

    def test_boxes_variant_emits_single_timestamped_line(self, tmp_path):
        """The ``boxes=[...]`` path goes through ``create_box_message`` which
        returns ``\\n{box}\\n`` — this secondary trigger must also collapse
        to one timestamped physical line."""
        log_file = tmp_path / "boxes.log"
        logger = AsyncLogger(log_file=str(log_file), verbose=False)
        logger.error(
            "{error}",
            tag="ERROR",
            params={"error": "Connection refused"},
            boxes=["error"],
        )
        lines = log_file.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1, (
            f"boxes=['error'] must yield one physical line, got {len(lines)}: "
            f"{lines!r}"
        )
        assert lines[0].startswith("[20"), f"Missing timestamp prefix: {lines[0]!r}"
        assert "Connection refused" in lines[0], (
            f"Box payload lost from flattened record: {lines[0]!r}"
        )
        # The continuation marker should appear because the box injects \n.
        assert "⏎" in lines[0]

    def test_carriage_returns_are_also_normalized(self, tmp_path):
        """``\\r\\n`` and lone ``\\r`` (Windows / legacy) must collapse the
        same way lone ``\\n`` does, so file output stays one-record-per-line
        regardless of newline convention."""
        log_file = tmp_path / "crlf.log"
        logger = AsyncLogger(log_file=str(log_file), verbose=False)
        logger.error("line1\r\nline2\rline3\nline4", tag="TEST")
        lines = log_file.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1, (
            f"CRLF/CR must flatten to one line, got {len(lines)}: {lines!r}"
        )
        for frag in ("line1", "line2", "line3", "line4"):
            assert frag in lines[0], f"Fragment {frag!r} lost: {lines[0]!r}"

    def test_blank_line_collapsing_preserves_nonblank_content(self, tmp_path):
        """A message that is only blank lines must not produce an orphan
        empty line in the file (which would be an un-timestamped record)."""
        log_file = tmp_path / "blanks.log"
        logger = AsyncLogger(log_file=str(log_file), verbose=False)
        logger.error("\n\n\n", tag="TEST")
        logger.error("real\n\n\nmessage", tag="TEST")
        lines = log_file.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2, (
            f"Expected 2 physical lines, got {len(lines)}: {lines!r}"
        )
        # The all-blank call must still carry a timestamp (empty payload ok).
        assert lines[0].startswith("[20"), f"Missing timestamp on L1: {lines[0]!r}"
        # The mixed call must retain the real content.
        assert "real" in lines[1] and "message" in lines[1], (
            f"Real content lost from mixed call: {lines[1]!r}"
        )
        # An empty-output call should not inject a continuation marker.
        assert "⏎" not in lines[0], (
            f"Continuation marker on empty record: {lines[0]!r}"
        )


class TestAsyncFileLoggerNewlineNormalization:
    """The file-only ``AsyncFileLogger`` shares the one-timestamp-per-line
    format and must preserve the same invariant when callers pass multi-line
    messages directly."""

    def test_multiline_message_emits_single_timestamped_line(self, tmp_path):
        log_file = tmp_path / "filelogger.log"
        logger = AsyncFileLogger(str(log_file))
        logger.error("line1\nline2\nline3", tag="CUSTOM")
        lines = log_file.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1, (
            f"Expected 1 physical line, got {len(lines)}: {lines!r}"
        )
        assert lines[0].startswith("[20"), f"Missing timestamp: {lines[0]!r}"
        assert "[ERROR]" in lines[0] and "[CUSTOM]" in lines[0]
        for frag in ("line1", "line2", "line3"):
            assert frag in lines[0], f"Fragment {frag!r} lost: {lines[0]!r}"
        assert "⏎" in lines[0]

    def test_single_line_messages_are_unaffected(self, tmp_path):
        log_file = tmp_path / "filelogger_single.log"
        logger = AsyncFileLogger(str(log_file))
        logger.info("plain", tag="T")
        lines = log_file.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        assert "plain" in lines[0]
        assert "⏎" not in lines[0], (
            f"Continuation marker on single-line message: {lines[0]!r}"
        )

    def test_multiple_calls_produce_equal_line_count(self, tmp_path):
        log_file = tmp_path / "filelogger_multi.log"
        logger = AsyncFileLogger(str(log_file))
        msgs = ["a", "b\nc", "d\ne\nf"]
        for m in msgs:
            logger.warning(m, tag="W")
        lines = log_file.read_text(encoding="utf-8").splitlines()
        assert len(lines) == len(msgs), (
            f"Expected {len(msgs)} lines, got {len(lines)}: {lines!r}"
        )
        assert all(l.startswith("[20") and "[WARNING]" in l for l in lines)
