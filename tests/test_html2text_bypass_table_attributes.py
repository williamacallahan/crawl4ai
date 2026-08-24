from crawl4ai.html2text import HTML2Text


def test_bypass_tables_escape_every_emitted_attribute_value():
    html = """
    <table data-table="A &quot; B &amp; C &lt; D &gt; E &#x27; F">
      <tr data-row="A &quot; B &amp; C &lt; D &gt; E &#x27; F">
        <th colspan="2" title="A &quot; B &amp; C &lt; D &gt; E &#x27; F">Head</th>
        <td disabled data-cell="A &quot; B &amp; C &lt; D &gt; E &#x27; F">Cell</td>
      </tr>
    </table>
    """
    converter = HTML2Text()
    converter.body_width = 0
    converter.bypass_tables = True

    result = converter.handle(html)
    escaped = "A &quot; B &amp; C &lt; D &gt; E &#x27; F"

    assert f'<table data-table="{escaped}">' in result
    assert f'<tr data-row="{escaped}">' in result
    assert f'<th colspan="2" title="{escaped}">' in result
    assert f'<td disabled data-cell="{escaped}">' in result
    assert 'data-cell="A " B' not in result


def test_bypass_tables_drop_event_handler_attributes():
    converter = HTML2Text()
    converter.body_width = 0
    converter.bypass_tables = True

    result = converter.handle(
        '<table onload="steal()"><tr><td onclick="steal()" data-id="1">Cell</td></tr></table>'
    )

    assert "onload" not in result
    assert "onclick" not in result
    assert 'data-id="1"' in result
