from dynamicalgorithmselection.analysis import parse_ert_from_html


def test_parse_ert_from_html_basic() -> None:
    """Basic COCO-style table with finite ERT ratios for two targets."""
    html = """
    <html>
      <body>
        <p>Dimension = 5</p>
        <table>
          <thead>
            <tr>
              <th></th>
              <th>1e-5</th>
              <th>1e-7</th>
            </tr>
          </thead>
          <tbody>
            <!-- First row per function: absolute ERT etc., ignored by parser -->
            <tr>
              <th>f1</th>
              <td>10 (2)</td>
              <td>20 (3)</td>
            </tr>
            <!-- Second row per function: ratio row, parsed by parser -->
            <tr>
              <th></th>
              <td>1.5 (0.2)</td>
              <td>2.0 (0.3)</td>
            </tr>
          </tbody>
        </table>
      </body>
    </html>
    """
    result = parse_ert_from_html(html, targets=("1e-5", "1e-7"))
    expected_keys = {
        "DIM5_f01_target_1e-5",
        "DIM5_f01_target_1e-7",
    }
    assert set(result.keys()) == expected_keys
    assert result["DIM5_f01_target_1e-5"] == 1.5
    assert result["DIM5_f01_target_1e-7"] == 2.0


def test_parse_ert_from_html_infinite_and_missing_values() -> None:
    """Handle ∞ values and rows with fewer cells than headers."""
    html = """
    <html>
      <body>
        <p>Dimension = 10</p>
        <table>
          <thead>
            <tr>
              <th></th>
              <th>1e-5</th>
              <th>1e-7</th>
              <th>1e-9</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <th>f2</th>
              <td>100 (10)</td>
              <td>200 (20)</td>
              <td>300 (30)</td>
            </tr>
            <!-- Ratio row with an infinite entry and one missing cell -->
            <tr>
              <th></th>
              <td>∞ (100)</td>
              <td>3.0 (0.4)</td>
            </tr>
          </tbody>
        </table>
      </body>
    </html>
    """
    result = parse_ert_from_html(html, targets=("1e-5", "1e-7", "1e-9"))
    # 1e-5 should be parsed as infinity
    key_inf = "DIM10_f02_target_1e-5"
    assert key_inf in result
    assert result[key_inf] == float("inf")
    # 1e-7 should be finite
    key_finite = "DIM10_f02_target_1e-7"
    assert key_finite in result
    assert result[key_finite] == 3.0
    # 1e-9 has no corresponding cell in the ratio row and should be absent
    key_missing = "DIM10_f02_target_1e-9"
    assert key_missing not in result


def test_parse_dimension_ignores_tables_without_dimension_paragraph() -> None:
    """Tables without a preceding 'Dimension = ...' paragraph are ignored."""
    html = """
    <html>
      <body>
        <table>
          <thead>
            <tr>
              <th></th>
              <th>1e-5</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <th>f1</th>
              <td>1.0</td>
            </tr>
            <tr>
              <th></th>
              <td>1.2</td>
            </tr>
          </tbody>
        </table>
      </body>
    </html>
    """
    result = parse_ert_from_html(html, targets=("1e-5",))
    # Without a 'Dimension = X' paragraph, the table should be ignored
    assert result == {}
