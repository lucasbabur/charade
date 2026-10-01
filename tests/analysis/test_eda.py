from pathlib import Path

from charade.analysis import eda
from tests.conftest import FIXTURES


def test_eda_report_covers_every_section(tmp_path: Path) -> None:
    eda.run(FIXTURES, tmp_path)
    report = (tmp_path / "eda.md").read_text()
    for title in ("Daily CTR", "Genre", "LEAK", "Character x campaign interaction"):
        assert title in report
