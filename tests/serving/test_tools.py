from pathlib import Path

from charade.analysis import loadtest, parity
from charade.config import Settings
from tests.conftest import FIXTURES


def test_parity_on_fixture(bundle: Settings) -> None:
    report = parity.run(bundle, FIXTURES)
    assert report["n_rows"] > 0
    assert report["train_serve_max_abs_diff"] == 0.0
    assert report["categorical_mismatches"] == 0


def test_loadtest_summary_reads_locust_stats(tmp_path: Path) -> None:
    csv = tmp_path / "stats.csv"
    csv.write_text(
        "Type,Name,Request Count,Failure Count,Requests/s,50%,95%,99%\n"
        "POST,/v1/rank,1000,1,100,9,17,28\nPOST,/v1/events/impression,100,0,10,1,2,3\n"
    )
    summary = loadtest.summarize(csv)
    assert summary["p99_ms"] == 28.0
    assert summary["error_rate"] == 0.001
