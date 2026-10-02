import io
from pathlib import Path

import pytest

from charade.analysis import loadtest, parity
from charade.config import Settings
from charade.scoring.scorer import bundle_sha256
from tests.conftest import FIXTURES


def test_parity_on_fixture(bundle: Settings) -> None:
    report = parity.run(bundle, FIXTURES)
    assert report["n_rows"] != 0
    assert report["train_serve_max_abs_diff"] == 0.0
    assert report["categorical_mismatches"] == 0
    assert report["bundle_sha256"] == bundle_sha256(bundle.artifacts_dir)


def test_loadtest_summary_reads_locust_stats_and_names_the_served_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv = tmp_path / "stats.csv"
    csv.write_text(
        "Type,Name,Request Count,Failure Count,Requests/s,50%,95%,99%\n"
        "POST,/v1/rank,1000,1,100,9,17,28\nPOST,/v1/events/impression,100,10,10,1,2,3\n"
        ",Aggregated,1100,11,110,8,16,27\n"
    )

    def model_info(url: str, timeout: float) -> io.BytesIO:
        assert url == "http://api/v1/model"
        del timeout
        return io.BytesIO(b'{"bundle_sha256": "abc"}')

    monkeypatch.setattr(loadtest.urllib.request, "urlopen", model_info)
    summary = loadtest.summarize(csv, "http://api")
    assert summary["p99_ms"] == 28.0
    assert summary["error_rate"] == 11 / 1100  # events count too
    assert summary["bundle_sha256"] == "abc"
