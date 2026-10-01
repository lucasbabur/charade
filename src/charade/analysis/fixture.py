"""Build the committed test fixture: every impression of a random set of users (`uv run poe fixture`).

Sampling users (not rows) keeps per-user histories intact, so counters and their parity with the
online store are testable on the fixture.
"""

from pathlib import Path

import polars as pl

from charade.data.load import IMPRESSION_COLUMNS, load_characters, load_impressions
from charade.features.derive import user_proxy

FIXTURE_DIR = Path("tests/fixtures")
N_USERS = 2500
SEED = 7


def build(data_dir: Path, out_dir: Path = FIXTURE_DIR) -> None:
    """Write `impressions.csv` and `characters.csv` for the sampled users."""
    impressions = load_impressions(data_dir / "impressions.csv").with_columns(user_proxy())
    users = impressions["user"].unique().sort().sample(N_USERS, seed=SEED)
    # Keep the heaviest users too, otherwise long histories never reach the fixture.
    heavy = impressions.group_by("user").len().sort(["len", "user"], descending=True).head(25)["user"]
    sample = impressions.filter(pl.col("user").is_in(pl.concat([users, heavy]).implode())).sort(["ts", "id"])
    characters = load_characters(data_dir / "characters.csv").filter(
        pl.col("character_id").is_in(sample["character_id"].unique().implode())
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    sample.select(list(IMPRESSION_COLUMNS)).write_csv(out_dir / "impressions.csv")
    characters.with_columns(pl.col("created_at").dt.strftime("%Y-%m-%d")).drop("genre").write_csv(
        out_dir / "characters.csv"
    )


if __name__ == "__main__":
    build(Path())
