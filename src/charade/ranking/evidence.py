"""Evidence per (campaign, genre): how many training impressions back the model's estimate.

The policy turns this into Beta-posterior width: a campaign never shown on a genre gets a wide
interval and a real chance in the exploration bucket. Stored as `evidence.json` in the bundle.
"""

import polars as pl
from pydantic import BaseModel


class Evidence(BaseModel):
    """`"<C17>|<genre>" -> training impressions`."""

    counts: dict[str, int]

    def lookup(self, campaign_id: str, genre: str) -> float:
        """Impressions for the pair (0 if never seen)."""
        return float(self.counts.get(f"{campaign_id}|{genre}", 0))


def build_evidence(train: pl.DataFrame) -> Evidence:
    """Count training impressions per (C17, genre)."""
    counts = train.group_by("C17", "genre").len()
    return Evidence(counts={f"{r['C17']}|{r['genre']}": int(r["len"]) for r in counts.iter_rows(named=True)})
