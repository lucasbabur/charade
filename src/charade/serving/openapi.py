"""Write the OpenAPI contract to docs/api/openapi.json (`uv run poe openapi`)."""

import json
from pathlib import Path

from charade.serving.app import create_app

CONTRACT_PATH = Path("docs/api/openapi.json")


def render() -> str:
    """OpenAPI document as stable, diff-friendly JSON."""
    return json.dumps(create_app().openapi(), indent=2, sort_keys=True) + "\n"


def main() -> None:
    """Regenerate the committed contract."""
    CONTRACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONTRACT_PATH.write_text(render())


if __name__ == "__main__":
    main()
