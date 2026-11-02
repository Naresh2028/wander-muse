"""Start the Wander web UI.

    python scripts/run_ui.py            # http://127.0.0.1:8000
    python scripts/run_ui.py --port 8010
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import uvicorn  # noqa: E402

from wander.cache import DiskCache  # noqa: E402
from wander.client import MetaClient  # noqa: E402
from wander.config import Settings  # noqa: E402
from wander.ledger import Ledger  # noqa: E402
from wander.web.app import create_app  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--out", default="out")
    args = ap.parse_args()

    s = Settings.from_env()
    ledger = Ledger(s.state_dir / "ledger.jsonl", s.budget_usd)
    client = MetaClient(s, ledger, DiskCache(s.state_dir / "cache"))
    app = create_app(client, ledger, s.state_dir, Path(args.out))
    print(f"Wander is waking up at http://{args.host}:{args.port}  (cap ${s.budget_usd:.2f}, spent ${ledger.total:.3f})")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
