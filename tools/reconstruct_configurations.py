"""Reproduce the same reconstruction used by the Electron button.

Example: python tools/reconstruct_configurations.py --direction both
         --output artifacts/reconstruction/both.json
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mirrorlab.service import optimize
from mirrorlab.state import save_document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direction", choices=("both", "west_to_east", "east_to_west"), default="both")
    parser.add_argument("--kind", choices=("real",), default="real")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-holdout", action="store_true", help="Diagnostic run only; explicitly disables validation")
    args = parser.parse_args()
    result = optimize(dict(direction=args.direction, kind=args.kind,
                           options={"validate_holdout": not args.skip_holdout}),
                      progress=lambda event: print(json.dumps(event, ensure_ascii=False), file=sys.stderr, flush=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    saved = save_document(result["document"], args.output)
    print(json.dumps({**saved, "status": result["status"],
                      "sides": {d: r.get("counts", {}) for d, r in result["document"]["datasets"].items()}},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
