"""Run the same manual import/refinement operations as the desktop worker."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mirrorlab.service import dispatch
from mirrorlab.state import save_document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--import-only", action="store_true")
    parser.add_argument("--direction", choices=["both", "east_to_west", "west_to_east"], default="both")
    parser.add_argument("--options", type=Path, help="JSON refinement options")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = dispatch("import_manual", dict(path=str(args.input), direction=args.direction))
    if not args.import_only:
        options = json.loads(args.options.read_text(encoding="utf-8")) if args.options else None
        result = dispatch("refine_manual", dict(candidate_configuration=result["document"],
            direction=args.direction, options=options), progress=lambda event: print(
                json.dumps(event, ensure_ascii=False), file=sys.stderr))
    save_document(result["document"], args.output)
    summary = dict(status=result["status"], datasets={direction: dict(status=d["status"],
        mirror_count=len(d["mirrors"]), diagnostics=d["diagnostics"], mechanics=d.get("mechanics"))
        for direction, d in result["document"]["datasets"].items()})
    text = json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)
    if args.report:
        args.report.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
