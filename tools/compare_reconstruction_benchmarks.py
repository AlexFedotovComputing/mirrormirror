"""Compare complete numerical outputs, excluding only versioned field fingerprints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def compare(before, after, *, atol=0.):
    failures, numeric_count, max_error, fingerprints = [], 0, 0., 0
    def visit(a, b, path):
        nonlocal numeric_count, max_error, fingerprints
        if path.endswith('.field_hash'):
            # The cache fingerprint now also covers the effective interaction
            # limit. Its intentional change is not a physical result change.
            if not all(isinstance(v, str) and len(v) == 64 for v in (a, b)):
                failures.append((path, 'invalid fingerprint'))
            fingerprints += a != b
        elif isinstance(a, dict) and isinstance(b, dict):
            if a.keys() != b.keys():
                failures.append((path, 'different keys'))
            for key in a.keys() & b.keys():
                visit(a[key], b[key], f'{path}.{key}')
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                failures.append((path, 'different length'))
            for i, (x, y) in enumerate(zip(a, b)):
                visit(x, y, f'{path}[{i}]')
        elif type(a) in (int, float) and type(b) in (int, float):
            numeric_count += 1
            error = abs(a-b)
            max_error = max(max_error, error)
            if error > atol:
                failures.append((path, f'{a!r} != {b!r}'))
        elif type(a) != type(b) or a != b:
            failures.append((path, f'{a!r} != {b!r}'))
    visit(before, after, 'result')
    return dict(passed=not failures, numeric_values=numeric_count, max_absolute_error=max_error,
                updated_field_fingerprints=fingerprints, failure_count=len(failures), failures=failures[:20])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('before', type=Path)
    parser.add_argument('after', type=Path)
    parser.add_argument('--atol', type=float, default=0.)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    reports = {}
    for path in sorted(args.before.glob('*.json')):
        if path.name == 'timing.json':
            continue
        other = args.after / path.name
        reports[path.name] = compare(json.loads(path.read_text(encoding='utf-8')),
            json.loads(other.read_text(encoding='utf-8')), atol=args.atol)
    if not reports:
        raise ValueError('No baseline snapshots found')
    text = json.dumps(reports, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(text + '\n', encoding='utf-8')
    print(text)
    if not all(r['passed'] for r in reports.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
