#!/usr/bin/env python3
"""
Query focal diversity value for a given FR combination from a YAML result file.

Features:
- Order-insensitive matching of FR names separated by '|'.
- Supports input via a single --combo string or multiple --fr arguments.
- Tries to parse with PyYAML if available; otherwise uses a robust fallback
  parser that supports both "key: value" and YAML "? key" / ": value" forms.
- Optional case-insensitive matching.

Examples:
  python tools/query_focal_diversity.py \
    --file results/eval/focal_diversity_soft_new.yaml \
    --combo "ir18_adaface_webface|ir101_adaface_webface|inception_facenet_vgg|ir100_magface_ms1mv2"

  python tools/query_focal_diversity.py \
    -f results/eval/focal_diversity_soft_new.yaml \
    --fr ir18_adaface_webface --fr ir101_adaface_webface --fr inception_facenet_vgg --fr ir100_magface_ms1mv2

    # List-format string (JSON or Python list literal)
    python tools/query_focal_diversity.py \
        -f results/eval/focal_diversity_soft_new.yaml \
        --fr-list '["ir18_adaface_webface", "ir101_adaface_webface", "inception_facenet_vgg", "ir100_magface_ms1mv2"]'
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
import json
import ast
from pathlib import Path
from typing import Dict, Tuple


def _try_parse_with_pyyaml(text: str) -> Dict[str, float] | None:
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
        if isinstance(data, dict):
            # Coerce keys to str and values to float where possible
            out: Dict[str, float] = {}
            for k, v in data.items():
                if k is None:
                    continue
                key = str(k)
                try:
                    out[key] = float(v)
                except Exception:
                    # Skip unparsable values
                    continue
            return out
        return None
    except Exception:
        return None


def _fallback_parse(text: str) -> Dict[str, float]:
    """Parse simple YAML-like mapping supporting two forms:
    - "key: value" on one line
    - "? key" newline ": value" (YAML complex key form)
    Returns dict[str, float]. Lines starting with '#' are ignored.
    """
    result: Dict[str, float] = {}
    pending_key: str | None = None

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue

        # Handle two-line key form
        if line.startswith('?'):
            pending_key = line[1:].strip()
            continue

        if pending_key is not None:
            # Expect ": value" (optionally with spaces before colon)
            # Normalize to start with ':' when stripped
            if line.startswith(':'):
                val_str = line[1:].strip()
            else:
                # Allow variants like " : 0.123"
                colon_idx = line.find(':')
                if colon_idx == -1:
                    # Malformed; reset pending
                    pending_key = None
                    continue
                val_str = line[colon_idx + 1 :].strip()
            try:
                result[pending_key] = float(val_str)
            except Exception:
                # Skip unparsable value
                pass
            finally:
                pending_key = None
            continue

        # Handle single-line form: key: value
        if ':' in line:
            # Split at the last ':' to be safer if key accidentally contains ':'
            key_part, val_part = line.rsplit(':', 1)
            key = key_part.strip()
            val_str = val_part.strip()
            if key:
                try:
                    result[key] = float(val_str)
                except Exception:
                    # Skip unparsable value
                    pass

    return result


def load_mapping(path: Path) -> Dict[str, float]:
    text = path.read_text(encoding='utf-8')
    parsed = _try_parse_with_pyyaml(text)
    if parsed is not None:
        return parsed
    return _fallback_parse(text)


def normalize_token(token: str, ignore_case: bool) -> str:
    t = token.strip()
    return t.lower() if ignore_case else t


def combo_to_counter(combo: str, ignore_case: bool) -> Counter[str]:
    parts = [normalize_token(p, ignore_case) for p in combo.split('|') if p.strip()]
    return Counter(parts)


def find_value(mapping: Dict[str, float], frs: Tuple[str, ...], ignore_case: bool) -> Tuple[str, float] | None:
    """Find value by order-insensitive match using a multiset (Counter) comparison.
    Returns (matched_key, value) or None if not found.
    """
    target = Counter(normalize_token(x, ignore_case) for x in frs)
    for key, val in mapping.items():
        key_counter = combo_to_counter(key, ignore_case)
        if key_counter == target:
            return key, val
    return None


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Query focal diversity value by FR combination (order-insensitive)")
    p.add_argument('-f', '--file', required=True, help='Path to YAML results file')
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('-c', '--combo', help='Combination string like "a|b|c"')
    g.add_argument('--fr', action='append', help='Specify FR name; repeat to build the combination')
    g.add_argument('--fr-list', help='FR list in JSON/Python list format, e.g., ["a","b","c"]')
    p.add_argument('-i', '--ignore-case', action='store_true', help='Case-insensitive matching')
    p.add_argument('--list', action='store_true', help='List all available keys and exit')
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    path = Path(args.file)
    if not path.is_file():
        print(f"Error: file not found: {path}", file=sys.stderr)
        return 2

    mapping = load_mapping(path)
    if args.list:
        for k in sorted(mapping.keys()):
            print(k)
        return 0

    def _parse_fr_list_str(s: str) -> Tuple[str, ...]:
        # Try JSON first
        try:
            obj = json.loads(s)
            if isinstance(obj, (list, tuple)):
                return tuple(str(x).strip() for x in obj if str(x).strip())
        except Exception:
            pass
        # Fallback to Python literal
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, (list, tuple)):
                return tuple(str(x).strip() for x in obj if str(x).strip())
        except Exception:
            pass
        # Last resort: split by comma
        return tuple([t.strip() for t in s.split(',') if t.strip()])

    if args.combo:
        frs = tuple(p.strip() for p in args.combo.split('|') if p.strip())
    elif args.fr_list:
        frs = _parse_fr_list_str(args.fr_list)
    else:
        assert args.fr, 'Argument parsing bug: --fr should be provided when --combo is absent'
        frs = tuple(args.fr)

    if not frs:
        print('Error: empty FR combination', file=sys.stderr)
        return 2

    res = find_value(mapping, frs, ignore_case=args.ignore_case)
    if res is None and not args.ignore_case:
        # Best-effort retry with case-insensitive match
        res = find_value(mapping, frs, ignore_case=True)

    if res is None:
        combo_display = '|'.join(frs)
        print(f"Not found: {combo_display}", file=sys.stderr)
        # Suggest close candidates sharing same set size
        target_len = len(frs)
        suggestions = [k for k in mapping.keys() if len(k.split('|')) == target_len]
        if suggestions:
            print("Did you mean one of:", file=sys.stderr)
            for s in suggestions[:10]:
                print(f"  {s}", file=sys.stderr)
        return 1

    matched_key, value = res
    # Print just the numeric value to make it easy to pipe
    print(value)
    # If needed for debugging, uncomment:
    # print(f"Matched key: {matched_key}", file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
