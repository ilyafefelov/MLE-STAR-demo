#!/usr/bin/env python
"""
Scan repository for potential hard-coded API keys and tokens.

Search patterns include Google API keys (AIza), OpenAI keys (sk-), and other suspect keys.

Usage:
  python scripts/scan_hardcoded_keys.py --path .
"""

import argparse
import re
import os
from pathlib import Path

PATTERNS = {
    'Google API Key (AIza...)': re.compile(r"AIza[0-9A-Za-z-_]{35}") ,
    'OpenAI Key (sk-)': re.compile(r"sk-[A-Za-z0-9-_]{32,}", re.IGNORECASE),
    'Generic API Key (AKIA...)': re.compile(r"AKIA[0-9A-Z]{16}"),
}

EXCLUDE_DIRS = {'.git', 'venv', '__pycache__', 'node_modules', 'docs_build'}


def scan_directory(base_path: Path):
    results = []
    for root, dirs, files in os.walk(base_path):
        # skip excluded dirs
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for file in files:
            if file.endswith(('.py', '.env', '.sh', '.md', '.yaml', '.yml', '.json', '.txt')):
                file_path = Path(root) / file
                try:
                    text = file_path.read_text(encoding='utf-8')
                except Exception:
                    continue
                for name, pattern in PATTERNS.items():
                    for m in pattern.finditer(text):
                        results.append((file_path.as_posix(), name, m.group(0)))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='.', help='Path to scan')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    base = Path(args.path)
    hits = scan_directory(base)
    if not hits:
        if not args.quiet:
            print('No potential keys found by patterns.')
        return

    print('Potential hard-coded keys found:')
    for fp, name, token in hits:
        print(f"  - File: {fp} | Pattern: {name} | Token: {token}")

    print('\nPlease verify and remove/replace these tokens. Consider moving to .env and using environment variables.')

if __name__ == '__main__':
    main()
