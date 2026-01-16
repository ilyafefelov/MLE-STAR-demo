#!/usr/bin/env python
"""
Find hard-coded API keys and tokens in the repository.

Looks for patterns like API keys starting with 'AIza', 'genai.configure(api_key=', and 'GEMINI_API_TOKEN='
and prints a report with file names and line numbers.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
PATTERNS = [
    (re.compile(r"AIza[0-9A-Za-z\-_]{35}"), "Google API key (AIza...)"),
    (re.compile(r"genai\.configure\(api_key=\s*['\"](.*?)['\"]\)"), "genai.configure(api_key=... )"),
    (re.compile(r"GEMINI_API_TOKEN\s*=\s*['\"](.*?)['\"]"), "GEMINI_API_TOKEN in file"),
    (re.compile(r"GOOGLE_API_KEY\s*=\s*['\"](AIza[0-9A-Za-z\-_]{35})['\"]"), "GOOGLE_API_KEY hard-coded"),
]


def scan_files(root: Path):
    matches = []
    for path in root.rglob('*'):
        if path.is_file():
            try:
                text = path.read_text(encoding='utf-8')
            except Exception:
                continue
            for pat, label in PATTERNS:
                for m in pat.finditer(text):
                    context = text[max(0, m.start()-80):min(len(text), m.end()+80)].replace('\n',' ')[:200]
                    matches.append((path.relative_to(root), label, m.group(0), context))
    return matches


def main():
    root = ROOT
    print(f"Scanning repository: {root}")
    matches = scan_files(root)
    if not matches:
        print("No hard-coded keys found.")
        return 0

    print("Found hard-coded keys / tokens:")
    for path, label, value, ctx in matches:
        print(f" - {path}: {label}: {value}")
        print(f"  Context: {ctx[:160]}")
    return 1


if __name__ == '__main__':
    sys.exit(main())
