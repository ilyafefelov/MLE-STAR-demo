#!/usr/bin/env python
"""
Check logs and verify that each variant/dataset registered an external pipeline (Gemini-generated)

Usage:
 python scripts/check_variants_usage.py --log logs/variants_n10.log
"""
import argparse
from pathlib import Path

def parse_log(logpath: Path):
    found = []
    with open(logpath, 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            if 'Registered external pipeline:' in line:
                # extract the filename
                parts = line.strip().split(':', 1)
                if len(parts) >= 2:
                    pipeline_file = parts[1].strip()
                    found.append((i+1, pipeline_file))
    return found

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str, default='logs/variants_n10.log')
    args = parser.parse_args()
    logpath = Path(args.log)
    if not logpath.exists():
        print('Log file not found:', logpath)
        return
    entries = parse_log(logpath)
    if not entries:
        print('No registered external pipelines found in log:', logpath)
        return
    print('Found registered external pipelines (line, file):')
    for ln, fname in entries:
        print(f'  {ln:6d}  {fname}')

if __name__ == '__main__':
    main()
