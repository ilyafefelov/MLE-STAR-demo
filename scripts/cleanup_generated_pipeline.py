#!/usr/bin/env python
"""
Extracts the python code block from a Gemini-generated file that contains explanatory text + code blocks.
Saves a cleaned Python script file containing only the code from the first Python block.

Usage:
  python scripts/cleanup_generated_pipeline.py --src generated_pipelines/pipeline_iris_gemini_direct.py --out generated_pipelines/pipeline_iris_clean.py
"""
import argparse
from pathlib import Path
import re
import sys
import ast

def extract_python_code(text: str):
    # Find all code blocks marked as ```python or ```
    candidates = []
    # Handle typical code block fence variants and multiple code blocks
    for m in re.finditer(r"```(?:python)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE):
        candidates.append(m.group(1))

    # If we found candidates, pick the first that parses as valid Python, otherwise pick the longest
    for c in candidates:
        try:
            ast.parse(c)
            return c
        except Exception:
            continue
    if candidates:
        # fallback to longest candidate if none parsed
        candidates.sort(key=lambda x: len(x), reverse=True)
        return candidates[0]

    # Fallback: attempt to extract starting from the first import statement onward
    m_import = re.search(r"(import\s+[\w\.,\s]+|from\s+[\w\.]+\s+import\s+)[\s\S]*", text)
    if m_import:
        snippet = m_import.group(0)
        try:
            ast.parse(snippet)
            return snippet
        except Exception:
            # If parsing fails, try to extract from first 'def' or from 'def build_full_pipeline'
            m_def = re.search(r"(def\s+build_full_pipeline\([^\)]*\):[\s\S]*)", text)
            if m_def:
                snippet2 = m_def.group(0)
                try:
                    ast.parse(snippet2)
                    return snippet2
                except Exception:
                    return None
    # As last fallback, try to extract from def main
    m3 = re.search(r"(def main\(\):[\s\S]*)", text)
    if m3:
        return m3.group(1)
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    if not src.exists():
        print(f"Source file not found: {src}")
        sys.exit(1)
    txt = src.read_text(encoding='utf-8')
    code = extract_python_code(txt)
    if not code:
        print('No python code block found in source file; copying original content to output')
        code = txt
    out.write_text(code, encoding='utf-8')
    print(f'Written cleaned pipeline to {out}')

if __name__ == '__main__':
    main()
