#!/usr/bin/env python
"""
Check the GenAI / Gemini API key by making a light request to the client.

Usage: python scripts/check_api_key.py --env-file .env
"""
import argparse
import os
from pathlib import Path

def load_env(env_file: Path):
    if not env_file.exists():
        return
    with open(env_file, 'r', encoding='utf-8') as f:
        for line in f:
            if '=' in line and not line.strip().startswith('#'):
                k, v = line.strip().split('=', 1)
                os.environ.setdefault(k, v)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-file', type=str, default='.env')
    args = parser.parse_args()

    load_env(Path(args.env_file))
    api_key = os.environ.get('GOOGLE_API_KEY') or os.environ.get('GEMINI_API_TOKEN')
    if not api_key:
        print('No API key found in environment. Please set GOOGLE_API_KEY or GEMINI_API_TOKEN in .env or the environment')
        return 1

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        # Attempt a minimal generation call to verify permissions (small prompt)
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content('Hello world (this is a minimal permission test).')
        print('Generation succeeded; sample response length:', len(str(response)) if response else 0)
        return 0
    except Exception as e:
        print('API key check failed:', e)
        return 2

if __name__ == '__main__':
    raise SystemExit(main())
