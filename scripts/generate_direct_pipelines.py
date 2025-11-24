#!/usr/bin/env python
"""
Generate pipelines for datasets and variants with direct Gemini SDK calls (non-ADK path).

Usage:
  python scripts/generate_direct_pipelines.py --datasets iris,iris --variants flash,pro --out-dir generated_pipelines
"""
import argparse
import subprocess
import sys
from pathlib import Path

VARIANT_MODEL_MAP = {
    'flash_lite': 'gemini-2.5-flash-lite',
    'flash': 'gemini-2.5-flash',
    'pro': 'gemini-2.5-pro'
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, default='iris', help='Comma-separated dataset names')
    parser.add_argument('--variants', type=str, default='flash_lite,flash,pro')
    parser.add_argument('--out-dir', type=str, default='generated_pipelines')
    parser.add_argument('--env-file', type=str, default='.env')
    parser.add_argument('--python', type=str, default=sys.executable)
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(',')]
    variants = [v.strip() for v in args.variants.split(',')]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for ds in datasets:
        for variant in variants:
            model = VARIANT_MODEL_MAP.get(variant, 'gemini-2.5-flash')
            out_file = out_dir / f'pipeline_{ds}_{variant}.py'
            cmd = [args.python, 'scripts/simple_mle_star.py', '--dataset', ds, '--out-dir', str(out_dir), '--env-file', args.env_file, '--model', model]
            print('Generating:', ' '.join(cmd))
            proc = subprocess.run(cmd)
            if proc.returncode != 0:
                print(f'Generation failed for {ds}/{variant}')
                continue
            # Now cleanup the generated file to standard name
            # The simple script writes pipeline_{dataset}_gemini_direct.py; find and rename
            generated_raw = out_dir / f'pipeline_{ds}_gemini_direct.py'
            generated_clean = out_dir / f'pipeline_{ds}_clean.py'
            generated = None
            # Prefer the cleaned version if available; otherwise fallback to the raw generation
            if generated_clean.exists():
                generated = generated_clean
            elif generated_raw.exists():
                generated = generated_raw
            if generated and generated.exists():
                import shutil
                if out_file.exists():
                    out_file.unlink()
                shutil.move(str(generated), str(out_file))
                print('Wrote', out_file)
            else:
                print('Generation did not produce expected file:', generated)

if __name__ == '__main__':
    main()
