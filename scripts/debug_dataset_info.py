#!/usr/bin/env python
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.mle_star_ablation.datasets import DatasetLoader

def main():
    info = DatasetLoader.get_dataset_info('iris')
    print(info)

if __name__ == '__main__':
    main()
