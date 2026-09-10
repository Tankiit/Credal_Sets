"""HF image dataset -> frozen DINOv2/timm feature cache.

Run python feature_extraction.py --help for dataset, split, and annotation options.
"""
from experiments.real.extract import main


if __name__ == "__main__":
    main()
