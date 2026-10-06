"""Download and standardize datasets into data/<name>/.

    python download.py all
    python download.py cebab imdb_cad
"""

import argparse

from concept_datasets import DATASETS, download

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("names", nargs="+", choices=DATASETS + ["all"])
args = parser.parse_args()

for name in DATASETS if "all" in args.names else args.names:
    download(name)
    print()
