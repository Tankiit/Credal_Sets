"""Image-dataset extraction entry point for the real-data experiment family."""
import sys
from concept_audit.data.extraction import main as extract


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not any(arg == "--out-dir" or arg.startswith("--out-dir=") for arg in argv):
        argv += ["--out-dir", "features/real/default"]
    return extract(argv)


if __name__ == "__main__":
    main()
