"""Preview the dataset without the original undefined-variable crash."""

import argparse
import json
from data_loader import load_data
from config import Config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input")
    parser.add_argument("--split", default="validation", choices=["train", "validation"])
    args = parser.parse_args()
    sample = load_data(
        1, input_file=args.input, split=args.split, revision=Config().dataset_revision
    )[0]
    print(json.dumps(sample, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
