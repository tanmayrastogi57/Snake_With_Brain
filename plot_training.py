"""Generate an analysis graph from a JSONL training log."""

import argparse
import json
from pathlib import Path

from helper import plot_training_history


def load_records(path):
    records = []
    with Path(path).open(encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f'Invalid JSON on line {line_number} of {path}') from error
    return records


def main():
    parser = argparse.ArgumentParser(description='Generate a training analysis PNG from a JSONL log.')
    parser.add_argument('--log-file', default='runs/shared/shared-training.jsonl')
    parser.add_argument('--output', default='runs/shared/shared-training.png')
    args = parser.parse_args()

    records = load_records(args.log_file)
    if not plot_training_history(records, args.output):
        raise SystemExit(f'No episode records found in {args.log_file}')
    print(f'Wrote training graph to {args.output}')


if __name__ == '__main__':
    main()
