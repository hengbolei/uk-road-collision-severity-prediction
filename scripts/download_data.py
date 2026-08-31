'''Download the official DfT five-year collision extract.'''

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from road_severity.download import OFFICIAL_COLLISIONS_URL, download_file


def main() -> None:
    '''Resolve the configured raw path and download the dataset atomically.'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/default.yaml')
    parser.add_argument('--url', default=OFFICIAL_COLLISIONS_URL)
    parser.add_argument('--output', type=Path)
    parser.add_argument(
        '--expected-sha256',
        help='Optional checksum pin; the download fails if it does not match.',
    )
    parser.add_argument(
        '--force', action='store_true',
        help='Replace an existing source file instead of only verifying it.',
    )
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    output = args.output or ROOT / config['data']['raw_path']
    metadata = download_file(
        args.url,
        output,
        force=args.force,
        expected_sha256=args.expected_sha256,
    )
    action = 'Downloaded' if metadata['downloaded'] else 'Verified existing'
    print(
        f"{action} file: {output}\n"
        f"Size: {metadata['size_bytes']:,} bytes\n"
        f"SHA-256: {metadata['sha256']}"
    )


if __name__ == '__main__':
    main()
