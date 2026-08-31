'''Download and verify source datasets without loading them into memory.'''

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen


OFFICIAL_COLLISIONS_URL = (
    'https://data.dft.gov.uk/road-accidents-safety-data/'
    'dft-road-casualty-statistics-collision-last-5-years.csv'
)


def sha256_file(path: str | Path) -> str:
    '''Return the SHA-256 digest of a file using bounded-memory streaming.'''
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _write_metadata(metadata_path: Path, metadata: dict) -> None:
    '''Write download metadata atomically beside the source file.'''
    temporary = metadata_path.with_suffix(metadata_path.suffix + '.part')
    temporary.write_text(
        json.dumps(metadata, indent=2) + '\n', encoding='utf-8'
    )
    temporary.replace(metadata_path)


def download_file(
    url: str,
    destination: str | Path,
    *,
    force: bool = False,
    expected_sha256: str | None = None,
) -> dict:
    '''Download a URL atomically and return reproducibility metadata.'''
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = destination.with_suffix(destination.suffix + '.download.json')

    if destination.exists() and not force:
        digest = sha256_file(destination)
        if expected_sha256 and digest.lower() != expected_sha256.lower():
            raise ValueError(
                f'Existing file SHA-256 mismatch: expected {expected_sha256}, got {digest}'
            )
        metadata = {
            'source_url': url,
            'destination': str(destination),
            'size_bytes': destination.stat().st_size,
            'sha256': digest,
            'verified_at_utc': datetime.now(timezone.utc).isoformat(),
            'downloaded': False,
        }
        _write_metadata(metadata_path, metadata)
        return metadata

    temporary = destination.with_suffix(destination.suffix + '.part')
    request = Request(url, headers={'User-Agent': 'uk-road-collision-severity/0.1'})
    digest = hashlib.sha256()
    size = 0
    try:
        with urlopen(request, timeout=120) as response, temporary.open('wb') as output:
            for chunk in iter(lambda: response.read(1024 * 1024), b''):
                output.write(chunk)
                digest.update(chunk)
                size += len(chunk)
        actual_sha256 = digest.hexdigest()
        if expected_sha256 and actual_sha256.lower() != expected_sha256.lower():
            raise ValueError(
                f'Downloaded file SHA-256 mismatch: expected {expected_sha256}, '
                f'got {actual_sha256}'
            )
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    metadata = {
        'source_url': url,
        'destination': str(destination),
        'size_bytes': size,
        'sha256': actual_sha256,
        'downloaded_at_utc': datetime.now(timezone.utc).isoformat(),
        'downloaded': True,
    }
    _write_metadata(metadata_path, metadata)
    return metadata
