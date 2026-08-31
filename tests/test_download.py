'''Test atomic source-data downloads and checksum verification.'''

from __future__ import annotations

import hashlib
import json

import pytest

from road_severity.download import download_file, sha256_file


def test_download_file_from_local_url(tmp_path):
    payload = b'collision_index,collision_year\nA,2025\n'
    source = tmp_path / 'source.csv'
    source.write_bytes(payload)
    destination = tmp_path / 'raw' / 'collisions.csv'
    expected = hashlib.sha256(payload).hexdigest()

    metadata = download_file(
        source.as_uri(), destination, expected_sha256=expected
    )

    assert destination.read_bytes() == payload
    assert sha256_file(destination) == expected
    assert metadata['downloaded'] is True
    recorded = json.loads(
        destination.with_suffix('.csv.download.json').read_text(encoding='utf-8')
    )
    assert recorded['sha256'] == expected


def test_existing_file_is_verified_and_metadata_is_recorded(tmp_path):
    destination = tmp_path / 'collisions.csv'
    destination.write_bytes(b'already present')
    expected = sha256_file(destination)

    metadata = download_file(
        'https://example.invalid/collisions.csv',
        destination,
        expected_sha256=expected,
    )
    assert metadata['downloaded'] is False
    assert destination.with_suffix('.csv.download.json').exists()


def test_download_rejects_checksum_mismatch(tmp_path):
    source = tmp_path / 'source.csv'
    source.write_bytes(b'unexpected')
    destination = tmp_path / 'raw.csv'

    with pytest.raises(ValueError, match='SHA-256 mismatch'):
        download_file(source.as_uri(), destination, expected_sha256='0' * 64)

    assert not destination.exists()
    assert not destination.with_suffix('.csv.part').exists()
