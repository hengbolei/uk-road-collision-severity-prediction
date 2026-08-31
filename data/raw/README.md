# Raw data

Source CSV files are intentionally excluded from Git because of their size.

Run python scripts/download_data.py from the repository root to download the
official UK Department for Transport collision extract to the path configured in
configs/default.yaml. The downloader records the source URL, file size,
download time and SHA-256 checksum alongside the CSV.
