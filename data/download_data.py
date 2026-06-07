"""Download the Jena Climate dataset and save it to data/raw/."""

import ssl
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/gilbutITbook/006975/master/datasets/jena_climate/jena_climate_2009_2016.csv"
DEST = Path(__file__).parent / "raw" / "jena_climate_2009_2016.csv"


def _make_ssl_context() -> ssl.SSLContext:
    """Return an SSL context that works on macOS without system cert issues."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        # Fallback: disable verification (acceptable for a public dataset)
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx


def download(url: str = URL, dest: Path = DEST) -> None:
    """Download *url* and write it to *dest*, creating parent dirs if needed.

    Args:
        url: Remote URL of the dataset CSV.
        dest: Local file path where the CSV will be saved.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"Already exists: {dest}")
        return

    print(f"Downloading {url} …")
    ctx = _make_ssl_context()
    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))
    with opener.open(url) as response, open(dest, "wb") as out_file:
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        chunk = 1024 * 64
        while True:
            data = response.read(chunk)
            if not data:
                break
            out_file.write(data)
            downloaded += len(data)
            if total:
                pct = downloaded / total * 100
                print(f"\r  {pct:.1f}%  ({downloaded:,} / {total:,} bytes)", end="", flush=True)
    print(f"\nSaved to {dest}")


if __name__ == "__main__":
    download()
