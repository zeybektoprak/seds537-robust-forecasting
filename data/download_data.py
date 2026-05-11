"""Download the Jena Climate dataset and save it to data/raw/."""

import urllib.request
from pathlib import Path

URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv"
DEST = Path(__file__).parent / "raw" / "jena_climate_2009_2016.csv"


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
    urllib.request.urlretrieve(url, dest)
    print(f"Saved to {dest}")


if __name__ == "__main__":
    download()
