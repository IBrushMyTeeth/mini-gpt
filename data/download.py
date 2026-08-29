"""
Download the Tiny Shakespeare dataset used for the NLP project.

The dataset is downloaded from Andrej Karpathy's char-rnn repository
and saved locally as `shakespeare.txt`.

The script is safe to run multiple times: if the dataset already
exists locally, it will not be downloaded again.
"""

from pathlib import Path
from urllib.request import urlopen


URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = Path(__file__).parent / "shakespeare.txt"


def main() -> None:
    if DATA_PATH.exists():
        print(f"Dataset already exists at {DATA_PATH}")
        return

    print("Downloading Shakespeare dataset...")

    with urlopen(URL) as response:
        text = response.read().decode("utf-8")

    DATA_PATH.write_text(text, encoding="utf-8")

    print(f"Downloaded {len(text):,} characters.")


if __name__ == "__main__":
    main()