"""
Inspect the downloaded Shakespeare dataset and print basic statistics.
"""

from pathlib import Path


def inspect()-> None:
    """Print basic information about the Shakespeare dataset."""

    data_path = Path(__file__).parent / "shakespeare.txt"
    if not data_path.exists():
        raise FileNotFoundError(
            "Data-path is empty. Please first download the data."
        )

    data = data_path.read_text(encoding="utf-8")

    text_length = len(data)
    unique_chars = sorted(set(data))

    print("=" * 40)
    print("Summary:")
    print()
    print(f"Number of characters: {text_length}")
    print(f"Number of unique characters: {len(unique_chars)}")
    print()
    print("Vocabulary:")
    print(unique_chars)

    print()
    print("First 500 characters:")
    print(data[:500])

if __name__ == "__main__":
    inspect()