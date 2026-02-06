from pathlib import Path
import requests


def download_dataset(url: str, dest: Path, *, overwrite: bool = False) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not overwrite:
        return dest

    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    return dest

path = download_dataset(
    url="https://fever.ai/download/feverous/feverous_dev_challenges.jsonl",
    dest=Path("datasets/feverous/feverous_dev_challenges.jsonl"),
)
print(f"Downloaded to {path}")
