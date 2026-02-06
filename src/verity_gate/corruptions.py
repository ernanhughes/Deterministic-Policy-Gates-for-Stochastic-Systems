def truncate(text: str, n: int = 100) -> str:
    return text[:n]


def shuffle_words(text: str) -> str:
    words = text.split()
    return " ".join(sorted(words))
