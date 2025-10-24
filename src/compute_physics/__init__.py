def print_header(index: str, desc: str = ""):
    title = f"Problem {index}" + (f" — {desc}" if desc else "")
    sep = "=" * len(title)
    print(f"\n{sep}\n{title}\n{sep}")

def main() -> int:
    return 0
