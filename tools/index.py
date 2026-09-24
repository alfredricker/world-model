"""Print a table of every experiment card from its front matter."""
from pathlib import Path

FIELDS = ("id", "rung", "serves", "status", "verdict", "arch_version", "title")


def front_matter(path: Path) -> dict:
    lines = path.read_text().splitlines()
    if not lines or lines[0] != "---":
        return {}
    meta = {}
    for line in lines[1:]:
        if line == "---":
            break
        key, _, value = line.partition(":")
        meta[key.strip()] = value.split("#")[0].strip().strip('"')
    return meta


def main():
    root = Path(__file__).resolve().parent.parent / "experiments"
    rows = [front_matter(p) for p in sorted(root.glob("[0-9]*/card.md"))]
    print(" | ".join(FIELDS))
    for meta in rows:
        print(" | ".join(meta.get(f, "") for f in FIELDS))


if __name__ == "__main__":
    main()
