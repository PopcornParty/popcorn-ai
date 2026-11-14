import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

SECRET_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|access[_-]?token|client[_-]?secret)\\s*[:=]\\s*['\\"]?[^\\s'\\"]{8,}"),
    re.compile(r"(?i)-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"(?i)password\\s*[:=]\\s*\\S{6,}"),
]
EMAIL_PATTERN = re.compile(r"(?i)\\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\\.[A-Z]{2,}\\b")
REQUIRED_METADATA = ("source", "license", "permission")


def _normalise(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def inspect_dataset(path: Path):
    seen = set()
    source_counts = Counter()
    license_counts = Counter()
    duplicates = empty = total = characters = 0
    missing_metadata, unreviewed, suspicious, oversized = [], [], [], []

    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        total += 1
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
        if not isinstance(item, dict) or not isinstance(item.get("text"), str):
            raise ValueError(f"Line {line_number} must contain a string text field")

        value = item["text"].strip()
        if not value:
            empty += 1
            continue
        characters += len(value)
        if len(value) > 100_000:
            oversized.append(line_number)

        normalised = _normalise(value)
        if normalised in seen:
            duplicates += 1
        seen.add(normalised)

        metadata = item.get("metadata")
        if not isinstance(metadata, dict) or any(
            not isinstance(metadata.get(field), str) or not metadata[field].strip()
            for field in REQUIRED_METADATA
        ):
            missing_metadata.append(line_number)
        else:
            source_counts[metadata["source"].strip()] += 1
            license_counts[metadata["license"].strip()] += 1
        if not isinstance(metadata, dict) or metadata.get("reviewed") is not True:
            unreviewed.append(line_number)

        if any(pattern.search(value) for pattern in SECRET_PATTERNS) or EMAIL_PATTERN.search(value):
            suspicious.append(line_number)

    ready = (
        total >= 3 and len(seen) >= 3 and duplicates == 0 and empty == 0
        and not missing_metadata and not unreviewed and not suspicious and not oversized
    )
    return {
        "file": str(path), "records": total, "unique_nonempty_records": len(seen),
        "duplicate_records": duplicates, "empty_records": empty, "characters": characters,
        "source_counts": dict(source_counts), "license_counts": dict(license_counts),
        "missing_metadata_line_numbers": missing_metadata, "unreviewed_line_numbers": unreviewed,
        "possible_sensitive_content_line_numbers": suspicious, "oversized_record_line_numbers": oversized,
        "ready_for_training": ready,
        "warning": "Automated checks cannot establish copyright ownership, licence validity or absence of all personal data. A human must review every source and its permission before training."
    }


def main():
    parser = argparse.ArgumentParser(description="Validate JSONL training data without printing its text")
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    report = inspect_dataset(args.data)
    print(json.dumps(report, indent=2))
    if not report["ready_for_training"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
