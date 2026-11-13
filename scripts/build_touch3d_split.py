"""Rebuild the GeoTAC object-level training and validation split."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Sequence

from dinov2.tacdino.file_records import write


def _category(instance_id: str) -> str:
    if not isinstance(instance_id, str) or not instance_id or "/" not in instance_id:
        raise ValueError(f"invalid TacDINO instance ID: {instance_id!r}")
    return instance_id.split("/", 1)[0]


def _validate_unique(values: Sequence[str], split_name: str) -> None:
    duplicates = sorted(value for value, count in Counter(values).items() if count > 1)
    if duplicates:
        raise ValueError(
            f"{split_name} contains {len(duplicates)} duplicate instance IDs "
            f"(first: {duplicates[:5]})"
        )


def _stable_rank(seed: int, category: str, instance_id: str) -> bytes:
    payload = f"{int(seed)}\0{category}\0{instance_id}".encode("utf-8")
    return hashlib.sha256(payload).digest()


def build_category_stratified_split(
    train_entries: Sequence[str],
    *,
    val_fraction: float = 0.2,
    seed: int = 0,
):
    """Partition training instances while retaining every category in train.

    Categories with at least two instances contribute at least one validation
    instance.  A singleton remains train-only.  Selection uses SHA-256 ranking
    rather than process-dependent RNG state, and output order follows the input
    split so downstream dataset enumeration remains stable.
    """

    if not 0.0 < float(val_fraction) < 1.0:
        raise ValueError("val_fraction must be strictly between 0 and 1")
    train_entries = list(train_entries)
    _validate_unique(train_entries, "source train split")
    by_category = defaultdict(list)
    for instance_id in train_entries:
        by_category[_category(instance_id)].append(instance_id)

    validation_set = set()
    allocation = {}
    for category in sorted(by_category):
        entries = by_category[category]
        if len(entries) == 1:
            val_count = 0
        else:
            val_count = max(1, int(round(len(entries) * float(val_fraction))))
            val_count = min(val_count, len(entries) - 1)
        ranked = sorted(
            entries,
            key=lambda value: (_stable_rank(seed, category, value), value),
        )
        selected = ranked[:val_count]
        validation_set.update(selected)
        allocation[category] = {
            "source": len(entries),
            "train_core": len(entries) - val_count,
            "validation": val_count,
            "singleton_train_only": len(entries) == 1,
        }

    train_core = [value for value in train_entries if value not in validation_set]
    validation = [value for value in train_entries if value in validation_set]
    if set(train_core) & set(validation):
        raise AssertionError("train_core/validation overlap")
    if set(train_core) | set(validation) != set(train_entries):
        raise AssertionError("train_core/validation do not partition the source split")
    if any(value["train_core"] < 1 for value in allocation.values()):
        raise AssertionError("a category was removed from train_core")
    return train_core, validation, allocation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-train-split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    args = parser.parse_args()
    source = json.loads(args.source_train_split.read_text())
    train, validation, allocation = build_category_stratified_split(
        source, seed=args.seed, val_fraction=args.validation_fraction)
    write(args.output_dir / "train_core_split.json", train)
    write(args.output_dir / "val_split.json", validation)
    write(args.output_dir / "split_summary.json", dict(seed=args.seed,
          validation_fraction=args.validation_fraction, train_count=len(train),
          validation_count=len(validation), categories=allocation))
    print(f"Training objects: {len(train)}, validation objects: {len(validation)}")


if __name__ == "__main__":
    main()
