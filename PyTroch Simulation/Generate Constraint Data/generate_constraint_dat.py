#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate a Verilog $readmemh constraint.dat file from:

1. storage_audit.json
2. ber_constraint_regions.csv

The output order is fixed:

    line 0: Base_Max_Faults
    line 1: Inter_Max_Faults
    line 2: Intra_Max_Faults

Default selection:
    target_retention = 0.98
    selection_policy = selected_pooled_wilson_monotonic
    storage_bits     = valid_storage_bits
    rounding         = floor
    output width     = 24 bits
"""

from __future__ import annotations

import argparse
import csv
import json
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP
from pathlib import Path
from typing import Dict, List


MODE_ORDER = ("base", "inter", "intra")
ROUNDING_MODES = {
    "floor": ROUND_FLOOR,
    "ceil": ROUND_CEILING,
    "nearest": ROUND_HALF_UP,
}


def decimal_arg(text: str) -> Decimal:
    try:
        return Decimal(text)
    except Exception as exc:
        raise argparse.ArgumentTypeError(f"Invalid decimal value: {text}") from exc


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"storage audit does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def choose_storage_bits(audit: dict, source: str) -> int:
    if source == "valid":
        key = "valid_storage_bits"
        if key not in audit:
            raise KeyError(f"storage_audit.json is missing {key!r}")
        return int(audit[key])

    if source == "fbb_allocated":
        key = "allocated_fbb_bytes_including_padding"
        if key not in audit:
            raise KeyError(f"storage_audit.json is missing {key!r}")
        return int(audit[key]) * 8

    if source == "page_allocated":
        key = "occupied_16KiB_pages_including_padding"
        if key not in audit:
            raise KeyError(f"storage_audit.json is missing {key!r}")
        return int(audit[key]) * 16 * 1024 * 8

    raise ValueError(f"Unsupported storage-bit source: {source}")


def load_selected_regions(
    path: Path,
    target_retention: Decimal,
    selection_policy: str,
) -> Dict[str, Decimal]:
    if not path.exists():
        raise FileNotFoundError(f"constraint regions CSV does not exist: {path}")

    selected: Dict[str, List[Decimal]] = {mode: [] for mode in MODE_ORDER}

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "target_retention",
            "selection_policy",
            "mode",
            "ber_start_inclusive",
            "ber_end_inclusive",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"CSV is missing columns {sorted(missing)}; "
                f"available={reader.fieldnames}"
            )

        for row in reader:
            try:
                row_retention = Decimal(str(row["target_retention"]).strip())
            except Exception:
                continue

            if row_retention != target_retention:
                continue
            if str(row["selection_policy"]).strip() != selection_policy:
                continue

            mode = str(row["mode"]).strip().lower()
            if mode in selected:
                selected[mode].append(
                    Decimal(str(row["ber_end_inclusive"]).strip())
                )

    missing_modes = [mode for mode in MODE_ORDER if not selected[mode]]
    if missing_modes:
        raise ValueError(
            "The selected target/policy does not provide usable regions for "
            f"{missing_modes}. target_retention={target_retention}, "
            f"selection_policy={selection_policy!r}. "
            "Choose another policy or inspect ber_constraint_regions.csv."
        )

    # A mode may theoretically appear in multiple disjoint rows. The threshold
    # is the greatest BER endpoint assigned to that mode.
    return {mode: max(values) for mode, values in selected.items()}


def convert_threshold(
    storage_bits: int,
    max_ber: Decimal,
    rounding: str,
) -> int:
    value = Decimal(storage_bits) * max_ber
    return int(value.to_integral_value(rounding=ROUNDING_MODES[rounding]))


def validate_thresholds(thresholds: Dict[str, int], width: int) -> None:
    if width <= 0:
        raise ValueError("--width must be positive")

    if not (
        thresholds["base"]
        <= thresholds["inter"]
        <= thresholds["intra"]
    ):
        raise ValueError(
            "Generated thresholds are not monotonic: "
            f"{thresholds}"
        )

    max_value = (1 << width) - 1
    for mode, value in thresholds.items():
        if value < 0:
            raise ValueError(f"{mode} threshold is negative: {value}")
        if value > max_value:
            raise OverflowError(
                f"{mode} threshold {value} exceeds {width}-bit maximum "
                f"{max_value}. Increase --width and match TOTAL_COUNT_WIDTH."
            )


def write_dat(path: Path, thresholds: Dict[str, int], width: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    hex_digits = (width + 3) // 4
    with path.open("w", encoding="ascii", newline="\n") as handle:
        for mode in MODE_ORDER:
            handle.write(f"{thresholds[mode]:0{hex_digits}X}\n")


def write_metadata(
    path: Path,
    *,
    audit_path: Path,
    regions_path: Path,
    target_retention: Decimal,
    selection_policy: str,
    storage_source: str,
    storage_bits: int,
    ber_limits: Dict[str, Decimal],
    thresholds: Dict[str, int],
    width: int,
    rounding: str,
    output_dat: Path,
) -> None:
    metadata = {
        "storage_audit": str(audit_path),
        "constraint_regions": str(regions_path),
        "target_retention": str(target_retention),
        "selection_policy": selection_policy,
        "storage_bit_source": storage_source,
        "storage_bits": storage_bits,
        "rounding": rounding,
        "total_count_width": width,
        "output_dat": str(output_dat),
        "line_order": [
            "Base_Max_Faults",
            "Inter_Max_Faults",
            "Intra_Max_Faults",
        ],
        "max_ber": {
            mode: str(ber_limits[mode]) for mode in MODE_ORDER
        },
        "threshold_decimal": {
            mode: thresholds[mode] for mode in MODE_ORDER
        },
        "threshold_hex": {
            mode: f"0x{thresholds[mode]:0{(width + 3)//4}X}"
            for mode in MODE_ORDER
        },
        "mode_intervals": {
            "base": f"0 <= count <= {thresholds['base']}",
            "inter": (
                f"{thresholds['base'] + 1} <= count "
                f"<= {thresholds['inter']}"
            ),
            "intra": (
                f"{thresholds['inter'] + 1} <= count "
                f"<= {thresholds['intra']}"
            ),
            "unrepairable": f"count >= {thresholds['intra'] + 1}",
        },
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--storage-audit",
        type=Path,
        default=Path("storage_audit.json"),
    )
    parser.add_argument(
        "--regions-csv",
        type=Path,
        default=Path("ber_constraint_regions.csv"),
    )
    parser.add_argument(
        "--target-retention",
        type=decimal_arg,
        default=Decimal("0.98"),
    )
    parser.add_argument(
        "--selection-policy",
        default="selected_pooled_wilson_monotonic",
    )
    parser.add_argument(
        "--storage-bit-source",
        choices=("valid", "fbb_allocated", "page_allocated"),
        default="valid",
        help=(
            "Use 'valid' to match the current Python experiment's "
            "packed.total_valid_bits BER population."
        ),
    )
    parser.add_argument(
        "--rounding",
        choices=tuple(ROUNDING_MODES),
        default="floor",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=24,
        help="TOTAL_COUNT_WIDTH used by RTL/testbench",
    )
    parser.add_argument(
        "--output-dat",
        type=Path,
        default=Path("vgg16_cifar10_r980_constraint.dat"),
    )
    parser.add_argument(
        "--output-metadata",
        type=Path,
        default=Path("vgg16_cifar10_r980_constraint_metadata.json"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    audit = load_json(args.storage_audit)
    storage_bits = choose_storage_bits(
        audit, args.storage_bit_source
    )
    ber_limits = load_selected_regions(
        args.regions_csv,
        args.target_retention,
        args.selection_policy,
    )
    thresholds = {
        mode: convert_threshold(
            storage_bits,
            ber_limits[mode],
            args.rounding,
        )
        for mode in MODE_ORDER
    }

    validate_thresholds(thresholds, args.width)
    write_dat(args.output_dat, thresholds, args.width)
    write_metadata(
        args.output_metadata,
        audit_path=args.storage_audit,
        regions_path=args.regions_csv,
        target_retention=args.target_retention,
        selection_policy=args.selection_policy,
        storage_source=args.storage_bit_source,
        storage_bits=storage_bits,
        ber_limits=ber_limits,
        thresholds=thresholds,
        width=args.width,
        rounding=args.rounding,
        output_dat=args.output_dat,
    )

    print("=" * 72)
    print("Constraint generation completed")
    print("=" * 72)
    print(f"Storage bits     : {storage_bits:,}")
    print(f"Target retention : {args.target_retention}")
    print(f"Policy           : {args.selection_policy}")
    print(f"Rounding         : {args.rounding}")
    print("-" * 72)
    for index, mode in enumerate(MODE_ORDER):
        print(
            f"line {index} {mode:5s}: "
            f"BER <= {ber_limits[mode]}  "
            f"threshold={thresholds[mode]:,}  "
            f"hex=0x{thresholds[mode]:0{(args.width+3)//4}X}"
        )
    print("-" * 72)
    print(f"DAT      : {args.output_dat.resolve()}")
    print(f"Metadata : {args.output_metadata.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
