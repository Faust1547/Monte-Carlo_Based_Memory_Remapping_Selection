#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RTL-aligned Monte Carlo experiment with separate EXP and S&M ES profiles.

The same candidate Inter shift or Intra CW is evaluated using:

    total_ES = exp_scale * EXP_ES + sm_scale * S&M_ES

The candidate control is selected from total_ES, while EXP, S&M, and total
components are all retained in the CSV outputs. Masked modes can independently
mask selected EXP or S&M significance indices; the default configuration masks
only EXP.

Physical fault coordinates remain sparse:
    page / bitmap block / bank / RG / word / cell bit

Logical-to-physical remapping remains RTL aligned:
    inter: circular logical-bank shift
    intra: XOR logical-word remapping

The mapping can use one shared significance-index layout for EXP and S&M, or
separate layouts. Field assignment supports:
    paired      - every mapped slot contributes both EXP and S&M ES
    by_cell_bit - each cell-bit lane is assigned EXP, S&M, BOTH, or NONE
    explicit_3d - field assignment is specified for every logical coordinate

Use "paired" only when one fault-map entry represents a paired EXP/S&M logical
slot. If EXP and S&M occupy distinct cells, use by_cell_bit or explicit_3d.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


FAULT_NAME_TO_ID = {"SA0": 0, "SA1": 1, "TRANSIENT": 2}
DEFAULT_MODE_ORDER = ["masked_intra", "intra", "masked_inter", "inter"]


@dataclass(frozen=True)
class Geometry:
    num_pages: int
    page_bytes: int
    banks: int
    rgs: int
    words: int
    bits: int

    @property
    def page_bits(self) -> int:
        return self.page_bytes * 8

    @property
    def cells_per_bitmap_block(self) -> int:
        return self.banks * self.rgs * self.words * self.bits

    @property
    def bitmap_blocks_per_page(self) -> int:
        if self.page_bits % self.cells_per_bitmap_block != 0:
            raise ValueError(
                f"page_bits={self.page_bits} is not divisible by the RTL "
                f"fault-bitmap size={self.cells_per_bitmap_block}."
            )
        return self.page_bits // self.cells_per_bitmap_block

    @property
    def total_bits(self) -> int:
        return self.num_pages * self.page_bits

    def validate(self) -> None:
        for name in ("num_pages", "page_bytes", "banks", "rgs", "words", "bits"):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")
        if self.words & (self.words - 1):
            raise ValueError("XOR CW remapping requires words to be a power of two.")
        if self.banks & (self.banks - 1):
            warnings.warn(
                "banks is not a power of two. Circular inter shift still works, "
                "but a 3-bit RTL implementation normally assumes 8 banks."
            )
        _ = self.bitmap_blocks_per_page


@dataclass(frozen=True)
class RTLMapping:
    exp_index_map: np.ndarray  # [logical_bank, logical_word, logical_cell_bit]
    sm_index_map: np.ndarray   # [logical_bank, logical_word, logical_cell_bit]
    exp_active_map: np.ndarray # bool, same shape
    sm_active_map: np.ndarray  # bool, same shape
    inter_scope: str
    intra_scope: str
    inter_shift_direction: str
    field_assignment_kind: str

    def validate(
        self,
        geometry: Geometry,
        exp_count: int,
        sm_count: int,
    ) -> None:
        expected = (geometry.banks, geometry.words, geometry.bits)
        arrays = {
            "exp_index_map": self.exp_index_map,
            "sm_index_map": self.sm_index_map,
            "exp_active_map": self.exp_active_map,
            "sm_active_map": self.sm_active_map,
        }
        for name, array in arrays.items():
            if array.shape != expected:
                raise ValueError(f"{name}.shape={array.shape}, expected {expected}.")

        if np.any(self.exp_index_map < 0) or np.any(self.exp_index_map >= exp_count):
            raise ValueError(
                f"EXP logical index map must contain values in [0, {exp_count - 1}]."
            )
        if np.any(self.sm_index_map < 0) or np.any(self.sm_index_map >= sm_count):
            raise ValueError(
                f"S&M logical index map must contain values in [0, {sm_count - 1}]."
            )
        if not np.any(self.exp_active_map | self.sm_active_map):
            raise ValueError("The field assignment disables both EXP and S&M everywhere.")
        if self.inter_scope not in {"flash_page", "bitmap_block"}:
            raise ValueError("inter_scope must be 'flash_page' or 'bitmap_block'.")
        if self.intra_scope not in {
            "bitmap_block_bank_rg",
            "flash_page_bank_rg",
        }:
            raise ValueError(
                "intra_scope must be 'bitmap_block_bank_rg' or "
                "'flash_page_bank_rg'."
            )
        if self.inter_shift_direction not in {"right", "left"}:
            raise ValueError("inter_shift_direction must be 'right' or 'left'.")


@dataclass(frozen=True)
class ESProfile:
    name: str
    exp_labels: Tuple[str, ...]
    sm_labels: Tuple[str, ...]
    exp_es: np.ndarray
    sm_es: np.ndarray
    exp_p_one: np.ndarray
    sm_p_one: np.ndarray
    masked_exp: np.ndarray
    masked_sm: np.ndarray
    exp_scale: float
    sm_scale: float
    exp_mask_factor: float
    sm_mask_factor: float

    @property
    def exp_count(self) -> int:
        return int(self.exp_es.size)

    @property
    def sm_count(self) -> int:
        return int(self.sm_es.size)

    def validate(self) -> None:
        for field_name, labels, es, p_one, masked in (
            ("EXP", self.exp_labels, self.exp_es, self.exp_p_one, self.masked_exp),
            ("S&M", self.sm_labels, self.sm_es, self.sm_p_one, self.masked_sm),
        ):
            n = int(es.size)
            if n <= 0:
                raise ValueError(f"Profile {self.name!r} has no {field_name} ES classes.")
            if len(labels) != n:
                raise ValueError(
                    f"Profile {self.name!r}: {field_name} labels length "
                    f"{len(labels)} != {n}."
                )
            if p_one.shape != (n,):
                raise ValueError(
                    f"Profile {self.name!r}: {field_name} p_one must have length {n}."
                )
            if masked.shape != (n,):
                raise ValueError(
                    f"Profile {self.name!r}: {field_name} mask must have length {n}."
                )
            if np.any(es < 0):
                raise ValueError(f"{field_name} ES values must be nonnegative.")
            if np.any((p_one < 0) | (p_one > 1)):
                raise ValueError(f"{field_name} p_one values must be within [0, 1].")

        if self.exp_scale < 0 or self.sm_scale < 0:
            raise ValueError("exp_scale and sm_scale must be nonnegative.")
        if not 0.0 <= self.exp_mask_factor <= 1.0:
            raise ValueError("exp_mask_factor must be within [0, 1].")
        if not 0.0 <= self.sm_mask_factor <= 1.0:
            raise ValueError("sm_mask_factor must be within [0, 1].")

    @staticmethod
    def _fault_factor(p_one: np.ndarray, fault_type: np.ndarray) -> np.ndarray:
        factor = np.ones_like(p_one, dtype=np.float64)
        sa0 = fault_type == FAULT_NAME_TO_ID["SA0"]
        sa1 = fault_type == FAULT_NAME_TO_ID["SA1"]
        transient = fault_type == FAULT_NAME_TO_ID["TRANSIENT"]
        factor[sa0] = p_one[sa0]
        factor[sa1] = 1.0 - p_one[sa1]
        factor[transient] = 1.0
        if np.any(~(sa0 | sa1 | transient)):
            raise ValueError("Unknown fault type ID encountered.")
        return factor

    def cost_components(
        self,
        exp_index: np.ndarray,
        sm_index: np.ndarray,
        exp_active: np.ndarray,
        sm_active: np.ndarray,
        fault_type: np.ndarray,
        masked: bool,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        exp_index = np.asarray(exp_index, dtype=np.int64)
        sm_index = np.asarray(sm_index, dtype=np.int64)
        exp_active = np.asarray(exp_active, dtype=bool)
        sm_active = np.asarray(sm_active, dtype=bool)

        exp_base = self.exp_scale * self.exp_es[exp_index].astype(np.float64, copy=True)
        sm_base = self.sm_scale * self.sm_es[sm_index].astype(np.float64, copy=True)

        if masked:
            exp_selected = self.masked_exp[exp_index]
            sm_selected = self.masked_sm[sm_index]
            exp_base[exp_selected] *= self.exp_mask_factor
            sm_base[sm_selected] *= self.sm_mask_factor

        exp_factor = self._fault_factor(self.exp_p_one[exp_index], fault_type)
        sm_factor = self._fault_factor(self.sm_p_one[sm_index], fault_type)

        exp_cost = exp_base * exp_factor * exp_active.astype(np.float64)
        sm_cost = sm_base * sm_factor * sm_active.astype(np.float64)
        return exp_cost, sm_cost, exp_cost + sm_cost

@dataclass(frozen=True)
class FaultSet:
    page: np.ndarray
    cell: np.ndarray
    fault_type: np.ndarray
    poisson_count: int
    unique_count: int

    def __len__(self) -> int:
        return int(self.unique_count)


@dataclass(frozen=True)
class DecodedFaults:
    page: np.ndarray
    page_compact: np.ndarray
    unique_pages: np.ndarray
    block: np.ndarray
    bank: np.ndarray
    rg: np.ndarray
    word: np.ndarray
    bit: np.ndarray
    fault_type: np.ndarray

    @property
    def faulty_page_count(self) -> int:
        return int(self.unique_pages.size)

    def __len__(self) -> int:
        return int(self.fault_type.size)


@dataclass(frozen=True)
class ModeEvaluation:
    page_es: np.ndarray
    page_exp_es: np.ndarray
    page_sm_es: np.ndarray
    group_page_compact: np.ndarray
    best_control: np.ndarray

def _float_vector(value: Sequence[float], label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{label} must be a one-dimensional array.")
    return array


def _labels(profile_cfg: Mapping[str, Any], key: str, prefix: str, n: int) -> Tuple[str, ...]:
    values = profile_cfg.get(key)
    if values is None:
        return tuple(f"{prefix}_{i}" for i in range(n))
    return tuple(str(value) for value in values)


def _mask_vector(profile_cfg: Mapping[str, Any], key: str, n: int, label: str) -> np.ndarray:
    result = np.zeros(n, dtype=bool)
    for raw_index in profile_cfg.get(key, []):
        index = int(raw_index)
        if not 0 <= index < n:
            raise ValueError(f"{label} mask index {index} is outside [0, {n - 1}].")
        result[index] = True
    return result


def build_profile(profile_cfg: Mapping[str, Any]) -> ESProfile:
    name = str(profile_cfg["name"])
    forbidden = {"bank_weights", "word_weights", "bit_weights", "es_map"}
    used_forbidden = sorted(forbidden.intersection(profile_cfg.keys()))
    if used_forbidden:
        raise ValueError(
            f"Profile {name!r} still uses legacy fields {used_forbidden}. "
            "Use exp_es and sm_es instead."
        )

    # Backward compatibility with the previous single-table RTL version.
    if "exp_es" not in profile_cfg and "sm_es" not in profile_cfg:
        if "significance_es" not in profile_cfg:
            raise ValueError(
                f"Profile {name!r} must provide exp_es and sm_es."
            )
        warnings.warn(
            f"Profile {name!r} uses legacy significance_es. It is interpreted "
            "as EXP ES, while S&M ES is set to zero."
        )
        exp_es = _float_vector(profile_cfg["significance_es"], f"{name}.significance_es")
        sm_es = np.zeros_like(exp_es)
        exp_labels = tuple(str(v) for v in profile_cfg.get(
            "significance_labels", [f"EXP_{i}" for i in range(exp_es.size)]
        ))
        sm_labels = tuple(f"SM_{i}" for i in range(sm_es.size))
        exp_p_one = _float_vector(
            profile_cfg.get("p_one_by_significance", [0.5] * exp_es.size),
            f"{name}.p_one_by_significance",
        )
        sm_p_one = np.full(sm_es.size, 0.5, dtype=np.float64)
        masked_exp = _mask_vector(
            profile_cfg, "masked_significance_indices", exp_es.size, f"{name}.EXP"
        )
        masked_sm = np.zeros(sm_es.size, dtype=bool)
    else:
        exp_es = _float_vector(profile_cfg["exp_es"], f"{name}.exp_es")
        sm_es = _float_vector(profile_cfg["sm_es"], f"{name}.sm_es")
        exp_labels = _labels(profile_cfg, "exp_labels", "EXP", int(exp_es.size))
        sm_labels = _labels(profile_cfg, "sm_labels", "SM", int(sm_es.size))
        exp_p_one = _float_vector(
            profile_cfg.get("exp_p_one", [0.5] * exp_es.size),
            f"{name}.exp_p_one",
        )
        sm_p_one = _float_vector(
            profile_cfg.get("sm_p_one", [0.5] * sm_es.size),
            f"{name}.sm_p_one",
        )
        masked_exp = _mask_vector(
            profile_cfg, "masked_exp_indices", exp_es.size, f"{name}.EXP"
        )
        masked_sm = _mask_vector(
            profile_cfg, "masked_sm_indices", sm_es.size, f"{name}.S&M"
        )

    common_mask_factor = float(profile_cfg.get("mask_factor", 0.0))
    profile = ESProfile(
        name=name,
        exp_labels=exp_labels,
        sm_labels=sm_labels,
        exp_es=exp_es,
        sm_es=sm_es,
        exp_p_one=exp_p_one,
        sm_p_one=sm_p_one,
        masked_exp=masked_exp,
        masked_sm=masked_sm,
        exp_scale=float(profile_cfg.get("exp_scale", 1.0)),
        sm_scale=float(profile_cfg.get("sm_scale", 1.0)),
        exp_mask_factor=float(profile_cfg.get("exp_mask_factor", common_mask_factor)),
        sm_mask_factor=float(profile_cfg.get("sm_mask_factor", common_mask_factor)),
    )
    profile.validate()
    return profile


def _build_index_map(
    layout_cfg: Mapping[str, Any],
    geometry: Geometry,
    base_dir: Path,
    label: str,
) -> np.ndarray:
    if layout_cfg.get("npy"):
        result = np.load(base_dir / str(layout_cfg["npy"])).astype(np.int64)
        kind = "npy"
    else:
        kind = str(layout_cfg.get("kind", "explicit_2d")).lower()
        result = None

    b = np.arange(geometry.banks, dtype=np.int64)[:, None, None]
    w = np.arange(geometry.words, dtype=np.int64)[None, :, None]
    bit = np.arange(geometry.bits, dtype=np.int64)[None, None, :]
    shape = (geometry.banks, geometry.words, geometry.bits)

    if kind == "npy":
        result = np.asarray(result, dtype=np.int64)
    elif kind == "explicit_2d":
        raw = np.asarray(layout_cfg["table"], dtype=np.int64)
        if raw.shape != (geometry.banks, geometry.words):
            raise ValueError(
                f"{label} explicit_2d table must have shape "
                f"({geometry.banks}, {geometry.words}), got {raw.shape}."
            )
        result = np.broadcast_to(raw[:, :, None], shape).copy()
    elif kind == "explicit_3d":
        result = np.asarray(layout_cfg["table"], dtype=np.int64)
    elif kind == "bank":
        result = np.broadcast_to(b, shape).copy()
    elif kind == "word":
        result = np.broadcast_to(w, shape).copy()
    elif kind == "cell_bit":
        result = np.broadcast_to(bit, shape).copy()
    elif kind == "bank_xor_word":
        result = np.broadcast_to(np.bitwise_xor(b, w), shape).copy()
    elif kind == "bank_plus_word_mod":
        modulus = int(layout_cfg.get("modulus", max(geometry.banks, geometry.words)))
        result = np.broadcast_to((b + w) % modulus, shape).copy()
    else:
        raise ValueError(
            f"Unknown {label} layout kind {kind!r}. Supported: explicit_2d, "
            "explicit_3d, bank, word, cell_bit, bank_xor_word, "
            "bank_plus_word_mod, or npy."
        )

    if result.shape != shape:
        raise ValueError(f"{label} map shape={result.shape}, expected {shape}.")
    return result


def _decode_field_token(value: Any) -> Tuple[bool, bool]:
    if isinstance(value, (int, np.integer)):
        code = int(value)
        if code == 0:
            return False, False
        if code == 1:
            return True, False
        if code == 2:
            return False, True
        if code == 3:
            return True, True
    token = str(value).strip().upper().replace("&", "").replace("_", "")
    if token in {"NONE", "N", "0"}:
        return False, False
    if token in {"EXP", "E", "1"}:
        return True, False
    if token in {"SM", "S&M", "SANDM", "2"}:
        return False, True
    if token in {"BOTH", "PAIRED", "EXP+SM", "3"}:
        return True, True
    raise ValueError(f"Unknown field-assignment token: {value!r}.")


def _build_field_active_maps(
    assignment_cfg: Mapping[str, Any],
    geometry: Geometry,
    base_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, str]:
    shape = (geometry.banks, geometry.words, geometry.bits)
    kind = str(assignment_cfg.get("kind", "paired")).lower()

    if kind == "paired":
        return np.ones(shape, dtype=bool), np.ones(shape, dtype=bool), kind

    if kind == "by_cell_bit":
        fields = assignment_cfg.get("cell_bit_fields")
        if fields is None or len(fields) != geometry.bits:
            raise ValueError(
                f"by_cell_bit requires cell_bit_fields with length {geometry.bits}."
            )
        exp_bits = np.zeros(geometry.bits, dtype=bool)
        sm_bits = np.zeros(geometry.bits, dtype=bool)
        for bit_index, token in enumerate(fields):
            exp_bits[bit_index], sm_bits[bit_index] = _decode_field_token(token)
        exp_map = np.broadcast_to(exp_bits[None, None, :], shape).copy()
        sm_map = np.broadcast_to(sm_bits[None, None, :], shape).copy()
        return exp_map, sm_map, kind

    if kind in {"explicit_3d", "npy"}:
        if kind == "npy":
            raw = np.load(base_dir / str(assignment_cfg["npy"]), allow_pickle=True)
        else:
            raw = np.asarray(assignment_cfg["table"], dtype=object)
        if raw.shape != shape:
            raise ValueError(
                f"field_assignment {kind} shape={raw.shape}, expected {shape}."
            )
        exp_map = np.zeros(shape, dtype=bool)
        sm_map = np.zeros(shape, dtype=bool)
        for coordinate in np.ndindex(shape):
            exp_map[coordinate], sm_map[coordinate] = _decode_field_token(raw[coordinate])
        return exp_map, sm_map, kind

    raise ValueError(
        "field_assignment.kind must be paired, by_cell_bit, explicit_3d, or npy."
    )


def build_rtl_mapping(
    mapping_cfg: Mapping[str, Any],
    geometry: Geometry,
    base_dir: Path,
) -> RTLMapping:
    common_layout = mapping_cfg.get("logical_significance_layout")
    exp_layout = mapping_cfg.get("exp_logical_significance_layout", common_layout)
    sm_layout = mapping_cfg.get("sm_logical_significance_layout", common_layout)
    if exp_layout is None or sm_layout is None:
        raise ValueError(
            "Provide logical_significance_layout, or separate "
            "exp_logical_significance_layout and sm_logical_significance_layout."
        )

    exp_index_map = _build_index_map(exp_layout, geometry, base_dir, "EXP")
    sm_index_map = _build_index_map(sm_layout, geometry, base_dir, "S&M")
    exp_active, sm_active, assignment_kind = _build_field_active_maps(
        mapping_cfg.get("field_assignment", {"kind": "paired"}),
        geometry,
        base_dir,
    )

    return RTLMapping(
        exp_index_map=exp_index_map,
        sm_index_map=sm_index_map,
        exp_active_map=exp_active,
        sm_active_map=sm_active,
        inter_scope=str(mapping_cfg.get("inter_scope", "bitmap_block")),
        intra_scope=str(mapping_cfg.get("intra_scope", "bitmap_block_bank_rg")),
        inter_shift_direction=str(mapping_cfg.get("inter_shift_direction", "right")),
        field_assignment_kind=assignment_kind,
    )

def normalize_fault_mix(mix: Mapping[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    names = ("SA0", "SA1", "TRANSIENT")
    probabilities = np.asarray([float(mix.get(name, 0.0)) for name in names])
    if np.any(probabilities < 0):
        raise ValueError("Fault-type probabilities must be nonnegative.")
    total = float(probabilities.sum())
    if total <= 0:
        raise ValueError("At least one fault-type probability must be positive.")
    probabilities /= total
    identifiers = np.asarray([FAULT_NAME_TO_ID[name] for name in names], dtype=np.uint8)
    return identifiers, probabilities


def sample_faults(
    rng: np.random.Generator,
    geometry: Geometry,
    ber: float,
    fault_mix: Mapping[str, float],
    spatial_cfg: Mapping[str, Any],
    max_faults_per_trial: Optional[int],
) -> FaultSet:
    if ber < 0:
        raise ValueError("BER must be nonnegative.")

    poisson_count = int(rng.poisson(geometry.total_bits * ber))
    if max_faults_per_trial is not None and poisson_count > max_faults_per_trial:
        raise RuntimeError(
            f"Sampled {poisson_count:,} fault events, exceeding "
            f"max_faults_per_trial={max_faults_per_trial:,}. The program does not "
            "silently cap the count because that would bias the distribution."
        )

    if poisson_count == 0:
        return FaultSet(
            page=np.empty(0, dtype=np.int64),
            cell=np.empty(0, dtype=np.int64),
            fault_type=np.empty(0, dtype=np.uint8),
            poisson_count=0,
            unique_count=0,
        )

    kind = str(spatial_cfg.get("kind", "uniform")).lower()

    if kind == "uniform":
        pages = rng.integers(0, geometry.num_pages, poisson_count, dtype=np.int64)
        cells = rng.integers(0, geometry.page_bits, poisson_count, dtype=np.int64)

    elif kind == "page_clustered":
        hot_page_fraction = float(spatial_cfg.get("hot_page_fraction", 0.01))
        hot_fault_fraction = float(spatial_cfg.get("hot_fault_fraction", 0.80))
        if not 0 < hot_page_fraction <= 1:
            raise ValueError("hot_page_fraction must be within (0, 1].")
        if not 0 <= hot_fault_fraction <= 1:
            raise ValueError("hot_fault_fraction must be within [0, 1].")

        hot_page_count = max(1, int(round(geometry.num_pages * hot_page_fraction)))
        hot_pages = rng.choice(
            geometry.num_pages, size=hot_page_count, replace=False
        ).astype(np.int64)
        choose_hot = rng.random(poisson_count) < hot_fault_fraction
        pages = rng.integers(0, geometry.num_pages, poisson_count, dtype=np.int64)
        pages[choose_hot] = rng.choice(
            hot_pages, size=int(choose_hot.sum()), replace=True
        )
        cells = rng.integers(0, geometry.page_bits, poisson_count, dtype=np.int64)

    elif kind == "shared_offsets":
        offset_count = int(spatial_cfg.get("num_shared_offsets", 64))
        offset_count = min(max(1, offset_count), geometry.page_bits)
        offsets = rng.choice(
            geometry.page_bits, size=offset_count, replace=False
        ).astype(np.int64)
        pages = rng.integers(0, geometry.num_pages, poisson_count, dtype=np.int64)
        cells = rng.choice(offsets, size=poisson_count, replace=True).astype(np.int64)

    else:
        raise ValueError(
            f"Unknown spatial distribution {kind!r}. Supported: uniform, "
            "page_clustered, shared_offsets."
        )

    fault_ids, probabilities = normalize_fault_mix(fault_mix)
    fault_types = rng.choice(
        fault_ids, size=poisson_count, p=probabilities
    ).astype(np.uint8)

    # Keep one physical fault per cell. Duplicate events otherwise over-count a
    # single faulty cell, especially in clustered/shared-offset scenarios.
    linear = pages * geometry.page_bits + cells
    _, first_indices = np.unique(linear, return_index=True)
    first_indices.sort()

    return FaultSet(
        page=pages[first_indices],
        cell=cells[first_indices],
        fault_type=fault_types[first_indices],
        poisson_count=poisson_count,
        unique_count=int(first_indices.size),
    )


def decode_faults(faults: FaultSet, geometry: Geometry) -> DecodedFaults:
    unique_pages, page_compact = np.unique(faults.page, return_inverse=True)

    remaining = faults.cell.astype(np.int64, copy=True)
    block = remaining // geometry.cells_per_bitmap_block
    remaining %= geometry.cells_per_bitmap_block

    bank_stride = geometry.rgs * geometry.words * geometry.bits
    bank = remaining // bank_stride
    remaining %= bank_stride

    rg_stride = geometry.words * geometry.bits
    rg = remaining // rg_stride
    remaining %= rg_stride

    word = remaining // geometry.bits
    bit = remaining % geometry.bits

    return DecodedFaults(
        page=faults.page,
        page_compact=page_compact.astype(np.int64),
        unique_pages=unique_pages.astype(np.int64),
        block=block.astype(np.int64),
        bank=bank.astype(np.int64),
        rg=rg.astype(np.int64),
        word=word.astype(np.int64),
        bit=bit.astype(np.int64),
        fault_type=faults.fault_type.astype(np.uint8),
    )


def channel_mapping_for_coordinates(
    mapping: RTLMapping,
    logical_bank: np.ndarray,
    logical_word: np.ndarray,
    logical_bit: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    coordinates = (logical_bank, logical_word, logical_bit)
    return (
        mapping.exp_index_map[coordinates],
        mapping.sm_index_map[coordinates],
        mapping.exp_active_map[coordinates],
        mapping.sm_active_map[coordinates],
    )


def _page_sum(
    page_compact: np.ndarray,
    values: np.ndarray,
    page_count: int,
) -> np.ndarray:
    return np.bincount(
        page_compact, weights=values, minlength=page_count
    ).astype(np.float64)


def no_remap_page_components(
    decoded: DecodedFaults,
    mapping: RTLMapping,
    profile: ESProfile,
    masked: bool,
) -> ModeEvaluation:
    exp_index, sm_index, exp_active, sm_active = channel_mapping_for_coordinates(
        mapping, decoded.bank, decoded.word, decoded.bit
    )
    exp_cost, sm_cost, total_cost = profile.cost_components(
        exp_index,
        sm_index,
        exp_active,
        sm_active,
        decoded.fault_type,
        masked=masked,
    )
    page_count = decoded.faulty_page_count
    return ModeEvaluation(
        page_es=_page_sum(decoded.page_compact, total_cost, page_count),
        page_exp_es=_page_sum(decoded.page_compact, exp_cost, page_count),
        page_sm_es=_page_sum(decoded.page_compact, sm_cost, page_count),
        group_page_compact=np.arange(page_count, dtype=np.int64),
        best_control=np.zeros(page_count, dtype=np.int64),
    )


def _aggregate_candidate_costs(
    group_inverse: np.ndarray,
    group_count: int,
    candidate_costs: List[np.ndarray],
) -> np.ndarray:
    matrix = np.empty((len(candidate_costs), group_count), dtype=np.float64)
    for index, costs in enumerate(candidate_costs):
        matrix[index] = np.bincount(
            group_inverse, weights=costs, minlength=group_count
        )
    return matrix


def _select_group_components(
    exp_matrix: np.ndarray,
    sm_matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    total_matrix = exp_matrix + sm_matrix
    best_control = np.argmin(total_matrix, axis=0).astype(np.int64)
    columns = np.arange(total_matrix.shape[1], dtype=np.int64)
    best_exp = exp_matrix[best_control, columns]
    best_sm = sm_matrix[best_control, columns]
    return best_control, best_exp, best_sm, best_exp + best_sm


def evaluate_inter_vectorized(
    decoded: DecodedFaults,
    geometry: Geometry,
    mapping: RTLMapping,
    profile: ESProfile,
    masked: bool,
) -> ModeEvaluation:
    page_count = decoded.faulty_page_count
    if len(decoded) == 0:
        empty = np.empty(0, dtype=np.int64)
        zeros = np.zeros(page_count, dtype=np.float64)
        return ModeEvaluation(zeros, zeros.copy(), zeros.copy(), empty, empty)

    if mapping.inter_scope == "flash_page":
        group_key = decoded.page_compact
        group_page_compact = np.arange(page_count, dtype=np.int64)
    else:
        group_key = decoded.page_compact * geometry.bitmap_blocks_per_page + decoded.block
        unique_group_key = np.unique(group_key)
        group_page_compact = (
            unique_group_key // geometry.bitmap_blocks_per_page
        ).astype(np.int64)

    _, group_inverse = np.unique(group_key, return_inverse=True)
    group_count = int(group_inverse.max()) + 1
    exp_candidates: List[np.ndarray] = []
    sm_candidates: List[np.ndarray] = []

    for shift in range(geometry.banks):
        if mapping.inter_shift_direction == "right":
            logical_bank = (decoded.bank - shift) % geometry.banks
        else:
            logical_bank = (decoded.bank + shift) % geometry.banks

        exp_index, sm_index, exp_active, sm_active = channel_mapping_for_coordinates(
            mapping, logical_bank, decoded.word, decoded.bit
        )
        exp_cost, sm_cost, _ = profile.cost_components(
            exp_index,
            sm_index,
            exp_active,
            sm_active,
            decoded.fault_type,
            masked=masked,
        )
        exp_candidates.append(exp_cost)
        sm_candidates.append(sm_cost)

    exp_matrix = _aggregate_candidate_costs(group_inverse, group_count, exp_candidates)
    sm_matrix = _aggregate_candidate_costs(group_inverse, group_count, sm_candidates)
    best_control, best_exp, best_sm, best_total = _select_group_components(
        exp_matrix, sm_matrix
    )

    return ModeEvaluation(
        page_es=_page_sum(group_page_compact, best_total, page_count),
        page_exp_es=_page_sum(group_page_compact, best_exp, page_count),
        page_sm_es=_page_sum(group_page_compact, best_sm, page_count),
        group_page_compact=group_page_compact,
        best_control=best_control,
    )


def evaluate_intra_vectorized(
    decoded: DecodedFaults,
    geometry: Geometry,
    mapping: RTLMapping,
    profile: ESProfile,
    masked: bool,
) -> ModeEvaluation:
    page_count = decoded.faulty_page_count
    if len(decoded) == 0:
        empty = np.empty(0, dtype=np.int64)
        zeros = np.zeros(page_count, dtype=np.float64)
        return ModeEvaluation(zeros, zeros.copy(), zeros.copy(), empty, empty)

    if mapping.intra_scope == "flash_page_bank_rg":
        group_key = (
            (decoded.page_compact * geometry.banks + decoded.bank) * geometry.rgs
            + decoded.rg
        )
        group_divisor = geometry.banks * geometry.rgs
    else:
        group_key = (
            (((decoded.page_compact * geometry.bitmap_blocks_per_page + decoded.block)
              * geometry.banks + decoded.bank) * geometry.rgs)
            + decoded.rg
        )
        group_divisor = geometry.bitmap_blocks_per_page * geometry.banks * geometry.rgs

    unique_group_key, group_inverse = np.unique(group_key, return_inverse=True)
    group_count = int(unique_group_key.size)
    group_page_compact = (unique_group_key // group_divisor).astype(np.int64)
    exp_candidates: List[np.ndarray] = []
    sm_candidates: List[np.ndarray] = []

    for cw in range(geometry.words):
        logical_word = np.bitwise_xor(decoded.word, cw)
        exp_index, sm_index, exp_active, sm_active = channel_mapping_for_coordinates(
            mapping, decoded.bank, logical_word, decoded.bit
        )
        exp_cost, sm_cost, _ = profile.cost_components(
            exp_index,
            sm_index,
            exp_active,
            sm_active,
            decoded.fault_type,
            masked=masked,
        )
        exp_candidates.append(exp_cost)
        sm_candidates.append(sm_cost)

    exp_matrix = _aggregate_candidate_costs(group_inverse, group_count, exp_candidates)
    sm_matrix = _aggregate_candidate_costs(group_inverse, group_count, sm_candidates)
    best_control, best_exp, best_sm, best_total = _select_group_components(
        exp_matrix, sm_matrix
    )

    return ModeEvaluation(
        page_es=_page_sum(group_page_compact, best_total, page_count),
        page_exp_es=_page_sum(group_page_compact, best_exp, page_count),
        page_sm_es=_page_sum(group_page_compact, best_sm, page_count),
        group_page_compact=group_page_compact,
        best_control=best_control,
    )


def evaluate_all_modes_vectorized(
    decoded: DecodedFaults,
    geometry: Geometry,
    mapping: RTLMapping,
    profile: ESProfile,
) -> Tuple[
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Dict[str, ModeEvaluation],
]:
    no_remap = no_remap_page_components(decoded, mapping, profile, masked=False)
    masked_intra = evaluate_intra_vectorized(decoded, geometry, mapping, profile, True)
    intra = evaluate_intra_vectorized(decoded, geometry, mapping, profile, False)
    masked_inter = evaluate_inter_vectorized(decoded, geometry, mapping, profile, True)
    inter = evaluate_inter_vectorized(decoded, geometry, mapping, profile, False)

    evaluations = {
        "no_remap": no_remap,
        "masked_intra": masked_intra,
        "intra": intra,
        "masked_inter": masked_inter,
        "inter": inter,
    }
    total = {name: value.page_es for name, value in evaluations.items()}
    exp = {name: value.page_exp_es for name, value in evaluations.items()}
    sm = {name: value.page_sm_es for name, value in evaluations.items()}
    return total, exp, sm, evaluations

def page_threshold_vector(
    no_remap_es: np.ndarray,
    selection_cfg: Mapping[str, Any],
) -> np.ndarray:
    absolute = selection_cfg.get("absolute_es_limit")
    relative = selection_cfg.get("max_relative_es")

    if absolute is not None:
        return np.full_like(no_remap_es, float(absolute), dtype=np.float64)
    if relative is not None:
        return no_remap_es * float(relative)
    return np.full_like(no_remap_es, math.inf, dtype=np.float64)


def choose_oracle_modes_vectorized(
    page_es: Mapping[str, np.ndarray],
    selection_cfg: Mapping[str, Any],
    mode_costs: Mapping[str, float],
    mode_order: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], np.ndarray]:
    threshold = page_threshold_vector(page_es["no_remap"], selection_cfg)
    page_count = threshold.size

    candidates = list(mode_order)
    if bool(selection_cfg.get("include_no_remap_candidate", False)):
        candidates = ["no_remap"] + candidates

    feasible = {
        mode: np.asarray(page_es[mode] <= threshold + 1e-12, dtype=bool)
        for mode in candidates
    }

    # Lowest cost feasible mode. Ties preserve candidate order.
    ordered = sorted(
        candidates,
        key=lambda mode: (float(mode_costs.get(mode, math.inf)), candidates.index(mode)),
    )
    selected = np.full(page_count, "", dtype=object)
    for mode in ordered:
        take = (selected == "") & feasible[mode]
        selected[take] = mode

    unrepairable = selected == ""
    if np.any(unrepairable):
        es_matrix = np.vstack([page_es[mode] for mode in candidates])
        minimum_index = np.argmin(es_matrix[:, unrepairable], axis=0)
        selected[unrepairable] = np.asarray(candidates, dtype=object)[minimum_index]

    return selected, threshold, feasible, unrepairable


def _axis_max_fraction(
    page_compact: np.ndarray,
    axis_value: np.ndarray,
    page_count: int,
    axis_size: int,
    fault_count: np.ndarray,
) -> np.ndarray:
    flat = page_compact * axis_size + axis_value
    counts = np.bincount(flat, minlength=page_count * axis_size).reshape(
        page_count, axis_size
    )
    return counts.max(axis=1) / np.maximum(fault_count, 1)


def compute_page_features(
    decoded: DecodedFaults,
    geometry: Geometry,
    mapping: RTLMapping,
    profile: ESProfile,
    raw_es: np.ndarray,
    raw_exp_es: np.ndarray,
    raw_sm_es: np.ndarray,
) -> Dict[str, np.ndarray]:
    page_count = decoded.faulty_page_count
    fault_count = np.bincount(
        decoded.page_compact, minlength=page_count
    ).astype(np.float64)
    denominator = np.maximum(fault_count, 1)

    result: Dict[str, np.ndarray] = {
        "fault_count": fault_count,
        "raw_es": raw_es,
        "raw_exp_es": raw_exp_es,
        "raw_sm_es": raw_sm_es,
        "max_bank_fraction": _axis_max_fraction(
            decoded.page_compact, decoded.bank, page_count, geometry.banks, fault_count
        ),
        "max_word_fraction": _axis_max_fraction(
            decoded.page_compact, decoded.word, page_count, geometry.words, fault_count
        ),
        "max_rg_fraction": _axis_max_fraction(
            decoded.page_compact, decoded.rg, page_count, geometry.rgs, fault_count
        ),
        "max_block_fraction": _axis_max_fraction(
            decoded.page_compact,
            decoded.block,
            page_count,
            geometry.bitmap_blocks_per_page,
            fault_count,
        ),
    }

    for name, identifier in FAULT_NAME_TO_ID.items():
        counts = np.bincount(
            decoded.page_compact[decoded.fault_type == identifier],
            minlength=page_count,
        )
        result[f"{name.lower()}_fraction"] = counts / denominator

    exp_index, sm_index, exp_active, sm_active = channel_mapping_for_coordinates(
        mapping, decoded.bank, decoded.word, decoded.bit
    )
    exp_active_count = np.bincount(
        decoded.page_compact[exp_active], minlength=page_count
    )
    sm_active_count = np.bincount(
        decoded.page_compact[sm_active], minlength=page_count
    )
    result["exp_fault_fraction"] = exp_active_count / denominator
    result["sm_fault_fraction"] = sm_active_count / denominator

    exp_flat = decoded.page_compact[exp_active] * profile.exp_count + exp_index[exp_active]
    exp_counts = np.bincount(
        exp_flat, minlength=page_count * profile.exp_count
    ).reshape(page_count, profile.exp_count)
    for index in range(profile.exp_count):
        result[f"exp_sig_{index}_fraction"] = exp_counts[:, index] / denominator

    sm_flat = decoded.page_compact[sm_active] * profile.sm_count + sm_index[sm_active]
    sm_counts = np.bincount(
        sm_flat, minlength=page_count * profile.sm_count
    ).reshape(page_count, profile.sm_count)
    for index in range(profile.sm_count):
        result[f"sm_sig_{index}_fraction"] = sm_counts[:, index] / denominator

    exp_weight = profile.exp_scale * profile.exp_es
    sm_weight = profile.sm_scale * profile.sm_es
    exp_cut = float(np.quantile(exp_weight, 0.75))
    sm_cut = float(np.quantile(sm_weight, 0.75))
    exp_critical = exp_active & (exp_weight[exp_index] >= exp_cut)
    sm_critical = sm_active & (sm_weight[sm_index] >= sm_cut)
    exp_critical_count = np.bincount(
        decoded.page_compact[exp_critical], minlength=page_count
    )
    sm_critical_count = np.bincount(
        decoded.page_compact[sm_critical], minlength=page_count
    )
    result["exp_critical_fault_fraction"] = exp_critical_count / denominator
    result["sm_critical_fault_fraction"] = sm_critical_count / denominator
    result["critical_fault_fraction"] = np.bincount(
        decoded.page_compact[exp_critical | sm_critical], minlength=page_count
    ) / denominator
    return result

def _control_histogram_for_page(
    evaluation: ModeEvaluation,
    page_index: int,
    control_count: int,
) -> str:
    controls = evaluation.best_control[
        evaluation.group_page_compact == page_index
    ]
    histogram = np.bincount(controls, minlength=control_count).tolist()
    return json.dumps(histogram, separators=(",", ":"))


def evaluate_trial_profile(
    decoded: DecodedFaults,
    faults: FaultSet,
    geometry: Geometry,
    mapping: RTLMapping,
    profile: ESProfile,
    selection_cfg: Mapping[str, Any],
    mode_costs: Mapping[str, float],
    mode_order: Sequence[str],
    sampled_page_indices: np.ndarray,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    page_count = decoded.faulty_page_count
    candidate_modes = ["no_remap"] + list(mode_order)

    if page_count == 0:
        summary: Dict[str, Any] = {
            "profile": profile.name,
            "faulty_pages": 0,
            "no_fault_pages": geometry.num_pages,
            "poisson_fault_events": faults.poisson_count,
            "unique_physical_faults": faults.unique_count,
            "duplicate_event_count": faults.poisson_count - faults.unique_count,
            "unrepairable_pages": 0,
            "unrepairable_fraction": 0.0,
        }
        for mode in candidate_modes:
            summary[f"total_es_{mode}"] = 0.0
            summary[f"total_exp_es_{mode}"] = 0.0
            summary[f"total_sm_es_{mode}"] = 0.0
            summary[f"mean_es_per_faulty_page_{mode}"] = 0.0
            summary[f"mean_exp_es_per_faulty_page_{mode}"] = 0.0
            summary[f"mean_sm_es_per_faulty_page_{mode}"] = 0.0
            summary[f"feasible_fraction_{mode}"] = 1.0
            summary[f"selected_fraction_{mode}"] = 0.0
            summary[f"minimum_es_fraction_{mode}"] = 0.0
        return summary, []

    page_es, page_exp_es, page_sm_es, evaluations = evaluate_all_modes_vectorized(
        decoded, geometry, mapping, profile
    )
    selected, threshold, feasible, unrepairable = choose_oracle_modes_vectorized(
        page_es, selection_cfg, mode_costs, mode_order
    )

    summary = {
        "profile": profile.name,
        "faulty_pages": page_count,
        "no_fault_pages": geometry.num_pages - page_count,
        "poisson_fault_events": faults.poisson_count,
        "unique_physical_faults": faults.unique_count,
        "duplicate_event_count": faults.poisson_count - faults.unique_count,
        "unrepairable_pages": int(unrepairable.sum()),
        "unrepairable_fraction": float(unrepairable.mean()),
    }
    min_es = np.min(np.vstack([page_es[m] for m in mode_order]), axis=0)

    for mode in candidate_modes:
        values = page_es[mode]
        exp_values = page_exp_es[mode]
        sm_values = page_sm_es[mode]
        summary[f"total_es_{mode}"] = float(values.sum())
        summary[f"total_exp_es_{mode}"] = float(exp_values.sum())
        summary[f"total_sm_es_{mode}"] = float(sm_values.sum())
        summary[f"mean_es_per_faulty_page_{mode}"] = float(values.mean())
        summary[f"mean_exp_es_per_faulty_page_{mode}"] = float(exp_values.mean())
        summary[f"mean_sm_es_per_faulty_page_{mode}"] = float(sm_values.mean())
        summary[f"feasible_fraction_{mode}"] = float(
            np.mean(values <= threshold + 1e-12)
        )
        summary[f"selected_fraction_{mode}"] = float(np.mean(selected == mode))
        summary[f"minimum_es_fraction_{mode}"] = (
            float(np.mean(np.isclose(values, min_es, atol=1e-12, rtol=0.0)))
            if mode in mode_order
            else 0.0
        )

    features = compute_page_features(
        decoded,
        geometry,
        mapping,
        profile,
        page_es["no_remap"],
        page_exp_es["no_remap"],
        page_sm_es["no_remap"],
    )

    page_rows: List[Dict[str, Any]] = []
    for compact_index_raw in sampled_page_indices:
        compact_index = int(compact_index_raw)
        row: Dict[str, Any] = {
            "page_id": int(decoded.unique_pages[compact_index]),
            "profile": profile.name,
            "oracle_mode": str(selected[compact_index]),
            "oracle_status": (
                "unrepairable" if unrepairable[compact_index] else "feasible"
            ),
            "es_threshold": float(threshold[compact_index]),
            "inter_shift_histogram": _control_histogram_for_page(
                evaluations["inter"], compact_index, geometry.banks
            ),
            "masked_inter_shift_histogram": _control_histogram_for_page(
                evaluations["masked_inter"], compact_index, geometry.banks
            ),
            "intra_cw_histogram": _control_histogram_for_page(
                evaluations["intra"], compact_index, geometry.words
            ),
            "masked_intra_cw_histogram": _control_histogram_for_page(
                evaluations["masked_intra"], compact_index, geometry.words
            ),
        }
        for name, values in features.items():
            row[name] = float(values[compact_index])
        for mode in candidate_modes:
            row[f"es_{mode}"] = float(page_es[mode][compact_index])
            row[f"exp_es_{mode}"] = float(page_exp_es[mode][compact_index])
            row[f"sm_es_{mode}"] = float(page_sm_es[mode][compact_index])
            row[f"feasible_{mode}"] = int(
                page_es[mode][compact_index] <= threshold[compact_index] + 1e-12
            )
        page_rows.append(row)

    return summary, page_rows

def add_metadata(
    row: Dict[str, Any],
    scenario: str,
    spatial: str,
    ber: float,
    repeat: int,
    seed_value: int,
) -> Dict[str, Any]:
    result = dict(row)
    result.update(
        {
            "scenario": scenario,
            "spatial_distribution": spatial,
            "ber": float(ber),
            "repeat": int(repeat),
            "trial_seed": int(seed_value),
        }
    )
    return result


def normal_ci_half_width(values: pd.Series) -> float:
    array = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if array.size <= 1:
        return math.nan
    return float(1.96 * array.std(ddof=1) / math.sqrt(array.size))


def summarize_trials(trial_df: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["profile", "scenario", "spatial_distribution", "ber"]
    metric_columns = [
        column
        for column in trial_df.columns
        if column.startswith(
            (
                "total_es_",
                "total_exp_es_",
                "total_sm_es_",
                "mean_es_per_faulty_page_",
                "mean_exp_es_per_faulty_page_",
                "mean_sm_es_per_faulty_page_",
                "feasible_fraction_",
                "selected_fraction_",
                "minimum_es_fraction_",
            )
        )
        or column
        in {
            "faulty_pages",
            "unique_physical_faults",
            "unrepairable_fraction",
        }
    ]

    rows: List[Dict[str, Any]] = []
    for keys, group in trial_df.groupby(group_columns, dropna=False, sort=True):
        row = dict(zip(group_columns, keys))
        row["repeats"] = int(len(group))
        for column in metric_columns:
            values = pd.to_numeric(group[column], errors="coerce")
            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else math.nan
            )
            row[f"{column}_ci95_half_width"] = normal_ci_half_width(values)
        rows.append(row)
    return pd.DataFrame(rows)


def recommend_modes_by_condition(
    summary_df: pd.DataFrame,
    mode_costs: Mapping[str, float],
    mode_order: Sequence[str],
    required_feasible_rate: float,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, source in summary_df.iterrows():
        qualified = [
            mode
            for mode in mode_order
            if float(source[f"feasible_fraction_{mode}_mean"])
            >= required_feasible_rate
        ]
        if qualified:
            selected = min(
                qualified,
                key=lambda mode: (
                    float(mode_costs.get(mode, math.inf)),
                    mode_order.index(mode),
                ),
            )
            status = "recommended"
        else:
            selected = min(
                mode_order,
                key=lambda mode: (
                    -float(source[f"feasible_fraction_{mode}_mean"]),
                    float(mode_costs.get(mode, math.inf)),
                ),
            )
            status = "target_not_met"

        row: Dict[str, Any] = {
            "profile": source["profile"],
            "scenario": source["scenario"],
            "spatial_distribution": source["spatial_distribution"],
            "ber": source["ber"],
            "required_feasible_rate": required_feasible_rate,
            "recommended_mode": selected,
            "recommendation_status": status,
        }
        for mode in mode_order:
            row[f"feasible_rate_{mode}"] = source[
                f"feasible_fraction_{mode}_mean"
            ]
        rows.append(row)
    return pd.DataFrame(rows)


def compress_mode_regions(recommendations: pd.DataFrame) -> pd.DataFrame:
    if recommendations.empty:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    group_columns = ["profile", "scenario", "spatial_distribution"]
    for keys, group in recommendations.groupby(group_columns, sort=True):
        group = group.sort_values("ber").reset_index(drop=True)
        start = 0
        for end in range(1, len(group) + 1):
            boundary = (
                end == len(group)
                or group.loc[end, "recommended_mode"]
                != group.loc[start, "recommended_mode"]
                or group.loc[end, "recommendation_status"]
                != group.loc[start, "recommendation_status"]
            )
            if boundary:
                block = group.iloc[start:end]
                row = dict(zip(group_columns, keys))
                row.update(
                    {
                        "ber_min": float(block["ber"].min()),
                        "ber_max": float(block["ber"].max()),
                        "recommended_mode": block.iloc[0]["recommended_mode"],
                        "recommendation_status": block.iloc[0][
                            "recommendation_status"
                        ],
                    }
                )
                rows.append(row)
                start = end
    return pd.DataFrame(rows)


def train_threshold_selector(
    page_df: pd.DataFrame,
    output_dir: Path,
    random_seed: int,
) -> None:
    if page_df.empty:
        return

    try:
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        from sklearn.model_selection import train_test_split
        from sklearn.tree import DecisionTreeClassifier, export_text
    except Exception as exc:
        (output_dir / "selector_skipped.txt").write_text(
            f"scikit-learn is unavailable; selector training was skipped.\n{exc}\n",
            encoding="utf-8",
        )
        return

    fixed_features = [
        "fault_count",
        "raw_es",
        "raw_exp_es",
        "raw_sm_es",
        "exp_fault_fraction",
        "sm_fault_fraction",
        "sa0_fraction",
        "sa1_fraction",
        "transient_fraction",
        "max_bank_fraction",
        "max_word_fraction",
        "max_rg_fraction",
        "max_block_fraction",
        "critical_fault_fraction",
        "exp_critical_fault_fraction",
        "sm_critical_fault_fraction",
    ]
    significance_features = sorted(
        column for column in page_df.columns
        if (column.startswith("exp_sig_") or column.startswith("sm_sig_"))
        and column.endswith("_fraction")
    )
    features = [column for column in fixed_features + significance_features if column in page_df.columns]

    clean = page_df.dropna(subset=features + ["oracle_mode"]).copy()
    if len(clean) < 50 or clean["oracle_mode"].nunique() < 2:
        (output_dir / "selector_skipped.txt").write_text(
            "Not enough sampled pages or mode classes to train a selector.\n",
            encoding="utf-8",
        )
        return

    metrics: List[Dict[str, Any]] = []
    importances: List[Dict[str, Any]] = []
    group_columns = ["profile", "scenario", "spatial_distribution"]

    for keys, group in clean.groupby(group_columns, sort=True):
        if len(group) < 50 or group["oracle_mode"].nunique() < 2:
            continue

        x = group[features]
        y = group["oracle_mode"].astype(str)
        stratify = y if y.value_counts().min() >= 2 else None
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.25,
            random_state=random_seed,
            stratify=stratify,
        )

        model = DecisionTreeClassifier(
            max_depth=4,
            min_samples_leaf=max(10, int(0.01 * len(x_train))),
            class_weight="balanced",
            random_state=random_seed,
        )
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)

        profile, scenario, spatial = keys
        prefix = f"{profile}__{scenario}__{spatial}".replace("/", "_")
        (output_dir / f"selector_rules__{prefix}.txt").write_text(
            export_text(model, feature_names=features), encoding="utf-8"
        )

        labels = sorted(set(y_test) | set(prediction))
        matrix = confusion_matrix(y_test, prediction, labels=labels)
        pd.DataFrame(matrix, index=labels, columns=labels).to_csv(
            output_dir / f"selector_confusion__{prefix}.csv"
        )
        pd.DataFrame(
            classification_report(
                y_test, prediction, output_dict=True, zero_division=0
            )
        ).transpose().to_csv(output_dir / f"selector_report__{prefix}.csv")

        metrics.append(
            {
                "profile": profile,
                "scenario": scenario,
                "spatial_distribution": spatial,
                "sample_rows": len(group),
                "test_accuracy": accuracy_score(y_test, prediction),
                "classes": ",".join(labels),
            }
        )
        for feature, importance in zip(features, model.feature_importances_):
            importances.append(
                {
                    "profile": profile,
                    "scenario": scenario,
                    "spatial_distribution": spatial,
                    "feature": feature,
                    "importance": float(importance),
                }
            )

    if metrics:
        pd.DataFrame(metrics).to_csv(
            output_dir / "selector_metrics.csv", index=False
        )
        pd.DataFrame(importances).to_csv(
            output_dir / "selector_feature_importance.csv", index=False
        )


def analyze_accuracy_calibration(
    csv_path: Path,
    output_dir: Path,
    max_accuracy_drop: float,
    max_false_safe_rate: float,
) -> None:
    df = pd.read_csv(csv_path)
    required = {"profile", "mode", "es", "accuracy"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Accuracy calibration CSV is missing columns: {sorted(missing)}"
        )

    if "baseline_accuracy" not in df.columns:
        df["baseline_accuracy"] = df.groupby("profile")["accuracy"].transform("max")

    df["accuracy_drop"] = df["baseline_accuracy"] - df["accuracy"]
    df["safe"] = df["accuracy_drop"] <= float(max_accuracy_drop)

    correlation_rows: List[Dict[str, Any]] = []
    threshold_rows: List[Dict[str, Any]] = []

    for (profile, mode), group in df.groupby(["profile", "mode"], sort=True):
        correlation_rows.append(
            {
                "profile": profile,
                "mode": mode,
                "samples": len(group),
                "pearson_es_vs_accuracy_drop": group["es"].corr(
                    group["accuracy_drop"], method="pearson"
                ),
                "spearman_es_vs_accuracy_drop": group["es"].corr(
                    group["accuracy_drop"], method="spearman"
                ),
            }
        )

        best: Optional[Dict[str, Any]] = None
        for threshold in np.sort(group["es"].dropna().unique()):
            predicted_safe = group["es"] <= threshold
            covered = int(predicted_safe.sum())
            if covered == 0:
                continue
            false_safe = int((predicted_safe & ~group["safe"]).sum())
            false_safe_rate = false_safe / covered
            coverage = covered / len(group)
            if false_safe_rate <= max_false_safe_rate:
                candidate = {
                    "profile": profile,
                    "mode": mode,
                    "es_threshold": float(threshold),
                    "false_safe_rate": false_safe_rate,
                    "coverage": coverage,
                    "max_accuracy_drop": max_accuracy_drop,
                }
                if best is None or candidate["coverage"] > best["coverage"]:
                    best = candidate

        threshold_rows.append(
            best
            if best is not None
            else {
                "profile": profile,
                "mode": mode,
                "es_threshold": math.nan,
                "false_safe_rate": math.nan,
                "coverage": 0.0,
                "max_accuracy_drop": max_accuracy_drop,
            }
        )

    pd.DataFrame(correlation_rows).to_csv(
        output_dir / "accuracy_es_correlation.csv", index=False
    )
    pd.DataFrame(threshold_rows).to_csv(
        output_dir / "accuracy_es_thresholds.csv", index=False
    )
    df.to_csv(output_dir / "accuracy_calibration_enriched.csv", index=False)

    if plt is not None:
        for (profile, mode), group in df.groupby(["profile", "mode"], sort=True):
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(group["es"], group["accuracy_drop"], alpha=0.65)
            ax.set_xlabel("ES")
            ax.set_ylabel("Accuracy drop")
            ax.set_title(f"{profile} / {mode}: ES vs accuracy drop")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            safe = f"{profile}__{mode}".replace("/", "_")
            fig.savefig(output_dir / f"accuracy_vs_es__{safe}.png", dpi=160)
            plt.close(fig)


def make_plots(
    summary_df: pd.DataFrame,
    mode_order: Sequence[str],
    output_dir: Path,
) -> None:
    if plt is None or summary_df.empty:
        return

    for keys, group in summary_df.groupby(
        ["profile", "scenario", "spatial_distribution"], sort=True
    ):
        profile, scenario, spatial = keys
        group = group.sort_values("ber")
        safe = f"{profile}__{scenario}__{spatial}".replace("/", "_")

        fig, ax = plt.subplots(figsize=(8, 5))
        for mode in ["no_remap"] + list(mode_order):
            ax.plot(
                group["ber"],
                group[f"mean_es_per_faulty_page_{mode}_mean"],
                marker="o",
                label=mode,
            )
        ax.set_xscale("log")
        ax.set_xlabel("BER")
        ax.set_ylabel("Mean ES per faulty page")
        ax.set_title(f"{profile} / {scenario} / {spatial}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"es_vs_ber__{safe}.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        for mode in mode_order:
            ax.plot(
                group["ber"],
                group[f"selected_fraction_{mode}_mean"],
                marker="o",
                label=mode,
            )
        ax.set_xscale("log")
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("BER")
        ax.set_ylabel("Oracle-selected fraction among faulty pages")
        ax.set_title(f"{profile} / {scenario} / {spatial}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"selected_mode_fraction__{safe}.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        for mode in mode_order:
            ax.plot(
                group["ber"],
                group[f"feasible_fraction_{mode}_mean"],
                marker="o",
                label=mode,
            )
        ax.set_xscale("log")
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("BER")
        ax.set_ylabel("Feasible fraction among faulty pages")
        ax.set_title(f"{profile} / {scenario} / {spatial}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"feasible_mode_fraction__{safe}.png", dpi=160)
        plt.close(fig)


def export_mapping_audit(
    output_dir: Path,
    geometry: Geometry,
    mapping: RTLMapping,
    profiles: Sequence[ESProfile],
) -> None:
    rows: List[Dict[str, Any]] = []
    for profile in profiles:
        mapping.validate(geometry, profile.exp_count, profile.sm_count)
        for bank in range(geometry.banks):
            for word in range(geometry.words):
                for bit in range(geometry.bits):
                    exp_index = int(mapping.exp_index_map[bank, word, bit])
                    sm_index = int(mapping.sm_index_map[bank, word, bit])
                    exp_active = bool(mapping.exp_active_map[bank, word, bit])
                    sm_active = bool(mapping.sm_active_map[bank, word, bit])
                    exp_nominal = (
                        profile.exp_scale * profile.exp_es[exp_index] if exp_active else 0.0
                    )
                    sm_nominal = (
                        profile.sm_scale * profile.sm_es[sm_index] if sm_active else 0.0
                    )
                    rows.append(
                        {
                            "profile": profile.name,
                            "logical_bank": bank,
                            "logical_word": word,
                            "logical_cell_bit": bit,
                            "field_assignment_kind": mapping.field_assignment_kind,
                            "exp_active": int(exp_active),
                            "exp_index": exp_index,
                            "exp_label": profile.exp_labels[exp_index],
                            "exp_es_unscaled": float(profile.exp_es[exp_index]),
                            "exp_scale": profile.exp_scale,
                            "exp_es_scaled": float(exp_nominal),
                            "exp_p_one": float(profile.exp_p_one[exp_index]),
                            "exp_masked": int(profile.masked_exp[exp_index]),
                            "sm_active": int(sm_active),
                            "sm_index": sm_index,
                            "sm_label": profile.sm_labels[sm_index],
                            "sm_es_unscaled": float(profile.sm_es[sm_index]),
                            "sm_scale": profile.sm_scale,
                            "sm_es_scaled": float(sm_nominal),
                            "sm_p_one": float(profile.sm_p_one[sm_index]),
                            "sm_masked": int(profile.masked_sm[sm_index]),
                            "combined_nominal_es": float(exp_nominal + sm_nominal),
                        }
                    )
    pd.DataFrame(rows).to_csv(output_dir / "logical_mapping_audit.csv", index=False)


def run_self_check(
    geometry: Geometry,
    mapping: RTLMapping,
    profiles: Sequence[ESProfile],
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    fault_count = 300
    pages = rng.integers(0, min(4, geometry.num_pages), fault_count, dtype=np.int64)
    cells = rng.integers(0, geometry.page_bits, fault_count, dtype=np.int64)
    types = rng.integers(0, 3, fault_count, dtype=np.uint8)
    linear = pages * geometry.page_bits + cells
    _, first = np.unique(linear, return_index=True)
    faults = FaultSet(
        page=pages[first],
        cell=cells[first],
        fault_type=types[first],
        poisson_count=fault_count,
        unique_count=int(first.size),
    )
    decoded = decode_faults(faults, geometry)

    for profile in profiles:
        mapping.validate(geometry, profile.exp_count, profile.sm_count)
        page_es, page_exp_es, page_sm_es, _ = evaluate_all_modes_vectorized(
            decoded, geometry, mapping, profile
        )
        for mode in ("intra", "inter"):
            if np.any(page_es[mode] > page_es["no_remap"] + 1e-9):
                raise AssertionError(
                    f"Self-check failed for {profile.name}: {mode} worsened total ES."
                )
        if np.any(page_es["masked_intra"] > page_es["intra"] + 1e-9):
            raise AssertionError(
                f"Self-check failed for {profile.name}: masked intra > intra."
            )
        if np.any(page_es["masked_inter"] > page_es["inter"] + 1e-9):
            raise AssertionError(
                f"Self-check failed for {profile.name}: masked inter > inter."
            )
        for mode in page_es:
            if not np.allclose(page_es[mode], page_exp_es[mode] + page_sm_es[mode]):
                raise AssertionError(
                    f"Self-check failed for {profile.name}: component sum mismatch in {mode}."
                )

        paired_uniform = (
            mapping.field_assignment_kind == "paired"
            and np.allclose(profile.exp_es, profile.exp_es[0])
            and np.allclose(profile.sm_es, profile.sm_es[0])
            and np.allclose(profile.exp_p_one, profile.exp_p_one[0])
            and np.allclose(profile.sm_p_one, profile.sm_p_one[0])
        )
        if paired_uniform:
            if not np.allclose(page_es["intra"], page_es["no_remap"]):
                raise AssertionError(
                    "Uniform paired ES sanity profile should not benefit from intra."
                )
            if not np.allclose(page_es["inter"], page_es["no_remap"]):
                raise AssertionError(
                    "Uniform paired ES sanity profile should not benefit from inter."
                )

def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_experiment(config_path: Path, output_override: Optional[Path]) -> Path:
    config = load_config(config_path)
    base_dir = config_path.parent

    geometry_cfg = config["geometry"]
    geometry = Geometry(
        num_pages=int(geometry_cfg["num_pages"]),
        page_bytes=int(geometry_cfg["page_bytes"]),
        banks=int(geometry_cfg["banks"]),
        rgs=int(geometry_cfg["rgs"]),
        words=int(geometry_cfg["words"]),
        bits=int(geometry_cfg["bits"]),
    )
    geometry.validate()

    profiles = [build_profile(item) for item in config["es_profiles"]]
    if not profiles:
        raise ValueError("At least one ES profile is required.")

    mapping = build_rtl_mapping(config["rtl_mapping"], geometry, base_dir)

    seed = int(config.get("seed", 20260801))
    run_self_check(geometry, mapping, profiles, seed)

    output_dir = (
        output_override
        if output_override is not None
        else base_dir / config.get("output_dir", "remap_monte_carlo_rtl_output")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    export_mapping_audit(output_dir, geometry, mapping, profiles)

    mode_order = list(config.get("mode_order", DEFAULT_MODE_ORDER))
    mode_costs = {
        str(name): float(value)
        for name, value in config.get(
            "mode_costs",
            {
                "no_remap": 0.0,
                "masked_intra": 1.0,
                "intra": 2.0,
                "masked_inter": 3.0,
                "inter": 4.0,
            },
        ).items()
    }
    selection_cfg = config.get(
        "selection",
        {
            "absolute_es_limit": None,
            "max_relative_es": 0.5,
            "include_no_remap_candidate": False,
            "required_feasible_rate": 0.95,
        },
    )

    ber_values = [float(value) for value in config["ber_values"]]
    repeats = int(config.get("repeats", 20))
    sample_pages_per_trial = int(config.get("sample_pages_per_trial", 200))
    max_faults = config.get("max_faults_per_trial")
    max_faults = None if max_faults is None else int(max_faults)

    fault_scenarios = config["fault_scenarios"]
    spatial_distributions = config["spatial_distributions"]

    condition_count = (
        len(fault_scenarios)
        * len(spatial_distributions)
        * len(ber_values)
        * repeats
    )
    child_seeds = np.random.SeedSequence(seed).spawn(condition_count)
    seed_index = 0

    trial_rows: List[Dict[str, Any]] = []
    page_rows: List[Dict[str, Any]] = []

    for scenario_cfg in fault_scenarios:
        scenario_name = str(scenario_cfg["name"])
        fault_mix = scenario_cfg["fault_mix"]

        for spatial_cfg_full in spatial_distributions:
            spatial_name = str(spatial_cfg_full["name"])
            spatial_cfg = dict(spatial_cfg_full)
            spatial_cfg.pop("name", None)

            for ber in ber_values:
                for repeat in range(repeats):
                    child_seed = child_seeds[seed_index]
                    seed_index += 1
                    trial_rng = np.random.default_rng(child_seed)
                    sample_rng = np.random.default_rng(child_seed.spawn(1)[0])

                    faults = sample_faults(
                        trial_rng,
                        geometry,
                        ber,
                        fault_mix,
                        spatial_cfg,
                        max_faults_per_trial=max_faults,
                    )
                    decoded = decode_faults(faults, geometry)

                    if decoded.faulty_page_count > 0 and sample_pages_per_trial > 0:
                        sampled_page_indices = np.sort(
                            sample_rng.choice(
                                decoded.faulty_page_count,
                                size=min(
                                    sample_pages_per_trial,
                                    decoded.faulty_page_count,
                                ),
                                replace=False,
                            ).astype(np.int64)
                        )
                    else:
                        sampled_page_indices = np.empty(0, dtype=np.int64)

                    seed_value = int(child_seed.generate_state(1)[0])

                    for profile in profiles:
                        mapping.validate(geometry, profile.exp_count, profile.sm_count)
                        summary, sampled_rows = evaluate_trial_profile(
                            decoded=decoded,
                            faults=faults,
                            geometry=geometry,
                            mapping=mapping,
                            profile=profile,
                            selection_cfg=selection_cfg,
                            mode_costs=mode_costs,
                            mode_order=mode_order,
                            sampled_page_indices=sampled_page_indices,
                        )
                        trial_rows.append(
                            add_metadata(
                                summary,
                                scenario_name,
                                spatial_name,
                                ber,
                                repeat,
                                seed_value,
                            )
                        )
                        for row in sampled_rows:
                            page_rows.append(
                                add_metadata(
                                    row,
                                    scenario_name,
                                    spatial_name,
                                    ber,
                                    repeat,
                                    seed_value,
                                )
                            )

                    print(
                        f"[done] scenario={scenario_name}, spatial={spatial_name}, "
                        f"BER={ber:.3e}, repeat={repeat + 1}/{repeats}, "
                        f"faults={faults.unique_count:,}, "
                        f"faulty_pages={decoded.faulty_page_count:,}"
                    )

    trial_df = pd.DataFrame(trial_rows)
    page_df = pd.DataFrame(page_rows)
    trial_df.to_csv(output_dir / "trial_results.csv", index=False)
    page_df.to_csv(output_dir / "sampled_page_results.csv", index=False)

    summary_df = summarize_trials(trial_df)
    summary_df.to_csv(output_dir / "summary.csv", index=False)

    required_rate = float(selection_cfg.get("required_feasible_rate", 0.95))
    recommendations = recommend_modes_by_condition(
        summary_df,
        mode_costs,
        mode_order,
        required_rate,
    )
    recommendations.to_csv(
        output_dir / "recommended_mode_by_ber.csv", index=False
    )
    compress_mode_regions(recommendations).to_csv(
        output_dir / "mode_regions.csv", index=False
    )

    train_threshold_selector(page_df, output_dir, seed)
    make_plots(summary_df, mode_order, output_dir)

    metadata = {
        "config_path": str(config_path.resolve()),
        "geometry": geometry_cfg,
        "rtl_fault_bitmap_bits": geometry.cells_per_bitmap_block,
        "bitmap_blocks_per_16KB_page": geometry.bitmap_blocks_per_page,
        "total_bits": geometry.total_bits,
        "inter_scope": mapping.inter_scope,
        "intra_scope": mapping.intra_scope,
        "inter_shift_direction": mapping.inter_shift_direction,
        "field_assignment_kind": mapping.field_assignment_kind,
        "profiles": [profile.name for profile in profiles],
        "mode_order": mode_order,
        "mode_costs": mode_costs,
        "selection": selection_cfg,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RTL-aligned Monte Carlo ES experiment for remap selection."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("remap_experiment_rtl_config.json"),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--accuracy-csv",
        type=Path,
        default=None,
        help="CSV columns: profile, mode, es, accuracy[, baseline_accuracy].",
    )
    parser.add_argument("--max-accuracy-drop", type=float, default=1.0)
    parser.add_argument("--max-false-safe-rate", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = run_experiment(args.config, args.output)
    if args.accuracy_csv is not None:
        analyze_accuracy_calibration(
            args.accuracy_csv,
            output_dir,
            args.max_accuracy_drop,
            args.max_false_safe_rate,
        )
    print(f"\nExperiment complete. Results: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
