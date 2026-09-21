#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Global Base/Inter/Intra accuracy-constrained remap experiment.

Purpose
-------
This program implements the offline experiment discussed for AlexNet + CIFAR-10:

1. Only the storage occupied by model parameters participates in BER sampling.
2. One *global mode* is applied to the whole occupied model image:
       Base  : EXP masking only; S&M no remap
       Inter : EXP masking + one best 3-bit shift per FBB
       Intra : EXP masking + one best 3-bit CW per (FBB, bank, RG)
3. Control values remain local exactly as before; global mode only selects which
   remap family is used by every FBB.
4. For every BER / SAF mix / spatial distribution / repeat, the same fault map
   is evaluated under Base, Inter, and Intra.
5. Accuracy requirements are expressed as a retention ratio of the fault-free
   codec baseline, e.g. 0.95 / 0.97 / 0.98.
6. The lowest-cost feasible mode follows Base > Inter > Intra.
7. BER constraint tables are generated from mode pass rates.

Important modeling boundary
---------------------------
The included codec is an explicit B=4 block floating-point storage model:
one shared 8-bit exponent byte and four 8-bit sign/magnitude bytes. It creates
exactly the continuous FBB pattern EXP, SM0, SM1, SM2, SM3. If the final project
uses a different MSFP encoder, replace only BlockMSFCodec; fault generation,
remap control search, global-mode evaluation, and reporting can remain intact.

The remap's logical-bit effect intentionally matches the earlier Monte Carlo
abstraction:
    baseline logical bit = physical bit
    intra    logical bit = physical bit XOR CW
    inter    logical bit = circularly rotated physical bit
Physical faults never move.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import random
import re
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.nn.utils import parameters_to_vector
    from torch.utils.data import DataLoader, Subset
    import torchvision
    import torchvision.transforms as transforms
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "PyTorch and torchvision are required. Install them before running this program."
    ) from exc

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


MODE_ORDER = ("base", "inter", "intra")
MODE_RANK = {"base": 0, "inter": 1, "intra": 2, "unrepairable": 3}
RANK_MODE = {value: key for key, value in MODE_RANK.items()}
FAULT_NAME_TO_ID = {"SA0": 0, "SA1": 1}
FAULT_ID_TO_NAME = {0: "SA0", 1: "SA1"}
FIELD_EXP = 0
FIELD_SM = 1
FIELD_NAME = {FIELD_EXP: "EXP", FIELD_SM: "SM"}


# ---------------------------------------------------------------------------
# Basic configuration and reproducibility
# ---------------------------------------------------------------------------


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, obj: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    requested = str(requested).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def deep_update(base: MutableMapping[str, Any], extra: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for key, value in extra.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), MutableMapping):
            deep_update(base[key], value)  # type: ignore[index]
        else:
            base[key] = value
    return base


# ---------------------------------------------------------------------------
# CIFAR AlexNet model support
# ---------------------------------------------------------------------------


class CIFARAlexNet(nn.Module):
    """AlexNet variant adapted to 32x32 CIFAR-10 inputs."""

    def __init__(self, num_classes: int = 10, batch_norm: bool = False):
        super().__init__()
        feature_layers: List[nn.Module] = [
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
        ]
        if batch_norm:
            feature_layers.append(nn.BatchNorm2d(64))
        feature_layers += [
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
        ]
        if batch_norm:
            feature_layers.append(nn.BatchNorm2d(192))
        feature_layers += [
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
        ]
        if batch_norm:
            feature_layers.append(nn.BatchNorm2d(384))
        feature_layers += [
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
        ]
        if batch_norm:
            feature_layers.append(nn.BatchNorm2d(256))
        feature_layers += [
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
        ]
        if batch_norm:
            feature_layers.append(nn.BatchNorm2d(256))
        feature_layers += [
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ]
        self.features = nn.Sequential(*feature_layers)
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)


def strip_state_dict_prefixes(state: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    prefixes = ("module.", "model.", "net.")
    output: Dict[str, torch.Tensor] = {}
    for key, value in state.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True
        output[new_key] = value
    return output


def extract_state_dict(checkpoint: Any, explicit_key: str = "auto") -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, nn.Module):
        return strip_state_dict_prefixes(checkpoint.state_dict())
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint must be a state_dict, a checkpoint mapping, or an nn.Module.")

    if explicit_key != "auto":
        if explicit_key not in checkpoint:
            raise KeyError(f"state_dict_key={explicit_key!r} does not exist in checkpoint.")
        candidate = checkpoint[explicit_key]
        if not isinstance(candidate, Mapping):
            raise TypeError(f"checkpoint[{explicit_key!r}] is not a mapping.")
        return strip_state_dict_prefixes(candidate)  # type: ignore[arg-type]

    common_keys = ("state_dict", "model_state_dict", "net", "model", "weights")
    for key in common_keys:
        candidate = checkpoint.get(key)
        if isinstance(candidate, Mapping) and candidate:
            if all(torch.is_tensor(value) for value in candidate.values()):
                return strip_state_dict_prefixes(candidate)  # type: ignore[arg-type]

    if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        return strip_state_dict_prefixes(checkpoint)  # type: ignore[arg-type]

    raise ValueError("Could not locate a tensor state_dict inside the checkpoint.")


def infer_architecture(state: Mapping[str, torch.Tensor]) -> str:
    keys = set(state.keys())
    has_bn = any("running_mean" in key for key in keys)

    # torchvision AlexNet uses classifier.6 for the final layer and its first
    # convolution is normally 64x3x11x11.
    if "classifier.6.weight" in keys and "features.0.weight" in keys:
        first = state["features.0.weight"]
        if first.ndim == 4 and tuple(first.shape[2:]) == (11, 11):
            return "torchvision_alexnet"
        return "cifar_alexnet_bn" if has_bn else "cifar_alexnet"

    # Common CIFAR AlexNet checkpoints use linear.* or classifier.* names.
    if any(key.startswith("linear.") for key in keys):
        return "cifar_alexnet_linear"
    if "classifier.weight" in keys:
        return "cifar_alexnet_single_fc"

    raise ValueError(
        "Could not infer AlexNet architecture from checkpoint keys. Set "
        "model.architecture explicitly or provide model.custom_factory."
    )


def import_factory(spec: str):
    if ":" not in spec:
        raise ValueError("custom_factory must use 'module:function' syntax.")
    module_name, function_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, function_name)
    if not callable(factory):
        raise TypeError(f"{spec!r} is not callable.")
    return factory


def build_model(model_cfg: Mapping[str, Any], state: Mapping[str, torch.Tensor]) -> nn.Module:
    num_classes = int(model_cfg.get("num_classes", 10))
    custom_factory = model_cfg.get("custom_factory")
    if custom_factory:
        factory = import_factory(str(custom_factory))
        kwargs = dict(model_cfg.get("custom_factory_kwargs", {}))
        model = factory(**kwargs)
        if not isinstance(model, nn.Module):
            raise TypeError("custom_factory did not return nn.Module.")
        return model

    architecture = str(model_cfg.get("architecture", "auto"))
    if architecture == "auto":
        architecture = infer_architecture(state)
        print(f"[model] Auto-detected architecture: {architecture}")

    if architecture == "cifar_alexnet":
        return CIFARAlexNet(num_classes=num_classes, batch_norm=False)
    if architecture == "cifar_alexnet_bn":
        return CIFARAlexNet(num_classes=num_classes, batch_norm=True)
    if architecture == "torchvision_alexnet":
        return torchvision.models.alexnet(weights=None, num_classes=num_classes)

    if architecture in {"cifar_alexnet_linear", "cifar_alexnet_single_fc"}:
        raise ValueError(
            f"architecture={architecture!r} was detected, but its exact classifier layout "
            "cannot be reconstructed safely from names alone. Set model.custom_factory "
            "to the Python factory used to train this checkpoint."
        )

    raise ValueError(f"Unsupported model.architecture={architecture!r}.")


def load_model_and_checkpoint(model_cfg: Mapping[str, Any], device: torch.device) -> Tuple[nn.Module, Dict[str, Any]]:
    checkpoint_path = Path(str(model_cfg["checkpoint_path"])).expanduser()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = extract_state_dict(checkpoint, explicit_key=str(model_cfg.get("state_dict_key", "auto")))
    model = build_model(model_cfg, state)
    strict = bool(model_cfg.get("strict_load", True))
    incompatible = model.load_state_dict(state, strict=strict)
    if not strict:
        if incompatible.missing_keys:
            print(f"[model] Missing keys ({len(incompatible.missing_keys)}): {incompatible.missing_keys[:12]}")
        if incompatible.unexpected_keys:
            print(f"[model] Unexpected keys ({len(incompatible.unexpected_keys)}): {incompatible.unexpected_keys[:12]}")
    model.to(device)
    model.eval()
    metadata = {
        "checkpoint_path": str(checkpoint_path.resolve()),
        "state_dict_tensor_count": len(state),
        "strict_load": strict,
    }
    return model, metadata


def build_test_loader(dataset_cfg: Mapping[str, Any], device: torch.device) -> DataLoader:
    mean = tuple(float(x) for x in dataset_cfg.get("mean", [0.4914, 0.4822, 0.4465]))
    std = tuple(float(x) for x in dataset_cfg.get("std", [0.2470, 0.2435, 0.2616]))
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    root = str(Path(str(dataset_cfg.get("root", "./data"))).expanduser())
    dataset = torchvision.datasets.CIFAR10(
        root=root,
        train=False,
        download=bool(dataset_cfg.get("download", True)),
        transform=transform,
    )
    max_samples = dataset_cfg.get("max_eval_samples")
    if max_samples is not None:
        max_samples = int(max_samples)
        if max_samples <= 0:
            raise ValueError("dataset.max_eval_samples must be positive or null.")
        dataset = Subset(dataset, list(range(min(max_samples, len(dataset)))))
    return DataLoader(
        dataset,
        batch_size=int(dataset_cfg.get("batch_size", 256)),
        shuffle=False,
        num_workers=int(dataset_cfg.get("num_workers", 2)),
        pin_memory=(device.type == "cuda"),
        persistent_workers=bool(dataset_cfg.get("persistent_workers", False))
        and int(dataset_cfg.get("num_workers", 2)) > 0,
    )


@torch.inference_mode()
def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(inputs)
        predictions = logits.argmax(dim=1)
        correct += int((predictions == targets).sum().item())
        total += int(targets.numel())
    if total == 0:
        raise RuntimeError("Evaluation loader is empty.")
    return correct / total


# ---------------------------------------------------------------------------
# Parameter flattening and restoration
# ---------------------------------------------------------------------------


@dataclass
class ParameterSegment:
    name: str
    parameter: nn.Parameter
    start: int
    end: int
    shape: Tuple[int, ...]


@dataclass
class ParameterImage:
    segments: List[ParameterSegment]
    original_vector: torch.Tensor

    @property
    def count(self) -> int:
        return int(self.original_vector.numel())

    def assign(self, flat_cpu: np.ndarray, device: torch.device) -> None:
        if flat_cpu.ndim != 1 or flat_cpu.size != self.count:
            raise ValueError(f"Expected flat vector of length {self.count}, got {flat_cpu.shape}.")
        source = torch.from_numpy(np.asarray(flat_cpu, dtype=np.float32))
        with torch.no_grad():
            for segment in self.segments:
                tensor = source[segment.start:segment.end].reshape(segment.shape)
                segment.parameter.copy_(tensor.to(device=device, dtype=segment.parameter.dtype))

    def restore(self, device: torch.device) -> None:
        self.assign(self.original_vector.numpy(), device)


def build_parameter_image(model: nn.Module, model_cfg: Mapping[str, Any]) -> ParameterImage:
    includes = [re.compile(pattern) for pattern in model_cfg.get("parameter_include_regex", [".*"])]
    excludes = [re.compile(pattern) for pattern in model_cfg.get("parameter_exclude_regex", [])]
    segments: List[ParameterSegment] = []
    chunks: List[torch.Tensor] = []
    offset = 0
    for name, parameter in model.named_parameters():
        if not parameter.is_floating_point():
            continue
        if not any(pattern.search(name) for pattern in includes):
            continue
        if any(pattern.search(name) for pattern in excludes):
            continue
        flat = parameter.detach().cpu().float().reshape(-1)
        start = offset
        end = start + int(flat.numel())
        segments.append(ParameterSegment(name, parameter, start, end, tuple(parameter.shape)))
        chunks.append(flat)
        offset = end
    if not segments:
        raise ValueError("No model parameters matched the include/exclude rules.")
    vector = torch.cat(chunks).contiguous()
    print(f"[storage] Selected {len(segments)} parameter tensors, {vector.numel():,} scalar weights.")
    return ParameterImage(segments=segments, original_vector=vector)


# ---------------------------------------------------------------------------
# B=4 block floating-point codec and occupied FBB geometry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FBBGeometry:
    banks: int = 8
    rgs: int = 8
    words: int = 8
    bits: int = 8

    @property
    def bytes_per_fbb(self) -> int:
        return self.banks * self.rgs * self.words

    @property
    def bits_per_fbb(self) -> int:
        return self.bytes_per_fbb * self.bits

    def validate(self) -> None:
        if (self.banks, self.rgs, self.words, self.bits) != (8, 8, 8, 8):
            warnings.warn("This program is validated for 8x8x8x8 FBB geometry.")
        if self.bytes_per_fbb != 512:
            raise ValueError("Expected 512 bytes per FBB for the current project geometry.")


@dataclass
class PackedWeights:
    bytes_by_fbb: np.ndarray       # uint8 [num_fbb, 512]
    valid_bytes_per_fbb: np.ndarray  # int64 [num_fbb]
    fbb_field: np.ndarray          # uint8 [num_fbb]
    weight_count: int
    block_count: int
    bounding_box_size: int
    exponent_bias: int
    mantissa_scale: float
    original_shape: Tuple[int, ...]

    @property
    def num_fbb(self) -> int:
        return int(self.bytes_by_fbb.shape[0])

    @property
    def total_valid_bytes(self) -> int:
        return int(self.valid_bytes_per_fbb.sum())

    @property
    def total_valid_bits(self) -> int:
        return self.total_valid_bytes * 8

    @property
    def occupied_bytes_including_padding(self) -> int:
        return int(self.bytes_by_fbb.size)


class BlockMSFCodec:
    """
    B-element block floating-point codec.

    For each B-weight block:
      - shared signed binary exponent e is stored as an unsigned byte e+bias;
      - each weight stores sign in bit7 and a 7-bit magnitude q;
      - q = round(|w| / 2**e * mantissa_scale), clamped to 0..127;
      - decoded w = sign * q / mantissa_scale * 2**e.

    With mantissa_scale=64 and e=floor(log2(max_abs)), q remains below 128.
    exponent_bias=64 keeps normal exponent bytes in [0,127], so EXP-MSB masking
    does not alter a fault-free encoded model under the expected exponent range.
    """

    def __init__(self, cfg: Mapping[str, Any], geometry: FBBGeometry):
        self.bounding_box_size = int(cfg.get("bounding_box_size", 4))
        self.exponent_bias = int(cfg.get("exponent_bias", 64))
        self.exponent_min = int(cfg.get("exponent_min", -64))
        self.exponent_max = int(cfg.get("exponent_max", 63))
        self.mantissa_scale = float(cfg.get("mantissa_scale", 64.0))
        self.geometry = geometry
        if self.bounding_box_size <= 0:
            raise ValueError("codec.bounding_box_size must be positive.")
        if self.exponent_min + self.exponent_bias < 0:
            raise ValueError("Exponent minimum and bias would underflow uint8.")
        if self.exponent_max + self.exponent_bias > 127:
            raise ValueError(
                "Exponent maximum + bias must be <=127 so fault-free EXP MSB remains zero."
            )
        if self.mantissa_scale <= 0:
            raise ValueError("codec.mantissa_scale must be positive.")

    def encode(self, values: np.ndarray) -> PackedWeights:
        flat = np.asarray(values, dtype=np.float32).reshape(-1)
        n = int(flat.size)
        b = self.bounding_box_size
        blocks = int(math.ceil(n / b))
        padded = np.zeros(blocks * b, dtype=np.float32)
        padded[:n] = flat
        matrix = padded.reshape(blocks, b)

        max_abs = np.max(np.abs(matrix), axis=1)
        exponent = np.zeros(blocks, dtype=np.int16)
        nonzero = max_abs > 0
        exponent[nonzero] = np.floor(np.log2(max_abs[nonzero])).astype(np.int16)
        exponent = np.clip(exponent, self.exponent_min, self.exponent_max)
        scale = np.exp2(exponent.astype(np.float64))[:, None]
        magnitude = np.rint(np.abs(matrix.astype(np.float64)) / scale * self.mantissa_scale)
        magnitude = np.clip(magnitude, 0, 127).astype(np.uint8)
        sign = (matrix < 0).astype(np.uint8) << 7
        sm = sign | magnitude
        exp_byte = (exponent + self.exponent_bias).astype(np.uint8)

        bytes_per_fbb = self.geometry.bytes_per_fbb
        periods = int(math.ceil(blocks / bytes_per_fbb))
        fbb_per_period = 1 + b
        num_fbb = periods * fbb_per_period
        storage = np.zeros((num_fbb, bytes_per_fbb), dtype=np.uint8)
        valid = np.zeros(num_fbb, dtype=np.int64)
        field = np.full(num_fbb, FIELD_SM, dtype=np.uint8)

        for period in range(periods):
            start = period * bytes_per_fbb
            end = min(start + bytes_per_fbb, blocks)
            count = end - start
            base_fbb = period * fbb_per_period
            field[base_fbb] = FIELD_EXP
            storage[base_fbb, :count] = exp_byte[start:end]
            valid[base_fbb] = count
            for lane in range(b):
                fbb = base_fbb + 1 + lane
                storage[fbb, :count] = sm[start:end, lane]
                lane_valid = np.arange(start, end, dtype=np.int64) * b + lane < n
                valid[fbb] = int(lane_valid.sum())
                if valid[fbb] < count:
                    storage[fbb, valid[fbb]:count] = 0

        return PackedWeights(
            bytes_by_fbb=storage,
            valid_bytes_per_fbb=valid,
            fbb_field=field,
            weight_count=n,
            block_count=blocks,
            bounding_box_size=b,
            exponent_bias=self.exponent_bias,
            mantissa_scale=self.mantissa_scale,
            original_shape=tuple(values.shape),
        )

    def decode(self, packed: PackedWeights, bytes_by_fbb: np.ndarray) -> np.ndarray:
        storage = np.asarray(bytes_by_fbb, dtype=np.uint8)
        if storage.shape != packed.bytes_by_fbb.shape:
            raise ValueError("Corrupted storage shape does not match packed storage shape.")
        b = packed.bounding_box_size
        bytes_per_fbb = self.geometry.bytes_per_fbb
        periods = packed.num_fbb // (1 + b)
        exp_bytes = np.empty(packed.block_count, dtype=np.uint8)
        sm = np.empty((packed.block_count, b), dtype=np.uint8)
        for period in range(periods):
            start = period * bytes_per_fbb
            end = min(start + bytes_per_fbb, packed.block_count)
            count = end - start
            base_fbb = period * (1 + b)
            exp_bytes[start:end] = storage[base_fbb, :count]
            for lane in range(b):
                sm[start:end, lane] = storage[base_fbb + 1 + lane, :count]
        exponent = exp_bytes.astype(np.int16) - packed.exponent_bias
        magnitude = (sm & 0x7F).astype(np.float64)
        sign = np.where((sm & 0x80) != 0, -1.0, 1.0)
        values = sign * magnitude / packed.mantissa_scale * np.exp2(exponent.astype(np.float64))[:, None]
        return values.reshape(-1)[:packed.weight_count].astype(np.float32).reshape(packed.original_shape)


# ---------------------------------------------------------------------------
# SAF generation over valid model storage only
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FaultMap:
    fbb: np.ndarray
    byte: np.ndarray
    bit: np.ndarray       # logical indexing convention: 0=MSB, 7=LSB
    fault_type: np.ndarray
    sampled_event_count: int
    unique_fault_count: int

    def __len__(self) -> int:
        return self.unique_fault_count


@dataclass
class StorageSampler:
    packed: PackedWeights
    geometry: FBBGeometry

    def __post_init__(self) -> None:
        counts = self.packed.valid_bytes_per_fbb.astype(np.int64)
        self._counts = counts
        self._cumulative = np.cumsum(counts)
        self._nonempty = np.flatnonzero(counts > 0)
        if self._cumulative.size == 0 or int(self._cumulative[-1]) <= 0:
            raise ValueError("Packed model contains no valid bytes.")

    @property
    def total_valid_bytes(self) -> int:
        return int(self._cumulative[-1])

    def sample_fbb_weighted(self, rng: np.random.Generator, size: int, candidates: Optional[np.ndarray] = None) -> np.ndarray:
        if candidates is None:
            slots = rng.integers(0, self.total_valid_bytes, size=size, dtype=np.int64)
            return np.searchsorted(self._cumulative, slots, side="right").astype(np.int64)
        candidates = np.asarray(candidates, dtype=np.int64)
        weights = self._counts[candidates].astype(np.float64)
        if weights.sum() <= 0:
            raise ValueError("Candidate FBBs have no valid bytes.")
        return rng.choice(candidates, size=size, replace=True, p=weights / weights.sum()).astype(np.int64)

    def sample_byte_in_fbb(self, rng: np.random.Generator, fbb: int, bank: Optional[int] = None) -> int:
        count = int(self._counts[fbb])
        if bank is None:
            return int(rng.integers(0, count))
        bank_start = bank * self.geometry.rgs * self.geometry.words
        bank_end = min(bank_start + self.geometry.rgs * self.geometry.words, count)
        if bank_end <= bank_start:
            return int(rng.integers(0, count))
        return int(rng.integers(bank_start, bank_end))


def normalize_fault_mix(mix: Mapping[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    ids: List[int] = []
    probs: List[float] = []
    for name in ("SA0", "SA1"):
        probability = float(mix.get(name, 0.0))
        if probability < 0:
            raise ValueError("Fault probabilities must be nonnegative.")
        if probability > 0:
            ids.append(FAULT_NAME_TO_ID[name])
            probs.append(probability)
    if not probs:
        raise ValueError("fault_mix must contain a positive SA0 or SA1 probability.")
    p = np.asarray(probs, dtype=np.float64)
    p /= p.sum()
    return np.asarray(ids, dtype=np.uint8), p


def empty_fault_map() -> FaultMap:
    empty_i = np.empty(0, dtype=np.int64)
    empty_u = np.empty(0, dtype=np.uint8)
    return FaultMap(empty_i, empty_i.copy(), empty_i.copy(), empty_u, 0, 0)


def deduplicate_faults(faults: FaultMap, geometry: FBBGeometry) -> FaultMap:
    if faults.sampled_event_count == 0:
        return faults
    linear = (faults.fbb * geometry.bytes_per_fbb + faults.byte) * geometry.bits + faults.bit
    _, first = np.unique(linear, return_index=True)
    first.sort()
    return FaultMap(
        fbb=faults.fbb[first],
        byte=faults.byte[first],
        bit=faults.bit[first],
        fault_type=faults.fault_type[first],
        sampled_event_count=faults.sampled_event_count,
        unique_fault_count=int(first.size),
    )


def sample_faults(
    rng: np.random.Generator,
    packed: PackedWeights,
    geometry: FBBGeometry,
    ber: float,
    fault_mix: Mapping[str, float],
    spatial_cfg: Mapping[str, Any],
    max_faults: Optional[int],
) -> FaultMap:
    if ber < 0:
        raise ValueError("BER must be nonnegative.")
    sampled_count = int(rng.poisson(packed.total_valid_bits * ber))
    if sampled_count == 0:
        return empty_fault_map()
    if max_faults is not None and sampled_count > max_faults:
        raise RuntimeError(
            f"Sampled {sampled_count:,} SAF events over the occupied model storage, "
            f"exceeding max_faults_per_trial={max_faults:,}."
        )

    sampler = StorageSampler(packed, geometry)
    kind = str(spatial_cfg.get("kind", "uniform"))
    fbb = np.empty(sampled_count, dtype=np.int64)
    byte = np.empty(sampled_count, dtype=np.int64)

    if kind == "uniform":
        fbb[:] = sampler.sample_fbb_weighted(rng, sampled_count)
        for index in range(sampled_count):
            byte[index] = sampler.sample_byte_in_fbb(rng, int(fbb[index]))

    elif kind == "fbb_bank_clustered":
        hot_fbb_fraction = float(spatial_cfg.get("hot_fbb_fraction", 0.01))
        hot_fault_fraction = float(spatial_cfg.get("hot_fault_fraction", 0.80))
        hot_bank_fraction = float(spatial_cfg.get("hot_bank_fraction", 0.875))
        if not 0 < hot_fbb_fraction <= 1:
            raise ValueError("hot_fbb_fraction must be in (0,1].")
        if not 0 <= hot_fault_fraction <= 1 or not 0 <= hot_bank_fraction <= 1:
            raise ValueError("hot_fault_fraction and hot_bank_fraction must be in [0,1].")
        nonempty = np.flatnonzero(packed.valid_bytes_per_fbb > 0)
        hot_count = max(1, int(round(nonempty.size * hot_fbb_fraction)))
        hot_count = min(hot_count, nonempty.size)
        hot_fbbs = rng.choice(nonempty, size=hot_count, replace=False).astype(np.int64)
        hot_bank_for_fbb: Dict[int, int] = {}
        for hot_fbb in hot_fbbs:
            valid_count = int(packed.valid_bytes_per_fbb[hot_fbb])
            available_banks = max(1, int(math.ceil(valid_count / (geometry.rgs * geometry.words))))
            hot_bank_for_fbb[int(hot_fbb)] = int(rng.integers(0, min(available_banks, geometry.banks)))

        choose_hot = rng.random(sampled_count) < hot_fault_fraction
        hot_n = int(choose_hot.sum())
        if hot_n:
            fbb[choose_hot] = sampler.sample_fbb_weighted(rng, hot_n, candidates=hot_fbbs)
        if hot_n < sampled_count:
            fbb[~choose_hot] = sampler.sample_fbb_weighted(rng, sampled_count - hot_n)
        choose_hot_bank = choose_hot & (rng.random(sampled_count) < hot_bank_fraction)
        for index in range(sampled_count):
            current_fbb = int(fbb[index])
            bank = hot_bank_for_fbb.get(current_fbb) if choose_hot_bank[index] else None
            byte[index] = sampler.sample_byte_in_fbb(rng, current_fbb, bank=bank)
    else:
        raise ValueError(f"Unknown spatial distribution kind={kind!r}.")

    bit = rng.integers(0, geometry.bits, size=sampled_count, dtype=np.int64)
    type_ids, type_probs = normalize_fault_mix(fault_mix)
    fault_type = rng.choice(type_ids, size=sampled_count, p=type_probs).astype(np.uint8)
    return deduplicate_faults(
        FaultMap(fbb, byte, bit, fault_type, sampled_count, sampled_count), geometry
    )


# ---------------------------------------------------------------------------
# Remap control search and fault application
# ---------------------------------------------------------------------------


@dataclass
class RemapControls:
    best_inter_shift: np.ndarray   # [num_fbb]
    best_intra_cw: np.ndarray      # [num_fbb, banks, rgs]
    baseline_es_total: float
    inter_es_total: float
    intra_es_total: float


def bit_occupancy(packed: PackedWeights) -> Tuple[np.ndarray, np.ndarray]:
    exp_bytes: List[np.ndarray] = []
    sm_bytes: List[np.ndarray] = []
    for fbb in range(packed.num_fbb):
        count = int(packed.valid_bytes_per_fbb[fbb])
        if count == 0:
            continue
        values = packed.bytes_by_fbb[fbb, :count]
        if packed.fbb_field[fbb] == FIELD_EXP:
            exp_bytes.append(values)
        else:
            sm_bytes.append(values)

    def occupancy(chunks: List[np.ndarray]) -> np.ndarray:
        if not chunks:
            return np.full(8, 0.5, dtype=np.float64)
        values = np.concatenate(chunks)
        return np.asarray([np.mean((values & (1 << (7 - bit))) != 0) for bit in range(8)])

    return occupancy(exp_bytes), occupancy(sm_bytes)


def logical_index_inter(bit: np.ndarray, shift: int, direction: str) -> np.ndarray:
    if direction == "right":
        return (bit - int(shift)) % 8
    if direction == "left":
        return (bit + int(shift)) % 8
    raise ValueError("remap.inter_direction must be 'right' or 'left'.")


def expected_fault_cost(
    fields: np.ndarray,
    logical_index: np.ndarray,
    fault_type: np.ndarray,
    exp_es: np.ndarray,
    sm_es: np.ndarray,
    exp_p_one: np.ndarray,
    sm_p_one: np.ndarray,
) -> np.ndarray:
    is_exp = fields == FIELD_EXP
    base = np.where(is_exp, exp_es[logical_index], sm_es[logical_index]).astype(np.float64)
    p_one = np.where(is_exp, exp_p_one[logical_index], sm_p_one[logical_index])
    factor = np.where(fault_type == FAULT_NAME_TO_ID["SA0"], p_one, 1.0 - p_one)
    return base * factor


def search_controls(
    faults: FaultMap,
    packed: PackedWeights,
    geometry: FBBGeometry,
    remap_cfg: Mapping[str, Any],
    measured_exp_p_one: np.ndarray,
    measured_sm_p_one: np.ndarray,
) -> RemapControls:
    shifts = geometry.bits
    cws = geometry.words
    best_inter = np.zeros(packed.num_fbb, dtype=np.uint8)
    best_intra = np.zeros((packed.num_fbb, geometry.banks, geometry.rgs), dtype=np.uint8)
    if len(faults) == 0:
        return RemapControls(best_inter, best_intra, 0.0, 0.0, 0.0)

    exp_es = np.asarray(remap_cfg.get("exp_es", [31, 7, 7, 7, 7, 15, 15, 3]), dtype=np.float64)
    sm_es = np.asarray(remap_cfg.get("sm_es", [31, 15, 7, 3, 1, 1, 1, 1]), dtype=np.float64)
    if exp_es.shape != (8,) or sm_es.shape != (8,):
        raise ValueError("remap.exp_es and remap.sm_es must each contain 8 values.")
    exp_msb_index = int(remap_cfg.get("exp_msb_index", 0))
    exp_mask_factor = float(remap_cfg.get("exp_mask_factor", 0.0))
    exp_es = exp_es.copy()
    exp_es[exp_msb_index] *= exp_mask_factor

    p_source = str(remap_cfg.get("p_one_source", "measured"))
    if p_source == "measured":
        exp_p_one = measured_exp_p_one
        sm_p_one = measured_sm_p_one
    elif p_source == "config":
        exp_p_one = np.asarray(remap_cfg.get("exp_p_one", [0.5] * 8), dtype=np.float64)
        sm_p_one = np.asarray(remap_cfg.get("sm_p_one", [0.5] * 8), dtype=np.float64)
    else:
        raise ValueError("remap.p_one_source must be 'measured' or 'config'.")

    fields = packed.fbb_field[faults.fbb]
    baseline_index = faults.bit.astype(np.int64)
    baseline_costs = expected_fault_cost(
        fields, baseline_index, faults.fault_type, exp_es, sm_es, exp_p_one, sm_p_one
    )
    baseline_total = float(baseline_costs.sum())

    # One best shift independently per FBB.
    faulty_fbbs, fbb_inverse = np.unique(faults.fbb, return_inverse=True)
    inter_matrix = np.zeros((shifts, faulty_fbbs.size), dtype=np.float64)
    direction = str(remap_cfg.get("inter_direction", "right"))
    for shift in range(shifts):
        index = logical_index_inter(faults.bit, shift, direction)
        costs = expected_fault_cost(
            fields, index, faults.fault_type, exp_es, sm_es, exp_p_one, sm_p_one
        )
        inter_matrix[shift] = np.bincount(fbb_inverse, weights=costs, minlength=faulty_fbbs.size)
    selected_shift = np.argmin(inter_matrix, axis=0).astype(np.uint8)
    best_inter[faulty_fbbs] = selected_shift
    inter_total = float(inter_matrix[selected_shift, np.arange(faulty_fbbs.size)].sum())

    # One best CW independently per (FBB, bank, RG).
    bank = faults.byte // (geometry.rgs * geometry.words)
    within_bank = faults.byte % (geometry.rgs * geometry.words)
    rg = within_bank // geometry.words
    group_key = (faults.fbb * geometry.banks + bank) * geometry.rgs + rg
    unique_group, group_inverse = np.unique(group_key, return_inverse=True)
    intra_matrix = np.zeros((cws, unique_group.size), dtype=np.float64)
    for cw in range(cws):
        index = np.bitwise_xor(faults.bit.astype(np.int64), cw)
        costs = expected_fault_cost(
            fields, index, faults.fault_type, exp_es, sm_es, exp_p_one, sm_p_one
        )
        intra_matrix[cw] = np.bincount(group_inverse, weights=costs, minlength=unique_group.size)
    selected_cw = np.argmin(intra_matrix, axis=0).astype(np.uint8)
    group_rg = unique_group % geometry.rgs
    temp = unique_group // geometry.rgs
    group_bank = temp % geometry.banks
    group_fbb = temp // geometry.banks
    best_intra[group_fbb, group_bank, group_rg] = selected_cw
    intra_total = float(intra_matrix[selected_cw, np.arange(unique_group.size)].sum())

    return RemapControls(best_inter, best_intra, baseline_total, inter_total, intra_total)


def apply_fault_map(
    packed: PackedWeights,
    faults: FaultMap,
    controls: RemapControls,
    mode: str,
    geometry: FBBGeometry,
    remap_cfg: Mapping[str, Any],
) -> np.ndarray:
    if mode not in MODE_ORDER:
        raise ValueError(f"Unknown global mode {mode!r}.")
    corrupted = packed.bytes_by_fbb.copy()
    if len(faults):
        if mode == "base":
            logical_bit = faults.bit.astype(np.int64)
        elif mode == "inter":
            shift = controls.best_inter_shift[faults.fbb].astype(np.int64)
            direction = str(remap_cfg.get("inter_direction", "right"))
            if direction == "right":
                logical_bit = (faults.bit - shift) % geometry.bits
            elif direction == "left":
                logical_bit = (faults.bit + shift) % geometry.bits
            else:
                raise ValueError("remap.inter_direction must be 'right' or 'left'.")
        else:
            bank = faults.byte // (geometry.rgs * geometry.words)
            rg = (faults.byte % (geometry.rgs * geometry.words)) // geometry.words
            cw = controls.best_intra_cw[faults.fbb, bank, rg].astype(np.int64)
            logical_bit = np.bitwise_xor(faults.bit.astype(np.int64), cw)

        bit_mask = (1 << (7 - logical_bit)).astype(np.uint8)
        for index in range(len(faults)):
            fbb = int(faults.fbb[index])
            byte = int(faults.byte[index])
            mask = int(bit_mask[index])
            if int(faults.fault_type[index]) == FAULT_NAME_TO_ID["SA0"]:
                corrupted[fbb, byte] = np.uint8(int(corrupted[fbb, byte]) & (~mask & 0xFF))
            else:
                corrupted[fbb, byte] = np.uint8(int(corrupted[fbb, byte]) | mask)

    # EXP-MSB masking is always active for every global mode.
    exp_fbbs = np.flatnonzero(packed.fbb_field == FIELD_EXP)
    for fbb in exp_fbbs:
        count = int(packed.valid_bytes_per_fbb[fbb])
        if count:
            corrupted[fbb, :count] &= np.uint8(0x7F)
    return corrupted


def fault_statistics(faults: FaultMap, packed: PackedWeights, geometry: FBBGeometry) -> Dict[str, Any]:
    if len(faults) == 0:
        return {
            "sampled_fault_events": 0,
            "unique_faults": 0,
            "exp_faults": 0,
            "sm_faults": 0,
            "sa0_faults": 0,
            "sa1_faults": 0,
            "faulty_fbbs": 0,
            "faulty_banks": 0,
            "faulty_rgs": 0,
            "max_faults_per_fbb": 0,
            "max_faults_per_bank": 0,
            "max_faults_per_rg": 0,
            "exp_msb_faults": 0,
        }
    fields = packed.fbb_field[faults.fbb]
    bank = faults.byte // (geometry.rgs * geometry.words)
    rg = (faults.byte % (geometry.rgs * geometry.words)) // geometry.words
    fbb_counts = np.unique(faults.fbb, return_counts=True)[1]
    bank_key = faults.fbb * geometry.banks + bank
    bank_counts = np.unique(bank_key, return_counts=True)[1]
    rg_key = bank_key * geometry.rgs + rg
    rg_counts = np.unique(rg_key, return_counts=True)[1]
    return {
        "sampled_fault_events": int(faults.sampled_event_count),
        "unique_faults": int(faults.unique_fault_count),
        "exp_faults": int(np.sum(fields == FIELD_EXP)),
        "sm_faults": int(np.sum(fields == FIELD_SM)),
        "sa0_faults": int(np.sum(faults.fault_type == FAULT_NAME_TO_ID["SA0"])),
        "sa1_faults": int(np.sum(faults.fault_type == FAULT_NAME_TO_ID["SA1"])),
        "faulty_fbbs": int(np.unique(faults.fbb).size),
        "faulty_banks": int(np.unique(bank_key).size),
        "faulty_rgs": int(np.unique(rg_key).size),
        "max_faults_per_fbb": int(fbb_counts.max()),
        "max_faults_per_bank": int(bank_counts.max()),
        "max_faults_per_rg": int(rg_counts.max()),
        "exp_msb_faults": int(np.sum((fields == FIELD_EXP) & (faults.bit == 0))),
    }


# ---------------------------------------------------------------------------
# Statistics and constraint extraction
# ---------------------------------------------------------------------------


def wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> Tuple[float, float]:
    if trials <= 0:
        return float("nan"), float("nan")
    # 1.959963984540054 is the two-sided 95% normal quantile. The config keeps
    # confidence for output compatibility; 95% is the validated setting.
    if not math.isclose(confidence, 0.95):
        warnings.warn("wilson_interval currently uses z=1.95996 (95% CI).")
    z = 1.959963984540054
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def choose_lowest_cost_mode(pass_values: Mapping[str, float], required: float) -> str:
    for mode in MODE_ORDER:
        value = float(pass_values.get(mode, float("nan")))
        if not math.isnan(value) and value >= required:
            return mode
    return "unrepairable"


def lowest_cost_feasible_from_accuracy(accuracies: Mapping[str, float], threshold: float) -> str:
    for mode in MODE_ORDER:
        if float(accuracies[mode]) >= threshold:
            return mode
    return "unrepairable"


def summarize_trials(
    trial_df: pd.DataFrame,
    baseline_accuracy: float,
    target_cfg: Mapping[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ratios = [float(x) for x in target_cfg.get("retention_ratios", [0.95, 0.97, 0.98])]
    required_pass = float(target_cfg.get("required_trial_pass_rate", 0.95))
    confidence = float(target_cfg.get("confidence", 0.95))
    group_columns = ["ber", "fault_scenario", "spatial_distribution"]

    pass_rows: List[Dict[str, Any]] = []
    for keys, group in trial_df.groupby(group_columns, dropna=False):
        ber, scenario, spatial = keys
        for ratio in ratios:
            threshold = baseline_accuracy * ratio
            for mode in MODE_ORDER:
                values = group[f"accuracy_{mode}"].astype(float)
                successes = int((values >= threshold).sum())
                count = int(values.size)
                low, high = wilson_interval(successes, count, confidence)
                pass_rows.append({
                    "ber": float(ber),
                    "fault_scenario": scenario,
                    "spatial_distribution": spatial,
                    "target_retention": ratio,
                    "target_accuracy": threshold,
                    "mode": mode,
                    "trials": count,
                    "successes": successes,
                    "pass_rate": successes / count if count else float("nan"),
                    "wilson_low": low,
                    "wilson_high": high,
                    "accuracy_mean": float(values.mean()),
                    "accuracy_median": float(values.median()),
                    "accuracy_std": float(values.std(ddof=1)) if count > 1 else 0.0,
                    "accuracy_min": float(values.min()),
                    "accuracy_max": float(values.max()),
                })
    pass_df = pd.DataFrame(pass_rows)

    constraint_rows: List[Dict[str, Any]] = []
    for ratio in ratios:
        threshold = baseline_accuracy * ratio
        ratio_df = pass_df[pass_df["target_retention"] == ratio]
        for ber, ber_df in ratio_df.groupby("ber"):
            pooled_group = trial_df[trial_df["ber"] == ber]
            pooled: Dict[str, float] = {}
            pooled_wilson: Dict[str, float] = {}
            worst: Dict[str, float] = {}
            worst_wilson: Dict[str, float] = {}
            for mode in MODE_ORDER:
                values = pooled_group[f"accuracy_{mode}"].astype(float)
                successes = int((values >= threshold).sum())
                n = int(values.size)
                pooled[mode] = successes / n if n else float("nan")
                pooled_wilson[mode] = wilson_interval(successes, n, confidence)[0]
                condition_mode = ber_df[ber_df["mode"] == mode]
                worst[mode] = float(condition_mode["pass_rate"].min())
                worst_wilson[mode] = float(condition_mode["wilson_low"].min())
            constraint_rows.append({
                "target_retention": ratio,
                "target_accuracy": threshold,
                "ber": float(ber),
                **{f"pooled_pass_{m}": pooled[m] for m in MODE_ORDER},
                **{f"pooled_wilson_low_{m}": pooled_wilson[m] for m in MODE_ORDER},
                **{f"worst_case_pass_{m}": worst[m] for m in MODE_ORDER},
                **{f"worst_case_wilson_low_{m}": worst_wilson[m] for m in MODE_ORDER},
                "selected_pooled_observed": choose_lowest_cost_mode(pooled, required_pass),
                "selected_pooled_wilson": choose_lowest_cost_mode(pooled_wilson, required_pass),
                "selected_conservative_observed": choose_lowest_cost_mode(worst, required_pass),
                "selected_conservative_wilson": choose_lowest_cost_mode(worst_wilson, required_pass),
            })
    constraint_df = pd.DataFrame(constraint_rows).sort_values(["target_retention", "ber"])

    # Monotonic mode tables: as BER rises, do not allow a lower-cost downgrade.
    for column in [
        "selected_pooled_observed",
        "selected_pooled_wilson",
        "selected_conservative_observed",
        "selected_conservative_wilson",
    ]:
        output_column = f"{column}_monotonic"
        values: List[str] = []
        for _, group in constraint_df.groupby("target_retention", sort=False):
            running_rank = 0
            for mode in group[column]:
                running_rank = max(running_rank, MODE_RANK[str(mode)])
                values.append(RANK_MODE[running_rank])
        constraint_df[output_column] = values

    region_rows: List[Dict[str, Any]] = []
    selection_columns = [column for column in constraint_df.columns if column.endswith("_monotonic")]
    for ratio, ratio_df in constraint_df.groupby("target_retention"):
        ratio_df = ratio_df.sort_values("ber")
        for column in selection_columns:
            current_mode: Optional[str] = None
            start_ber: Optional[float] = None
            previous_ber: Optional[float] = None
            for _, row in ratio_df.iterrows():
                mode = str(row[column])
                ber = float(row["ber"])
                if current_mode is None:
                    current_mode, start_ber = mode, ber
                elif mode != current_mode:
                    region_rows.append({
                        "target_retention": float(ratio),
                        "selection_policy": column,
                        "mode": current_mode,
                        "ber_start_inclusive": start_ber,
                        "ber_end_inclusive": previous_ber,
                    })
                    current_mode, start_ber = mode, ber
                previous_ber = ber
            if current_mode is not None:
                region_rows.append({
                    "target_retention": float(ratio),
                    "selection_policy": column,
                    "mode": current_mode,
                    "ber_start_inclusive": start_ber,
                    "ber_end_inclusive": previous_ber,
                })
    region_df = pd.DataFrame(region_rows)
    return pass_df, constraint_df, region_df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_results(
    trial_df: pd.DataFrame,
    constraint_df: pd.DataFrame,
    baseline_accuracy: float,
    target_cfg: Mapping[str, Any],
    output_dir: Path,
) -> None:
    if plt is None:
        warnings.warn("matplotlib is unavailable; skipping plots.")
        return
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    pooled = trial_df.groupby("ber")[[f"accuracy_{m}" for m in MODE_ORDER]].agg(["mean", "std", "count"])
    bers = np.asarray(sorted(trial_df["ber"].unique()), dtype=np.float64)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for mode in MODE_ORDER:
        means = pooled[(f"accuracy_{mode}", "mean")].reindex(bers).to_numpy()
        std = pooled[(f"accuracy_{mode}", "std")].reindex(bers).fillna(0).to_numpy()
        count = pooled[(f"accuracy_{mode}", "count")].reindex(bers).to_numpy()
        ci = 1.96 * std / np.sqrt(np.maximum(count, 1))
        ax.errorbar(bers, means / baseline_accuracy, yerr=ci / baseline_accuracy, marker="o", label=mode)
    ax.set_xscale("log")
    ax.set_xlabel("BER over occupied model storage")
    ax.set_ylabel("Accuracy retention")
    ax.set_title("Global-mode accuracy retention vs BER")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    fig.tight_layout(rect=[0, 0, 0.84, 1])
    fig.savefig(plot_dir / "accuracy_retention_vs_ber.png", dpi=180)
    plt.close(fig)

    ratios = [float(x) for x in target_cfg.get("retention_ratios", [0.95, 0.97, 0.98])]
    for ratio in ratios:
        subset = constraint_df[constraint_df["target_retention"] == ratio].sort_values("ber")
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for mode in MODE_ORDER:
            ax.plot(subset["ber"], subset[f"pooled_pass_{mode}"], marker="o", label=mode)
        ax.axhline(float(target_cfg.get("required_trial_pass_rate", 0.95)), linestyle="--", label="required pass rate")
        ax.set_xscale("log")
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("BER over occupied model storage")
        ax.set_ylabel("Trial pass rate")
        ax.set_title(f"Mode pass rate, target retention={ratio:.3f}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
        fig.tight_layout(rect=[0, 0, 0.82, 1])
        fig.savefig(plot_dir / f"pass_rate_target_{ratio:.3f}.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 4.8))
        selection_column = "selected_conservative_observed_monotonic"
        rank = subset[selection_column].map(MODE_RANK).to_numpy()
        ax.step(subset["ber"], rank, where="mid", marker="o")
        ax.set_xscale("log")
        ax.set_yticks([0, 1, 2, 3], ["Base", "Inter", "Intra", "Unrepairable"])
        ax.set_xlabel("BER over occupied model storage")
        ax.set_ylabel("Recommended global mode")
        ax.set_title(f"Conservative monotonic BER constraint, target={ratio:.3f}")
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_dir / f"selected_mode_target_{ratio:.3f}.png", dpi=180)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def default_config() -> Dict[str, Any]:
    return {
        "seed": 20260801,
        "output_dir": "alexnet_global_mode_accuracy_output",
        "model": {
            "checkpoint_path": "CHANGE_ME_ALEXNET_CIFAR10.pth",
            "architecture": "auto",
            "num_classes": 10,
            "device": "auto",
            "state_dict_key": "auto",
            "strict_load": True,
            "custom_factory": None,
            "custom_factory_kwargs": {},
            "parameter_include_regex": [".*"],
            "parameter_exclude_regex": [],
        },
        "dataset": {
            "root": "./data",
            "download": True,
            "batch_size": 256,
            "num_workers": 2,
            "persistent_workers": False,
            "max_eval_samples": None,
            "mean": [0.4914, 0.4822, 0.4465],
            "std": [0.2470, 0.2435, 0.2616],
        },
        "geometry": {"banks": 8, "rgs": 8, "words": 8, "bits": 8},
        "codec": {
            "kind": "block_msf_bfp8",
            "bounding_box_size": 4,
            "exponent_bias": 64,
            "exponent_min": -64,
            "exponent_max": 63,
            "mantissa_scale": 64.0,
        },
        "remap": {
            "inter_direction": "right",
            "exp_es": [31, 7, 7, 7, 7, 15, 15, 3],
            "sm_es": [31, 15, 7, 3, 1, 1, 1, 1],
            "exp_msb_index": 0,
            "exp_mask_factor": 0.0,
            "p_one_source": "measured",
            "exp_p_one": [0.5] * 8,
            "sm_p_one": [0.5] * 8,
        },
        "offline_saf": {
            "ber_values": [1e-9, 3e-9, 1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5],
            "repeats": 20,
            "max_faults_per_trial": 250000,
            "fault_scenarios": [
                {"name": "saf_balanced", "fault_mix": {"SA0": 0.5, "SA1": 0.5}},
                {"name": "sa0_dominant", "fault_mix": {"SA0": 0.8, "SA1": 0.2}},
                {"name": "sa1_dominant", "fault_mix": {"SA0": 0.2, "SA1": 0.8}},
            ],
            "spatial_distributions": [
                {"name": "uniform", "kind": "uniform"},
                {
                    "name": "fbb_bank_clustered",
                    "kind": "fbb_bank_clustered",
                    "hot_fbb_fraction": 0.01,
                    "hot_fault_fraction": 0.80,
                    "hot_bank_fraction": 0.875,
                },
            ],
        },
        "accuracy_targets": {
            "retention_ratios": [0.95, 0.97, 0.98],
            "required_trial_pass_rate": 0.95,
            "confidence": 0.95,
        },
        "execution": {
            "resume": True,
            "save_plots": True,
            "evaluate_original_fp32": True,
        },
    }


def validate_config(cfg: Mapping[str, Any]) -> None:
    ratios = [float(x) for x in cfg["accuracy_targets"]["retention_ratios"]]
    if not ratios or any(not 0 < ratio <= 1 for ratio in ratios):
        raise ValueError("accuracy_targets.retention_ratios must be in (0,1].")
    required = float(cfg["accuracy_targets"].get("required_trial_pass_rate", 0.95))
    if not 0 < required <= 1:
        raise ValueError("required_trial_pass_rate must be in (0,1].")
    bers = [float(x) for x in cfg["offline_saf"]["ber_values"]]
    if not bers or any(value < 0 for value in bers):
        raise ValueError("offline_saf.ber_values must contain nonnegative values.")
    if int(cfg["offline_saf"].get("repeats", 1)) <= 0:
        raise ValueError("offline_saf.repeats must be positive.")


def evaluate_storage_mode(
    mode: str,
    packed: PackedWeights,
    codec: BlockMSFCodec,
    faults: FaultMap,
    controls: RemapControls,
    parameter_image: ParameterImage,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    geometry: FBBGeometry,
    remap_cfg: Mapping[str, Any],
) -> float:
    corrupted = apply_fault_map(packed, faults, controls, mode, geometry, remap_cfg)
    decoded = codec.decode(packed, corrupted).reshape(-1)
    parameter_image.assign(decoded, device)
    return evaluate_accuracy(model, loader, device)


def build_trial_key(ber: float, scenario: str, spatial: str, repeat: int) -> Tuple[float, str, str, int]:
    return float(ber), str(scenario), str(spatial), int(repeat)


def run_experiment(cfg: Mapping[str, Any]) -> None:
    validate_config(cfg)
    seed = int(cfg.get("seed", 20260806))
    set_global_seed(seed)
    output_dir = Path(str(cfg.get("output_dir", "alexnet_global_mode_accuracy_output"))).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "config_resolved.json", cfg)

    device = resolve_device(str(cfg["model"].get("device", "auto")))
    print(f"[runtime] Device: {device}")
    model, model_metadata = load_model_and_checkpoint(cfg["model"], device)
    loader = build_test_loader(cfg["dataset"], device)
    parameter_image = build_parameter_image(model, cfg["model"])

    geometry = FBBGeometry(**{key: int(value) for key, value in cfg["geometry"].items()})
    geometry.validate()
    codec_kind = str(cfg["codec"].get("kind", "block_msf_bfp8"))
    if codec_kind != "block_msf_bfp8":
        raise ValueError("The packaged version currently supports codec.kind='block_msf_bfp8'.")
    codec = BlockMSFCodec(cfg["codec"], geometry)
    original_values = parameter_image.original_vector.numpy().astype(np.float32, copy=True)
    packed = codec.encode(original_values)
    exp_p_one, sm_p_one = bit_occupancy(packed)

    # Storage audit.
    page_bytes = 16 * 1024
    occupied_pages = int(math.ceil(packed.occupied_bytes_including_padding / page_bytes))
    audit = {
        **model_metadata,
        "selected_parameter_count": parameter_image.count,
        "bounding_box_size": packed.bounding_box_size,
        "block_count": packed.block_count,
        "occupied_fbb_count": packed.num_fbb,
        "valid_storage_bytes": packed.total_valid_bytes,
        "valid_storage_bits": packed.total_valid_bits,
        "allocated_fbb_bytes_including_padding": packed.occupied_bytes_including_padding,
        "occupied_16KiB_pages_including_padding": occupied_pages,
        "exp_p_one_measured": exp_p_one.tolist(),
        "sm_p_one_measured": sm_p_one.tolist(),
    }
    save_json(output_dir / "storage_audit.json", audit)
    pd.DataFrame({
        "bit_index_msb0": np.arange(8),
        "exp_p_one": exp_p_one,
        "sm_p_one": sm_p_one,
    }).to_csv(output_dir / "bit_occupancy.csv", index=False)
    print(
        f"[storage] {packed.num_fbb:,} occupied FBBs, {packed.total_valid_bits:,} valid bits, "
        f"approximately {occupied_pages:,} physical 16-KiB pages."
    )

    # Evaluate original FP32 and fault-free codec+masking baseline.
    execution_cfg = cfg.get("execution", {})
    original_accuracy: Optional[float] = None
    if bool(execution_cfg.get("evaluate_original_fp32", True)):
        parameter_image.restore(device)
        original_accuracy = evaluate_accuracy(model, loader, device)
        print(f"[baseline] Original checkpoint accuracy: {original_accuracy:.6f}")

    no_faults = empty_fault_map()
    zero_controls = search_controls(
        no_faults, packed, geometry, cfg["remap"], exp_p_one, sm_p_one
    )
    baseline_accuracy = evaluate_storage_mode(
        "base", packed, codec, no_faults, zero_controls, parameter_image,
        model, loader, device, geometry, cfg["remap"]
    )
    print(f"[baseline] Fault-free codec + EXP masking accuracy A0: {baseline_accuracy:.6f}")
    if baseline_accuracy <= 0:
        raise RuntimeError("Fault-free codec baseline accuracy is zero; verify codec/model compatibility.")
    baseline_metadata = {
        "original_fp32_accuracy": original_accuracy,
        "fault_free_codec_masked_accuracy": baseline_accuracy,
        "accuracy_thresholds": {
            str(ratio): baseline_accuracy * float(ratio)
            for ratio in cfg["accuracy_targets"]["retention_ratios"]
        },
    }
    save_json(output_dir / "baseline_accuracy.json", baseline_metadata)

    trial_path = output_dir / "trial_results.csv"
    existing_keys: set[Tuple[float, str, str, int]] = set()
    existing_df: Optional[pd.DataFrame] = None
    if bool(execution_cfg.get("resume", True)) and trial_path.exists():
        existing_df = pd.read_csv(trial_path)
        for row in existing_df.itertuples(index=False):
            existing_keys.add(build_trial_key(row.ber, row.fault_scenario, row.spatial_distribution, row.repeat))
        print(f"[resume] Found {len(existing_keys)} completed trial conditions.")

    offline_cfg = cfg["offline_saf"]
    ber_values = [float(x) for x in offline_cfg["ber_values"]]
    repeats = int(offline_cfg["repeats"])
    max_faults_raw = offline_cfg.get("max_faults_per_trial")
    max_faults = None if max_faults_raw is None else int(max_faults_raw)
    scenarios = list(offline_cfg["fault_scenarios"])
    spatials = list(offline_cfg["spatial_distributions"])
    ratios = [float(x) for x in cfg["accuracy_targets"]["retention_ratios"]]

    new_rows: List[Dict[str, Any]] = []
    condition_index = 0
    total_conditions = len(ber_values) * repeats * len(scenarios) * len(spatials)
    for ber_index, ber in enumerate(ber_values):
        for scenario_index, scenario in enumerate(scenarios):
            scenario_name = str(scenario["name"])
            for spatial_index, spatial in enumerate(spatials):
                spatial_name = str(spatial["name"])
                for repeat in range(repeats):
                    condition_index += 1
                    key = build_trial_key(ber, scenario_name, spatial_name, repeat)
                    if key in existing_keys:
                        continue
                    trial_seed = (
                        seed
                        + ber_index * 1_000_003
                        + scenario_index * 100_003
                        + spatial_index * 10_007
                        + repeat * 101
                    )
                    rng = np.random.default_rng(trial_seed)
                    started = time.time()
                    faults = sample_faults(
                        rng, packed, geometry, ber, scenario["fault_mix"], spatial, max_faults
                    )
                    controls = search_controls(
                        faults, packed, geometry, cfg["remap"], exp_p_one, sm_p_one
                    )
                    accuracies: Dict[str, float] = {}
                    for mode in MODE_ORDER:
                        accuracies[mode] = evaluate_storage_mode(
                            mode, packed, codec, faults, controls, parameter_image,
                            model, loader, device, geometry, cfg["remap"]
                        )
                    stats = fault_statistics(faults, packed, geometry)
                    row: Dict[str, Any] = {
                        "ber": ber,
                        "fault_scenario": scenario_name,
                        "spatial_distribution": spatial_name,
                        "repeat": repeat,
                        "trial_seed": trial_seed,
                        "baseline_accuracy_A0": baseline_accuracy,
                        **stats,
                        "es_base_total": controls.baseline_es_total,
                        "es_inter_total": controls.inter_es_total,
                        "es_intra_total": controls.intra_es_total,
                        **{f"accuracy_{mode}": accuracies[mode] for mode in MODE_ORDER},
                        **{f"retention_{mode}": accuracies[mode] / baseline_accuracy for mode in MODE_ORDER},
                        "elapsed_seconds": time.time() - started,
                    }
                    for ratio in ratios:
                        threshold = baseline_accuracy * ratio
                        label = lowest_cost_feasible_from_accuracy(accuracies, threshold)
                        suffix = f"r{int(round(ratio * 1000)):03d}"
                        row[f"target_accuracy_{suffix}"] = threshold
                        row[f"lowest_cost_mode_{suffix}"] = label
                    new_rows.append(row)

                    # Append each row immediately so long experiments are restartable.
                    pd.DataFrame([row]).to_csv(
                        trial_path,
                        mode="a",
                        header=not trial_path.exists(),
                        index=False,
                    )
                    print(
                        f"[{condition_index}/{total_conditions}] BER={ber:.1e}, "
                        f"{scenario_name}/{spatial_name}, repeat={repeat}, faults={len(faults):,}, "
                        f"acc=({accuracies['base']:.4f}, {accuracies['inter']:.4f}, {accuracies['intra']:.4f}), "
                        f"time={row['elapsed_seconds']:.1f}s"
                    )

    parameter_image.restore(device)
    if trial_path.exists():
        trial_df = pd.read_csv(trial_path)
    elif existing_df is not None:
        trial_df = existing_df
    else:
        raise RuntimeError("No trial rows were produced.")

    pass_df, constraint_df, region_df = summarize_trials(
        trial_df, baseline_accuracy, cfg["accuracy_targets"]
    )
    pass_df.to_csv(output_dir / "mode_pass_summary.csv", index=False)
    constraint_df.to_csv(output_dir / "ber_constraint_table.csv", index=False)
    region_df.to_csv(output_dir / "ber_constraint_regions.csv", index=False)

    # Lowest-cost mode distribution per trial and target.
    distribution_rows: List[Dict[str, Any]] = []
    for ratio in ratios:
        suffix = f"r{int(round(ratio * 1000)):03d}"
        column = f"lowest_cost_mode_{suffix}"
        for ber, group in trial_df.groupby("ber"):
            counts = group[column].value_counts()
            total = int(group.shape[0])
            for mode in (*MODE_ORDER, "unrepairable"):
                distribution_rows.append({
                    "target_retention": ratio,
                    "ber": float(ber),
                    "mode": mode,
                    "count": int(counts.get(mode, 0)),
                    "fraction": float(counts.get(mode, 0) / total) if total else float("nan"),
                })
    pd.DataFrame(distribution_rows).to_csv(output_dir / "lowest_cost_mode_distribution.csv", index=False)

    if bool(execution_cfg.get("save_plots", True)):
        plot_results(trial_df, constraint_df, baseline_accuracy, cfg["accuracy_targets"], output_dir)

    print(f"[done] Results written to: {output_dir.resolve()}")


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def self_test() -> None:
    print("[self-test] Starting codec/fault/remap smoke test...")
    geometry = FBBGeometry()
    geometry.validate()
    codec = BlockMSFCodec(
        {
            "bounding_box_size": 4,
            "exponent_bias": 64,
            "exponent_min": -64,
            "exponent_max": 63,
            "mantissa_scale": 64.0,
        },
        geometry,
    )
    rng = np.random.default_rng(123)
    values = rng.normal(0, 0.1, size=2051).astype(np.float32)
    packed = codec.encode(values)
    decoded = codec.decode(packed, packed.bytes_by_fbb)
    assert decoded.shape == values.shape
    assert packed.total_valid_bytes == packed.block_count + values.size
    exp_p, sm_p = bit_occupancy(packed)
    faults = sample_faults(
        rng,
        packed,
        geometry,
        ber=1e-3,
        fault_mix={"SA0": 0.5, "SA1": 0.5},
        spatial_cfg={"kind": "fbb_bank_clustered", "hot_fbb_fraction": 0.2},
        max_faults=10000,
    )
    controls = search_controls(
        faults,
        packed,
        geometry,
        {
            "inter_direction": "right",
            "exp_es": [31, 7, 7, 7, 7, 15, 15, 3],
            "sm_es": [31, 15, 7, 3, 1, 1, 1, 1],
            "exp_msb_index": 0,
            "exp_mask_factor": 0.0,
            "p_one_source": "measured",
        },
        exp_p,
        sm_p,
    )
    for mode in MODE_ORDER:
        corrupted = apply_fault_map(
            packed, faults, controls, mode, geometry, {"inter_direction": "right"}
        )
        output = codec.decode(packed, corrupted)
        assert output.shape == values.shape
        assert np.isfinite(output).all()
    print(
        f"[self-test] PASS: weights={values.size}, FBBs={packed.num_fbb}, "
        f"valid_bits={packed.total_valid_bits}, faults={len(faults)}"
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Path to experiment JSON configuration.")
    parser.add_argument("--write-default-config", type=Path, help="Write a default JSON config and exit.")
    parser.add_argument("--self-test", action="store_true", help="Run an internal smoke test and exit.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.write_default_config:
        save_json(args.write_default_config, default_config())
        print(f"Default configuration written to {args.write_default_config}")
        return 0
    if args.self_test:
        self_test()
        return 0
    if args.config is None:
        raise SystemExit("Provide --config, --write-default-config, or --self-test.")
    cfg = default_config()
    user_cfg = load_json(args.config)
    deep_update(cfg, user_cfg)
    run_experiment(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
