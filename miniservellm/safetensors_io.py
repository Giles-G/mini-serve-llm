"""Minimal streaming reader for safetensors shards.

``safetensors.safe_open`` mmaps the whole file. That is the fastest option and
works fine on a workstation, but it fails outright when the shard is larger
than the free address space or RAM the host can spare — a 10 GB Gemma4 shard
on a 16 GB laptop, for example — with ``Cannot allocate memory``.

This reader parses the header once and then serves explicit byte ranges, so
peak host memory is one tensor (or one requested row range) instead of the
whole file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import torch

# safetensors stores little-endian tensors in C order; map the dtypes the
# Gemma4 checkpoint actually uses so rows can be decoded from raw bytes.
SAFETENSORS_DTYPES = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "I8": torch.int8,
    "U8": torch.uint8,
}



class StreamingSafetensors:
    """Read tensor byte ranges without mapping the whole shard.

    ``safe_open`` mmaps the complete file, which fails outright on
    RAM-constrained hosts for a 10 GB shard ("Cannot allocate memory").
    Reading explicit ``pread`` ranges keeps peak host memory at one tensor —
    or at ``rows`` for the multi-gigabyte PLE table.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        with self.path.open("rb") as handle:
            header_len = int.from_bytes(handle.read(8), "little")
            self._header = json.loads(handle.read(header_len).decode("utf-8"))
        self._data_start = 8 + header_len
        self._header.pop("__metadata__", None)

    def names(self) -> list[str]:
        return list(self._header)

    def shape(self, name: str) -> tuple[int, ...]:
        return tuple(self._header[name]["shape"])

    def dtype(self, name: str) -> torch.dtype:
        return SAFETENSORS_DTYPES[self._header[name]["dtype"]]

    def read(self, name: str, rows: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """Read a tensor, or the half-open row range ``[rows[0], rows[1])``."""
        entry = self._header[name]
        start, end = entry["data_offsets"]
        dtype = SAFETENSORS_DTYPES[entry["dtype"]]
        item_size = torch.tensor([], dtype=dtype).element_size()

        if rows is None:
            with self.path.open("rb") as handle:
                handle.seek(self._data_start + start)
                payload = handle.read(end - start)
            return torch.frombuffer(bytearray(payload), dtype=dtype).reshape(entry["shape"])

        row_start, row_stop = rows
        if not 0 <= row_start < row_stop <= entry["shape"][0]:
            raise ValueError(f"{name}: invalid row range {rows} for shape {entry['shape']}")
        row_bytes = (end - start) // entry["shape"][0]
        with self.path.open("rb") as handle:
            handle.seek(self._data_start + start + row_start * row_bytes)
            payload = handle.read((row_stop - row_start) * row_bytes)
        return torch.frombuffer(bytearray(payload), dtype=dtype).reshape(
            (row_stop - row_start, *entry["shape"][1:])
        )


