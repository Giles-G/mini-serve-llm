import mlx.core as mx
import torch

from miniservellm.mlx.gemma4 import _quantize_rows
from miniservellm.runtime.gemma4_quant import Int4Weight, _pack_int4


def test_gemma4_ple_row_quantization_shape_and_lookup():
    source = torch.linspace(-1.0, 1.0, 2 * 128, dtype=torch.float32).reshape(2, 128)
    table = _quantize_rows(source, group_size=64)
    rows = table[mx.array([0, 1], dtype=mx.uint32)]
    mx.eval(rows)
    assert rows.shape == (2, 128)
    assert float(mx.max(mx.abs(rows)).item()) <= 1.1


def test_gemma4_ple_row_quantization_keeps_rows_distinct():
    source = torch.zeros((2, 128), dtype=torch.float32)
    source[1] = 1.0
    table = _quantize_rows(source, group_size=64)
    rows = table[mx.array([0, 1], dtype=mx.uint32)]
    mx.eval(rows)
    assert float(mx.max(mx.abs(rows[0])).item()) == 0.0
    assert float(mx.min(rows[1]).item()) > 0.9


def test_int4_weight_respects_fp16_compute_dtype():
    values = torch.tensor(
        [[-8, -3, 0, 7], [6, 1, -2, 4]],
        dtype=torch.int8,
    )
    scales = torch.ones((2, 1), dtype=torch.float16)
    weight = Int4Weight(_pack_int4(values), scales, group_size=4, compute_dtype=torch.float16)

    dense = weight.dequant_dense()
    rows = weight.dequant_rows(torch.tensor([0, 1]))

    assert dense.dtype == torch.float16
    assert rows.dtype == torch.float16
    torch.testing.assert_close(dense.float(), values.float(), atol=0, rtol=0)
    torch.testing.assert_close(rows.float(), values.float(), atol=0, rtol=0)
