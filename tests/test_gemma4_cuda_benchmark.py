import importlib.util
from pathlib import Path

import pytest
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "bench_gemma4_cuda.py"
SPEC = importlib.util.spec_from_file_location("bench_gemma4_cuda", SCRIPT)
BENCH = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BENCH)


def make_small_case(dtype=torch.float16):
    q = torch.tensor(
        [[-8, -1, 0, 7], [7, 1, -2, -8], [3, -4, 5, -6], [-7, 6, -5, 2]],
        dtype=torch.int8,
    )
    packed = BENCH._pack_int4(q)
    scales = torch.tensor(
        [
            [0.125, 0.03125],
            [0.0625, 0.015625],
            [0.09375, 0.046875],
            [0.078125, 0.0234375],
        ],
        dtype=torch.float16,
    )
    x = torch.tensor(
        [[1.0, -2.0, 3.0, -4.0], [0.3333, 1.75, -2.5, 0.125]],
        dtype=dtype,
    )
    return x, packed, scales


def test_pack_and_cpu_fp64_oracle_match_manual_values():
    x, packed, scales = make_small_case(torch.float32)
    exact, rounded, _ = BENCH._high_precision_references(x, packed, scales, chunk_rows=2)
    oracle = BENCH._cpu_fp64_oracle(x, packed, scales)

    assert packed.dtype == torch.uint8
    assert torch.allclose(exact.double(), oracle, atol=2e-6, rtol=2e-6)
    assert torch.equal(rounded, exact)


def test_pack_rejects_out_of_range_int4_values():
    with pytest.raises(ValueError, match=r"\[-8, 7\]"):
        BENCH._pack_int4(torch.tensor([[8, 0], [0, 0]], dtype=torch.int8))


def test_oracle_rejects_nan_and_inf():
    x, packed, scales = make_small_case()
    with pytest.raises(ValueError, match="finite"):
        BENCH._high_precision_references(x, packed, scales * float("nan"))
    with pytest.raises(ValueError, match="finite"):
        BENCH._cpu_fp64_oracle(x * float("inf"), packed, scales)


def test_chunk_rows_must_preserve_packed_pair_boundaries():
    x, packed, scales = make_small_case()
    with pytest.raises(ValueError, match="even"):
        BENCH._high_precision_references(x, packed, scales, chunk_rows=3)
    with pytest.raises(ValueError, match="positive"):
        BENCH._high_precision_references(x, packed, scales, chunk_rows=0)


def test_wrong_result_is_rejected_and_reports_worst_over_budget():
    x, packed, scales = make_small_case()
    exact, rounded, _ = BENCH._high_precision_references(x, packed, scales, chunk_rows=2)
    report = BENCH._check_parity(
        x,
        packed,
        scales,
        rounded,
        exact.to(x.dtype) + 1,
    )

    assert not report["passed"]
    fused = report["fused_vs_exact"]
    assert fused["failed_elements"] > 0
    assert fused["worst_over_budget"]["ratio"] > 1


def test_fp16_rounding_difference_is_reported_without_false_failure():
    x, packed, scales = make_small_case(torch.float16)
    scales = scales.clone()
    scales[0, 0] = 0.1235
    scales[2, 1] = 0.0377
    exact, rounded, _ = BENCH._high_precision_references(x, packed, scales, chunk_rows=2)
    report = BENCH._check_parity(x, packed, scales, rounded, exact.to(x.dtype))

    assert report["weight_rounding_max_abs"] > 0
    assert report["fused_vs_exact"]["failed_elements"] == 0
    assert report["fallback_vs_rounded"]["failed_elements"] == 0
    assert report["passed"]


def test_tf32_reference_setting_is_restored():
    before_matmul = torch.backends.cuda.matmul.allow_tf32
    before_cudnn = torch.backends.cudnn.allow_tf32
    x, packed, scales = make_small_case()

    BENCH._high_precision_references(x, packed, scales)

    assert torch.backends.cuda.matmul.allow_tf32 == before_matmul
    assert torch.backends.cudnn.allow_tf32 == before_cudnn
