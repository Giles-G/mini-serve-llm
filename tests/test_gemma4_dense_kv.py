"""Dense KV mirror growth across 128/256 capacity boundaries."""

from __future__ import annotations

import pytest
import torch

from miniservellm.runtime.gemma4_engine_runner import Gemma4EngineModelRunner


def _runner_for_dense_append():
    # _dense_append only needs the static capacity helper.
    return object.__new__(Gemma4EngineModelRunner)


def test_dense_append_capacity_growth_matches_concatenation():
    runner = _runner_for_dense_append()
    dense = {}
    chunks = [1, 126, 1, 1, 127, 1, 1, 2]
    expected_k = []
    expected_v = []
    for step, length in enumerate(chunks):
        k = torch.randn(length, 1, 16)
        v = torch.randn(length, 1, 16)
        k_out, v_out = runner._dense_append(dense, 3, k, v)
        expected_k.append(k)
        expected_v.append(v)
        expected = torch.cat(expected_k, dim=0)
        assert k_out.shape[0] == expected.shape[0]
        torch.testing.assert_close(k_out, expected)
        torch.testing.assert_close(v_out, torch.cat(expected_v, dim=0))
        assert dense[3].length == expected.shape[0]


def test_dense_append_rejects_shape_mismatch():
    runner = _runner_for_dense_append()
    dense = {}
    with pytest.raises(ValueError, match="shape mismatch"):
        runner._dense_append(
            dense, 0, torch.randn(1, 1, 16), torch.randn(1, 2, 16)
        )


def test_dense_append_empty_chunk_is_noop():
    runner = _runner_for_dense_append()
    dense = {}
    k = torch.randn(1, 1, 16)
    v = torch.randn(1, 1, 16)
    runner._dense_append(dense, 0, k, v)
    before = dense[0].length
    k_out, v_out = runner._dense_append(
        dense, 0, torch.empty(0, 1, 16), torch.empty(0, 1, 16)
    )
    assert dense[0].length == before
    assert k_out.shape[0] == before
    assert v_out.shape[0] == before


def test_block_table_tensor_reuses_fixed_length_and_invalidates_on_growth():
    from types import SimpleNamespace

    from miniservellm.cache.gemma4_paged_kv_cache import Gemma4PagedKVCacheManager

    manager = object.__new__(Gemma4PagedKVCacheManager)
    manager.engine_config = SimpleNamespace(device="cpu")
    manager.req_block_tables = {"req": [3, 4]}
    manager._block_table_tensors = {}
    req = SimpleNamespace(request_id="req", block_table=[3, 4])

    first = manager.block_table_tensor(req)
    second = manager.block_table_tensor(req)
    assert first is second
    assert first.tolist() == [3, 4]

    manager.req_block_tables["req"] = [3, 4, 5]
    grown = manager.block_table_tensor(req)
    assert grown is not first
    assert grown.tolist() == [3, 4, 5]

    manager.free_block_ids = []
    manager.free_request(req)
    assert "req" not in manager._block_table_tensors
