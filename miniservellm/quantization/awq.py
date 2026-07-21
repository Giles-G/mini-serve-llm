"""AWQ (Activation-aware Weight Quantization) calibration and inference.

AWQ protects salient weight channels (those with large activation magnitudes)
by scaling them up before quantization, then scaling input activations down
at inference time.  This keeps the important channels' rounding error small.

Reference: Lin et al., "AWQ: Activation-aware Weight Quantization for
LLM Compression and Acceleration", 2023.

Usage::

    calib = AWQCalibrator(runner, salient_ratio=0.05, beta=0.5)
    calib.calibrate(calibration_prompts, num_samples=32)
    awq_configs = calib.compute_awq_scales()
    awq_quantize_model(runner, awq_configs, bits=4, group_size=64)
"""

from __future__ import annotations

import torch
from typing import Dict, List, Tuple, Optional, Callable


class AWQCalibrator:
    """Offline AWQ calibration.

    Steps:
        1. Run a small set of calibration prompts through the model.
        2. For each linear layer, collect the input activation statistics
           (channel-wise L2 norm across all calibration tokens).
        3. Compute per-channel AWQ scales that amplify important channels.

    Attributes:
        runner: TransformerModelRunner instance.
        salient_ratio: fraction of channels treated as "salient" (default 0.05).
        beta: activation importance exponent (default 0.5).
        act_stats: dict of layer_path → [in_features] float32 accumulated L2 norms.
    """

    def __init__(
        self,
        runner,
        salient_ratio: float = 0.05,
        beta: float = 0.5,
    ):
        self.runner = runner
        self.salient_ratio = salient_ratio
        self.beta = beta
        self.act_stats: Dict[str, torch.Tensor] = {}  # layer_path → accumulated stat
        self._num_samples = 0

    def calibrate(
        self,
        calibration_loader: List[torch.Tensor],
        num_samples: int = 32,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        """Run calibration data through the model and collect activation stats.

        Args:
            calibration_loader: list of tokenised prompts, each is [seq_len] int.
            num_samples: how many samples to run.
            progress_callback: optional (current, total) callback.
        """
        self.act_stats = {}
        self._num_samples = 0

        for idx, input_ids in enumerate(calibration_loader):
            if idx >= num_samples:
                break

            # Truncate to max_position_embeddings
            max_len = self.runner.model_config.max_position_embeddings
            if len(input_ids) > max_len:
                input_ids = input_ids[:max_len]

            self._run_one_sample(input_ids.to(self.runner.engine_config.device))
            self._num_samples += 1

            if progress_callback:
                progress_callback(idx + 1, num_samples)

    def _run_one_sample(self, input_ids: torch.Tensor) -> None:
        """Run one token sequence through the model, collecting input stats.

        We need per-layer input activation statistics. The approach:
        use a *forward* with output hooks on every layer's attention norm input.

        For simplicity, we run the raw embed → layer forward path by patching
        the RMSNorm function to collect statistics.
        """
        device = self.runner.engine_config.device
        hidden = self.runner.weights.embed_tokens[input_ids]  # [seq, hidden]

        for layer_idx, block in enumerate(self.runner.blocks):
            lw = self.runner.weights.layers[layer_idx]
            # Collect statistics at the input to each layer's attention norm
            # (this is the input activation to qkv_proj, gate_up_proj)
            key = f"layer_{layer_idx}"
            self._accumulate_stat(key, hidden)

            # Simplified forward (just embeddings, no KV cache needed for stats)
            # We don't need actual outputs — just the activation values
            # So we skip the full forward and just do a quick pass
            residual = hidden

            # Apply RMSNorm to get the "x_normed" that goes into linear layers
            from miniservellm.runtime.nn_ops import rms_norm as _rms_norm
            x_normed = _rms_norm(hidden, lw.input_layernorm, self.runner.model_config.rms_norm_eps)

            # Simulate a quick forward (no KV write needed)
            # Project to QKV shape to consume the activation
            qkv = torch.mm(x_normed, lw.qkv_proj.T.half() if lw.qkv_proj.dtype != x_normed.dtype
                           else lw.qkv_proj.T)
            q_dim = self.runner.model_config.num_attention_heads * self.runner.model_config.head_dim
            q_raw = qkv[:, :q_dim]

            # Simulate attention output
            # Use a simple identity-like pass to preserve activation statistics
            attn_out = torch.mm(
                q_raw.view(-1, q_dim),
                lw.o_proj.T.half() if lw.o_proj.dtype != q_raw.dtype else lw.o_proj.T,
            )

            # Fused residual + FFN norm
            from miniservellm.runtime.nn_ops import fused_add_rms_norm
            x_normed2, hidden = fused_add_rms_norm(
                attn_out, residual,
                lw.post_attention_layernorm, self.runner.model_config.rms_norm_eps,
            )

            # FFN (SwiGLU) — get the input to gate_up_proj for statistics
            self._accumulate_stat(f"{key}_ffn", x_normed2)

    def _accumulate_stat(self, key: str, tensor: torch.Tensor) -> None:
        """Accumulate channel-wise L2 norm over all token positions."""
        # tensor: [seq_len, hidden_size] or [seq_len, intermediate_size]
        stat = torch.norm(tensor.float(), p=2, dim=0)  # [hidden_size]
        if key not in self.act_stats:
            self.act_stats[key] = stat
        else:
            self.act_stats[key] += stat

    def compute_awq_scales(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Compute per-channel AWQ scale for every linear layer.

        Returns:
            Dict layer_path → {
                'qkv_proj':   [in_features] fp16 scale,
                'o_proj':     [in_features] fp16 scale,
                'gate_up_proj': [in_features] fp16 scale,
                'down_proj':  [in_features] fp16 scale,
            }
        """
        if self._num_samples == 0:
            raise RuntimeError("call calibrate() first")

        awq_configs: Dict[str, Dict[str, torch.Tensor]] = {}

        for layer_idx, block in enumerate(self.runner.blocks):
            key = f"layer_{layer_idx}"
            act_attn = self.act_stats.get(key)
            act_ffn  = self.act_stats.get(f"{key}_ffn")

            if act_attn is None or act_ffn is None:
                continue

            # Average over samples
            act_attn = act_attn / self._num_samples
            act_ffn  = act_ffn / self._num_samples

            lw = block.layer_weights
            h = self.runner.model_config.hidden_size
            inter = self.runner.model_config.intermediate_size
            qkv_dim = lw.qkv_proj.shape[1]

            config = {
                'qkv_proj':   self._compute_scale(lw.qkv_proj, act_attn),
                'o_proj':     self._compute_scale(lw.o_proj, act_attn[:lw.o_proj.shape[1]]),
                'gate_up_proj': self._compute_scale(lw.gate_up_proj, act_ffn),
                'down_proj':  self._compute_scale(lw.down_proj, act_ffn[:lw.down_proj.shape[1]]),
            }
            awq_configs[key] = config

        return awq_configs

    def _compute_scale(
        self, weight: torch.Tensor, act_stats: torch.Tensor
    ) -> torch.Tensor:
        """Compute per-channel AWQ scale for one weight matrix.

        Formula:
            s[c] = mean(|w[:,c]|) × (act_stats[c] / mean(act_stats))^beta
                 × smooth_factor[c]

        Args:
            weight: [out_features, in_features] fp16/bfloat16
            act_stats: [in_features] float32

        Returns:
            [in_features] fp16 scale tensor.
        """
        in_features = weight.shape[1]

        # Truncate/pad act_stats to match weight's in_features
        if act_stats.numel() > in_features:
            act = act_stats[:in_features]
        elif act_stats.numel() < in_features:
            act = torch.zeros(in_features, device=act_stats.device, dtype=torch.float32)
            act[:act_stats.numel()] = act_stats
        else:
            act = act_stats

        # Weight magnitude per input channel
        w_abs_mean = weight.float().abs().mean(dim=0)  # [in_features]

        # Relative activation importance
        act_mean = act.mean()
        act_norm = act / act_mean.clamp(min=1e-8)  # [in_features]

        # Smooth factor: salient channels get extra boost
        threshold = torch.quantile(act, 1.0 - self.salient_ratio)
        smooth = torch.where(
            act >= threshold,
            torch.tensor(1.2, device=act.device, dtype=torch.float32),
            torch.tensor(0.8, device=act.device, dtype=torch.float32),
        )

        # AWQ scale
        s = w_abs_mean * (act_norm ** self.beta) * smooth
        s = s.clamp(min=1e-6)

        return s.half()


def awq_quantize_model(
    runner,
    awq_configs: Dict[str, Dict[str, torch.Tensor]],
    bits: int = 4,
    group_size: int = 64,
) -> None:
    """Apply AWQ scales and quantize all linear weights in-place.

    After this call:
      - Each layer's weights are replaced by (w_q_packed, group_scales) tuples.
      - AWQ per-channel scales are stored in ``layer_weights._awq_scales``.
      - The ``linear()`` function auto-detects AWQ-quantized weights.

    Args:
        runner: TransformerModelRunner instance.
        awq_configs: AWQ per-layer scale config from compute_awq_scales().
        bits: quantization bit-width (4 or 8).
        group_size: group size for group quantization.
    """
    from miniservellm.runtime.nn_ops import quantize_weight_group as _qwg

    max_val = 2 ** (bits - 1) - 1  # 7 for INT4

    for layer_idx, block in enumerate(runner.blocks):
        key = f"layer_{layer_idx}"
        cfg = awq_configs.get(key)
        if cfg is None:
            continue

        lw = block.layer_weights

        for weight_name in ['qkv_proj', 'o_proj', 'gate_up_proj', 'down_proj']:
            w_orig = getattr(lw, weight_name)
            scale = cfg.get(weight_name)
            if scale is None or scale.numel() != w_orig.shape[1]:
                continue

            # 1. Apply AWQ scale: w' = w / s
            w_scaled = w_orig.float() / scale.float().unsqueeze(0).clamp(min=1e-6)
            w_scaled = w_scaled.half()

            # 2. Group quantize the AWQ-scaled weights
            w_q, group_scales = _qwg(w_scaled, bits=bits, group_size=group_size)

            # 3. Pack INT4
            w_packed = _pack_int4(w_q)

            # 4. Store as tuple: (w_packed, group_scales, awq_scales)
            setattr(lw, weight_name, (w_packed, group_scales, scale))

        # Also quantize lm_head on final iteration
        if layer_idx == len(runner.blocks) - 1:
            # lm_head is not in a layer — handle separately if needed
            pass


def _pack_int4(w_q: torch.Tensor) -> torch.Tensor:
    """Pack 2 consecutive INT4 values into 1 byte (low nibble first).

    Args:
        w_q: [N, K] int8, values in [-7, 7] (or [-8, 7])

    Returns:
        [N//2, K] uint8, where output[2i, j] and output[2i+1, j] share 1 byte.
    """
    N, K = w_q.shape
    if N % 2 != 0:
        # Pad with zeros
        pad = torch.zeros(1, K, dtype=w_q.dtype, device=w_q.device)
        w_q = torch.cat([w_q, pad], dim=0)
        N += 1

    # Offset from [-8, 7] to [0, 15] unsigned
    w_u = (w_q + 8).clamp(0, 15).to(torch.uint8)

    # Pack: low nibble = even row, high nibble = odd row
    even = w_u[0::2, :]
    odd  = w_u[1::2, :]
    return (even | (odd << 4)).contiguous()
