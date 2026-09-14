"""Load an NVFP4 (ModelOpt) GLM-5.3 checkpoint into transformers' GlmMoeDsaForCausalLM, spread
over every visible GPU, with the routed experts kept packed in FP4 and dequantized on the fly.

The served checkpoint quantizes only the 256 routed experts per MoE layer (4-bit weights, one
FP8 scale per 16 weights, one FP32 scale per output row). Everything else (attention, indexer,
shared expert, dense layers, embeddings) is plain bf16. Keeping the experts packed leaves
~60 GB per GPU used on an 8 x B300 node instead of ~190 GB dequantized, and LoRA never touches
them anyway: gradients only have to flow *through* the experts, which is what the weight-only
dequantization inside `NVFP4Experts.forward` provides.
"""

import math
import re
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import dispatch_model
from accelerate.utils import set_module_tensor_to_device
from safetensors import safe_open
from torch import nn
from torch.utils.checkpoint import checkpoint

# FP4 E2M1 code points, indexed by the 4-bit nibble (bit 3 = sign).
_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
# Byte -> (low nibble value, high nibble value): the low nibble holds the even element.
_BYTE_TABLE = torch.stack((_E2M1[torch.arange(256) & 0xF], _E2M1[torch.arange(256) >> 4]), dim=-1)

BLOCK = 16
_EXPERT_KEY = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(\w+)$")


def _dequant_table(packed: torch.Tensor, block_scale: torch.Tensor, row_scale: torch.Tensor) -> torch.Tensor:
    """Eager reference: nibble lookup table, fp32 scales. packed uint8 [E, N, K/2] -> bf16 [E, N, K]."""
    experts, rows, _ = packed.shape
    values = _BYTE_TABLE.to(packed.device)[packed.int()].view(experts, rows, -1, BLOCK)
    scale = block_scale.to(torch.float32) * row_scale.unsqueeze(-1)
    return (values * scale.unsqueeze(-1)).view(experts, rows, -1).to(torch.bfloat16)


def _dequant_math(packed: torch.Tensor, block_scale: torch.Tensor, row_scale: torch.Tensor) -> torch.Tensor:
    """Same result as `_dequant_table`, written as arithmetic so torch.compile fuses it into one kernel."""
    codes = torch.stack((packed & 0x0F, packed >> 4), dim=-1).flatten(-2)  # low nibble = even element
    sign = torch.where((codes & 8) != 0, -1.0, 1.0)
    exponent = ((codes >> 1) & 3).to(torch.float32)
    mantissa = (codes & 1).to(torch.float32)
    magnitude = torch.where(exponent == 0, mantissa * 0.5, (1.0 + mantissa * 0.5) * torch.exp2(exponent - 1.0))
    scale = block_scale.to(torch.float32) * row_scale.unsqueeze(-1)
    weights = (sign * magnitude).view(*codes.shape[:-1], -1, BLOCK) * scale.unsqueeze(-1)
    return weights.view(codes.shape).to(torch.bfloat16)


_dequant_cuda = None


def _dequant_chunk(packed: torch.Tensor, block_scale: torch.Tensor, row_scale: torch.Tensor) -> torch.Tensor:
    global _dequant_cuda
    if not packed.is_cuda:
        return _dequant_table(packed, block_scale, row_scale)
    if _dequant_cuda is None:
        try:
            # Dynamo specialises on the device index, so eight GPUs need eight cache entries
            # (plus one per weight shape); the default limit of 8 would silently fall back to eager.
            for name in ("recompile_limit", "cache_size_limit"):
                if hasattr(torch._dynamo.config, name):
                    setattr(torch._dynamo.config, name, max(getattr(torch._dynamo.config, name), 64))
            compiled = torch.compile(_dequant_math, dynamic=True)
            compiled(packed[:1], block_scale[:1], row_scale[:1])
            _dequant_cuda = compiled
        except Exception as exc:  # no Triton, no compiler: 2.7x slower but the same numbers
            warnings.warn(f"torch.compile unavailable ({type(exc).__name__}: {exc}); using the eager dequantizer")
            _dequant_cuda = _dequant_table
    return _dequant_cuda(packed, block_scale, row_scale)


def dequantize(packed: torch.Tensor, block_scale: torch.Tensor, row_scale: torch.Tensor, chunk: int = 32) -> torch.Tensor:
    """packed uint8 [E, N, K/2], block_scale float8 [E, N, K/16], row_scale float32 [E, N] -> bf16 [E, N, K]."""
    experts, rows, half = packed.shape
    out = torch.empty(experts, rows, half * 2, dtype=torch.bfloat16, device=packed.device)
    for start in range(0, experts, chunk):
        stop = min(start + chunk, experts)
        out[start:stop] = _dequant_chunk(packed[start:stop], block_scale[start:stop], row_scale[start:stop])
    return out


def _grouped_linear(x: torch.Tensor, weight: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """x [S, in] sorted by group, weight [G, out, in], offsets [G] cumulative -> [S, out]."""
    if x.is_cuda:
        return F.grouped_mm(x, weight.transpose(-2, -1), offs=offsets)
    pieces, start = [], 0
    for group, stop in enumerate(offsets.tolist()):
        pieces.append(x[start:stop] @ weight[group].T)
        start = stop
    return torch.cat(pieces)


class NVFP4Experts(nn.Module):
    """Drop-in replacement for GlmMoeDsaExperts holding the routed experts in packed NVFP4."""

    def __init__(self, num_experts: int, hidden_dim: int, intermediate_dim: int, device):
        super().__init__()
        self.num_experts, self.hidden_dim, self.intermediate_dim = num_experts, hidden_dim, intermediate_dim
        u8 = dict(dtype=torch.uint8, device=device)
        f8 = dict(dtype=torch.float8_e4m3fn, device=device)
        f32 = dict(dtype=torch.float32, device=device)
        self.gate_up_w = nn.Buffer(torch.empty(num_experts, 2 * intermediate_dim, hidden_dim // 2, **u8), persistent=False)
        self.gate_up_s = nn.Buffer(torch.empty(num_experts, 2 * intermediate_dim, hidden_dim // BLOCK, **f8), persistent=False)
        self.gate_up_g = nn.Buffer(torch.empty(num_experts, 2 * intermediate_dim, **f32), persistent=False)
        self.down_w = nn.Buffer(torch.empty(num_experts, hidden_dim, intermediate_dim // 2, **u8), persistent=False)
        self.down_s = nn.Buffer(torch.empty(num_experts, hidden_dim, intermediate_dim // BLOCK, **f8), persistent=False)
        self.down_g = nn.Buffer(torch.empty(num_experts, hidden_dim, **f32), persistent=False)
        self.loaded = torch.zeros(num_experts, 3, 3, dtype=torch.bool)  # [expert, proj, tensor kind]

    _PROJ = {"gate_proj": 0, "up_proj": 1, "down_proj": 2}
    _KIND = {"weight": 0, "weight_scale": 1, "weight_scale_2": 2}

    def load(self, expert: int, proj: str, kind: str, tensor: torch.Tensor) -> bool:
        """Copy one checkpoint tensor into place. Returns False for tensors that are not needed."""
        if kind not in self._KIND:
            return False  # input_scale: activation quantization, unused for weight-only dequantization
        rows = slice(0, self.intermediate_dim) if proj == "gate_proj" else slice(self.intermediate_dim, None)
        if proj == "down_proj":
            w, s, g, rows = self.down_w, self.down_s, self.down_g, slice(None)
        else:
            w, s, g = self.gate_up_w, self.gate_up_s, self.gate_up_g
        with torch.no_grad():
            if kind == "weight":
                w[expert, rows].copy_(tensor.view(torch.uint8), non_blocking=True)
            elif kind == "weight_scale":
                s[expert, rows].copy_(tensor.view(torch.float8_e4m3fn), non_blocking=True)
            else:
                g[expert, rows].fill_(float(tensor.float()))
        self.loaded[expert, self._PROJ[proj], self._KIND[kind]] = True
        return True

    def forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled() and hidden_states.requires_grad:
            # Recompute (and re-dequantize) in backward instead of keeping bf16 expert weights alive.
            return checkpoint(self._forward, hidden_states, top_k_index, top_k_weights, use_reentrant=False)
        return self._forward(hidden_states, top_k_index, top_k_weights)

    def _forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor) -> torch.Tensor:
        tokens, top_k = top_k_index.shape
        hit, compact = torch.unique(top_k_index, return_inverse=True)  # only dequantize experts that received tokens
        gate_up = dequantize(self.gate_up_w[hit], self.gate_up_s[hit], self.gate_up_g[hit])
        down = dequantize(self.down_w[hit], self.down_s[hit], self.down_g[hit])

        flat = compact.reshape(-1)
        order = torch.argsort(flat, stable=True)
        token_of = order // top_k
        offsets = torch.bincount(flat, minlength=hit.numel()).cumsum(0).to(torch.int32)

        x = hidden_states[token_of]
        gate, up = _grouped_linear(x, gate_up, offsets).chunk(2, dim=-1)
        y = _grouped_linear(F.silu(gate) * up, down, offsets)
        y = y * top_k_weights.reshape(-1)[order].unsqueeze(-1).to(y.dtype)

        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(order.numel(), device=order.device)
        return y[inverse].view(tokens, top_k, -1).sum(dim=1).to(hidden_states.dtype)


def plan_devices(num_layers: int, devices: list) -> dict:
    """Contiguous blocks of decoder layers per GPU; embeddings first, head last."""
    device_map = {"model.embed_tokens": devices[0], "model.rotary_emb": devices[0], "model.norm": devices[-1], "lm_head": devices[-1]}
    for layer in range(num_layers):
        device_map[f"model.layers.{layer}"] = devices[min(layer * len(devices) // num_layers, len(devices) - 1)]
    return device_map


def load_model(model_dir, devices=None, workers: int = 8, attn_implementation: str = "eager", log=print):
    """Build GlmMoeDsaForCausalLM on the meta device, swap in NVFP4Experts, stream the checkpoint in.

    Attention defaults to the eager implementation: on B300 the SDPA backward produced NaN
    gradients in the last decoder layer for some inputs, and with the short sequences a LoRA
    run uses, eager attention costs nothing.
    """
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaRotaryEmbedding

    model_dir = Path(model_dir)
    config = AutoConfig.from_pretrained(model_dir)
    quant = getattr(config, "quantization_config", None) or {}
    if quant.get("quant_algo") != "NVFP4" or quant.get("group_size", BLOCK) != BLOCK:
        raise ValueError(f"expected a ModelOpt NVFP4 checkpoint with group size {BLOCK}, got {quant}")
    config.quantization_config = None
    config.dtype = torch.bfloat16

    devices = devices or [torch.device("cuda", i) for i in range(torch.cuda.device_count())]
    device_map = plan_devices(config.num_hidden_layers, devices)

    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, attn_implementation=attn_implementation)
    model.model.rotary_emb = GlmMoeDsaRotaryEmbedding(config).to(device_map["model.rotary_emb"])
    experts = {}
    for index, layer in enumerate(model.model.layers):
        if config.mlp_layer_types[index] == "sparse":
            layer.mlp.experts = experts[index] = NVFP4Experts(
                config.n_routed_experts, config.hidden_size, config.moe_intermediate_size, device_map[f"model.layers.{index}"]
            )

    index = __import__("json").loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    shards = sorted(set(index.values()))
    expected = set(model.state_dict().keys())
    seen, skipped = set(), set()
    started = time.time()

    def route(key: str):
        match = _EXPERT_KEY.match(key)
        if match:
            layer, expert, proj, kind = int(match[1]), int(match[2]), match[3], match[4]
            return ("expert", layer, expert, proj, kind) if layer in experts else None
        if key in expected:
            return ("param", device_map[".".join(key.split(".")[:3])] if key.startswith("model.layers.") else device_map.get(key.rsplit(".", 1)[0], devices[0]))
        return None

    def load_shard(shard: str):
        with safe_open(model_dir / shard, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                target = route(key)
                if target is None:
                    skipped.add(key)
                    continue
                tensor = handle.get_tensor(key)
                if target[0] == "param":
                    set_module_tensor_to_device(model, key, target[1], value=tensor)
                    seen.add(key)
                elif not experts[target[1]].load(*target[2:], tensor):
                    skipped.add(key)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for done, _ in enumerate(pool.map(load_shard, shards), 1):
            if done % 8 == 0 or done == len(shards):
                log(f"  {done}/{len(shards)} shards, {time.time() - started:.0f}s")

    missing = expected - seen
    if missing:
        raise RuntimeError(f"{len(missing)} tensors missing from the checkpoint, e.g. {sorted(missing)[:5]}")
    for layer, module in experts.items():
        if not module.loaded.all():
            raise RuntimeError(f"layer {layer}: {int((~module.loaded).sum())} expert tensors missing")
    if "model.layers.78.eh_proj.weight" in skipped:
        pass  # MTP draft layer: used by vLLM for speculative decoding only
    log(f"loaded {len(seen)} bf16 tensors + {sum(int(m.loaded.sum()) for m in experts.values())} expert tensors "
        f"in {time.time() - started:.0f}s; skipped {len(skipped)} (MTP layer, activation scales)")

    model = dispatch_model(model, device_map=device_map)
    model.eval()
    return model


def memory_report() -> str:
    lines = []
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        lines.append(f"  cuda:{i}  {(total - free) / 2**30:6.1f} GiB used of {total / 2**30:.0f} GiB")
    return "\n".join(lines)
