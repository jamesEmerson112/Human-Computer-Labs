"""
The `train_gpt.py` and `train_gpt_mlx.py` scripts are intended as good launching-off points for new participants, not SOTA configs. We'll accept PRs that tune, improve, or simplify these scripts without significantly increasing complexity, but competitive submissions should stay in the `/records` folder.

Hard stop: To keep readable for newcomers, let's make sure `train_gpt.py` and `train_gpt_mlx.py` never are longer than 1500 lines.
"""

from __future__ import annotations

import copy
import glob
import io
import math
import os
import random
import subprocess
import sys
import time
import uuid
import zlib
try:
    import brotli
except ImportError:
    brotli = None
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

# -----------------------------
# HYPERPARAMETERS
# -----------------------------
# Default Simple Baseline run:
# - 9 transformer blocks at width 512
# - 8 attention heads with 4 KV heads (GQA) and 2x MLP expansion
# - vocab size 1024, sequence length 1024, tied embeddings
# - 524,288 train tokens per step for 20,000 iterations with a ~10 minute cap

class Hyperparameters:
    # Data paths are shard globs produced by the existing preprocessing pipeline.
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    # Validation cadence and batch size. Validation always uses the full fineweb_val split.
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))

    # Training length.
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1200))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    # Model shape.
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 9))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 2))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    # Gated Attention (https://arxiv.org/abs/2505.06708, NeurIPS 2025 Best Paper)
    # "none" = disabled (baseline), "headwise" = 1 gate per head, "elementwise" = 1 gate per dim
    gated_attn = os.environ.get("GATED_ATTN", "none")

    # Optimizer hyperparameters.
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.05))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.0))

    # Activation: "relu2" (default), "leaky_relu2" = LeakyReLU(0.5)² (PG ranks 10-11)
    activation = os.environ.get("ACTIVATION", "relu2")

    # Selective Language Modeling (Rho-1, NeurIPS 2024 Best Paper Runner-Up)
    # "Not All Tokens Are What You Need" — https://arxiv.org/abs/2401.04056
    # Keep top k% tokens by loss, skip easy ones
    slm_enabled = os.environ.get("SLM_ENABLED", "0") == "1"
    slm_ratio = float(os.environ.get("SLM_RATIO", "0.6"))

    # Value Residual Learning (ResFormer, ACL 2025)
    # Blend layer 0's V with each subsequent layer's V: v = (1-alpha)*v + alpha*v0
    # Fights over-smoothing, equivalent to 16% more params. 0.0 = disabled.
    value_residual_alpha = float(os.environ.get("VALUE_RESIDUAL_ALPHA", "0.0"))

    # Quantization: "int8_zlib" (default), "int6_brotli" (PG ranks 1-9)
    # int6_brotli uses SDClip (k * std) instead of quantile clipping, int6 for matrices,
    # int8 for embeddings, and brotli compression. Saves ~25% artifact size.
    quant_mode = os.environ.get("QUANT_MODE", "int8_zlib")
    sdclip_k = float(os.environ.get("SDCLIP_K", "12.85"))
    embed_clip_k = float(os.environ.get("EMBED_CLIP_K", "20.0"))

    # GPTQ (Hessian-based error compensation for int6 quantization, PG ranks 1-9)
    # Column-by-column quantization that compensates for each error using inverse Hessian.
    # AR self-generation creates calibration data, no training data needed at eval time.
    use_gptq = os.environ.get("USE_GPTQ", "0") == "1"
    gptq_ar_samples = int(os.environ.get("GPTQ_AR_SAMPLES", "64"))
    gptq_ar_seqlen = int(os.environ.get("GPTQ_AR_SEQLEN", "2048"))
    gptq_temperature = float(os.environ.get("GPTQ_TEMPERATURE", "0.8"))
    gptq_batch_size = int(os.environ.get("GPTQ_BATCH_SIZE", "8"))
    gptq_reserve_ms = float(os.environ.get("GPTQ_RESERVE_MS", "30000"))
    gptq_clip_range = int(os.environ.get("GPTQ_CLIP_RANGE", "63"))  # 31=int6, 63=int7 (Kevin Clark rank 5)
    gptq_calib_source = os.environ.get("GPTQ_CALIB_SOURCE", "ar")  # "ar" = AR self-gen, "train" = training data
    gptq_sequential = os.environ.get("GPTQ_SEQUENTIAL", "0") == "1"  # Sequential block-by-block quantization
    gptq_use_hooks = os.environ.get("GPTQ_USE_HOOKS", "0") == "1"  # register_forward_hook for Hessian collection
    gptq_embed = os.environ.get("GPTQ_EMBED", "0") == "1"  # GPTQ on embedding layer (not just simple SDClip)
    gptq_percdamp = float(os.environ.get("GPTQ_PERCDAMP", "0.01"))  # Dampening factor for Cholesky

    # Test-Time Training (Score-First TTT, PG ranks 1-3)
    # "none" = disabled (baseline), "score_first" = legal score-before-update TTT
    ttt_mode = os.environ.get("TTT_MODE", "none")
    ttt_lr = float(os.environ.get("TTT_LR", "0.005"))
    ttt_momentum = float(os.environ.get("TTT_MOMENTUM", "0.9"))
    ttt_epochs = int(os.environ.get("TTT_EPOCHS", "3"))
    ttt_chunk_tokens = int(os.environ.get("TTT_CHUNK_TOKENS", "32768"))
    ttt_freeze_blocks = int(os.environ.get("TTT_FREEZE_BLOCKS", "0"))
    ttt_grad_clip = float(os.environ.get("TTT_GRAD_CLIP", "1.0"))
    ttt_batch_seqs = int(os.environ.get("TTT_BATCH_SEQS", "32"))

# -----------------------------
# MUON OPTIMIZER 
# -----------------------------
# 
# As borrowed from modded-nanogpt
# Background on Muon: https://kellerjordan.github.io/posts/muon/

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    # Orthogonalize a 2D update matrix with a fast Newton-Schulz iteration.
    # Muon uses this to normalize matrix-shaped gradients before applying them.
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0

        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]

            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)

            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    # Scale correction from Muon reference implementations.
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()

            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()

        return loss


# -----------------------------
# TOKENIZER-AGNOSTIC EVALUATION SETUP 
# -----------------------------
#
# It's common for small models have a large fraction of their parameters be embeddings, since the 2 * d_model * d_vocab vectors can be gigantic.
# Instead of locking the tokenizer, we let you bring your own and calculate our validation metrics on the average compression of the validation set.
# We calculate BPB (bits-per-byte) instead of validation loss, so we need methods to count the number of bits per token in the tokenizer.
# Note: Submissions that edit the tokenizer will be examined more carefully, since screwing this up might unjustly improve your score.

def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    # The export pipeline writes the fixed first-50k-doc validation set to fineweb_val_*.
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]


def eval_val(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
    # Validation computes two metrics:
    # - val_loss: token cross-entropy (natural log)
    # - val_bpb: tokenizer-agnostic compression metric used by the challenge
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < args.train_seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence per rank; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, WORLD_SIZE={world_size}, "
            f"GRAD_ACCUM_STEPS={grad_accum_steps}, TRAIN_SEQ_LEN={args.train_seq_len}"
        )
    local_batch_seqs = local_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * args.train_seq_len
            raw_end = batch_seq_end * args.train_seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)


# -----------------------------
# SCORE-FIRST TTT (Legal Test-Time Training)
# -----------------------------
# Protocol (PR #461 recipe, PG ranks 1-3):
# 1. Chunk validation data into ttt_chunk_tokens windows
# 2. For each chunk: SCORE all tokens under inference_mode(), then TRAIN on scored data
# 3. Last chunk: score only (never train on un-scored data)
# Guarantees: every token scored BEFORE any parameter update that could use it.

def eval_val_score_first_ttt(
    args: Hyperparameters,
    base_model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    log0=print,
) -> tuple[float, float]:
    seq_len = args.train_seq_len
    total_tokens = val_tokens.numel() - 1
    stride = seq_len  # non-overlapping windows for simplicity
    ttt_chunk = args.ttt_chunk_tokens

    # Pre-compute window starts
    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= 1]

    # Assign windows to chunks
    num_chunks = (total_tokens + ttt_chunk - 1) // ttt_chunk
    chunk_windows: list[list[int]] = [[] for _ in range(num_chunks)]
    for ws in window_starts:
        end = min(ws + seq_len, total_tokens)
        ci = min(ws // ttt_chunk, num_chunks - 1)
        chunk_windows[ci].append(ws)

    log0(f"ttt:start chunks={num_chunks} chunk_tokens={ttt_chunk} "
         f"windows={len(window_starts)} ttt_lr={args.ttt_lr} "
         f"ttt_epochs={args.ttt_epochs} freeze_blocks={args.ttt_freeze_blocks}")

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    # Freeze first N blocks
    frozen_block_ids = set(range(min(args.ttt_freeze_blocks, len(base_model.blocks))))
    ttt_params = []
    for name, p in base_model.named_parameters():
        freeze = any(f"blocks.{bi}." in name for bi in frozen_block_ids)
        if freeze:
            p.requires_grad_(False)
        else:
            p.requires_grad_(True)
            ttt_params.append(p)

    log0(f"ttt:params unfrozen={sum(p.numel() for p in ttt_params)} "
         f"frozen={sum(p.numel() for p in base_model.parameters() if not p.requires_grad)}")

    optimizer = torch.optim.SGD(ttt_params, lr=args.ttt_lr, momentum=args.ttt_momentum)
    t0 = time.perf_counter()

    for ci in range(num_chunks):
        windows = chunk_windows[ci]
        if not windows:
            continue

        # --- PHASE 1: SCORE (no gradients) ---
        my_s = (len(windows) * rank) // world_size
        my_e = (len(windows) * (rank + 1)) // world_size
        my_windows = windows[my_s:my_e]

        base_model.eval()
        with torch.inference_mode():
            for bi in range(0, len(my_windows), args.ttt_batch_seqs):
                batch_ws = my_windows[bi:bi + args.ttt_batch_seqs]
                bsz = len(batch_ws)
                x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                wlens: list[int] = []

                for i, ws in enumerate(batch_ws):
                    end = min(ws + seq_len, total_tokens)
                    wlen = end - ws
                    wlens.append(wlen)
                    chunk_tok = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
                    x_batch[i, :wlen] = chunk_tok[:-1]
                    y_batch[i, :wlen] = chunk_tok[1:]

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = base_model.forward_logits(x_batch)

                nll = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)).float(),
                    y_batch.reshape(-1), reduction="none",
                ).reshape(bsz, seq_len)

                for i in range(bsz):
                    wlen = wlens[i]
                    scored_nll = nll[i, :wlen].to(torch.float64)
                    loss_sum += scored_nll.sum()
                    token_count += float(wlen)
                    tgt = y_batch[i, :wlen]
                    prev = x_batch[i, :wlen]
                    tb = base_bytes_lut[tgt].to(torch.float64)
                    tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                    byte_count += tb.sum()

        # --- PHASE 2: TRAIN on scored chunk (skip last chunk) ---
        is_last_chunk = (ci == num_chunks - 1)
        if not is_last_chunk and args.ttt_epochs > 0:
            base_model.train()
            chunk_start = ci * ttt_chunk
            chunk_end = min((ci + 1) * ttt_chunk, total_tokens)
            chunk_seqs = (chunk_end - chunk_start) // seq_len
            if chunk_seqs > 0:
                # Cosine LR decay across chunks
                cos_lr = args.ttt_lr * 0.5 * (1.0 + math.cos(math.pi * ci / max(num_chunks - 1, 1)))
                for pg in optimizer.param_groups:
                    pg['lr'] = cos_lr

                my_seq_s = (chunk_seqs * rank) // world_size
                my_seq_e = (chunk_seqs * (rank + 1)) // world_size

                for _ep in range(args.ttt_epochs):
                    for bs in range(my_seq_s, my_seq_e, args.ttt_batch_seqs):
                        be = min(bs + args.ttt_batch_seqs, my_seq_e)
                        start_tok = chunk_start + bs * seq_len
                        end_tok = chunk_start + be * seq_len + 1
                        if end_tok > val_tokens.numel():
                            continue
                        local = val_tokens[start_tok:end_tok].to(device=device, dtype=torch.int64)
                        x = local[:-1].reshape(-1, seq_len)
                        y = local[1:].reshape(-1, seq_len)
                        optimizer.zero_grad(set_to_none=True)
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            loss = base_model(x, y)
                        loss.backward()
                        if world_size > 1:
                            for p in ttt_params:
                                if p.grad is not None:
                                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
                        torch.nn.utils.clip_grad_norm_(ttt_params, args.ttt_grad_clip)
                        optimizer.step()

        if rank == 0 and (ci % 10 == 0 or ci == num_chunks - 1):
            elapsed = time.perf_counter() - t0
            rl = loss_sum.item() / max(token_count.item(), 1)
            rbpb = rl / math.log(2.0) * (token_count.item() / max(byte_count.item(), 1)) if token_count.item() > 0 else 0.0
            log0(f"  ttt_chunk [{ci+1}/{num_chunks}] bpb={rbpb:.6f} time={elapsed:.1f}s")

    # AllReduce across ranks
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)

    val_loss = (loss_sum / token_count).item()
    val_bpb = val_loss / math.log(2.0) * (token_count.item() / byte_count.item())

    # Restore grad for all params
    for p in base_model.parameters():
        p.requires_grad_(True)
    base_model.eval()

    log0(f"ttt:done val_loss={val_loss:.6f} val_bpb={val_bpb:.6f} "
         f"elapsed={time.perf_counter() - t0:.1f}s")
    return val_loss, val_bpb


# -----------------------------
# POST-TRAINING QUANTIZATION
# -----------------------------
#
# It's silly to export our model, which is trained in bf16 and fp32, at that same precision.
# Instead, we get approximately the same model (with a small hit) by quantizing the model to int8 & zlib compressing.
# We can then decompress the model and run in higher precision for evaluation, after closing in under the size limit.

# Byte-shuffle: rearrange bytes by stride offset before Brotli to improve compression ratio.
# Ported from rank 9 (Marko Sisovic, 2026-03-31_ParallelResiduals_MiniDepthRecurrence).
def _byte_shuffle(data: bytes, stride: int = 2) -> bytes:
    n = len(data)
    arr = bytearray(n)
    dst = 0
    for s in range(stride):
        for i in range(s, n, stride):
            arr[dst] = data[i]
            dst += 1
    return b"BSHF" + stride.to_bytes(2, "little") + bytes(arr)

def _byte_unshuffle(data: bytes) -> bytes:
    if data[:4] != b"BSHF":
        return data
    stride = int.from_bytes(data[4:6], "little")
    payload = data[6:]
    n = len(payload)
    arr = bytearray(n)
    src = 0
    for s in range(stride):
        for i in range(s, n, stride):
            arr[i] = payload[src]
            src += 1
    return bytes(arr)

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights",
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0

# Embedding tensor name patterns — these get int8 even in int6 mode (more sensitive to quantization)
EMBED_TENSOR_NAME_PATTERNS = ("tok_emb", "lm_head")

def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())

def keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str]) -> Tensor:
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t

def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        # Matrices get one scale per row, which usually tracks output-channel
        # ranges much better than a single tensor-wide scale.
        clip_abs = (
            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()

    # Vectors / scalars use a simpler per-tensor scale.
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale

def quantize_float_tensor_int6_sdclip(t: Tensor, clip_k: float) -> tuple[Tensor, Tensor]:
    """Int6 quantization with SDClip: clip = k * std(row). Values in [-31, 31], stored as int8."""
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = t32.std(dim=1) * clip_k
        clip_abs = clip_abs.clamp_min(1e-8)
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 31.0).clamp_min(1.0 / 31.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -31, 31).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()
    # Vectors / scalars — per-tensor int6
    std_val = float(t32.std().item()) if t32.numel() > 1 else float(t32.abs().item())
    clip_abs = max(std_val * clip_k, 1e-8)
    scale = torch.tensor(clip_abs / 31.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -31, 31).to(torch.int8).contiguous()
    return q, scale

def quantize_float_tensor_int8_sdclip(t: Tensor, clip_k: float) -> tuple[Tensor, Tensor]:
    """Int8 quantization with SDClip: clip = k * std(row). For embeddings in int6 mode."""
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = t32.std(dim=1) * clip_k
        clip_abs = clip_abs.clamp_min(1e-8)
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()
    clip_abs = max(float(t32.std().item()) * clip_k, 1e-8) if t32.numel() > 1 else max(float(t32.abs().item()) * clip_k, 1e-8)
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale

# ─────────────────────────────────────────────────────────────────────────────
# GPTQ: Hessian-based int6 quantization with error compensation
# Ported from PG rank 9 (Marko Sisovic, 2026-03-31_ParallelResiduals_MiniDepthRecurrence)
# Algorithm: Frantar et al., "GPTQ: Accurate Post-Training Quantization for
#            Generative Pre-trained Transformers", ICLR 2023.
#            https://arxiv.org/abs/2210.17323
# ─────────────────────────────────────────────────────────────────────────────

def quantize_int6_per_row(t: Tensor, clip_range: int = 31) -> tuple[Tensor, Tensor]:
    """Naive int6 with multi-percentile clip search. Fallback for GPTQ Cholesky failure."""
    t32 = t.float()
    if t32.ndim == 2:
        best_q, best_s, best_err = None, None, float("inf")
        for pct in [0.999, 0.9995, 0.9999, 0.99999, 1.0]:
            if pct < 1.0:
                row_clip = torch.quantile(t32.abs(), pct, dim=1)
            else:
                row_clip = t32.abs().amax(dim=1)
            s = (row_clip / clip_range).clamp_min(1.0 / clip_range).to(torch.float16)
            q = torch.clamp(torch.round(t32 / s.float()[:, None]), -clip_range, clip_range).to(torch.int8)
            recon = q.float() * s.float()[:, None]
            err = (t32 - recon).pow(2).mean().item()
            if err < best_err:
                best_q, best_s, best_err = q, s, err
        return best_q, best_s
    amax = t32.abs().max().item()
    scale = torch.tensor(amax / clip_range if amax > 0 else 1.0, dtype=torch.float16)
    q = torch.clamp(torch.round(t32 / scale.float()), -clip_range, clip_range).to(torch.int8)
    return q, scale

def gptq_quantize_weight(W: Tensor, H: Tensor, clip_range: int = 63,
                         clip_sigmas: float = 12.85, block_size: int = 128,
                         percdamp: float = 0.01) -> tuple[Tensor, Tensor]:
    """GPTQ column-by-column quantization with Hessian error compensation.

    Uses k×std clipping (Kevin Clark, rank 5) instead of multi-percentile search:
      scale = clip_sigmas × std(row) / clip_range
    Single GPTQ pass (5× faster), directly controls compressed size via k.

    For each column j in weight matrix W (reordered by Hessian diagonal):
      1. Quantize column j: q_j = round(w_j / scale)
      2. Compute error: err_j = (w_j - q_j * scale) / H_inv[j,j]
      3. Compensate remaining columns: W[:, j+1:] -= err_j * H_inv[j, j+1:]
    Algorithm: Frantar et al., ICLR 2023. Clipping: Kevin Clark, PG rank 5.
    """
    W_orig = W.float().clone()
    rows, cols = W_orig.shape
    H = H.float().clone()
    # Handle dead columns (zero Hessian diagonal)
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    # Damping for numerical stability
    damp = percdamp * H.diag().mean()
    H.diagonal().add_(damp)
    # Column reordering by Hessian diagonal (descending)
    perm = torch.argsort(H.diag(), descending=True)
    invperm = torch.argsort(perm)
    W_perm = W_orig[:, perm].clone()
    W_perm[:, dead[perm]] = 0
    H = H[perm][:, perm]
    # Cholesky inverse for error compensation
    try:
        Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))
        Hinv = torch.linalg.cholesky(Hinv, upper=True)
    except torch.linalg.LinAlgError:
        return quantize_int6_per_row(W_orig, clip_range)
    # k×std clipping: scale = clip_sigmas × std(row) / clip_range
    # Directly controls entropy/compression tradeoff (Kevin Clark, PG rank 5 README).
    row_std = W_orig.std(dim=1)
    s = (clip_sigmas * row_std / clip_range).clamp_min(1e-10).to(torch.float16)
    sf = s.float()
    Q = torch.zeros(rows, cols, dtype=torch.int8)
    W_work = W_perm.clone()
    for i1 in range(0, cols, block_size):
        i2 = min(i1 + block_size, cols)
        W_block = W_work[:, i1:i2].clone()
        Hinv_block = Hinv[i1:i2, i1:i2]
        Err = torch.zeros(rows, i2 - i1)
        for j in range(i2 - i1):
            w_col = W_block[:, j]
            d = Hinv_block[j, j]
            q_col = torch.clamp(torch.round(w_col / sf), -clip_range, clip_range)
            Q[:, i1 + j] = q_col.to(torch.int8)
            err = (w_col - q_col.float() * sf) / d
            Err[:, j] = err
            W_block[:, j:] -= err.unsqueeze(1) * Hinv_block[j, j:].unsqueeze(0)
        if i2 < cols:
            W_work[:, i2:] -= Err @ Hinv[i1:i2, i2:]
    # Un-permute columns back to original order
    Q = Q[:, invperm]
    return Q, s


def generate_autoregressive_calib(
    model: nn.Module, device: torch.device, num_seqs: int = 64,
    seq_len: int = 2048, vocab_size: int = 1024, temperature: float = 0.8,
    batch_size: int = 8, seed: int = 42,
) -> list[Tensor]:
    """Generate calibration sequences autoregressively from the trained model.
    No training data needed — the model generates its own representative data.
    Ported from rank 9 (Marko Sisovic).
    """
    was_training = model.training
    model.eval()
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    all_tokens: list[Tensor] = []
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for batch_start in range(0, num_seqs, batch_size):
            bs = min(batch_size, num_seqs - batch_start)
            tokens = torch.randint(0, vocab_size, (bs, 1), device=device, generator=rng)
            for _ in range(seq_len - 1):
                logits = model.forward_logits(tokens)
                next_logit = logits[:, -1, :]
                probs = torch.softmax(next_logit / max(temperature, 1e-4), dim=-1)
                next_tok = torch.multinomial(probs, 1, generator=rng)
                tokens = torch.cat([tokens, next_tok], dim=1)
            for i in range(bs):
                all_tokens.append(tokens[i : i + 1].detach().clone())
    model.train(was_training)
    return all_tokens


def _gptq_init_hessians(model: nn.Module, device: torch.device) -> dict[str, Tensor]:
    """Allocate zero Hessian matrices for each quantizable weight layer."""
    hessians: dict[str, Tensor] = {}
    for i, block in enumerate(model.blocks):
        dim = block.attn.c_q.weight.shape[1]  # input dim (always model_dim)
        kv_dim = block.attn.c_k.weight.shape[1]  # same as dim
        hidden_dim = block.mlp.fc.weight.shape[0]  # mlp_mult * dim
        # QKV projections share the same input → same Hessian shape (dim, dim)
        for proj in ("c_q", "c_k", "c_v"):
            hessians[f"blocks.{i}.attn.{proj}.weight"] = torch.zeros(dim, dim, dtype=torch.float32, device=device)
        # Output projection: input is attention output (dim)
        hessians[f"blocks.{i}.attn.proj.weight"] = torch.zeros(dim, dim, dtype=torch.float32, device=device)
        # MLP up: input is normed residual (dim)
        hessians[f"blocks.{i}.mlp.fc.weight"] = torch.zeros(dim, dim, dtype=torch.float32, device=device)
        # MLP down: input is squared activation (hidden_dim)
        hessians[f"blocks.{i}.mlp.proj.weight"] = torch.zeros(hidden_dim, hidden_dim, dtype=torch.float32, device=device)
    return hessians


def _gptq_accum_hessians(hessians: dict[str, Tensor], model: nn.Module, block_idx: int | None = None) -> None:
    """Accumulate H = X^T X from saved activations (one forward pass).
    If block_idx is set, only accumulate for that block (sequential mode).
    """
    blocks = [(block_idx, model.blocks[block_idx])] if block_idx is not None else enumerate(model.blocks)
    for i, block in blocks:
        dim = block.attn.c_q.weight.shape[1]
        hidden_dim = block.mlp.fc.weight.shape[0]
        if hasattr(block.attn, "_gptq_qkv_in"):
            qkv_in = block.attn._gptq_qkv_in.float().reshape(-1, dim)
            h_qkv = qkv_in.t() @ qkv_in
            hessians[f"blocks.{i}.attn.c_q.weight"] += h_qkv
            hessians[f"blocks.{i}.attn.c_k.weight"] += h_qkv
            hessians[f"blocks.{i}.attn.c_v.weight"] += h_qkv
        if hasattr(block.attn, "_gptq_o_in"):
            o_in = block.attn._gptq_o_in.float().reshape(-1, dim)
            hessians[f"blocks.{i}.attn.proj.weight"] += o_in.t() @ o_in
        if hasattr(block.mlp, "_gptq_up_in"):
            up_in = block.mlp._gptq_up_in.float().reshape(-1, dim)
            hessians[f"blocks.{i}.mlp.fc.weight"] += up_in.t() @ up_in
        if hasattr(block.mlp, "_gptq_down_in"):
            down_in = block.mlp._gptq_down_in.float().reshape(-1, hidden_dim)
            hessians[f"blocks.{i}.mlp.proj.weight"] += down_in.t() @ down_in


def _gptq_finalize_hessians(hessians: dict[str, Tensor], num_batches: int, percdamp: float = 0.01) -> None:
    """Average Hessians, add damping, move to CPU."""
    for name in hessians:
        hessians[name] = hessians[name].cpu() / max(num_batches, 1)
        damp = percdamp * torch.diag(hessians[name]).mean().clamp_min(1e-6)
        hessians[name] += damp * torch.eye(hessians[name].shape[0])


def gptq_collect_hessians_from_tokens(
    model: nn.Module, token_seqs: list[Tensor], device: torch.device,
    percdamp: float = 0.01,
) -> dict[str, Tensor]:
    """Run forward passes on calibration tokens and collect Hessians for GPTQ."""
    hessians = _gptq_init_hessians(model, device)
    # Enable activation saving
    for block in model.blocks:
        block.attn._save_gptq = True
        block.mlp._save_gptq = True
    was_training = model.training
    model.eval()
    # Use no_grad (not inference_mode) — inference_mode creates special tensors that
    # poison the Rotary cache and crash TTT. Kevin Clark rank 5 uses no_grad for this reason.
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for seq in token_seqs:
            x = seq[:, :-1].to(device=device, dtype=torch.int64)
            y = seq[:, 1:].to(device=device, dtype=torch.int64)
            model(x, y)
            _gptq_accum_hessians(hessians, model)
    # Disable activation saving and clean up
    for block in model.blocks:
        block.attn._save_gptq = False
        block.mlp._save_gptq = False
        for attr in ("_gptq_qkv_in", "_gptq_o_in"):
            if hasattr(block.attn, attr):
                delattr(block.attn, attr)
        for attr in ("_gptq_up_in", "_gptq_down_in"):
            if hasattr(block.mlp, attr):
                delattr(block.mlp, attr)
        # Invalidate Rotary cos/sin cache — inference_mode() creates tensors that
        # can't be used in autograd. TTT needs to recompute them with gradients.
        block.attn.rotary._cos_cached = None
        block.attn.rotary._sin_cached = None
        block.attn.rotary._seq_len_cached = 0
    _gptq_finalize_hessians(hessians, len(token_seqs), percdamp=percdamp)
    model.train(was_training)
    return hessians


def gptq_collect_hessians_with_hooks(
    model: nn.Module, token_seqs: list[Tensor], device: torch.device,
    percdamp: float = 0.01,
) -> dict[str, Tensor]:
    """Collect Hessians using register_forward_hook (Kevin Clark approach).

    Instead of manual _save_gptq flags, uses PyTorch hooks on each Linear layer
    to capture exact inputs. More robust and guaranteed correct.
    """
    hessians: dict[str, Tensor] = {}
    hooks = []
    # Register hooks on every quantizable Linear layer
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        weight_name = f"{name}.weight"
        dim = module.weight.shape[1]
        hessians[weight_name] = torch.zeros(dim, dim, dtype=torch.float32, device=device)
        def make_hook(wname, d):
            def hook_fn(mod, inp, out):
                x = inp[0].detach().float().reshape(-1, d)
                hessians[wname] += x.t() @ x
            return hook_fn
        hooks.append(module.register_forward_hook(make_hook(weight_name, dim)))

    was_training = model.training
    model.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for seq in token_seqs:
            x = seq[:, :-1].to(device=device, dtype=torch.int64)
            y = seq[:, 1:].to(device=device, dtype=torch.int64)
            model(x, y)
    # Remove hooks
    for h in hooks:
        h.remove()
    # Invalidate Rotary cache
    for block in model.blocks:
        block.attn.rotary._cos_cached = None
        block.attn.rotary._sin_cached = None
        block.attn.rotary._seq_len_cached = 0
    _gptq_finalize_hessians(hessians, len(token_seqs), percdamp=percdamp)
    model.train(was_training)
    return hessians


def gptq_sequential_quantize(
    model: nn.Module, token_seqs: list[Tensor], device: torch.device,
    clip_range: int = 63, clip_sigmas: float = 12.85, percdamp: float = 0.01,
    use_hooks: bool = False,
) -> dict[str, Tensor]:
    """Sequential block-by-block GPTQ quantization.

    For each block i (0 → N):
      1. Run calibration through full model (blocks 0..i-1 already dequantized)
      2. Collect Hessian for block i only
      3. Quantize block i's weights with GPTQ
      4. Replace block i's weights with dequantized(quantized(weights))

    This ensures each block's Hessian reflects the actual quantized inputs it
    will see at inference, eliminating the full-precision/quantized mismatch.
    """
    import copy
    all_hessians: dict[str, Tensor] = {}
    num_blocks = len(model.blocks)

    was_training = model.training
    model.eval()

    # Save original weights — sequential quantization modifies them in-place for
    # calibration, but quantize_state_dict_int6_sdclip needs the originals.
    saved_weights: dict[str, Tensor] = {}
    for i, block in enumerate(model.blocks):
        for name, param in block.named_parameters():
            saved_weights[f"blocks.{i}.{name}"] = param.data.clone()

    for block_idx in range(num_blocks):
        block = model.blocks[block_idx]
        print(f"gptq_sequential: quantizing block {block_idx}/{num_blocks}", flush=True)

        if use_hooks:
            # Hook-based: register hooks only on this block's Linear layers
            block_hessians: dict[str, Tensor] = {}
            hooks = []
            prefix = f"blocks.{block_idx}."
            for name, module in block.named_modules():
                if not isinstance(module, nn.Linear):
                    continue
                weight_name = f"blocks.{block_idx}.{name}.weight"
                dim = module.weight.shape[1]
                block_hessians[weight_name] = torch.zeros(dim, dim, dtype=torch.float32, device=device)
                def make_hook(wname, d):
                    def hook_fn(mod, inp, out):
                        x = inp[0].detach().float().reshape(-1, d)
                        block_hessians[wname] += x.t() @ x
                    return hook_fn
                hooks.append(module.register_forward_hook(make_hook(weight_name, dim)))
        else:
            # Manual flag-based: enable only this block
            block.attn._save_gptq = True
            block.mlp._save_gptq = True
            block_hessians = _gptq_init_hessians(model, device)
            # Zero out all other blocks' Hessians (we only want this block)
            for key in list(block_hessians.keys()):
                if not key.startswith(f"blocks.{block_idx}."):
                    del block_hessians[key]
            # Re-init only this block's Hessians
            block_hessians = {}
            dim = block.attn.c_q.weight.shape[1]
            hidden_dim = block.mlp.fc.weight.shape[0]
            for suffix, size in [
                (f"blocks.{block_idx}.attn.c_q.weight", dim),
                (f"blocks.{block_idx}.attn.c_k.weight", dim),
                (f"blocks.{block_idx}.attn.c_v.weight", dim),
                (f"blocks.{block_idx}.attn.proj.weight", dim),
                (f"blocks.{block_idx}.mlp.fc.weight", dim),
                (f"blocks.{block_idx}.mlp.proj.weight", hidden_dim),
            ]:
                block_hessians[suffix] = torch.zeros(size, size, dtype=torch.float32, device=device)

        # Run calibration through full model (earlier blocks already have dequantized weights)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for seq in token_seqs:
                x = seq[:, :-1].to(device=device, dtype=torch.int64)
                y = seq[:, 1:].to(device=device, dtype=torch.int64)
                model(x, y)
                if not use_hooks:
                    _gptq_accum_hessians(block_hessians, model, block_idx=block_idx)

        # Clean up
        if use_hooks:
            for h in hooks:
                h.remove()
        else:
            block.attn._save_gptq = False
            block.mlp._save_gptq = False
            for attr in ("_gptq_qkv_in", "_gptq_o_in"):
                if hasattr(block.attn, attr):
                    delattr(block.attn, attr)
            for attr in ("_gptq_up_in", "_gptq_down_in"):
                if hasattr(block.mlp, attr):
                    delattr(block.mlp, attr)

        # Invalidate Rotary cache
        block.attn.rotary._cos_cached = None
        block.attn.rotary._sin_cached = None
        block.attn.rotary._seq_len_cached = 0

        _gptq_finalize_hessians(block_hessians, len(token_seqs), percdamp=percdamp)

        # Quantize this block's weights and replace with dequantized versions
        with torch.no_grad():
            for wname, H in block_hessians.items():
                # Navigate to the parameter
                parts = wname.replace(".weight", "").split(".")
                # parts = ["blocks", idx, "attn"/"mlp", layer_name]
                param_module = model
                for p in parts:
                    param_module = getattr(param_module, p) if not p.isdigit() else param_module[int(p)]
                W = param_module.weight.data.detach().cpu().float()
                Q, s = gptq_quantize_weight(W, H.cpu(), clip_range=clip_range,
                                            clip_sigmas=clip_sigmas, percdamp=percdamp)
                # Dequantize and replace: model sees "quantized" weights for subsequent blocks
                W_deq = Q.float() * s.float().unsqueeze(1)
                param_module.weight.data = W_deq.to(device=device, dtype=param_module.weight.dtype)
                all_hessians[wname] = H

    # Restore original weights — the in-place dequantized weights were only needed
    # for calibration. quantize_state_dict_int6_sdclip will do the final quantization
    # using the improved Hessians collected here.
    for wname, orig_data in saved_weights.items():
        parts = wname.split(".")
        module = model
        for p in parts[:-1]:
            module = getattr(module, p) if not p.isdigit() else module[int(p)]
        getattr(module, parts[-1]).data = orig_data

    model.train(was_training)
    return all_hessians


def quantize_state_dict_int8(state_dict: dict[str, Tensor]):
    # Single supported clean-script export format:
    # - per-row int8 for 2D float tensors
    # - per-tensor int8 for other float tensors
    # - exact passthrough for non-floats
    # - passthrough for small float tensors, stored as fp16 to save bytes
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "int8_payload_bytes"),
        0,
    )

    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)

        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int8_payload_bytes"] += tensor_nbytes(t)
            continue

        # Small float tensors are cheap enough to keep directly. We still downcast
        # fp32/bf16 passthrough tensors to fp16 so metadata does not dominate size.
        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue

        stats["num_float_tensors"] += 1
        q, s = quantize_float_tensor(t)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)

    obj: dict[str, object] = {
        "__quant_format__": "int8_clean_per_row_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats

def dequantize_state_dict_int8(obj: dict[str, object]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            s = s.to(dtype=torch.float32)
            # Broadcast the saved row scale back across trailing dimensions.
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
        else:
            scale = float(s.item())
            out[name] = (q.float() * scale).to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
        # Restore small tensors, undoing the temporary fp16 storage cast if needed.
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out

def quantize_state_dict_int6_sdclip(
    state_dict: dict[str, Tensor], sdclip_k: float, embed_clip_k: float,
    hessians: dict[str, Tensor] | None = None,
    gptq_clip_range: int = 63, gptq_clip_sigmas: float = 12.85,
    gptq_embed: bool = False, gptq_percdamp: float = 0.01,
):
    """Int6 SDClip quantization. When hessians provided, uses GPTQ for 2D weight matrices."""
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    quant_bits: dict[str, int] = {}
    gptq_count, naive_count = 0, 0
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "int8_payload_bytes"),
        0,
    )

    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)

        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int8_payload_bytes"] += tensor_nbytes(t)
            continue

        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue

        stats["num_float_tensors"] += 1
        is_embed = any(pat in name for pat in EMBED_TENSOR_NAME_PATTERNS)
        if is_embed and gptq_embed and hessians is not None and name in hessians and t.ndim == 2:
            # GPTQ on embeddings: Hessian-based error compensation (better than simple SDClip)
            q, s = gptq_quantize_weight(t, hessians[name].cpu(), clip_range=gptq_clip_range,
                                        clip_sigmas=gptq_clip_sigmas, percdamp=gptq_percdamp)
            quant_bits[name] = 6
            gptq_count += 1
        elif is_embed:
            q, s = quantize_float_tensor_int8_sdclip(t, embed_clip_k)
            quant_bits[name] = 8
        elif hessians is not None and name in hessians and t.ndim == 2:
            # GPTQ path: use Hessian-based error compensation (Frantar et al., ICLR 2023)
            q, s = gptq_quantize_weight(t, hessians[name].cpu(), clip_range=gptq_clip_range, clip_sigmas=gptq_clip_sigmas)
            quant_bits[name] = 6
            gptq_count += 1
        else:
            q, s = quantize_float_tensor_int6_sdclip(t, sdclip_k)
            quant_bits[name] = 6
            naive_count += 1
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)

    if hessians:
        print(f"gptq_quantize: {gptq_count} GPTQ layers, {naive_count} naive layers", flush=True)

    obj: dict[str, object] = {
        "__quant_format__": "int6_sdclip_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
        "quant_bits": quant_bits,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats

def dequantize_state_dict_int6(obj: dict[str, object]) -> dict[str, Tensor]:
    """Dequantize int6/int8 mixed quantized state dict."""
    out: dict[str, Tensor] = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            s = s.to(dtype=torch.float32)
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
        else:
            scale = float(s.item())
            out[name] = (q.float() * scale).to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out


# -----------------------------
# DATA LOADING
# -----------------------------

def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    # SHARD HEADER INTS & SHARD_MAGIC
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


class TokenStream:
    # Reads shards sequentially and wraps around forever. The training loop therefore
    # has deterministic, simple streaming behavior with no sampling or workers.
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    # Each call consumes a contiguous chunk from the shared token stream, then slices out
    # one disjoint span per rank. The extra "+1" token lets us build (x, y) by shifting.
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

# -----------------------------
# TRANSFORMER MODULES
# -----------------------------

class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    # Keep weights in fp32 for optimizer/state quality, cast at matmul time for bf16 compute.
    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    # Keep small/control parameters in fp32 even when the model body runs in bf16.
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()


class Rotary(nn.Module):
    # Caches cos/sin tables per sequence length on the current device.
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        # .clone() ensures cached tensors created under inference_mode() don't poison
        # autograd when TTT later needs gradients (fixes GPTQ → eval → TTT crash).
        return self._cos_cached.to(dtype=dtype).clone(), self._sin_cached.to(dtype=dtype).clone()


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


# ╔═══════════════════════════════════════════════════════════════════════════════════╗
# ║                     CAUSAL SELF-ATTENTION (with GQA)                              ║
# ║                                                                                   ║
# ║  Plain English: Each token asks "which earlier tokens should I pay attention to?" ║
# ║  It computes a compatibility score (Q·K) with every previous token, then takes    ║
# ║  a weighted average of their values (V). "Causal" = can only look backward.      ║
# ║                                                                                   ║
# ║  ┌──────────────── Input x: [batch, seq_len, 512] ────────────────┐               ║
# ║  │                            │                                    │               ║
# ║  │         ┌──────────────────┼──────────────────┐                 │               ║
# ║  │         │                  │                  │                 │               ║
# ║  │         ▼                  ▼                  ▼                 │               ║
# ║  │    c_q (512→512)     c_k (512→256)      c_v (512→256)          │               ║
# ║  │         │                  │                  │                 │               ║
# ║  │         ▼                  ▼                  ▼                 │               ║
# ║  │    Q: 8 heads × 64d  K: 4 heads × 64d  V: 4 heads × 64d      │               ║
# ║  │         │                  │                                    │               ║
# ║  │         ▼                  ▼                                    │               ║
# ║  │    RMSNorm(Q)         RMSNorm(K)       ← stabilize magnitudes  │               ║
# ║  │         │                  │                                    │               ║
# ║  │         ▼                  ▼                                    │               ║
# ║  │    RoPE(Q)            RoPE(K)           ← encode positions     │               ║
# ║  │         │                                                       │               ║
# ║  │         ▼                                                       │               ║
# ║  │    Q × q_gain                           ← learnable sharpness  │               ║
# ║  │         │                  │                  │                 │               ║
# ║  │         └──────────────────┼──────────────────┘                 │               ║
# ║  │                            ▼                                    │               ║
# ║  │              scaled_dot_product_attention                       │               ║
# ║  │              softmax(Q·Kᵀ / √64) · V                           │               ║
# ║  │              (causal mask + FlashAttention + GQA)               │               ║
# ║  │                            │                                    │               ║
# ║  │                            ▼                                    │               ║
# ║  │                     proj (512→512)                              │               ║
# ║  │                            │                                    │               ║
# ║  │                            ▼                                    │               ║
# ║  │              Output: [batch, seq_len, 512]                      │               ║
# ║  └─────────────────────────────────────────────────────────────────┘               ║
# ║                                                                                   ║
# ║  GROUPED QUERY ATTENTION (GQA) — why K,V have 4 heads but Q has 8:               ║
# ║                                                                                   ║
# ║    Standard (MHA, 8×8):           This model (GQA, 8×4):                          ║
# ║    Q₀ → K₀  Q₄ → K₄              Q₀ ─┐                                          ║
# ║    Q₁ → K₁  Q₅ → K₅              Q₁ ─┤→ K₀,V₀                                  ║
# ║    Q₂ → K₂  Q₆ → K₆              Q₂ ─┐                                          ║
# ║    Q₃ → K₃  Q₇ → K₇              Q₃ ─┤→ K₁,V₁                                  ║
# ║    (8 KV pairs = full cost)        Q₄ ─┐                                          ║
# ║                                    Q₅ ─┤→ K₂,V₂                                  ║
# ║                                    Q₆ ─┐                                          ║
# ║                                    Q₇ ─┤→ K₃,V₃                                  ║
# ║                                    (4 KV pairs = 50% memory)                      ║
# ║                                                                                   ║
# ║  CAUSAL MASK (is_causal=True) — what makes this a language model:                 ║
# ║                                                                                   ║
# ║       attends to→  t₀  t₁  t₂  t₃  t₄                                           ║
# ║    t₀              ✓   ✗   ✗   ✗   ✗     "The"  can only see itself              ║
# ║    t₁              ✓   ✓   ✗   ✗   ✗     "cat"  sees "The cat"                   ║
# ║    t₂              ✓   ✓   ✓   ✗   ✗     "sat"  sees "The cat sat"               ║
# ║    t₃              ✓   ✓   ✓   ✓   ✗     "on"   sees "The cat sat on"            ║
# ║    t₄              ✓   ✓   ✓   ✓   ✓     "the"  sees everything before it        ║
# ║                                                                                   ║
# ║    Each token predicts the NEXT token without peeking at the answer.              ║
# ║                                                                                   ║
# ║  WHY EACH STEP EXISTS:                                                            ║
# ║  • RMSNorm on Q,K  → prevents attention scores from exploding to ±inf            ║
# ║  • RoPE on Q,K     → encodes WHERE each token is (position 0, 1, 2...)           ║
# ║  • q_gain (4.0-5.25) → learnable per-head sharpness (higher = more focused)      ║
# ║  • √64 denominator → scales dot products so softmax doesn't saturate             ║
# ║  • proj            → mixes info from all 8 heads back to 512 dimensions          ║
# ╚═══════════════════════════════════════════════════════════════════════════════════╝
class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
        gated_attn: str = "none",
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads        # 512 / 8 = 64 dimensions per head
        self.gated_attn = gated_attn
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        kv_dim = self.num_kv_heads * self.head_dim  # 4 × 64 = 256 (half of Q's 512, GQA savings)
        # Gated Attention (arxiv.org/abs/2505.06708, NeurIPS 2025 Best Paper):
        # Widen Q projection to also produce gate logits from the same input.
        #   "none"        → c_q: 512→512,  no gate
        #   "headwise"    → c_q: 512→520,  +8 dims (1 gate scalar per head, ~9K extra params/layer)
        #   "elementwise" → c_q: 512→1024, +512 dims (1 gate per dimension, doubles Q projection)
        if gated_attn == "headwise":
            self.gate_dim = num_heads           # 8 extra outputs
        elif gated_attn == "elementwise":
            self.gate_dim = dim                 # 512 extra outputs
        else:
            self.gate_dim = 0
        self.c_q = CastedLinear(dim, dim + self.gate_dim, bias=False)  # Query + gate logits
        self.c_k = CastedLinear(dim, kv_dim, bias=False)    # Key:   512→256 (only 4 KV heads)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)    # Value: 512→256 (only 4 KV heads)
        self.proj = CastedLinear(dim, dim, bias=False)      # Output projection: 512→512
        self.proj._zero_init = True                          # Start at zero so skip connections dominate early
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x: Tensor, v0: Tensor | None = None, vr_alpha: float = 0.0) -> Tensor | tuple[Tensor, Tensor]:
        """Forward pass. If vr_alpha > 0 and v0 is None, returns (output, v) for V₀ caching.
        If v0 is provided, blends it with current V before attention (ResFormer)."""
        bsz, seqlen, dim = x.shape
        # GPTQ hook: save input to QKV projections for Hessian collection
        if getattr(self, "_save_gptq", False):
            self._gptq_qkv_in = x.detach()
        # Step 1: Project input → Q (+ gate logits if gated attention is enabled)
        q_out = self.c_q(x)
        if self.gate_dim > 0:
            q_raw, gate_logits = q_out.split([dim, self.gate_dim], dim=-1)
        else:
            q_raw = q_out
        # Reshape into multi-head format: [batch, heads, seq, head_dim]
        q = q_raw.reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        # ResFormer: cache V from layer 0, or blend V₀ into current layer's V
        cache_v = (vr_alpha > 0 and v0 is None)  # Layer 0: need to return V for caching
        if v0 is not None and vr_alpha > 0:
            v = (1 - vr_alpha) * v + vr_alpha * v0
        # Step 2: Normalize Q and K so dot products don't explode
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        # Step 3: Apply rotary position embeddings — encode WHERE each token sits
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        # Step 4: Scale Q by learnable per-head gain — controls attention sharpness
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        # Step 5: The actual attention — softmax(Q·Kᵀ/√d) · V with causal mask
        #         Uses FlashAttention kernel under the hood (memory-efficient, no full NxN matrix)
        #         GQA: repeat K,V heads to match Q heads (e.g. 4 KV → 8 Q, repeat 2×)
        v_before_repeat = v  # Save pre-repeat V for ResFormer caching
        if self.num_kv_heads != self.num_heads:
            rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=True,
        )
        # Step 6 (Gated Attention): sigmoid gate AFTER attention, BEFORE output projection
        #   Gate logits come from the Q projection (query-dependent, input-dependent).
        #   sigmoid(gate) ∈ [0,1] lets the model suppress uninformative heads/dims per token.
        #   Headwise:    [bsz, seq, 8]   → [bsz, 8, seq, 1]   (one scalar per head)
        #   Elementwise: [bsz, seq, 512] → [bsz, 8, seq, 64]  (one per dimension)
        if self.gated_attn == "headwise":
            gate = torch.sigmoid(gate_logits)                   # [bsz, seqlen, num_heads]
            gate = gate.transpose(1, 2).unsqueeze(-1)           # [bsz, num_heads, seqlen, 1]
            y = y * gate
        elif self.gated_attn == "elementwise":
            gate = torch.sigmoid(gate_logits)                   # [bsz, seqlen, dim]
            gate = gate.reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
            y = y * gate
        # Step 7: Reshape from multi-head back to [batch, seq, 512] and project
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        # GPTQ hook: save input to output projection for Hessian collection
        if getattr(self, "_save_gptq", False):
            self._gptq_o_in = y.detach()
        out = self.proj(y)
        if cache_v:
            return out, v_before_repeat  # Return V for caching (layer 0)
        return out


class MLP(nn.Module):
    # relu^2 MLP from the original modded-nanogpt setup
    def __init__(self, dim: int, mlp_mult: int, activation: str = "relu2"):
        super().__init__()
        self.activation = activation
        hidden = mlp_mult * dim
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        # GPTQ hook: save input to up-projection for Hessian collection
        if getattr(self, "_save_gptq", False):
            self._gptq_up_in = x.detach()
        x = self.fc(x)
        if self.activation == "leaky_relu2":
            x = F.leaky_relu(x, negative_slope=0.5)
        else:
            x = torch.relu(x)
        x = x.square()
        # GPTQ hook: save input to down-projection for Hessian collection
        if getattr(self, "_save_gptq", False):
            self._gptq_down_in = x.detach()
        return self.proj(x)


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
        gated_attn: str = "none",
        activation: str = "relu2",
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init, gated_attn)
        self.mlp = MLP(dim, mlp_mult, activation)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x: Tensor, x0: Tensor, v0: Tensor | None = None, vr_alpha: float = 0.0) -> Tensor | tuple[Tensor, Tensor]:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_result = self.attn(self.attn_norm(x), v0=v0, vr_alpha=vr_alpha)
        if isinstance(attn_result, tuple):
            attn_out, v_cached = attn_result  # Layer 0: attention returned V for caching
        else:
            attn_out, v_cached = attn_result, None
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        if v_cached is not None:
            return x, v_cached
        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
        gated_attn: str = "none",
        activation: str = "relu2",
        slm_enabled: bool = False,
        slm_ratio: float = 0.6,
        value_residual_alpha: float = 0.0,
    ):
        super().__init__()
        self.slm_enabled = slm_enabled
        self.slm_ratio = slm_ratio
        self.value_residual_alpha = value_residual_alpha
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList(
            [
                Block(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_mult,
                    rope_base,
                    qk_gain_init,
                    gated_attn,
                    activation,
                )
                for i in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self._init_weights()

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def _run_backbone(self, input_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        vr_alpha = self.value_residual_alpha
        v0: Tensor | None = None  # Cached V from layer 0 (ResFormer)
        skips: list[Tensor] = []
        for i in range(self.num_encoder_layers):
            result = self.blocks[i](x, x0, v0=v0, vr_alpha=vr_alpha)
            if isinstance(result, tuple):
                x, v0 = result  # Layer 0 returned V for caching
            else:
                x = result
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0, v0=v0, vr_alpha=vr_alpha)
        x = self.final_norm(x)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)

    def forward_logits(self, input_ids: Tensor) -> Tensor:
        """Return logits [batch, seq, vocab] without computing loss."""
        return self._run_backbone(input_ids)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        logits = self._run_backbone(input_ids)
        logits = logits.reshape(-1, logits.size(-1))
        targets = target_ids.reshape(-1)
        # Rho-1 Selective Language Modeling (https://arxiv.org/abs/2401.04056)
        # Train only on top-k% hardest tokens; skip well-predicted ones
        if self.training and self.slm_enabled:
            per_token_loss = F.cross_entropy(logits.float(), targets, reduction="none")
            k = max(1, int(per_token_loss.numel() * self.slm_ratio))
            topk_losses, _ = torch.topk(per_token_loss, k)
            return topk_losses.mean()
        return F.cross_entropy(logits.float(), targets, reduction="mean")


# -----------------------------
# TRAINING
# -----------------------------

def main() -> None:
    global zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    # -----------------------------
    # DISTRIBUTED + CUDA SETUP
    # -----------------------------

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral")
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    # Fast math knobs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp

    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(
        subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout,
        console=False,
    )
    log0("=" * 100, console=False)

    # -----------------------------
    # TOKENIZER + VALIDATION METRIC SETUP
    # -----------------------------

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"Script only setup for SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")

    # -----------------------------
    # MODEL + OPTIMIZER SETUP
    # -----------------------------

    base_model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        gated_attn=args.gated_attn,
        activation=args.activation,
        slm_enabled=args.slm_enabled,
        slm_ratio=args.slm_ratio,
        value_residual_alpha=args.value_residual_alpha,
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    model: nn.Module = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model

    # =====================================================================
    # OPTIMIZER SPLIT — Different param types get different optimizers & LRs
    # =====================================================================
    #
    # The model's parameters are sorted into 3-4 groups, each with its own
    # optimizer and learning rate. Think of it like a team where each role
    # needs a different management style:
    #
    #   ┌─────────────────────────────────────────────────────────────────┐
    #   │  Parameter Group        Optimizer   LR       % of Params       │
    #   │  ─────────────────────  ─────────   ──────   ───────────       │
    #   │  1. Embedding table     Adam        0.05     ~3%  (SP1024)     │
    #   │     (tok_emb.weight)                         ~12% (SP8192)     │
    #   │     "the dictionary"                                           │
    #   │                                                                │
    #   │  2. Matrix weights      MUON        0.04     ~95% of blocks    │
    #   │     (Q, K, V, proj,                          The heavy lifters │
    #   │      fc, mlp.proj)                           — 2D matrices     │
    #   │     "attention + MLP                         that do the real  │
    #   │      computation"                            learning          │
    #   │                                                                │
    #   │  3. Scalar params       Adam        0.04     <1%               │
    #   │     (attn_scale,                             Tiny knobs that   │
    #   │      mlp_scale,                              fine-tune how     │
    #   │      resid_mix,                              blocks blend      │
    #   │      q_gain,                                 their outputs     │
    #   │      skip_weights)                                             │
    #   │                                                                │
    #   │  4. Output head         Adam        0.008    Only if untied    │
    #   │     (lm_head.weight)                         embeddings        │
    #   │     "vector → token                          (not used in      │
    #   │      probability"                            baseline)         │
    #   └─────────────────────────────────────────────────────────────────┘
    #
    # WHY MUON FOR MATRICES?
    #   Muon orthogonalizes gradient updates via Newton-Schulz iterations.
    #   This only works on 2D matrices (needs rows & columns to orthogonalize).
    #   1D vectors and scalars are too small — they use standard Adam instead.
    #
    # WHY DIFFERENT LEARNING RATES?
    #   - Embeddings are a lookup table — aggressive updates cause instability
    #   - Matrix weights are the core compute — Muon handles the LR scaling
    #   - Scalars are sensitive knobs — moderate LR with Adam's adaptive step
    #   - Output head (if untied) maps back to vocab — needs gentler updates
    #
    block_named_params = list(base_model.blocks.named_parameters())

    # --- Group 2: Matrix params (2D tensors in transformer blocks) → Muon ---
    # Collects: c_q.weight, c_k.weight, c_v.weight, proj.weight, fc.weight, mlp.proj.weight
    # Excludes: control tensors like attn_scale, resid_mix (those are 2D but act as scalars)
    matrix_params = [
        p
        for name, p in block_named_params
        if p.ndim == 2 and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]

    # --- Group 3: Scalar params (1D vectors + control tensors) → Adam ---
    # Collects: attn_scale, mlp_scale, resid_mix, q_gain, skip_weights
    # These are the small "knobs" that tune how blocks combine their outputs
    scalar_params = [
        p
        for name, p in block_named_params
        if p.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)  # U-Net skip connection weights

    # --- Group 1: Embedding table → Adam (gentle LR) ---
    # The "dictionary" that maps token IDs to 512-dim vectors
    # If tied: same weight used for input embedding AND output prediction (LR=0.05)
    # If untied: separate input embedding (LR=0.6) and output head (LR=0.008)
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.Adam(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,  # fused=True: single GPU kernel for speed
    )

    # --- Group 2 optimizer: Muon for the heavy 2D matrices ---
    # Uses Newton-Schulz orthogonalization (5 steps) to normalize gradients
    # before applying them. This is the key innovation from modded-nanogpt.
    optimizer_muon = Muon(
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr  # Stash base LR for warmdown scheduling

    # --- Group 3 optimizer: Adam for small scalar/vector params ---
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )

    # Collect all optimizers — training loop will step() all of them each iteration
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]

    # --- Group 4 (optional): Output head → Adam (only when embeddings are untied) ---
    # In baseline, tie_embeddings=True so lm_head is None and this is skipped.
    # When untied, the output head gets its own gentle LR (0.008)
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        optimizers.insert(1, optimizer_head)

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0("sdp_backends:cudnn=False flash=True mem_efficient=False math=False")
    log0(f"attention_mode:gqa num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads}")
    log0(
        f"tie_embeddings:{args.tie_embeddings} embed_lr:{token_lr} "
        f"head_lr:{args.head_lr if base_model.lm_head is not None else 0.0} "
        f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr}"
    )
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
        f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log0(f"seed:{args.seed}")

    # -----------------------------
    # DATA LOADER & MODEL WARMUP
    # -----------------------------

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    # Reserve wall-clock time for GPTQ calibration (AR generation + Hessian collection + quantization)
    if args.use_gptq and max_wallclock_ms is not None:
        max_wallclock_ms -= args.gptq_reserve_ms
        log0(f"gptq:reserving {args.gptq_reserve_ms:.0f}ms from training budget, effective={max_wallclock_ms:.0f}ms")

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if warmdown_start <= step < args.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    # Warmup primes the compiled forward/backward/optimizer paths, then we restore the
    # initial weights/optimizer state so measured training starts from the true init.
    if args.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # -----------------------------
    # MAIN TRAINING LOOP
    # -----------------------------

    training_time_ms = 0.0
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)

        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(
                args,
                model,
                rank,
                world_size,
                device,
                grad_accum_steps,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            )
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(
                    f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms "
                    f"step:{step}/{args.iterations}"
                )
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum

        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        should_log_train = (
            args.train_log_every > 0
            and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        )
        if should_log_train:
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"
            )

        # Needed to sync whether we've reached the wallclock cap.
        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )

    # -----------------------------
    # SERIALIZATION + ROUNDTRIP VALIDATION
    # -----------------------------
    # Save the raw state (useful for debugging/loading in PyTorch directly), then produce
    # the compressed quantized artifact and validate the round-tripped weights.
    # Supports two modes: int8+zlib (default) and int6+brotli (PG ranks 1-9).

    use_int6 = args.quant_mode == "int6_brotli"
    quant_label = "int6+brotli" if use_int6 else "int8+zlib"
    quant_ext = ".int6.ptb" if use_int6 else ".int8.ptz"

    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size: {model_bytes + code_bytes} bytes")

    # GPTQ calibration: collect Hessians → pass to quantizer
    # Algorithm: Frantar et al., ICLR 2023. Clipping: Kevin Clark, PG rank 5.
    gptq_hessians = None
    if args.use_gptq and use_int6:
        t_gptq = time.perf_counter()
        log0(f"gptq:calib_source={args.gptq_calib_source} clip_range={args.gptq_clip_range} sdclip_k={args.sdclip_k}"
             f" sequential={args.gptq_sequential} hooks={args.gptq_use_hooks} embed={args.gptq_embed} percdamp={args.gptq_percdamp}")

        # Step 1: Get calibration tokens
        if args.gptq_calib_source == "train":
            log0(f"gptq:collecting calibration from {args.gptq_ar_samples} training batches...")
            calib_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
            calib_seqs = []
            with torch.no_grad():
                for _ in range(args.gptq_ar_samples):
                    x, y = calib_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                    calib_seqs.append(torch.cat([x, y[:, -1:]], dim=1).detach().clone())
            del calib_loader
        else:
            log0(f"gptq:starting AR self-generation ({args.gptq_ar_samples} seqs × {args.gptq_ar_seqlen} tokens)")
            calib_seqs = generate_autoregressive_calib(
                base_model, device,
                num_seqs=args.gptq_ar_samples,
                seq_len=args.gptq_ar_seqlen,
                vocab_size=args.vocab_size,
                temperature=args.gptq_temperature,
                batch_size=args.gptq_batch_size,
                seed=args.seed,
            )
            log0(f"gptq:AR generated {len(calib_seqs)} sequences in {time.perf_counter() - t_gptq:.1f}s")

        # Step 2: Collect Hessians (or do sequential quantization)
        if args.gptq_sequential:
            log0(f"gptq:sequential block-by-block quantization ({len(base_model.blocks)} blocks)...")
            gptq_hessians = gptq_sequential_quantize(
                base_model, calib_seqs, device,
                clip_range=args.gptq_clip_range, clip_sigmas=args.sdclip_k,
                percdamp=args.gptq_percdamp, use_hooks=args.gptq_use_hooks,
            )
        elif args.gptq_use_hooks:
            log0(f"gptq:hook-based Hessian collection from {len(calib_seqs)} sequences...")
            gptq_hessians = gptq_collect_hessians_with_hooks(
                base_model, calib_seqs, device, percdamp=args.gptq_percdamp,
            )
        else:
            log0(f"gptq:manual flag-based Hessian collection from {len(calib_seqs)} sequences...")
            gptq_hessians = gptq_collect_hessians_from_tokens(
                base_model, calib_seqs, device, percdamp=args.gptq_percdamp,
            )

        # Step 3: Embedding Hessian (token frequency-based)
        if args.gptq_embed:
            log0("gptq:computing embedding Hessian from calibration token frequencies...")
            vocab_size, dim = base_model.tok_emb.weight.shape
            # Compute frequency-weighted column correlation: H = W^T @ diag(freq) @ W → [dim, dim]
            # GPTQ needs H matching column count of W=[vocab, dim], so H must be [dim, dim]
            freq = torch.zeros(vocab_size, dtype=torch.float32)
            for seq in calib_seqs:
                tokens = seq.flatten().cpu()
                for tok in tokens:
                    freq[tok.item()] += 1.0
            freq = freq / freq.sum().clamp_min(1e-10)
            W = base_model.tok_emb.weight.data.detach().cpu().float()
            W_scaled = W * freq.unsqueeze(1).sqrt()  # [vocab, dim] weighted by sqrt(freq)
            embed_H = W_scaled.t() @ W_scaled  # [dim, dim]
            damp = args.gptq_percdamp * embed_H.diag().mean().clamp_min(1e-6)
            embed_H += damp * torch.eye(dim)
            gptq_hessians["tok_emb.weight"] = embed_H

        del calib_seqs
        gptq_elapsed = time.perf_counter() - t_gptq
        log0(f"gptq:done in {gptq_elapsed:.1f}s, {len(gptq_hessians)} Hessians collected")
        torch.cuda.empty_cache()

    if use_int6:
        quant_obj, quant_stats = quantize_state_dict_int6_sdclip(
            base_model.state_dict(), args.sdclip_k, args.embed_clip_k,
            hessians=gptq_hessians,
            gptq_clip_range=args.gptq_clip_range, gptq_clip_sigmas=args.sdclip_k,
            gptq_embed=args.gptq_embed if args.use_gptq else False,
            gptq_percdamp=args.gptq_percdamp if args.use_gptq else 0.01,
        )
        dequant_fn = dequantize_state_dict_int6
    else:
        quant_obj, quant_stats = quantize_state_dict_int8(base_model.state_dict())
        dequant_fn = dequantize_state_dict_int8

    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()

    if use_int6:
        if brotli is None:
            raise RuntimeError("brotli not installed. Run: pip install brotli")
        # Byte-shuffle before Brotli improves compression ratio for quantized data
        quant_blob = brotli.compress(_byte_shuffle(quant_raw), quality=11)
    else:
        quant_blob = zlib.compress(quant_raw, level=9)

    quant_raw_bytes = len(quant_raw)
    quant_filename = f"final_model{quant_ext}"
    if master_process:
        with open(quant_filename, "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = os.path.getsize(quant_filename)
        code_bytes = len(code.encode("utf-8"))
        ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
        log0(
            f"Serialized model {quant_label}: {quant_file_bytes} bytes "
            f"(payload:{quant_stats['int8_payload_bytes']} raw_torch:{quant_raw_bytes} payload_ratio:{ratio:.2f}x)"
        )
        log0(f"Total submission size {quant_label}: {quant_file_bytes + code_bytes} bytes")

    if distributed:
        dist.barrier()
    with open(quant_filename, "rb") as f:
        quant_blob_disk = f.read()
    if use_int6:
        quant_state = torch.load(io.BytesIO(_byte_unshuffle(brotli.decompress(quant_blob_disk))), map_location="cpu")
    else:
        quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob_disk)), map_location="cpu")
    base_model.load_state_dict(dequant_fn(quant_state), strict=True)
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        args,
        model,
        rank,
        world_size,
        device,
        grad_accum_steps,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    quant_rt_tag = "final_int6_brotli_roundtrip" if use_int6 else "final_int8_zlib_roundtrip"
    log0(
        f"{quant_rt_tag} val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    log0(f"{quant_rt_tag}_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")

    # Score-First TTT on quantized model (if enabled)
    if args.ttt_mode == "score_first":
        torch.cuda.synchronize()
        t_ttt = time.perf_counter()
        # Save quantized weights, TTT will modify them during adaptation
        saved_state = copy.deepcopy(base_model.state_dict())
        ttt_loss, ttt_bpb = eval_val_score_first_ttt(
            args, base_model, rank, world_size, device,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            log0=log0,
        )
        # Restore quantized weights (TTT modifies them in-place)
        base_model.load_state_dict(saved_state)
        torch.cuda.synchronize()
        ttt_tag = "final_int6_ttt" if use_int6 else "final_int8_ttt"
        log0(f"{ttt_tag} val_loss:{ttt_loss:.4f} val_bpb:{ttt_bpb:.4f} "
             f"eval_time:{1000.0 * (time.perf_counter() - t_ttt):.0f}ms")
        log0(f"{ttt_tag}_exact val_loss:{ttt_loss:.8f} val_bpb:{ttt_bpb:.8f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
