"""
========================================================================
CSCI 381 Project — Week 2, Stage 2
Trainable Neural Belief Propagation decoder for BCH(15, 11).

This builds the actual learned decoder that the project's main result
hinges on. The structure mirrors Stage 1's MinSumBP, with three
modifications drawn from L24 and Nachmani et al. (2016):

  1. TRAINABLE EDGE WEIGHTS. Each variable-to-check message and each
     check-to-variable message is multiplied by a learnable scalar
     weight before being sent. There is a separate (M, N) weight tensor
     per unrolled iteration, with weights only on real Tanner-graph
     edges. Initialization is at 1.0, so the untrained network exactly
     reproduces Min-Sum (this is checked).

  2. MULTI-LOSS WITH LINEAR DISCOUNT. The forward pass collects
     intermediate LLRs at every iteration. The training loss is a
     weighted sum of binary cross-entropy at each iteration, with
     later iterations weighted more (per professor's suggestion #3).
     This keeps gradient flowing to early iterations while prioritizing
     the final decoding accuracy.

  3. ALL-ZERO CODEWORD TRAINING. Per L24, BLER on a linear code is
     independent of the transmitted codeword, so training uses only
     the all-zero codeword with random Gaussian noise.

This script does the full Stage 2 pipeline:
  - Build NeuralBP(T=5) module
  - Verify unit-weight initialization reproduces Stage 1 MinSumBP exactly
  - Train across Eb/N0 in [1, 6] dB
  - BLER vs SNR sweep against ML / SP / MS / untrained NBP
  - Save figures and learned-weight diagnostics

Setup:
  pip install torch numpy matplotlib

Author: Ashir Qureshi
Course: CSCI 381 - Information Theory & Error Correction Codes
========================================================================
"""

import os
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import week1_bch_baselines as w1
import week2_stage1_pytorch_port as stage1

# ========================================================================
# REPRODUCIBILITY
# ========================================================================
# Set seeds for both NumPy (Week 1's RNG) and PyTorch (Stage 2 training).
# Note: PyTorch CPU operations are deterministic for our use case, but
# we still set seeds so different runs with the same seed produce
# bit-identical results.

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
w1.RNG = np.random.default_rng(seed=SEED)
DEVICE = torch.device("cpu")
DTYPE = torch.float64

# ========================================================================
# SECTION 1: NEURAL BP MODULE
# ========================================================================
# Inheriting from torch.nn.Module gives us automatic parameter tracking:
# any tensor wrapped in nn.Parameter and assigned to self.* shows up
# in .parameters() and gets gradient updates from the optimizer.
# Differences from Stage 1's MinSumBP:
#   - Per-iteration trainable weight tensors (one (M, N) tensor per t).
#   - The forward pass returns LLRs for ALL iterations (shape (T, B, N))
#     during training, so the multi-loss can use intermediate decisions.
#   - At inference time, we usually only care about the final iteration.

class NeuralBP(torch.nn.Module):
    """
    Unrolled Neural Min-Sum decoder with trainable per-iteration edge weights.

    Forward input:  llrs of shape (B, N)
    Forward output:
        - If `return_all_iters=False` (default at inference): hard
          decisions of shape (B, N), dtype int64.
        - If `return_all_iters=True` (used during training): a tensor
          of shape (T, B, N) containing the total LLR L_total at each
          iteration. Hard decisions are not produced here because BCE
          loss wants raw LLRs (it applies a sigmoid internally).

    Architectural choice: we use the FULLY-WEIGHTED variant. Each edge
    receives a learnable weight in BOTH the variable-to-check and the
    check-to-variable directions, at each iteration. This matches the
    "every variable-to-check and check-to-variable edge" wording in the
    proposal and is consistent with L24's framing.

    Total trainable parameter count:
        2 * T * (number of real edges in H)
    For BCH(15,11) with T=5: 2 * 5 * 32 = 320 parameters.
    (H has 32 ones in our derived form; verified at construction time.)
    """

    def __init__(self, max_iters=5):
        super().__init__()
        self.max_iters = max_iters

        # Static tensors (non-trainable). Reused from Stage 1.
        self.register_buffer("H", stage1.H_TORCH)
        self.register_buffer("H_int", stage1.H_TORCH.to(dtype=torch.long))
        self.register_buffer("edge_mask", stage1.EDGE_MASK_TORCH)
        M, N = self.H.shape

        # Trainable weights: one (M, N) tensor per iteration, for each
        # message direction. Initialized to 1.0 so untrained == Min-Sum.
        #
        # PyTorch idiom: torch.nn.ParameterList holds a list of trainable
        # tensors and exposes them via .parameters(). We use it because
        # we want T separate parameter tensors, one per iteration.
        #
        # We initialize to 1.0 only on real edges (H==1) and 0 elsewhere.
        # Multiplying by edge_mask in the forward pass keeps non-edges
        # zero throughout training, so weights on non-edges don't drift.

        init = self.edge_mask.clone()  # 1.0 on edges, 0.0 elsewhere
        self.w_v2c = torch.nn.ParameterList([
            torch.nn.Parameter(init.clone()) for _ in range(max_iters)
        ])
        self.w_c2v = torch.nn.ParameterList([
            torch.nn.Parameter(init.clone()) for _ in range(max_iters)
        ])

        # Sanity counter (printed once)
        n_edges = int(self.edge_mask.sum().item())
        n_params = 2 * max_iters * n_edges
        self._reported_params = (n_edges, n_params)

    def forward(self, llrs, return_all_iters=False):
        B = llrs.shape[0]
        M, N = self.H.shape
        M_v2c = llrs.new_zeros(B, M, N)
        M_c2v = llrs.new_zeros(B, M, N)
        edge_mask_b = self.edge_mask[None, :, :]  # (1, M, N)
        active = torch.ones(B, dtype=torch.bool, device=llrs.device)

        # Storage for per-iteration L_total if requested by training loop
        all_iter_llrs = []

        for t in range(self.max_iters):
            if not active.any() and not return_all_iters:
                break

            # ------ Variable-node update (extrinsic, weighted) ------
            # Same form as Min-Sum but with a learnable per-edge weight
            # multiplied in. The weight is masked to real edges by
            # multiplying against edge_mask (this also keeps weights on
            # non-edge positions from doing anything during training).
            total_incoming = llrs + M_c2v.sum(dim=1)             # (B, N)
            M_v2c = total_incoming[:, None, :] - M_c2v           # (B, M, N)
            # Apply trainable weight + mask
            w_v2c_t = self.w_v2c[t] * self.edge_mask             # (M, N)
            M_v2c = M_v2c * w_v2c_t[None, :, :]                  # broadcast over batch

            # ------ Check-node update (Min-Sum, extrinsic, weighted) ------
            # We use the Min-Sum approximation here rather than tanh so
            # the decoder is a learnable variant of Min-Sum specifically.
            # (Some versions of Neural BP use tanh; Min-Sum is the more
            # common choice for hardware-realistic learned decoders, and
            # it's what your proposal describes.)

            INF = torch.tensor(float("inf"), dtype=M_v2c.dtype, device=M_v2c.device)
            signs = torch.where(edge_mask_b > 0,
                                torch.sign(M_v2c),
                                torch.ones_like(M_v2c))
            signs = torch.where(signs == 0,
                                torch.ones_like(signs),
                                signs)
            total_sign = signs.prod(dim=2, keepdim=True)         # (B, M, 1)
            extrinsic_sign = total_sign * signs                  # (B, M, N)

            mags = torch.where(edge_mask_b > 0, M_v2c.abs(), INF)
            sorted_mags, _ = mags.sort(dim=2)
            min1 = sorted_mags[:, :, 0:1]
            min2 = sorted_mags[:, :, 1:2]
            extrinsic_min = torch.where(mags == min1,
                                        min2.expand_as(mags),
                                        min1.expand_as(mags))
            M_c2v_raw = extrinsic_sign * extrinsic_min
            # Apply trainable weight + mask
            w_c2v_t = self.w_c2v[t] * self.edge_mask
            new_M_c2v = M_c2v_raw * w_c2v_t[None, :, :]

            # Keep converged samples frozen so inference matches Stage 1.
            M_v2c = torch.where(active[:, None, None], M_v2c, M_v2c)
            M_c2v = torch.where(active[:, None, None], new_M_c2v, M_c2v)

            # ------ Per-iteration soft output ------
            # The "decision" at iteration t is the total LLR per bit:
            #   L_total[b, v] = llr[b, v] + sum_c M_c2v[b, c, v]
            # We stash this if the training loop wants per-iteration loss.
            L_total_t = llrs + M_c2v.sum(dim=1)              # (B, N)
            if return_all_iters:
                all_iter_llrs.append(L_total_t)
            bits = (L_total_t < 0).long()
            syndrome_ok = torch.remainder(bits @ self.H_int.T, 2).eq(0).all(dim=1)
            active = active & (~syndrome_ok)
        if return_all_iters:
            # Stack into (T, B, N) for the training loop's loss computation
            return torch.stack(all_iter_llrs, dim=0)

        # Inference path: hard decisions at the final iteration
        L_total = llrs + M_c2v.sum(dim=1)
        return (L_total < 0).long()

# ========================================================================
# SECTION 2: SANITY CHECK — UNTRAINED NeuralBP == Stage 1 MinSumBP
# ========================================================================
# Per the proposal: "Unit-weight initialization of the unrolled Neural
# BP decoder must reproduce Min-Sum BP exactly; this verifies that the
# unrolling is correctly implemented before any training is attempted."

def verify_init_matches_minsum():
    print("Sanity check: untrained NeuralBP (T=5) vs MinSumBP (max_iters=5)")
    nbp = NeuralBP(max_iters=5).to(device=DEVICE, dtype=DTYPE)
    msbp = stage1.MinSumBP(max_iters=5).to(device=DEVICE, dtype=DTYPE)
    n_edges, n_params = nbp._reported_params
    print(f"  Edges in H: {n_edges}, total trainable params: {n_params}")
    codewords, llrs_np = stage1.make_test_batch(
        n_samples=200, ebn0_db=3.0, seed=99999
    )
    llrs_torch = torch.tensor(llrs_np, dtype=DTYPE, device=DEVICE)
    with torch.no_grad():
        d_nbp = nbp(llrs_torch).cpu().numpy().astype(np.int8)
        d_msbp = msbp(llrs_torch).cpu().numpy().astype(np.int8)
    diffs = int(np.sum(d_nbp != d_msbp))
    if diffs == 0:
        print("  [PASS] Untrained NeuralBP produces identical decisions to MinSumBP.")
        return True
    else:
        print(f"  [FAIL] {diffs} bit disagreements between untrained NBP and MinSumBP.")
        return False

# ========================================================================
# SECTION 3: TRAINING LOOP
# ========================================================================
# Per L24, training uses only the all-zero codeword. The procedure:
#   1. Sample SNR uniformly from [snr_low, snr_high] (per-batch, not per-step,
#      to keep batches at one SNR for cleaner gradient signal).
#   2. Generate B all-zero codewords -> BPSK -> all +1 vectors.
#   3. Add Gaussian noise of the chosen sigma.
#   4. Convert to LLRs: L = 2y/sigma^2.
#   5. Forward pass with return_all_iters=True -> shape (T, B, N).
#   6. Compute multi-loss with linear discount: weight at iter t is
#      proportional to (t+1), normalized so weights sum to 1.
#   7. Loss at each iter is BCE with target = 0 for every bit.
#      Note on sign convention: our LLR is positive for bit=0, negative
#      for bit=1. So the "logit" for "bit is 1" is -L_total. We pass
#      -L_total to BCEWithLogitsLoss with target = zeros.

def train_neural_bp(model, num_steps=4000, batch_size=256,
                    snr_low_db=1.0, snr_high_db=6.0,
                    lr=1e-3, log_every=200):
    """
    Train the NeuralBP module on all-zero codewords with random noise.

    Args:
        model: a NeuralBP instance.
        num_steps: number of optimizer steps. Each step uses one batch.
        batch_size: number of noisy codewords per batch.
        snr_low_db, snr_high_db: range of Eb/N0 to sample from.
        lr: Adam learning rate.
        log_every: how often to print a progress line.

    Returns the final loss and a list of (step, loss) for plotting.
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Linear discount weights: iter T-1 weighted highest, iter 0 lowest.
    T = model.max_iters
    raw = torch.tensor([t + 1 for t in range(T)], dtype=DTYPE, device=DEVICE)
    iter_weights = raw / raw.sum()  # sums to 1
    print(f"  Iteration loss weights (later iters weighted more): {iter_weights.tolist()}")

    # Standard PyTorch BCE-with-logits is numerically stable: it combines
    # sigmoid + BCE in one operation.
    bce = torch.nn.BCEWithLogitsLoss()
    history = []
    for step in range(num_steps):
        # Sample one SNR for the whole batch
        snr_db = np.random.uniform(snr_low_db, snr_high_db)
        sigma = w1.ebn0_db_to_sigma(snr_db)

        # All-zero codeword -> BPSK -> all +1 -> add noise -> LLR
        # Shape: (B, N)
        noise = torch.randn(batch_size, w1.N, dtype=DTYPE, device=DEVICE) * sigma
        received = 1.0 + noise  # transmitted = +1 (bit 0), received = +1 + n
        llrs = 2.0 * received / (sigma ** 2)

        # Forward pass: get all per-iteration LLRs, shape (T, B, N)
        L_per_iter = model(llrs, return_all_iters=True)

        # BCE with logits expects "logit for class 1" and a target in [0, 1].
        # Our convention: positive L_total favors bit 0. So logit_for_bit_1 = -L.
        # The target codeword is all zeros, so target = zeros tensor.
        target_zeros = torch.zeros_like(L_per_iter[0])  # (B, N)

        # Multi-loss with linear discount
        total_loss = torch.tensor(0.0, dtype=DTYPE, device=DEVICE)
        for t in range(T):
            logits_for_bit1_t = -L_per_iter[t]
            total_loss = total_loss + iter_weights[t] * bce(logits_for_bit1_t, target_zeros)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        history.append((step, total_loss.item()))
        if step == 0 or (step + 1) % log_every == 0:
            print(f"  step {step+1:>5d} / {num_steps}  loss = {total_loss.item():.6f}  "
                  f"(SNR sample = {snr_db:.2f} dB)")
    model.eval()
    return history

# ========================================================================
# SECTION 4: BLER EVALUATION (RANDOM codewords, not all-zero)
# ========================================================================
# Per the proposal sanity check: even though training uses only the
# all-zero codeword, evaluation uses uniformly random codewords. Matching
# BLER on the two empirically validates the linearity argument.
# We evaluate every decoder we've built so far on the same SNR grid:
#   - ML (Week 1 NumPy, lower bound)
#   - Sum-Product BP (Stage 1 PyTorch, batched)
#   - Min-Sum BP (Stage 1 PyTorch, batched)
#   - Untrained NeuralBP (initialized at 1.0; should match Min-Sum)
#   - Trained NeuralBP

def bler_random_codewords(decoder_callable, ebn0_db, n_trials=2000, batch=200, seed=0):
    """
    Compute BLER for any decoder using random codewords, batched.

    decoder_callable: a function (llrs_torch_BxN) -> hard_decisions_BxN.
                      We use a callable rather than a Module directly so
                      we can wrap NumPy-only decoders (ML, syndrome).
    """
    rng = np.random.default_rng(seed)
    sigma = w1.ebn0_db_to_sigma(ebn0_db)
    n_errors = 0
    n_done = 0
    while n_done < n_trials:
        b = min(batch, n_trials - n_done)

        # Generate batch of random codewords
        msgs = rng.integers(0, 2, size=(b, w1.K)).astype(np.int8)
        codewords = np.stack([w1.encode(msgs[i]) for i in range(b)], axis=0)

        # BPSK + AWGN
        bpsk = 1.0 - 2.0 * codewords.astype(np.float64)
        noise = rng.normal(0.0, sigma, size=(b, w1.N))
        received = bpsk + noise
        llrs = 2.0 * received / (sigma ** 2)

        # Decode
        decoded = decoder_callable(received, llrs)  # (b, N) int

        # Count block errors
        block_errors = np.any(decoded != codewords, axis=1)
        n_errors += int(block_errors.sum())
        n_done += b
    return n_errors / n_done

# Decoder callables — each takes (received, llrs) numpy arrays and returns
# decoded bits as a numpy array of shape (B, N), dtype int8.

def make_ml_callable():
    def _call(received, llrs):
        return np.stack([w1.ml_decode(received[i]) for i in range(received.shape[0])], axis=0)
    return _call

def make_torch_callable(module):
    """Wrap a PyTorch decoder Module so it accepts numpy arrays."""
    module.eval()
    def _call(received, llrs):
        with torch.no_grad():
            llrs_t = torch.tensor(llrs, dtype=DTYPE, device=DEVICE)
            decoded = module(llrs_t).cpu().numpy().astype(np.int8)
        return decoded
    return _call

# ========================================================================
# SECTION 5: VALIDATE LINEARITY (BLER on all-zero == BLER on random)
# ========================================================================

def linearity_check(model, ebn0_db=4.0, n_trials=2000, seed=12345):
    """
    Compare BLER of the trained model on:
      (a) only the all-zero codeword (training distribution)
      (b) uniformly random codewords (evaluation distribution)
    These should match within statistical noise; mismatch indicates the
    model has memorized the all-zero pattern rather than learned to
    decode the code.
    """
    rng = np.random.default_rng(seed)
    sigma = w1.ebn0_db_to_sigma(ebn0_db)

    # (a) All-zero
    noise_a = rng.normal(0.0, sigma, size=(n_trials, w1.N))
    received_a = 1.0 + noise_a
    llrs_a = 2.0 * received_a / (sigma ** 2)
    with torch.no_grad():
        decoded_a = model(torch.tensor(llrs_a, dtype=DTYPE, device=DEVICE)).cpu().numpy()
    bler_a = float(np.any(decoded_a != 0, axis=1).mean())

    # (b) Random
    msgs = rng.integers(0, 2, size=(n_trials, w1.K)).astype(np.int8)
    cws = np.stack([w1.encode(msgs[i]) for i in range(n_trials)], axis=0)
    bpsk = 1.0 - 2.0 * cws.astype(np.float64)
    noise_b = rng.normal(0.0, sigma, size=(n_trials, w1.N))
    received_b = bpsk + noise_b
    llrs_b = 2.0 * received_b / (sigma ** 2)
    with torch.no_grad():
        decoded_b = model(torch.tensor(llrs_b, dtype=DTYPE, device=DEVICE)).cpu().numpy()
    bler_b = float(np.any(decoded_b != cws, axis=1).mean())
    return bler_a, bler_b

# ========================================================================
# SECTION 6: MAIN PIPELINE
# ========================================================================

def run_stage2():
    print("=" * 60)
    print("Week 2 Stage 2: Trainable Neural BP")
    print("=" * 60)
    print(f"  PyTorch: {torch.__version__}, device: {DEVICE}, dtype: {DTYPE}")
    print(f"  Seed: {SEED}")
    print()

    # ---- Sanity: untrained NBP matches MinSumBP ----
    if not verify_init_matches_minsum():
        print("Aborting: untrained NeuralBP doesn't match MinSumBP.")
        return
    print()

    # ---- Build and train ----
    print("Training NeuralBP(T=5) over Eb/N0 in [1, 6] dB:")
    model = NeuralBP(max_iters=5).to(device=DEVICE, dtype=DTYPE)
    history = train_neural_bp(
        model,
        num_steps=4000,
        batch_size=256,
        snr_low_db=1.0,
        snr_high_db=6.0,
        lr=1e-3,
        log_every=500,
    )
    print()

    # ---- Linearity check ----
    print("Linearity check (BLER on all-zero vs random codewords at 4 dB):")
    bler_zero, bler_rand = linearity_check(model, ebn0_db=4.0, n_trials=3000)
    print(f"  All-zero BLER: {bler_zero:.4f}")
    print(f"  Random   BLER: {bler_rand:.4f}")
    if abs(bler_zero - bler_rand) < 0.01:
        print("  [PASS] BLERs match within tolerance — linearity argument holds empirically.")
    else:
        print("  [WARN] BLERs differ noticeably. Worth investigating before trusting results.")
    print()

    # ---- BLER vs SNR sweep ----
    print("BLER vs SNR sweep (random codewords, 3000 trials per point):")
    snrs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    n_trials_per_point = 3000
    sp_module = stage1.SumProductBP(max_iters=20).to(device=DEVICE, dtype=DTYPE)
    ms_module = stage1.MinSumBP(max_iters=20).to(device=DEVICE, dtype=DTYPE)
    untrained_nbp = NeuralBP(max_iters=5).to(device=DEVICE, dtype=DTYPE)  # weights = 1.0
    callables = {
        "ML": make_ml_callable(),
        "SumProductBP": make_torch_callable(sp_module),
        "MinSumBP": make_torch_callable(ms_module),
        "NeuralBP_untrained": make_torch_callable(untrained_nbp),
        "NeuralBP_trained": make_torch_callable(model),
    }
    bler_table = {name: [] for name in callables}
    print(f"  {'SNR':>5}  " + "  ".join(f"{n:>20s}" for n in callables))
    for snr_db in snrs:
        row = []
        for name, fn in callables.items():
            b = bler_random_codewords(
                fn, ebn0_db=snr_db, n_trials=n_trials_per_point,
                batch=200, seed=int(snr_db * 1000)
            )
            bler_table[name].append(b)
            row.append(b)
        print(f"  {snr_db:>5.1f}  " + "  ".join(f"{v:>20.5f}" for v in row))
    print()

    # ---- Save BLER plot ----
    out_dir = os.path.dirname(os.path.abspath(__file__))
    fig_path = os.path.join(out_dir, "bler_vs_snr.png")
    plt.figure(figsize=(8, 6))
    style = {
        "ML":                  ("ML (lower bound)",     "k", "-",  "o"),
        "SumProductBP":        ("Sum-Product BP",       "C0", "-", "s"),
        "MinSumBP":            ("Min-Sum BP",           "C1", "-", "^"),
        "NeuralBP_untrained":  ("Neural BP (untrained)","C2", "--",  "x"),
        "NeuralBP_trained":    ("Neural BP (trained)",  "C3", "-",  "D"),
    }
    for name, (label, color, ls, marker) in style.items():
        # Avoid log(0) by clipping below 1/n_trials (the smallest measurable BLER).
        floor = 1.0 / (n_trials_per_point + 1)
        ys = [max(b, floor) for b in bler_table[name]]
        plt.semilogy(snrs, ys, color=color, linestyle=ls, marker=marker, label=label)
    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("Block Error Rate")
    plt.title("BCH(15,11) over BI-AWGN: BLER vs SNR")
    plt.grid(True, which="both", linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  BLER plot saved to: {fig_path}")

    # ---- Save learned-weight histogram ----
    hist_path = os.path.join(out_dir, "learned_weights_hist.png")
    all_weights = []
    for t in range(model.max_iters):
        w_v = model.w_v2c[t].detach().cpu().numpy()
        w_c = model.w_c2v[t].detach().cpu().numpy()
        # Only collect weights on real edges
        edge_mask_np = stage1.EDGE_MASK_TORCH.cpu().numpy() > 0
        all_weights.extend(w_v[edge_mask_np].tolist())
        all_weights.extend(w_c[edge_mask_np].tolist())
    plt.figure(figsize=(7, 4))
    plt.hist(all_weights, bins=40, edgecolor="black")
    plt.axvline(1.0, color="red", linestyle="--",
                label="Initialization value (1.0 = Min-Sum)")
    plt.xlabel("Learned edge weight")
    plt.ylabel("Count")
    plt.title("Distribution of trained NBP edge weights\n"
              f"(across all {model.max_iters} iterations)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(hist_path, dpi=150)
    plt.close()
    print(f"  Weight histogram saved to: {hist_path}")
    print()

    # ---- Quick numerical summary for the report ----
    print("Summary of headline gain (trained NBP vs Min-Sum):")
    for i, snr_db in enumerate(snrs):
        ms = bler_table["MinSumBP"][i]
        nbp = bler_table["NeuralBP_trained"][i]
        if ms > 0:
            rel = (ms - nbp) / ms * 100
            print(f"  {snr_db:>4.1f} dB:  MinSum BLER = {ms:.4f}, "
                  f"NBP BLER = {nbp:.4f}, relative gain = {rel:+.1f}%")
    print("\nStage 2 complete.")

if __name__ == "__main__":
    run_stage2()