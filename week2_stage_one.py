"""
========================================================================
CSCI 381 Project — Week 2, Stage 1
PyTorch port of Sum-Product and Min-Sum BP, with numerical-equivalence
verification against the Week 1 NumPy implementations.

This is INFRASTRUCTURE, not a new decoder. Stage 1 produces no new BLER
results; it only confirms that the PyTorch versions of the classical
decoders produce bit-for-bit identical outputs to the Week 1 versions.
Once that's verified, Stage 2 will modify the Min-Sum forward pass to
add trainable edge weights and turn it into Neural BP.

Why this two-stage approach matters:
  PyTorch tensor operations can fail silently — code that runs without
  error but produces incorrect outputs. By porting the message-passing
  pipeline first and verifying numerical equivalence to Week 1, we
  isolate "infrastructure bugs" from "training bugs" before they can
  compound.

Setup:
  pip install torch numpy
  Tested against PyTorch >= 2.0. Pure CPU (no GPU needed for N=15).

Author: Ashir Qureshi
Course: CSCI 381 - Information Theory & Error Correction Codes
========================================================================
"""

import numpy as np
import torch

# Import everything from Week 1 so we can reuse the encoder, channel,
# H matrix, and reference NumPy decoders.
import week1_bch_baselines as w1

# ========================================================================
# PYTORCH MENTAL MODEL — read this once before reading the code
# ========================================================================
# PyTorch is NumPy with two additions you need to know about:
#   1. AUTOGRAD: every tensor operation can be differentiated automatically.
#      As long as we use only torch.* operations (not numpy operations
#      mid-computation), PyTorch records what happened and can compute
#      gradients on demand. We don't use this in Stage 1 — but the way
#      we write the code now must be autograd-friendly so Stage 2 works.
#   2. nn.Module: a Python class that holds tensors marked as
#      "trainable parameters" (nn.Parameter) and has a .forward() method
#      describing the computation. Calling the module is calling forward.
#      We don't have parameters yet in Stage 1 — but we structure the
#      code as a Module now so Stage 2 only adds parameters, not refactors.
# Beyond those two: tensors instead of arrays (torch.Tensor vs np.ndarray),
# .device for GPU/CPU placement, .dtype for precision (float32 vs float64).
# Everything else is just NumPy with slightly different function names.
# A few common translations you'll see below:
#   np.sum(x, axis=0)   → torch.sum(x, dim=0)
#   np.where(c, a, b)   → torch.where(c, a, b)
#   np.tanh(x)          → torch.tanh(x)
#   x.copy()            → x.clone()
#   x.astype(np.float64)→ x.to(torch.float64)
# We use torch.float64 throughout to match Week 1's NumPy default and
# make the bit-for-bit equivalence test clean.

# ========================================================================
# SECTION 1: Tensor versions of the static code data
# ========================================================================
#
# We convert the H matrix (and derived edge mask) into tensors once.
# These never change during forward passes and never have gradients,
# so they're plain tensors (not Parameters). We register them as
# "buffers" inside the modules below — buffers are non-trainable
# tensors that move with the module when you do model.to(device).

DEVICE = torch.device("cpu")  # N=15 is tiny; CPU is plenty fast
DTYPE = torch.float64         # match Week 1 precision for clean comparison

# Convert H to a torch tensor. Note: w1.H is int8; we'll cast to DTYPE for math.
H_TORCH = torch.tensor(w1.H, dtype=DTYPE, device=DEVICE)        # shape (M, N)
H_INT_TORCH = torch.tensor(w1.H, dtype=torch.long, device=DEVICE)
EDGE_MASK_TORCH = (H_TORCH == 1).to(DTYPE)                       # 1.0 on edges, 0.0 elsewhere
EDGE_MASK_BOOL = (H_TORCH == 1)                                  # boolean version

# ========================================================================
# SECTION 2: PyTorch SUM-PRODUCT BP module
# ========================================================================
# This mirrors w1.sum_product_decode line for line, but using torch
# operations on tensors. No trainable parameters.
# nn.Module mechanics (just enough to read the code):
#   - __init__: set up any buffers/parameters (here just buffers).
#     Calling super().__init__() is required boilerplate.
#   - register_buffer(name, tensor): tells PyTorch "this is non-trainable
#     state belonging to this module." We use this for H and edge masks.
#   - forward(self, x): the actual computation. When you call
#     `module(x)`, PyTorch dispatches to forward(x).

class SumProductBP(torch.nn.Module):
    """
    Sum-product belief propagation decoder, PyTorch version.

    Forward input:  llrs of shape (B, N) — a batch of B received LLR vectors
    Forward output: hard decisions of shape (B, N), dtype int64 (0/1 values)

    Note the batched interface: Week 1's NumPy version decoded one vector
    at a time. The PyTorch version decodes a whole batch in one call,
    which will matter for training in Stage 2 (we'll train on batches of
    thousands of noisy codewords per gradient step).
    """

    def __init__(self, max_iters=20, tanh_clip=15.0):
        super().__init__()
        self.max_iters = max_iters
        self.tanh_clip = tanh_clip
        # Register H and the edge mask as buffers (non-trainable state)
        self.register_buffer("H", H_TORCH)
        self.register_buffer("H_int", H_INT_TORCH)
        self.register_buffer("edge_mask", EDGE_MASK_TORCH)

    def forward(self, llrs):
        # llrs shape: (B, N)
        B = llrs.shape[0]
        M, N = self.H.shape

        # Initialize messages to zero. Shape (B, M, N).
        # The B dimension is the batch dim; (M, N) is the same edge layout
        # as Week 1's NumPy code.
        # PyTorch idiom: torch.zeros takes a shape tuple; device and dtype
        # are inferred from input via tensor.new_zeros().
        M_v2c = llrs.new_zeros(B, M, N)
        M_c2v = llrs.new_zeros(B, M, N)
        active = torch.ones(B, dtype=torch.bool, device=llrs.device)

        # Edge mask broadcast helper: shape (1, M, N) so it can multiply
        # against (B, M, N) tensors via PyTorch broadcasting.
        # PyTorch idiom: tensor[None, :, :] adds a leading dimension of size 1.
        edge_mask_b = self.edge_mask[None, :, :]   # (1, M, N)
        for _ in range(self.max_iters):
            if not active.any():
                break
            # ------ Variable-node update (extrinsic) ------
            # For each (v, c): M_v2c[b, c, v] = llr[b, v] + sum_{c'!=c} M_c2v[b, c', v]
            #
            # Strategy: total per variable, then subtract this row's contribution.
            #   total[b, v] = llrs[b, v] + sum_c M_c2v[b, c, v]
            # Subtract M_c2v[b, c, v] to get the "all checks except this one" sum.
            #
            # PyTorch idiom: dim=1 sums over the M (check-node) dimension,
            # leaving shape (B, N). We then add a singleton M dim back via
            # [:, None, :] to broadcast against (B, M, N).
            total_incoming = llrs + M_c2v.sum(dim=1)               # (B, N)
            M_v2c = total_incoming[:, None, :] - M_c2v             # (B, M, N)
            M_v2c = M_v2c * edge_mask_b                            # zero non-edges
            # ------ Check-node update (tanh rule, extrinsic) ------
            # For each (c, v): M_c2v[b, c, v] = 2 * atanh( prod_{v'!=v} tanh(M_v2c[b, c, v']/2) )
            # Same "total product / this entry" trick as Week 1.
            clipped = torch.clamp(M_v2c / 2.0, -self.tanh_clip, self.tanh_clip)
            tanh_msgs = torch.tanh(clipped)                        # (B, M, N)

            # On non-edges, set tanh value to 1 so it's identity in product
            tanh_for_product = torch.where(edge_mask_b > 0,
                                           tanh_msgs,
                                           torch.ones_like(tanh_msgs))

            # Per-row total product, shape (B, M, 1)
            total_product = tanh_for_product.prod(dim=2, keepdim=True)

            # Extrinsic: divide out each entry. Guard against tiny values.
            EPS = 1e-12
            safe_tanh = torch.where(tanh_msgs.abs() < EPS,
                                    torch.full_like(tanh_msgs, EPS),
                                    tanh_msgs)
            extrinsic_product = total_product / safe_tanh           # (B, M, N)

            # Clip into (-1, 1) before atanh
            extrinsic_product = torch.clamp(extrinsic_product, -1 + EPS, 1 - EPS)
            new_M_c2v = 2.0 * torch.atanh(extrinsic_product)
            new_M_c2v = new_M_c2v * edge_mask_b                      # zero non-edges
            M_v2c = torch.where(active[:, None, None], M_v2c, M_v2c)
            M_c2v = torch.where(active[:, None, None], new_M_c2v, M_c2v)

            # Match Week 1: stop updating each sample as soon as its
            # syndrome is satisfied, while continuing any unfinished
            # batch elements.
            L_total = llrs + M_c2v.sum(dim=1)                       # (B, N)
            bits = (L_total < 0).long()                             # (B, N)
            syndrome_ok = torch.remainder(bits @ self.H_int.T, 2).eq(0).all(dim=1)
            active = active & (~syndrome_ok)

        # ------ Final hard decision after max_iters ------
        L_total = llrs + M_c2v.sum(dim=1)                          # (B, N)
        bits = (L_total < 0).long()                                # (B, N)
        return bits

# ========================================================================
# SECTION 3: PyTorch MIN-SUM BP module
# ========================================================================
# Identical to SumProductBP except for the check-node update.
# This is the module we'll modify in Stage 2 to add trainable weights.

class MinSumBP(torch.nn.Module):
    """
    Min-Sum belief propagation decoder, PyTorch version.
    """

    def __init__(self, max_iters=20):
        super().__init__()
        self.max_iters = max_iters
        self.register_buffer("H", H_TORCH)
        self.register_buffer("H_int", H_INT_TORCH)
        self.register_buffer("edge_mask", EDGE_MASK_TORCH)

    def forward(self, llrs):
        B = llrs.shape[0]
        M, N = self.H.shape
        M_v2c = llrs.new_zeros(B, M, N)
        M_c2v = llrs.new_zeros(B, M, N)
        edge_mask_b = self.edge_mask[None, :, :]                   # (1, M, N)
        active = torch.ones(B, dtype=torch.bool, device=llrs.device)
        for _ in range(self.max_iters):
            if not active.any():
                break

            # ------ Variable-node update (identical to sum-product) ------
            total_incoming = llrs + M_c2v.sum(dim=1)               # (B, N)
            M_v2c = total_incoming[:, None, :] - M_c2v             # (B, M, N)
            M_v2c = M_v2c * edge_mask_b

            # ------ Check-node update (Min-Sum, extrinsic) ------
            # For each (c, v): M_c2v = (prod_{v'!=v} sign) * (min_{v'!=v} |M|)
            #
            # Sign part: total product of signs / sign of this entry
            # (since signs are ±1, divide is the same as multiply).
            #
            # PyTorch idiom: torch.where(cond, a, b) is the elementwise
            # if-else. torch.sign returns -1, 0, or +1.

            signs = torch.where(edge_mask_b > 0,
                                torch.sign(M_v2c),
                                torch.ones_like(M_v2c))
            # Treat zero as +1 (avoid sign collapse)
            signs = torch.where(signs == 0,
                                torch.ones_like(signs),
                                signs)
            total_sign = signs.prod(dim=2, keepdim=True)            # (B, M, 1)
            extrinsic_sign = total_sign * signs                     # (B, M, N)

            # Magnitude part: smallest and second-smallest in each row
            INF = torch.tensor(float("inf"), dtype=M_v2c.dtype, device=M_v2c.device)
            mags = torch.where(edge_mask_b > 0, M_v2c.abs(), INF)   # (B, M, N)
            sorted_mags, _ = mags.sort(dim=2)                       # ascending
            min1 = sorted_mags[:, :, 0:1]                            # (B, M, 1)
            min2 = sorted_mags[:, :, 1:2]                            # (B, M, 1)
            extrinsic_min = torch.where(mags == min1, min2.expand_as(mags), min1.expand_as(mags))
            new_M_c2v = extrinsic_sign * extrinsic_min
            new_M_c2v = new_M_c2v * edge_mask_b
            M_v2c = torch.where(active[:, None, None], M_v2c, M_v2c)
            M_c2v = torch.where(active[:, None, None], new_M_c2v, M_c2v)
            L_total = llrs + M_c2v.sum(dim=1)
            bits = (L_total < 0).long()
            syndrome_ok = torch.remainder(bits @ self.H_int.T, 2).eq(0).all(dim=1)
            active = active & (~syndrome_ok)
        L_total = llrs + M_c2v.sum(dim=1)
        bits = (L_total < 0).long()
        return bits

# ========================================================================
# SECTION 4: NUMERICAL EQUIVALENCE TESTS
# ========================================================================
# These compare the PyTorch decoders to the Week 1 NumPy decoders on
# identical inputs. Pass criterion: the two implementations produce
# identical hard decisions on every test vector. We don't compare LLRs
# directly (floating-point arithmetic order can differ slightly) but
# decoded bits should be bit-for-bit identical.

def make_test_batch(n_samples, ebn0_db, seed=12345):
    """
    Generate (codewords, channel LLRs) using the same RNG/encoder as Week 1.
    Returns:
        codewords: shape (n_samples, N), int8 numpy array
        llrs_np:   shape (n_samples, N), float64 numpy array
    Both are passed unchanged to Week 1 decoders; the LLRs are also
    converted to a torch tensor for the PyTorch decoders.
    """
    # Use a fresh RNG so the test data is deterministic and independent
    # of any state Week 1's RNG might have accumulated.
    local_rng = np.random.default_rng(seed=seed)
    sigma = w1.ebn0_db_to_sigma(ebn0_db)
    codewords = np.zeros((n_samples, w1.N), dtype=np.int8)
    received = np.zeros((n_samples, w1.N), dtype=np.float64)
    for i in range(n_samples):
        msg = local_rng.integers(0, 2, size=w1.K).astype(np.int8)
        codewords[i] = w1.encode(msg)
        bpsk = w1.bpsk_modulate(codewords[i])
        # Use local_rng directly to avoid touching w1's global RNG
        noise = local_rng.normal(0.0, sigma, size=w1.N)
        received[i] = bpsk + noise
    llrs_np = 2.0 * received / (sigma ** 2)
    return codewords, llrs_np

def _test_sum_product_equivalence():
    """PyTorch SP must produce identical decisions to NumPy SP on every input."""
    codewords, llrs_np = make_test_batch(n_samples=100, ebn0_db=3.0, seed=11111)

    # NumPy reference
    np_decisions = np.zeros_like(codewords)
    for i in range(codewords.shape[0]):
        np_decisions[i] = w1.sum_product_decode(llrs_np[i])

    # PyTorch implementation
    llrs_torch = torch.tensor(llrs_np, dtype=DTYPE, device=DEVICE)
    sp_torch = SumProductBP(max_iters=20).to(device=DEVICE, dtype=DTYPE)
    with torch.no_grad():  # autograd not needed for inference
        pt_decisions = sp_torch(llrs_torch).cpu().numpy().astype(np.int8)

    # Count mismatches
    diffs = np.sum(np_decisions != pt_decisions)
    return diffs == 0, diffs, codewords.size

def _test_min_sum_equivalence():
    """PyTorch MS must produce identical decisions to NumPy MS on every input."""
    codewords, llrs_np = make_test_batch(n_samples=100, ebn0_db=3.0, seed=22222)
    np_decisions = np.zeros_like(codewords)
    for i in range(codewords.shape[0]):
        np_decisions[i] = w1.min_sum_decode(llrs_np[i])
    llrs_torch = torch.tensor(llrs_np, dtype=DTYPE, device=DEVICE)
    ms_torch = MinSumBP(max_iters=20).to(device=DEVICE, dtype=DTYPE)
    with torch.no_grad():
        pt_decisions = ms_torch(llrs_torch).cpu().numpy().astype(np.int8)
    diffs = np.sum(np_decisions != pt_decisions)
    return diffs == 0, diffs, codewords.size

def _test_bler_match():
    """
    Stronger test: at multiple SNRs, the PyTorch decoders must produce
    exactly the same BLER as the NumPy decoders (since they should produce
    exactly the same decoded bits on every input).
    """
    snrs = [2.0, 3.0, 4.0, 5.0]
    n_trials = 500
    all_match = True
    print("    SNR | NP-SP   PT-SP   match? | NP-MS   PT-MS   match?")
    for ebn0_db in snrs:
        codewords, llrs_np = make_test_batch(n_samples=n_trials,
                                              ebn0_db=ebn0_db,
                                              seed=int(ebn0_db * 1000))

        # NumPy decoders
        np_sp_errs = 0
        np_ms_errs = 0
        for i in range(n_trials):
            if not np.array_equal(w1.sum_product_decode(llrs_np[i]), codewords[i]):
                np_sp_errs += 1
            if not np.array_equal(w1.min_sum_decode(llrs_np[i]), codewords[i]):
                np_ms_errs += 1

        # PyTorch decoders (batched)
        llrs_torch = torch.tensor(llrs_np, dtype=DTYPE, device=DEVICE)
        cw_torch = torch.tensor(codewords, dtype=torch.long, device=DEVICE)
        sp_torch = SumProductBP(max_iters=20).to(device=DEVICE, dtype=DTYPE)
        ms_torch = MinSumBP(max_iters=20).to(device=DEVICE, dtype=DTYPE)
        with torch.no_grad():
            pt_sp_errs = (sp_torch(llrs_torch) != cw_torch).any(dim=1).sum().item()
            pt_ms_errs = (ms_torch(llrs_torch) != cw_torch).any(dim=1).sum().item()
        sp_match = "yes" if np_sp_errs == pt_sp_errs else "NO"
        ms_match = "yes" if np_ms_errs == pt_ms_errs else "NO"
        if sp_match == "NO" or ms_match == "NO":
            all_match = False
        print(f"    {ebn0_db:>3.1f} | {np_sp_errs:>5d}   {pt_sp_errs:>5d}   {sp_match:>5s}  | "
              f"{np_ms_errs:>5d}   {pt_ms_errs:>5d}   {ms_match:>5s}")
    return all_match


def run_stage1_tests():
    print("=" * 60)
    print("Week 2 Stage 1: PyTorch port verification")
    print("=" * 60)
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  Device: {DEVICE}, dtype: {DTYPE}")
    print()
    print("Test 1: Sum-Product PyTorch <==> NumPy (100 inputs at 3 dB)")
    ok, diffs, total = _test_sum_product_equivalence()
    status = "PASS" if ok else f"FAIL ({diffs}/{total} bits disagree)"
    print(f"  [{status}]")
    print()
    print("Test 2: Min-Sum PyTorch <==> NumPy (100 inputs at 3 dB)")
    ok, diffs, total = _test_min_sum_equivalence()
    status = "PASS" if ok else f"FAIL ({diffs}/{total} bits disagree)"
    print(f"  [{status}]")
    print()
    print("Test 3: BLER match across SNR sweep (500 trials per point)")
    bler_ok = _test_bler_match()
    print(f"  [{'PASS' if bler_ok else 'FAIL'}]")
    print()
    print("=" * 60)
    print("Stage 1 verification complete.")
    print("If all tests passed, the PyTorch infrastructure is correct.")
    print("Stage 2 will add trainable edge weights to MinSumBP to create")
    print("the Neural BP decoder.")
    print("=" * 60)

if __name__ == "__main__":
    run_stage1_tests()