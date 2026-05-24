"""
Diagnostic script for the Stage 2 anomaly:
trained NeuralBP performs worse than Min-Sum (and worse than untrained NBP)
at every SNR. We need to figure out why before fixing.

Things to check:
  1. Linearity check at every SNR (not just 4 dB).
  2. Training-time loss on all-zero codewords vs eval-time BLER on
     all-zero codewords. They should track each other.
  3. What do the learned weights actually look like? Mean, std, min, max.
  4. Try a quick re-run with a much smaller learning rate or a simpler
     loss to see whether the issue is the discount, the BCE, or the all-zero
     training itself.

Run from the same directory as week1/week2 files.
"""

import numpy as np
import torch
import week1_bch_baselines as w1
import week2_stage1_pytorch_port as stage1
import week2_stage2_neural_bp as stage2
DEVICE = torch.device("cpu")
DTYPE = torch.float64
torch.manual_seed(42)
np.random.seed(42)
w1.RNG = np.random.default_rng(seed=42)

def linearity_at_snr(model, ebn0_db, n_trials=5000, seed=0):
    """BLER on all-zero vs random codewords at one SNR."""
    rng = np.random.default_rng(seed)
    sigma = w1.ebn0_db_to_sigma(ebn0_db)

    # All-zero
    noise = rng.normal(0, sigma, size=(n_trials, w1.N))
    received_zero = 1.0 + noise
    llrs_zero = 2.0 * received_zero / sigma**2
    with torch.no_grad():
        d_zero = model(torch.tensor(llrs_zero, dtype=DTYPE, device=DEVICE)).cpu().numpy()
    bler_zero = float(np.any(d_zero != 0, axis=1).mean())

    # Random
    msgs = rng.integers(0, 2, size=(n_trials, w1.K)).astype(np.int8)
    cws = np.stack([w1.encode(msgs[i]) for i in range(n_trials)], axis=0)
    bpsk = 1.0 - 2.0 * cws.astype(np.float64)
    noise2 = rng.normal(0, sigma, size=(n_trials, w1.N))
    received_rand = bpsk + noise2
    llrs_rand = 2.0 * received_rand / sigma**2
    with torch.no_grad():
        d_rand = model(torch.tensor(llrs_rand, dtype=DTYPE, device=DEVICE)).cpu().numpy()
    bler_rand = float(np.any(d_rand != cws, axis=1).mean())
    return bler_zero, bler_rand

def weight_stats(model):
    """Print mean/std/min/max of learned weights at each iteration."""
    print("Learned weight statistics by iteration:")
    edge_mask = stage1.EDGE_MASK_TORCH.cpu().numpy() > 0
    for t in range(model.max_iters):
        wv = model.w_v2c[t].detach().cpu().numpy()[edge_mask]
        wc = model.w_c2v[t].detach().cpu().numpy()[edge_mask]
        print(f"  iter {t}: w_v2c mean={wv.mean():+.3f} std={wv.std():.3f} "
              f"min={wv.min():+.3f} max={wv.max():+.3f}  |  "
              f"w_c2v mean={wc.mean():+.3f} std={wc.std():.3f} "
              f"min={wc.min():+.3f} max={wc.max():+.3f}")

def quick_compare_to_minsum(model, ebn0_db=4.0, n_trials=5000, seed=42):
    """Side-by-side BLER on identical inputs."""
    rng = np.random.default_rng(seed)
    sigma = w1.ebn0_db_to_sigma(ebn0_db)
    msgs = rng.integers(0, 2, size=(n_trials, w1.K)).astype(np.int8)
    cws = np.stack([w1.encode(msgs[i]) for i in range(n_trials)], axis=0)
    bpsk = 1.0 - 2.0 * cws.astype(np.float64)
    noise = rng.normal(0, sigma, size=(n_trials, w1.N))
    received = bpsk + noise
    llrs = 2.0 * received / sigma**2
    llrs_t = torch.tensor(llrs, dtype=DTYPE, device=DEVICE)
    ms = stage1.MinSumBP(max_iters=5).to(device=DEVICE, dtype=DTYPE)  # NB: T=5 to match
    untrained = stage2.NeuralBP(max_iters=5).to(device=DEVICE, dtype=DTYPE)
    with torch.no_grad():
        d_ms = ms(llrs_t).cpu().numpy()
        d_un = untrained(llrs_t).cpu().numpy()
        d_tr = model(llrs_t).cpu().numpy()
    bler_ms = float(np.any(d_ms != cws, axis=1).mean())
    bler_un = float(np.any(d_un != cws, axis=1).mean())
    bler_tr = float(np.any(d_tr != cws, axis=1).mean())
    print(f"  At {ebn0_db} dB ({n_trials} trials, T=5 throughout):")
    print(f"    Min-Sum (T=5):           BLER = {bler_ms:.4f}")
    print(f"    Untrained NeuralBP (T=5): BLER = {bler_un:.4f}")
    print(f"    Trained NeuralBP (T=5):   BLER = {bler_tr:.4f}")
    print()
    print("  NOTE: Stage 2's earlier sweep compared trained NBP (T=5) against")
    print("  Min-Sum (T=20) — which is unfair! The two should both use T=5")
    print("  for an apples-to-apples comparison.")

if __name__ == "__main__":
    print("=" * 60)
    print("Diagnostic: why is trained NeuralBP underperforming?")
    print("=" * 60)
    print()

    # Re-train a fresh model so we have one to inspect
    print("Re-training NBP for diagnostics (4000 steps, seed=42)...")
    torch.manual_seed(42)
    np.random.seed(42)
    model = stage2.NeuralBP(max_iters=5).to(device=DEVICE, dtype=DTYPE)
    stage2.train_neural_bp(model, num_steps=4000, batch_size=256,
                            snr_low_db=1.0, snr_high_db=6.0,
                            lr=1e-3, log_every=2000)
    print()

    # 1. Weight statistics
    weight_stats(model)
    print()

    # 2. Linearity at every SNR
    print("Linearity check at each SNR (5000 trials each):")
    print(f"  {'SNR':>5}  {'BLER(all-0)':>14}  {'BLER(random)':>14}  {'gap':>10}")
    for snr in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]:
        b0, br = linearity_at_snr(model, snr, n_trials=5000, seed=int(snr * 1000))
        gap = br - b0
        print(f"  {snr:>5.1f}  {b0:>14.4f}  {br:>14.4f}  {gap:>+10.4f}")
    print()

    # 3. Apples-to-apples comparison: same T for everyone
    print("=" * 60)
    print("Apples-to-apples comparison (same T=5 for Min-Sum and NBP):")
    print("=" * 60)
    for snr in [2.0, 4.0, 6.0]:
        quick_compare_to_minsum(model, ebn0_db=snr, n_trials=5000, seed=int(snr * 100))
        print()