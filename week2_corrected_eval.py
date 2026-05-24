"""
========================================================================
Week 2 Stage 2 — Corrected Evaluation
Apples-to-apples BLER comparison and final figures.

The original Stage 2 sweep compared trained NBP (T=5) against Min-Sum
and Sum-Product running at T=20 iterations. This was an unfair
comparison: NBP got 1/4 the iteration budget of its baselines, which
made the original "trained NBP loses to Min-Sum" result an artifact
of iteration mismatch rather than a real performance gap.

This script:
  1. Trains NBP(T=5) — same training as before.
  2. Runs the BLER sweep with ALL decoders at T=5 for fairness.
  3. Uses adaptive trial counts: more trials at high SNR where errors
     are rare and confidence intervals are tight.
  4. Computes Wilson 95% confidence intervals for each BLER point.
  5. Saves a corrected BLER plot with error bars and a learned-weight
     histogram.

Run after week2_stage2_neural_bp.py if you want to regenerate the
final figures with the apples-to-apples comparison.

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
import week2_stage2_neural_bp as stage2

# ---- Reproducibility ----
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
w1.RNG = np.random.default_rng(seed=SEED)
DEVICE = torch.device("cpu")
DTYPE = torch.float64

# Common iteration budget for ALL message-passing decoders in this comparison
T_COMMON = 5

# ========================================================================
# Wilson 95% confidence interval for a binomial proportion
# ========================================================================
#
# Why Wilson and not normal-approximation? At low BLER the count of
# errors is small (a few dozen out of tens of thousands), and the
# normal approximation breaks down. The Wilson interval handles small
# error counts and proportions near 0 cleanly, including p_hat = 0.
#
# Formula:
#   Given x successes out of n trials and z = 1.96 for 95% confidence,
#   center = (p_hat + z^2/(2n)) / (1 + z^2/n)
#   spread = z * sqrt( p_hat*(1-p_hat)/n + z^2/(4n^2) ) / (1 + z^2/n)
#   CI = [center - spread, center + spread]

def wilson_ci(n_errors, n_trials, conf=0.95):
    """Returns (lower, upper) of the Wilson CI for an observed proportion."""
    if n_trials == 0:
        return 0.0, 1.0
    from scipy.stats import norm
    z = norm.ppf(1 - (1 - conf) / 2)  # 1.96 for 95%
    p_hat = n_errors / n_trials
    denom = 1 + z * z / n_trials
    center = (p_hat + z * z / (2 * n_trials)) / denom
    spread = z * math.sqrt(p_hat * (1 - p_hat) / n_trials + z * z / (4 * n_trials * n_trials)) / denom
    return max(0.0, center - spread), min(1.0, center + spread)

# ========================================================================
# Decoder callables
# ========================================================================

def make_ml_callable():
    def _call(received, llrs):
        return np.stack([w1.ml_decode(received[i]) for i in range(received.shape[0])], axis=0)
    return _call

def make_torch_callable(module):
    module.eval()
    def _call(received, llrs):
        with torch.no_grad():
            llrs_t = torch.tensor(llrs, dtype=DTYPE, device=DEVICE)
            return module(llrs_t).cpu().numpy().astype(np.int8)
    return _call

# ========================================================================
# Adaptive BLER measurement
# ========================================================================
#
# At low SNR, BLER is high so we need few trials. At high SNR, BLER is
# tiny so we need many trials to see any errors at all. We use a target
# of "at least N_TARGET_ERRORS errors observed, or N_MAX_TRIALS reached"
# — whichever comes first. This gives roughly equal statistical
# confidence at every SNR rather than wasted trials at low SNR and
# noisy estimates at high SNR.

def measure_bler_shared(decoder_callables, ebn0_db,
                        n_target_errors=200, n_max_trials=200000,
                        batch=500, seed=0):
    """
    Run a single shared Monte Carlo stream for ALL decoders.
    Every decoder sees the exact same transmitted codewords and AWGN samples
    at each SNR point, ensuring a fair apples-to-apples BLER comparison.
    Returns:
        n_errors: dict(name -> error_count)
        n_done: shared trial count used for all decoders
    """
    rng = np.random.default_rng(seed)
    sigma = w1.ebn0_db_to_sigma(ebn0_db)
    n_errors = {name: 0 for name in decoder_callables}
    n_done = 0
    while n_done < n_max_trials:
        b = min(batch, n_max_trials - n_done)
        msgs = rng.integers(0, 2, size=(b, w1.K)).astype(np.int8)
        codewords = np.stack([w1.encode(msgs[i]) for i in range(b)], axis=0)
        bpsk = 1.0 - 2.0 * codewords.astype(np.float64)
        noise = rng.normal(0.0, sigma, size=(b, w1.N))
        received = bpsk + noise
        llrs = 2.0 * received / sigma**2
        for name, fn in decoder_callables.items():
            decoded = fn(received, llrs)
            block_errors = np.any(decoded != codewords, axis=1)
            n_errors[name] += int(block_errors.sum())
        n_done += b
        if all(errs >= n_target_errors for errs in n_errors.values()):
            break
    return n_errors, n_done

# ========================================================================
# Main pipeline
# ========================================================================

def main():
    print("=" * 60)
    print("Stage 2 corrected evaluation")
    print("=" * 60)
    print(f"Common iteration budget for all decoders: T = {T_COMMON}")
    print(f"PyTorch: {torch.__version__}, device: {DEVICE}")
    print()

    # ---- Train NBP (same as Stage 2) ----
    print("Training NeuralBP(T=5) over Eb/N0 in [1, 6] dB...")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = stage2.NeuralBP(max_iters=T_COMMON).to(device=DEVICE, dtype=DTYPE)
    stage2.train_neural_bp(
        model, num_steps=4000, batch_size=256,
        snr_low_db=1.0, snr_high_db=6.0,
        lr=1e-3, log_every=2000,
    )
    print()

    # ---- Build all decoders at T = T_COMMON ----
    sp_module = stage1.SumProductBP(max_iters=T_COMMON).to(device=DEVICE, dtype=DTYPE)
    ms_module = stage1.MinSumBP(max_iters=T_COMMON).to(device=DEVICE, dtype=DTYPE)
    untrained_nbp = stage2.NeuralBP(max_iters=T_COMMON).to(device=DEVICE, dtype=DTYPE)
    callables = {
        "ML":                  make_ml_callable(),
        "SumProductBP":        make_torch_callable(sp_module),
        "MinSumBP":            make_torch_callable(ms_module),
        "NeuralBP_untrained":  make_torch_callable(untrained_nbp),
        "NeuralBP_trained":    make_torch_callable(model),
    }

    # ---- BLER sweep with adaptive trial counts ----
    snrs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    # At low SNR, errors are common — 5000 max trials enough. At high
    # SNR, errors are rare — let it run up to 50000 trials per point
    # if needed to gather 200 errors.
    sweep_max = {1.0: 10000, 2.0: 10000, 3.0: 15000, 4.0: 25000,
                 5.0: 50000, 6.0: 100000}
    bler_n_err = {name: [] for name in callables}
    bler_n_tri = {name: [] for name in callables}
    print("Adaptive BLER sweep (target: 200 errors per point, up to N_max trials):")
    print(f"{'SNR':>5}  " + "  ".join(f"{n:>22s}" for n in callables))
    for snr in snrs:
        row = []
        n_errs, n_t = measure_bler_shared(
            callables, ebn0_db=snr,
            n_target_errors=200,
            n_max_trials=sweep_max[snr],
            batch=500,
            seed=int(snr * 1000),
        )
        for name in callables:
            n_e = n_errs[name]
            bler_n_err[name].append(n_e)
            bler_n_tri[name].append(n_t)
            bler_hat = n_e / n_t
            row.append(f"{bler_hat:.5f} ({n_e}/{n_t})")
        print(f"{snr:>5.1f}  " + "  ".join(f"{r:>22s}" for r in row))
    print()

    # ---- Compute Wilson CIs ----
    bler_pt = {}
    bler_lo = {}
    bler_hi = {}
    for name in callables:
        pt, lo, hi = [], [], []
        for n_e, n_t in zip(bler_n_err[name], bler_n_tri[name]):
            p_hat = n_e / n_t if n_t > 0 else 0.0
            l, h = wilson_ci(n_e, n_t, conf=0.95)
            pt.append(p_hat)
            lo.append(l)
            hi.append(h)
        bler_pt[name] = pt
        bler_lo[name] = lo
        bler_hi[name] = hi

    # ---- Save plot ----
    out_dir = os.path.dirname(os.path.abspath(__file__))
    fig_path = os.path.join(out_dir, "bler_vs_snr.png")
    plt.figure(figsize=(8, 6))
    style = {
        "ML":                 ("ML (lower bound)",      "k",  "-",  "o"),
        "SumProductBP":       ("Sum-Product BP (T=5)",  "C0", "-",  "s"),
        "MinSumBP":           ("Min-Sum BP (T=5)",      "C1", "-",  "^"),
        "NeuralBP_untrained": ("Neural BP untrained (T=5)", "C2", "--", "x"),
        "NeuralBP_trained":   ("Neural BP trained (T=5)",   "C3", "-",  "D"),
    }
    for name, (label, color, ls, marker) in style.items():
        # Plot point estimate, with shaded CI band
        # Floor at 1/n to keep semilogy from blowing up at zero errors
        floor = 1.0 / max(bler_n_tri[name])
        ys = np.array([max(p, floor) for p in bler_pt[name]])
        lo = np.array([max(v, floor) for v in bler_lo[name]])
        hi = np.array([max(v, floor) for v in bler_hi[name]])

        # Use error bars (asymmetric in log space): [y - lo, hi - y]
        yerr_lower = ys - lo
        yerr_upper = hi - ys
        plt.errorbar(snrs, ys,
                     yerr=[yerr_lower, yerr_upper],
                     color=color, linestyle=ls, marker=marker,
                     label=label, capsize=3, markersize=6)
    plt.yscale("log")
    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("Block Error Rate")
    plt.title("BCH(15,11) over BI-AWGN: BLER vs SNR\n"
              "All message-passing decoders at T=5 iterations, 95% Wilson CIs")
    plt.grid(True, which="both", linestyle=":")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Saved corrected BLER plot: {fig_path}")

    # ---- Save weight histogram ----
    hist_path = os.path.join(out_dir, "learned_weights_hist.png")
    edge_mask_np = stage1.EDGE_MASK_TORCH.cpu().numpy() > 0
    all_w = []
    for t in range(model.max_iters):
        w_v = model.w_v2c[t].detach().cpu().numpy()
        w_c = model.w_c2v[t].detach().cpu().numpy()
        all_w.extend(w_v[edge_mask_np].tolist())
        all_w.extend(w_c[edge_mask_np].tolist())
    plt.figure(figsize=(7, 4))
    plt.hist(all_w, bins=40, edgecolor="black")
    plt.axvline(1.0, color="red", linestyle="--",
                label="Init value (1.0 = Min-Sum)")
    plt.axvline(np.mean(all_w), color="blue", linestyle=":",
                label=f"Trained mean = {np.mean(all_w):.3f}")
    plt.xlabel("Learned edge weight")
    plt.ylabel("Count")
    plt.title("Trained NBP edge weight distribution\n"
              f"(across all {model.max_iters} iterations, {len(all_w)} weights)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(hist_path, dpi=150)
    plt.close()
    print(f"Saved learned weights histogram: {hist_path}")
    print()

    # ---- Headline numerical summary ----
    print("Headline summary (NBP_trained vs MinSum, both at T=5):")
    print(f"  {'SNR':>5}  {'MinSum':>10}  {'NBP':>10}  {'rel gain':>10}  "
          f"{'NBP CI':>20}  {'MS CI':>20}")
    for i, snr in enumerate(snrs):
        ms = bler_pt["MinSumBP"][i]
        nbp = bler_pt["NeuralBP_trained"][i]
        rel = (ms - nbp) / ms * 100 if ms > 0 else 0
        nbp_ci = f"[{bler_lo['NeuralBP_trained'][i]:.4f}, {bler_hi['NeuralBP_trained'][i]:.4f}]"
        ms_ci = f"[{bler_lo['MinSumBP'][i]:.4f}, {bler_hi['MinSumBP'][i]:.4f}]"
        print(f"  {snr:>5.1f}  {ms:>10.5f}  {nbp:>10.5f}  {rel:>+9.1f}%  "
              f"{nbp_ci:>20}  {ms_ci:>20}")
    print("\nDone.")

if __name__ == "__main__":
    main()