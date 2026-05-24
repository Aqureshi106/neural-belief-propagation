"""
========================================================================
CSCI 381 Project — Week 1 Implementation
Neural Belief Propagation for Short BCH Codes: An Architecture Ablation Study

This module builds the foundation that every later experiment depends on:
  - BCH(15, 11) encoder
  - Parity-check matrix H (Tanner graph definition)
  - BI-AWGN channel simulator with BPSK modulation
  - LLR conversion
  - Syndrome decoder (used as a correctness sanity check)
  - Maximum-Likelihood decoder (absolute BLER lower bound, per professor's suggestion)
  - Sum-Product Belief Propagation decoder
  - Min-Sum Belief Propagation decoder
  - A self-test driver that validates every component above

Run as a script (`python week1_bch_baselines.py`) to execute the self-tests.
The script will print PASS/FAIL for each check and finish with a small
BLER table comparing all decoders at one SNR point.

Author: Ashir Qureshi
Course: CSCI 381 - Information Theory & Error Correction Codes
========================================================================
"""

import numpy as np

# Global random seed for reproducibility (course spec §4.1 requires this)
RNG = np.random.default_rng(seed=42)

# ========================================================================
# SECTION 1: BCH(15, 11) CODE DEFINITION
# ========================================================================
# BCH(15, 11) is constructed over GF(2^4) with primitive polynomial
#     p(x) = x^4 + x + 1
# It corrects any single bit error (t=1, minimum distance d_min=3),
# making it equivalent to the (15, 11) Hamming code, just with a
# polynomial-construction story rather than a parity-check construction.
# Generator polynomial: g(x) = x^4 + x + 1
# As a coefficient vector (lowest degree first): [1, 1, 0, 0, 1]
# The parity-check matrix H of the corresponding Hamming code has 4 rows
# and 15 columns. Each column is the binary representation of integers
# 1 through 15 — that is, every nonzero 4-bit pattern. We hard-code this
# explicitly here so the H matrix is auditable (per professor's suggestion #2).
# Generator polynomial g(x) = 1 + x + x^4, coefficients in increasing degree.
# Length 5 = (n - k + 1) = (15 - 11 + 1).
GENERATOR_POLY = np.array([1, 1, 0, 0, 1], dtype=np.int8)
N = 15   # codeword length
K = 11   # message length
M = N - K  # number of parity bits = 4
# Build H by deriving it from the generator polynomial. This guarantees
# encoder and H define the same code (a common bug source if they're
# constructed independently).
# Procedure (systematic-form construction):
#   1. The systematic encoder produces codeword [parity (M bits) | message (K bits)].
#   2. We want H such that H · c^T = 0 for every such codeword.
#   3. By construction, parity = (m·x^M) mod g(x). So if we expand the encoder
#      as a linear map, we get a generator matrix G of shape (K, N) in
#      systematic form: G = [P | I_K], where P is K x M.
#   4. The corresponding parity-check matrix is then H = [I_M | P^T],
#      which has shape (M, N) and satisfies H · G^T = 0.
# We build G column-by-column by encoding each unit message vector
# (i.e., a message that is all zeros except for a single 1).

def build_generator_matrix():
    """
    Construct the K x N generator matrix G in systematic form by encoding
    each unit message vector. After construction, G[i] is the codeword
    produced by a message that is 1 in position i and 0 everywhere else.
    """
    G = np.zeros((K, N), dtype=np.int8)
    for i in range(K):
        unit_msg = np.zeros(K, dtype=np.int8)
        unit_msg[i] = 1
        G[i] = encode(unit_msg)
    return G

def build_parity_check_matrix(G):
    """
    Given a systematic generator matrix G = [P | I_K] of shape (K, N),
    return the corresponding parity-check matrix H = [I_M | P^T]
    of shape (M, N).
    """
    # In our encoder convention: c = [parity (M bits) | message (K bits)],
    # so G[:, :M] is P (the parity part for each unit message)
    # and G[:, M:] should be I_K (the systematic identity part).
    P = G[:, :M]               # shape (K, M)
    I_M = np.eye(M, dtype=np.int8)
    H = np.concatenate([I_M, P.T], axis=1)  # shape (M, N)
    return H

# Note: the order matters. encode() doesn't reference H, so we can call it
# during G construction. Then we derive H from G.
G_MATRIX = None  # filled after encode() is defined
H = None         # filled after G_MATRIX is built

# ========================================================================
# SECTION 2: BCH ENCODER
# ========================================================================
# Systematic encoding via polynomial division:
#   1. Treat the 11-bit message m as a polynomial m(x) of degree < 11.
#   2. Multiply by x^4 (shift left by 4): m(x) * x^4
#   3. Divide by g(x) and take the remainder r(x), of degree < 4.
#   4. Codeword c(x) = m(x) * x^4 + r(x).
# In systematic form the codeword is [parity_bits | message_bits].
# All arithmetic is in GF(2), so addition is XOR.

def gf2_poly_mod(dividend, divisor):
    """
    Polynomial long division in GF(2). Returns the remainder.

    `dividend` and `divisor` are 1D int arrays of polynomial coefficients,
    lowest degree first. Returns an array of length (len(divisor) - 1)
    representing the remainder.

    NumPy idiom: we use .copy() because we'll modify the dividend in place,
    and we don't want to mutate the caller's array.
    """
    rem = dividend.copy().astype(np.int8)
    deg_div = len(divisor) - 1  # degree of divisor

    # Walk from highest degree of dividend down to degree of divisor.
    # Each iteration eliminates the highest-degree term of the current remainder
    # by XOR-ing in a shifted copy of the divisor.
    for i in range(len(rem) - 1, deg_div - 1, -1):
        if rem[i] == 1:
            # XOR divisor (shifted so its highest term aligns with rem[i]) into rem
            rem[i - deg_div : i + 1] ^= divisor
    # The remainder is everything below degree deg_div
    return rem[:deg_div]

def encode(message_bits):
    """
    Systematic BCH(15, 11) encoder.
    Input: message_bits, shape (11,), dtype int (0/1 values)
    Output: codeword, shape (15,), dtype int8
    Convention: codeword = [parity (4 bits) | message (11 bits)]
    """
    assert message_bits.shape == (K,), f"expected shape ({K},), got {message_bits.shape}"

    # Multiply m(x) by x^4: shift coefficients up by 4 positions.
    # In coefficient-array terms: prepend 4 zeros (since index 0 = lowest degree).
    shifted = np.concatenate([np.zeros(M, dtype=np.int8), message_bits.astype(np.int8)])

    # Compute parity = (m(x) * x^4) mod g(x)
    parity = gf2_poly_mod(shifted, GENERATOR_POLY)

    # Codeword = parity (low-degree positions) + shifted message (high-degree positions)
    # XOR works because parity occupies positions 0..3 and shifted has zeros there.
    codeword = shifted.copy()
    codeword[:M] ^= parity
    return codeword

# Now that encode() is defined, build G and derive H from it.
G_MATRIX = build_generator_matrix()
H = build_parity_check_matrix(G_MATRIX)

def is_codeword(c):
    """
    Verify that c satisfies all parity checks: H * c^T = 0 in GF(2).
    NumPy idiom: H @ c is matrix-vector product; % 2 reduces to GF(2).
    Returns True if c is a valid codeword, False otherwise.
    """
    syndrome = (H @ c) % 2
    return np.all(syndrome == 0)

# Precompute all 2^11 = 2048 codewords once, for use by the ML decoder.
# This is cheap (only 2048 × 15 = 30,720 bits ≈ 4 KB) and saves repeated work.
def build_codebook():
    """
    Returns an array of shape (2^K, N) = (2048, 15) containing every valid codeword.
    NumPy idiom: We iterate over integers 0..2047, convert each to its 11-bit
    binary representation (the message), and encode it. The result is stacked
    into a single 2D array.
    """
    codebook = np.zeros((2**K, N), dtype=np.int8)
    for i in range(2**K):
        # Convert integer i to a length-K binary array, lowest bit first
        msg = np.array([(i >> b) & 1 for b in range(K)], dtype=np.int8)
        codebook[i] = encode(msg)
    return codebook

CODEBOOK = build_codebook()  # shape (2048, 15)

# ========================================================================
# SECTION 3: BI-AWGN CHANNEL WITH BPSK MODULATION
# ========================================================================
# The channel:
#   1. Map each codeword bit b ∈ {0, 1} to a BPSK symbol s = 1 - 2b ∈ {+1, -1}
#      (so bit 0 → +1, bit 1 → -1)
#   2. Transmit s + n, where n ~ N(0, σ²)
#   3. The received symbol y is real-valued.
# Relating noise variance σ² to SNR (Eb/N0 in dB):
#   - Eb (energy per information bit) = (n/k) × Es, where Es = 1 for BPSK
#   - N0 = 2σ² for one-sided noise PSD, but for our purposes we use the
#     standard formula: σ² = 1 / (2 * R * Eb_N0_linear)
#     where R = k/n is the code rate and Eb_N0_linear = 10^(Eb_N0_dB / 10).

CODE_RATE = K / N  # 11/15 ≈ 0.733

def ebn0_db_to_sigma(ebn0_db):
    """
    Convert Eb/N0 in dB to AWGN noise standard deviation σ for BPSK
    transmission of a code with rate R = K/N.
    Formula: σ² = 1 / (2 * R * 10^(Eb/N0 / 10))
    """
    ebn0_linear = 10 ** (ebn0_db / 10)
    sigma_sq = 1.0 / (2 * CODE_RATE * ebn0_linear)
    return np.sqrt(sigma_sq)

def bpsk_modulate(codewords):
    """
    Map bits to BPSK symbols: 0 -> +1, 1 -> -1.
    Works on a single codeword (shape (N,)) or a batch (shape (B, N)).
    NumPy idiom: 1 - 2*x is vectorized — same code handles scalar, vector, batch.
    """
    return 1.0 - 2.0 * codewords.astype(np.float64)

def awgn(symbols, sigma):
    """
    Add zero-mean Gaussian noise of standard deviation sigma.
    NumPy idiom: RNG.normal(...) returns an array of the requested shape,
    independent of the input — broadcasting takes care of the addition.
    """
    noise = RNG.normal(0.0, sigma, size=symbols.shape)
    return symbols + noise

def llr_from_received(y, sigma):
    """
    Convert received symbols to log-likelihood ratios.
    For BPSK over BI-AWGN with the convention 0 -> +1, 1 -> -1:
        LLR = log(P(bit=0 | y) / P(bit=1 | y)) = 2y / σ²
    Convention: positive LLR favors bit 0, negative favors bit 1.
    Magnitude is the confidence.
    """
    return 2.0 * y / (sigma ** 2)

# ========================================================================
# SECTION 4: SYNDROME DECODER (sanity check, not one of the 4 main decoders)
# ========================================================================
# Hard-decision decoding via syndrome lookup.
#   1. Quantize LLRs to bits: positive -> 0, negative -> 1.
#   2. Compute syndrome s = H * b^T mod 2.
#   3. If s == 0, return b unchanged.
#   4. Otherwise, find the column of H matching s — that's the error position.
#   5. Flip that bit.
# This decodes any single-bit error perfectly. It's not competitive with BP
# at higher SNR but it's a useful baseline for verifying the encoder/H matrix
# combination is set up correctly.
# Precompute syndrome -> error-position lookup table.
# Each column of H is the syndrome you'd get if exactly that bit had flipped.
SYNDROME_TABLE = {}
for col_idx in range(N):
    syndrome_int = int("".join(str(b) for b in H[:, col_idx]), 2)
    SYNDROME_TABLE[syndrome_int] = col_idx

def syndrome_decode(llrs):
    """
    Hard-decision syndrome decoder for single-error correction.
    Input: llrs, shape (N,)
    Output: decoded codeword, shape (N,), dtype int8
    """
    # Hard-decide on the LLR sign: positive LLR -> bit 0, negative -> bit 1
    bits = (llrs < 0).astype(np.int8)
    syndrome = (H @ bits) % 2
    syndrome_int = int("".join(str(b) for b in syndrome), 2)
    if syndrome_int == 0:
        return bits  # already a valid codeword
    if syndrome_int in SYNDROME_TABLE:
        error_pos = SYNDROME_TABLE[syndrome_int]
        bits[error_pos] ^= 1  # flip the offending bit
    return bits

# ========================================================================
# SECTION 5: MAXIMUM-LIKELIHOOD DECODER (absolute BLER lower bound)
# ========================================================================
# Per professor's suggestion #4: enumerate all 2^11 = 2048 codewords,
# BPSK-modulate each, and pick the one closest in Euclidean distance to
# the received vector. This is optimal — no decoder can do better — so
# it gives us an absolute lower bound for the BLER plot.
# Important: ML decoding here directly compares received y to BPSK-modulated
# codewords (NOT to LLRs). The minimum-Euclidean-distance rule on y is
# equivalent to maximum-likelihood under AWGN.
# Precompute BPSK-modulated codebook once (avoid re-modulating per call).
CODEBOOK_BPSK = bpsk_modulate(CODEBOOK)  # shape (2048, 15), values in {+1, -1}

def ml_decode(y):
    """
    Maximum-likelihood decoder for BCH(15, 11) over BI-AWGN.
    Input: y, shape (N,) — received real-valued symbols
    Output: decoded codeword, shape (N,), dtype int8
    NumPy idiom: y is shape (15,), CODEBOOK_BPSK is shape (2048, 15).
    Subtracting them broadcasts y across all 2048 rows, giving differences
    of shape (2048, 15). Squaring elementwise and summing along axis=1
    gives squared distances of shape (2048,). argmin returns the row index
    of the closest codeword.
    """
    diffs = CODEBOOK_BPSK - y         # shape (2048, 15) via broadcasting
    sq_dists = np.sum(diffs ** 2, axis=1)  # shape (2048,)
    best_idx = np.argmin(sq_dists)
    return CODEBOOK[best_idx]

# ========================================================================
# SECTION 6: SUM-PRODUCT BELIEF PROPAGATION
# ========================================================================
# Iterative message passing on the Tanner graph. We use the log-domain
# (LLR) formulation throughout for numerical stability — the L24 lecture
# emphasizes this point.
# State maintained:
#   - L_ch[v]: channel LLR for variable node v (constant over iterations)
#   - M_v2c[v, c]: message from variable v to check c
#   - M_c2v[c, v]: message from check c to variable v
# The Tanner graph edge set is just {(c, v) : H[c, v] == 1}.
# We store M_v2c and M_c2v as full (M × N) matrices and zero out the
# entries where H == 0; this is wasteful for large/sparse codes but
# perfectly fine at N=15.
# Update rules (per L15/L24):
#   Variable node update (extrinsic):
#       M_v2c[v, c] = L_ch[v] + sum_{c' != c} M_c2v[c', v]
#     "Extrinsic" means we exclude the message from c itself when
#     building the message back to c — this prevents the echo-chamber
#     effect noted in L15.
#   Check node update (tanh rule):
#       M_c2v[c, v] = 2 * atanh( prod_{v' != v} tanh( M_v2c[v', c] / 2 ) )
#     This enforces parity in the LLR domain. Numerically delicate when
#     messages saturate; we clip arguments to tanh away from ±1.
#   Final decision:
#       L_total[v] = L_ch[v] + sum_c M_c2v[c, v]
#       bit[v] = 0 if L_total[v] >= 0 else 1
# Precompute the edge mask once.
EDGE_MASK = (H == 1)  # boolean array, shape (M, N)

def sum_product_decode(llrs, max_iters=20):
    """
    Sum-product belief propagation decoder.

    Input:
        llrs: shape (N,) — channel LLRs
        max_iters: number of message-passing iterations
    Output:
        bits: shape (N,) dtype int8 — hard decisions

    NumPy idiom: we use np.tanh and np.arctanh elementwise. The product
    over neighbors is computed by taking the product over the entire row,
    then dividing out the contribution of the excluded neighbor. To make
    the divide-out safe we clip incoming messages so their tanh isn't
    exactly ±1 (which would cause division by zero or saturation).
    """
    L_ch = llrs                             # shape (N,)
    M_c2v = np.zeros_like(EDGE_MASK, dtype=np.float64)  # shape (M, N)
    M_v2c = np.zeros_like(EDGE_MASK, dtype=np.float64)  # shape (M, N)
    # Tanh-clip threshold: tanh(±15) is already 0.99999...  Clipping the
    # *argument* of tanh to ±15 keeps tanh well away from ±1, which keeps
    # arctanh from blowing up.
    TANH_CLIP = 15.0

    for _ in range(max_iters):
        # ---- Variable-node update ----
        # M_v2c[v, c] = L_ch[v] + sum_{c' != c} M_c2v[c', v]
        # Strategy: compute total incoming LLR per variable, then subtract
        # the contribution from each individual check.
        # NumPy idiom: M_c2v.sum(axis=0) gives per-variable total, shape (N,).
        # Adding L_ch broadcasts cleanly. Then for each (c, v) we subtract
        # M_c2v[c, v]: this is just elementwise.
        total_incoming_v = L_ch + M_c2v.sum(axis=0)        # shape (N,)
        # Broadcast to (M, N), then subtract M_c2v elementwise
        M_v2c = total_incoming_v[np.newaxis, :] - M_c2v    # shape (M, N)
        # Zero out non-edges (no message on edges that don't exist)
        M_v2c = M_v2c * EDGE_MASK

        # ---- Check-node update (tanh rule, extrinsic) ----
        # M_c2v[c, v] = 2 * atanh( prod_{v' != v} tanh(M_v2c[v', c] / 2) )
        # Strategy: for each row (check c), compute tanh of all incoming
        # half-messages, then for each variable v in that row compute the
        # product of every OTHER variable's tanh. We do this via the
        # "total product / this entry" trick, with a small fudge to avoid
        # dividing by zero when an incoming message is 0.
        clipped = np.clip(M_v2c / 2.0, -TANH_CLIP, TANH_CLIP)
        tanh_msgs = np.tanh(clipped)  # shape (M, N)

        # On non-edges we want the product to be unaffected; set those
        # entries to 1 so they're identity under multiplication.
        tanh_msgs_for_product = np.where(EDGE_MASK, tanh_msgs, 1.0)

        # Per-row total product, shape (M,)
        total_product = np.prod(tanh_msgs_for_product, axis=1, keepdims=True)  # (M, 1)

        # Extrinsic: divide out each entry. Guard against division by zero
        # by replacing zero entries with a tiny epsilon.
        safe_tanh = np.where(np.abs(tanh_msgs) < 1e-12, 1e-12, tanh_msgs)
        extrinsic_product = total_product / safe_tanh  # shape (M, N)

        # Clip the product back into (-1, 1) so arctanh is finite
        extrinsic_product = np.clip(extrinsic_product, -1 + 1e-12, 1 - 1e-12)
        M_c2v = 2.0 * np.arctanh(extrinsic_product)
        M_c2v = M_c2v * EDGE_MASK  # zero non-edges

        # ---- Hard decision ----
        L_total = L_ch + M_c2v.sum(axis=0)
        bits = (L_total < 0).astype(np.int8)

        # Early termination: if syndrome is satisfied, stop iterating
        if np.all((H @ bits) % 2 == 0):
            return bits

    # Final hard decision after max_iters
    L_total = L_ch + M_c2v.sum(axis=0)
    return (L_total < 0).astype(np.int8)

# ========================================================================
# SECTION 7: MIN-SUM BELIEF PROPAGATION
# ========================================================================
# Same algorithm as sum-product, but the check-node update uses the
# Min-Sum approximation:
#       M_c2v[c, v] ≈ ( prod_{v' != v} sign(M_v2c[v', c]) )
#                     × min_{v' != v} |M_v2c[v', c]|
# This avoids tanh/arctanh entirely. It's the standard hardware
# approximation and is the operating point Neural BP will improve on.

def min_sum_decode(llrs, max_iters=20):
    """
    Min-Sum belief propagation decoder.

    Same interface as sum_product_decode. The only structural difference
    is the check-node update, which replaces the tanh rule with a sign
    product and minimum magnitude.
    """
    L_ch = llrs
    M_c2v = np.zeros_like(EDGE_MASK, dtype=np.float64)
    M_v2c = np.zeros_like(EDGE_MASK, dtype=np.float64)
    for _ in range(max_iters):
        # ---- Variable-node update (identical to sum-product) ----
        total_incoming_v = L_ch + M_c2v.sum(axis=0)
        M_v2c = total_incoming_v[np.newaxis, :] - M_c2v
        M_v2c = M_v2c * EDGE_MASK
        # ---- Check-node update (Min-Sum, extrinsic) ----
        # For each (c, v), we want the product of signs and minimum magnitude
        # of M_v2c[c, v'] for v' != v in row c.
        # Strategy:
        #   - Compute signs and magnitudes of M_v2c (shape (M, N))
        #   - For each row, get the total sign product and sort magnitudes
        #     to find the min and second-min (so we can exclude any single
        #     entry without recomputing).
        # NumPy idiom: np.where(mask, M_v2c, +inf) sets non-edge entries
        # to +inf so they never win the min. For sign product, set non-edge
        # entries to +1 (identity under multiplication).
        signs = np.where(EDGE_MASK, np.sign(M_v2c), 1.0)        # (M, N)
        # Treat zero as +1 in sign so the product doesn't collapse
        signs = np.where(signs == 0, 1.0, signs)
        # Total sign product per row, then divide out each entry's sign
        # (extrinsic). For ±1 values, division is the same as multiplication.
        total_sign = np.prod(signs, axis=1, keepdims=True)       # (M, 1)
        extrinsic_sign = total_sign * signs  # divide-by-±1 == multiply-by-±1
        mags = np.where(EDGE_MASK, np.abs(M_v2c), np.inf)        # (M, N)
        # For each row, find the smallest and second-smallest magnitudes.
        # Then for each entry: if it equals the smallest, the extrinsic min
        # is the second-smallest; otherwise it's the smallest.
        sorted_mags = np.sort(mags, axis=1)                      # (M, N), ascending
        min1 = sorted_mags[:, 0:1]                                # (M, 1) smallest
        min2 = sorted_mags[:, 1:2]                                # (M, 1) second smallest
        extrinsic_min = np.where(mags == min1, min2, min1)       # (M, N)
        M_c2v = extrinsic_sign * extrinsic_min
        M_c2v = M_c2v * EDGE_MASK

        # ---- Hard decision and early termination ----
        L_total = L_ch + M_c2v.sum(axis=0)
        bits = (L_total < 0).astype(np.int8)
        if np.all((H @ bits) % 2 == 0):
            return bits
    L_total = L_ch + M_c2v.sum(axis=0)
    return (L_total < 0).astype(np.int8)

# ========================================================================
# SECTION 8: SELF-TESTS — runs when this file is executed as a script
# ========================================================================
# These tests validate every component above before any "real" experiments
# happen. If any of these fail, downstream BLER curves cannot be trusted.

def _test_h_matrix():
    """H should be 4 × 15, full rank, and orthogonal to all of G."""
    assert H.shape == (M, N), f"H shape wrong: {H.shape}"
    # H @ G^T should be the all-zeros matrix (mod 2)
    product = (H @ G_MATRIX.T) % 2
    assert np.all(product == 0), "H is not orthogonal to G; encoder/H mismatch"
    # H should have full row rank over GF(2). We check by reducing row-by-row.
    # Quick approximation: every row of H should be nonzero (necessary, not sufficient).
    assert np.all(H.sum(axis=1) > 0), "H has a zero row"
    return True

def _test_encoder_round_trip():
    """Every encoded message should be a valid codeword (H · c = 0)."""
    fails = 0
    for trial in range(200):
        msg = RNG.integers(0, 2, size=K).astype(np.int8)
        c = encode(msg)
        if not is_codeword(c):
            fails += 1
        # Systematic check: the high-K positions should equal the message
        if not np.array_equal(c[M:], msg):
            fails += 1
    return fails == 0

def _test_codebook_size():
    """Codebook should contain exactly 2^11 = 2048 distinct codewords."""
    unique_rows = set(tuple(row) for row in CODEBOOK)
    return len(unique_rows) == 2**K

def _test_syndrome_decoder_no_noise():
    """At infinite SNR (no noise), syndrome decoder should never err."""
    fails = 0
    for trial in range(200):
        msg = RNG.integers(0, 2, size=K).astype(np.int8)
        c = encode(msg)
        # No noise: just convert bits to "infinitely confident" LLRs
        llrs = np.where(c == 0, 100.0, -100.0)
        decoded = syndrome_decode(llrs)
        if not np.array_equal(decoded, c):
            fails += 1
    return fails == 0

def _test_syndrome_decoder_single_error():
    """Syndrome decoder should correct any single-bit error."""
    fails = 0
    msg = RNG.integers(0, 2, size=K).astype(np.int8)
    c = encode(msg)
    for err_pos in range(N):
        corrupted = c.copy()
        corrupted[err_pos] ^= 1
        # High-confidence LLRs matching the corrupted bits
        llrs = np.where(corrupted == 0, 100.0, -100.0)
        decoded = syndrome_decode(llrs)
        if not np.array_equal(decoded, c):
            fails += 1
    return fails == 0

def _test_ml_decoder_no_noise():
    """ML decoder should be perfect at zero noise."""
    fails = 0
    for trial in range(50):
        msg = RNG.integers(0, 2, size=K).astype(np.int8)
        c = encode(msg)
        y = bpsk_modulate(c)  # noiseless
        decoded = ml_decode(y)
        if not np.array_equal(decoded, c):
            fails += 1
    return fails == 0

def _test_bp_decoders_no_noise():
    """Both BP decoders should be perfect at zero noise."""
    fails_sp = 0
    fails_ms = 0
    for trial in range(50):
        msg = RNG.integers(0, 2, size=K).astype(np.int8)
        c = encode(msg)
        y = bpsk_modulate(c)
        # Use a small but nonzero sigma to define LLRs (avoid division by zero)
        sigma = 0.01
        llrs = llr_from_received(y, sigma)
        if not np.array_equal(sum_product_decode(llrs), c):
            fails_sp += 1
        if not np.array_equal(min_sum_decode(llrs), c):
            fails_ms += 1
    return fails_sp == 0 and fails_ms == 0

def _bler_quick_sweep(ebn0_db, n_trials=2000):
    """
    Quick BLER measurement at one SNR point for all four decoders
    (syndrome, ML, sum-product, min-sum). Used to confirm the
    expected ordering: ML <= SP <= MS, and syndrome is the worst.
    """
    sigma = ebn0_db_to_sigma(ebn0_db)
    errors = {"syndrome": 0, "ML": 0, "sum_product": 0, "min_sum": 0}
    for _ in range(n_trials):
        msg = RNG.integers(0, 2, size=K).astype(np.int8)
        c = encode(msg)
        y = awgn(bpsk_modulate(c), sigma)
        llrs = llr_from_received(y, sigma)
        if not np.array_equal(syndrome_decode(llrs), c):
            errors["syndrome"] += 1
        if not np.array_equal(ml_decode(y), c):
            errors["ML"] += 1
        if not np.array_equal(sum_product_decode(llrs), c):
            errors["sum_product"] += 1
        if not np.array_equal(min_sum_decode(llrs), c):
            errors["min_sum"] += 1
    return {name: count / n_trials for name, count in errors.items()}

def run_self_tests():
    """Run every test, print PASS/FAIL, then a small BLER comparison."""
    tests = [
        ("H matrix structure",         _test_h_matrix),
        ("Encoder round-trip",         _test_encoder_round_trip),
        ("Codebook size = 2^11",       _test_codebook_size),
        ("Syndrome decoder (no noise)", _test_syndrome_decoder_no_noise),
        ("Syndrome decoder (1 error)",  _test_syndrome_decoder_single_error),
        ("ML decoder (no noise)",       _test_ml_decoder_no_noise),
        ("BP decoders (no noise)",      _test_bp_decoders_no_noise),
    ]
    print("=" * 60)
    print("Week 1 self-tests")
    print("=" * 60)
    all_passed = True
    for name, test_fn in tests:
        try:
            ok = test_fn()
            status = "PASS" if ok else "FAIL"
        except Exception as e:
            ok = False
            status = f"ERROR: {e}"
        print(f"  [{status}] {name}")
        if not ok:
            all_passed = False
    print()
    if not all_passed:
        print("Some tests failed. Stopping before BLER sweep.")
        return
    print("=" * 60)
    print("Quick BLER sweep at Eb/N0 = 4 dB (2000 trials each)")
    print("=" * 60)
    bler = _bler_quick_sweep(ebn0_db=4.0, n_trials=2000)
    print(f"  Syndrome decoder:  BLER = {bler['syndrome']:.4f}")
    print(f"  ML decoder:        BLER = {bler['ML']:.4f}  <- lower bound")
    print(f"  Sum-product BP:    BLER = {bler['sum_product']:.4f}")
    print(f"  Min-Sum BP:        BLER = {bler['min_sum']:.4f}")
    print()
    print("Expected ordering: ML <= sum-product <= min-sum, and")
    print("syndrome should be the worst (it's a hard-decision decoder).")

if __name__ == "__main__":
    run_self_tests()