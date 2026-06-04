"""
CR-RSMA vs CR-NOMA  —  MMF (Min-Rate) Simulation
===================================================
System  : 1 PU + K SUs, Underlay Cognitive Radio, SISO MAC
SVC     : 4 layers (BL + EL1 + EL2 + EL3), QCIF 30fps GOP-8
Rate    : Shannon capacity  (no FBL)
MMF obj : max  min_k  R_k   (bps/Hz), then map → PSNR via SVC layers
Compare : CR-RSMA  vs  CR-NOMA
Plots   : (1) MMF vs SNR   (2) MMF vs I_th   (3) MMF vs K
"""

import numpy as np
from scipy.optimize import linprog
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

from config import P          # global config object
np.random.seed(42)

# ════════════════════════════════════════════════════════════════════════════════
#  DERIVED CONSTANTS  (computed once from config)
# ════════════════════════════════════════════════════════════════════════════════
# Convert layer bitrates from kbps to bps/Hz  (normalised by bandwidth B)
LAYER_RATE_THR = np.array(P.layer_bitrate_kbps) * 1e3 / P.B   # bps/Hz per layer
CUM_RATE_THR   = np.cumsum(LAYER_RATE_THR)                     # cumulative thresholds
LAYER_PSNR     = np.array(P.layer_psnr_dB)                     # PSNR at each cum level
N_LAYERS       = P.n_layers                                     # 4

# Fixed SNR values for Ith and K sweeps
SNR_DB_FIXED   = 20          # dB — fixed operating point

# Ith sweep range (used in scenario 2)
ITH_RANGE      = np.linspace(0.05, 1.0, 15)

# K sweep range (used in scenario 3)
K_RANGE        = [2, 3, 4]



# Monte Carlo realizations per point
N_REAL         = 200

# SCA solver parameters
P_MIN          = 1e-9
SCA_MAX_ITER   = 35
SCA_TOL        = 1e-5
SCA_TRUST_INIT = 0.35
SCA_TRUST_MIN  = 0.03
SCA_TRUST_MAX  = 1.00
SCA_DAMPING    = 0.70

# I_th scaling factor for SNR sweep
# I_th scales with Pt so the CR constraint does not dominate at high SNR.
# Ratio I_th / Pt = I_TH_RATIO (kept constant across SNR)
I_TH_RATIO     = P.I_th / (10 ** (10 / 10))   # calibrated at SNR=10 dB

# ════════════════════════════════════════════════════════════════════════════════
#  SVC QUALITY MAPPING   (rate bps/Hz  →  PSNR dB)
# ════════════════════════════════════════════════════════════════════════════════
def rate_to_psnr(R):
    """
    Staircase mapping: decode as many SVC layers as the rate supports.
    Returns PSNR (dB) of the highest decodable layer set.
    R is normalised rate in bps/Hz.
    """
    psnr = 0.0
    for l in range(N_LAYERS):
        if R >= CUM_RATE_THR[l]:
            psnr = LAYER_PSNR[l]
        else:
            break
    return psnr

def mmf_rate_to_psnr(mmf_rate):
    """Convenience: scalar min-rate → PSNR."""
    return rate_to_psnr(mmf_rate)

# ════════════════════════════════════════════════════════════════════════════════
#  CHANNEL MODEL
# ════════════════════════════════════════════════════════════════════════════════
def generate_channels(K, seed=None):
    """
    Rayleigh fading channels.
    Returns:
        h2_su  : (K,)  |h_{SU_k → BS}|²
        h2_pu  : scalar |h_{PU → BS}|²
        g2_su  : (K,)  |g_{SU_k → PU_rx}|²   (interference channel)
    """
    rng = np.random.default_rng(seed)
    h_su = (rng.standard_normal(K) + 1j * rng.standard_normal(K)) / np.sqrt(2)
    h_pu = (rng.standard_normal()  + 1j * rng.standard_normal())  / np.sqrt(2)
    g_su = (rng.standard_normal(K) + 1j * rng.standard_normal(K)) / np.sqrt(2)
    return np.abs(h_su)**2, np.abs(h_pu)**2, np.abs(g_su)**2

# ════════════════════════════════════════════════════════════════════════════════
#  RATE FUNCTION  (Shannon — no FBL)
# ════════════════════════════════════════════════════════════════════════════════
def shannon_rate(sinr):
    """log2(1 + SINR) in bps/Hz."""
    return np.log2(1.0 + np.maximum(sinr, 1e-12))

# ════════════════════════════════════════════════════════════════════════════════
#  SINR  (SIC decoder)
# ════════════════════════════════════════════════════════════════════════════════
def compute_sinr_sic(powers, h2, order, pu_power, h2_pu):
    """
    Compute post-SIC SINR for each stream.
    powers : (M,) transmit powers
    h2     : (M,) channel gains
    order  : decoding order (indices decoded first → last)
    pu_power, h2_pu : PU interference term
    """
    M    = len(powers)
    sinr = np.zeros(M)
    decoded = set()
    for idx in order:
        interf = sum(powers[j] * h2[j] for j in range(M)
                     if j not in decoded and j != idx)
        interf += pu_power * h2_pu + P.sigma2
        sinr[idx] = powers[idx] * h2[idx] / interf
        decoded.add(idx)
    return sinr

# ════════════════════════════════════════════════════════════════════════════════
#  SCA SOLVER  (successive convex approximation for MMF power allocation)
# ════════════════════════════════════════════════════════════════════════════════
def _finite_diff_jacobian(rate_fn, p):
    """Numerical Jacobian dR/dp for the current SCA linearisation point."""
    p = np.asarray(p, dtype=float)
    r0 = np.asarray(rate_fn(p), dtype=float)
    jac = np.zeros((len(r0), len(p)))
    for i in range(len(p)):
        step = max(1e-6, 1e-5 * max(abs(p[i]), 1.0))
        p_hi = p.copy()
        p_lo = p.copy()
        p_hi[i] += step
        p_lo[i] = max(P_MIN, p_lo[i] - step)
        jac[:, i] = (np.asarray(rate_fn(p_hi)) - np.asarray(rate_fn(p_lo))) / (p_hi[i] - p_lo[i])
    return r0, jac

def _scale_to_feasible(p, A_power, b_power):
    """Scale a positive initial vector until all linear power constraints hold."""
    p = np.maximum(np.asarray(p, dtype=float), P_MIN)
    scale = 1.0
    for row, limit in zip(A_power, b_power):
        used = float(np.dot(row, p))
        if used > limit and used > 0:
            scale = min(scale, 0.95 * limit / used)
    return np.maximum(p * scale, P_MIN)

def _sca_mmf_optimize(rate_fn, n_var, A_power, b_power, Pt, init_seed_offset=0):
    """
    Solve max min_k R_k(p) by SCA.
    Each iteration replaces R_k(p) by its first-order approximation around
    the current point and solves the resulting linear MMF subproblem.
    """
    A_power = np.asarray(A_power, dtype=float)
    b_power = np.asarray(b_power, dtype=float)

    best_rate = -np.inf
    best_p = None

    for trial in range(3):
        rng = np.random.default_rng(init_seed_offset + trial)
        p = rng.uniform(max(P_MIN, Pt * 0.05), max(P_MIN * 10, Pt * 0.8), n_var)
        p = _scale_to_feasible(p, A_power, b_power)
        trust = SCA_TRUST_INIT
        prev_rate = float(np.min(rate_fn(p)))

        for _ in range(SCA_MAX_ITER):
            rates, jac = _finite_diff_jacobian(rate_fn, p)

            # Variables are [p_0, ..., p_{n-1}, t], maximize t -> minimize -t.
            c = np.zeros(n_var + 1)
            c[-1] = -1.0

            A_ub = []
            b_ub = []

            # Linearized MMF constraints: t <= R_k(p0) + grad_k @ (p - p0)
            for k in range(len(rates)):
                row = np.zeros(n_var + 1)
                row[:n_var] = -jac[k]
                row[-1] = 1.0
                A_ub.append(row)
                b_ub.append(float(rates[k] - np.dot(jac[k], p)))

            for row_power, limit in zip(A_power, b_power):
                row = np.zeros(n_var + 1)
                row[:n_var] = row_power
                A_ub.append(row)
                b_ub.append(float(limit))

            lower = np.maximum(P_MIN, p - trust * Pt)
            upper = np.minimum(Pt, p + trust * Pt)
            bounds = [(float(lo), float(hi)) for lo, hi in zip(lower, upper)]
            bounds.append((0.0, None))

            res = linprog(c, A_ub=np.asarray(A_ub), b_ub=np.asarray(b_ub),
                          bounds=bounds, method='highs')
            if not res.success:
                trust *= 0.5
                if trust < SCA_TRUST_MIN:
                    break
                continue

            p_candidate = _scale_to_feasible(res.x[:n_var], A_power, b_power)
            p_next = _scale_to_feasible(
                SCA_DAMPING * p_candidate + (1.0 - SCA_DAMPING) * p,
                A_power, b_power
            )
            curr_rate = float(np.min(rate_fn(p_next)))

            if curr_rate + SCA_TOL < prev_rate:
                trust *= 0.5
                if trust < SCA_TRUST_MIN:
                    break
                continue

            if abs(curr_rate - prev_rate) <= SCA_TOL * max(1.0, abs(prev_rate)):
                p = p_next
                prev_rate = curr_rate
                break

            p = p_next
            prev_rate = curr_rate
            trust = min(SCA_TRUST_MAX, trust * 1.15)

        final_rate = float(np.min(rate_fn(p)))
        if final_rate > best_rate:
            best_rate = final_rate
            best_p = p.copy()

    if best_p is None:
        return np.zeros(n_var), 0.0
    return best_p, max(best_rate, 0.0)

# ════════════════════════════════════════════════════════════════════════════════
#  CR-RSMA   (1 common + 1 private per SU — MMF optimisation)
# ════════════════════════════════════════════════════════════════════════════════
def rsma_mmf(h2_su, h2_pu, g2_su, Pt, I_th):
    """
    RSMA with one splitting user (best-channel SU).
    Streams: [c1, p_1, ..., p_{K-1}, c2]  where c1, c2 are two parts of the
    common stream for the split user (SU with max gain).

    Variables p = [p_c1, p_1, ..., p_{K-1}, p_c2]  (K+1)
    Objective: max  min_k  R_k
    """
    K = len(h2_su)
    split = int(np.argmax(h2_su))
    others = [k for k in range(K) if k != split]

    # stream channel gains: [split, other_0, ..., other_{K-2}, split]
    h2_s = np.array([h2_su[split]]
                    + [h2_su[k] for k in others]
                    + [h2_su[split]])    # length K+1
    g2_s = np.array([g2_su[split]]
                    + [g2_su[k] for k in others]
                    + [g2_su[split]])

    n_s    = K + 1
    order  = list(range(n_s))           # decode in index order (c1 first → c2 last)
    pu_pow = P.Pp_max

    def user_rates(p):
        sinr      = compute_sinr_sic(p, h2_s, order, pu_pow, h2_pu)
        rs        = shannon_rate(sinr)
        R         = np.zeros(K)
        R[split]  = rs[0] + rs[n_s - 1]   # both parts of common stream
        for i, k in enumerate(others):
            R[k] = rs[i + 1]
        return R

    A_power = []
    b_power = []

    row = np.zeros(n_s)
    row[0] = 1.0
    row[n_s - 1] = 1.0
    A_power.append(row)
    b_power.append(Pt)

    for i in range(1, K):
        row = np.zeros(n_s)
        row[i] = 1.0
        A_power.append(row)
        b_power.append(Pt)

    A_power.append(g2_s)
    b_power.append(I_th)

    _, mmf_rate = _sca_mmf_optimize(user_rates, n_s, A_power, b_power, Pt,
                                    init_seed_offset=0)
    return max(mmf_rate, 0.0)

# ════════════════════════════════════════════════════════════════════════════════
#  CR-NOMA  (MMF optimisation)
# ════════════════════════════════════════════════════════════════════════════════
def noma_mmf(h2_su, h2_pu, g2_su, Pt, I_th):
    """
    CR-NOMA: no stream splitting.
    Decoding order: descending channel gain (strongest decoded last → weakest first).
    """
    K      = len(h2_su)
    order  = list(np.argsort(h2_su))   # decode weakest first (standard NOMA SIC)
    pu_pow = P.Pp_max

    def user_rates(p):
        sinr = compute_sinr_sic(p, h2_su, order, pu_pow, h2_pu)
        return shannon_rate(sinr)

    A_power = []
    b_power = []
    # Total power constraint: sum(p) <= Pt
    A_power.append(np.ones(K))
    b_power.append(Pt)
    # Individual per-user limit (optional, but typical in NOMA)
    for k in range(K):
        row = np.zeros(K)
        row[k] = 1.0
        A_power.append(row)
        b_power.append(Pt)
    # CR interference constraint
    A_power.append(g2_su)
    b_power.append(I_th)

    _, mmf_rate = _sca_mmf_optimize(user_rates, K, A_power, b_power, Pt,
                                    init_seed_offset=100)
    return max(mmf_rate, 0.0)

# ════════════════════════════════════════════════════════════════════════════════
#  CR-OMA  (Orthogonal Multiple Access baseline)
# ════════════════════════════════════════════════════════════════════════════════
def oma_mmf(h2_su, h2_pu, g2_su, Pt, I_th):
    """
    OMA: orthogonal resource allocation with CR interference constraint.
    Each user gets orthogonal time/frequency slot.
    """
    K = len(h2_su)
    p = np.full(K, Pt / K)
    
    # Enforce CR interference constraint I_th
    interf = np.dot(p, g2_su)
    if interf > I_th:
        p *= (I_th / interf) * 0.99  # Scale down to satisfy constraint
    
    sinr = p * h2_su / (P.Pp_max * h2_pu + P.sigma2)
    rates = shannon_rate(sinr)
    return max(float(np.min(rates)), 0.0)

# ════════════════════════════════════════════════════════════════════════════════
#  MONTE CARLO HELPERS
# ════════════════════════════════════════════════════════════════════════════════
def _mc_mmf(K, Pt, I_th, n_real):
    """Returns (rsma_mmf_mean, noma_mmf_mean, oma_mmf_mean) averaged over n_real realisations."""
    rsma_vals, noma_vals, oma_vals = [], [], []
    for r in range(n_real):
        h2_su, h2_pu, g2_su = generate_channels(K, seed=r * 1000 + int(Pt * 100))
        try:
            rsma_vals.append(rsma_mmf(h2_su, h2_pu, g2_su, Pt, I_th))
        except Exception:
            rsma_vals.append(np.nan)
        try:
            noma_vals.append(noma_mmf(h2_su, h2_pu, g2_su, Pt, I_th))
        except Exception:
            noma_vals.append(np.nan)
        try:
            oma_vals.append(oma_mmf(h2_su, h2_pu, g2_su, Pt, I_th))
        except Exception:
            oma_vals.append(np.nan)
    return np.nanmean(rsma_vals), np.nanmean(noma_vals), np.nanmean(oma_vals)

# ════════════════════════════════════════════════════════════════════════════════
#  SCENARIO 1 — MMF vs SNR
#  Fixed: K=2, I_th from config
#  SNR sweep: P.snr_dB_range (0–30 dB, step 2 dB)
# ════════════════════════════════════════════════════════════════════════════════
def sim_vs_snr(K=2, n_real=N_REAL):
    """
    MMF vs SNR.
    I_th scales proportionally with Pt (fixed ratio I_TH_RATIO) so that
    the CR interference constraint does not become the sole bottleneck
    at high SNR, giving a monotone increasing MMF curve.
    """
    print(f"\n[1/3] Max-Min Fairness vs SNR  (K={K}, I_th scales with Pt) ...")
    rsma_r, noma_r, oma_r = [], [], []
    for snr_dB in P.snr_dB_range:
        Pt    = 10 ** (snr_dB / 10)
        I_th  = I_TH_RATIO * Pt      # scale with transmit power
        r, n, o  = _mc_mmf(K, Pt, I_th, n_real)
        rsma_r.append(r)
        noma_r.append(n)
        oma_r.append(o)
        print(f"   SNR={snr_dB:4.0f} dB | RSMA={r:.4f}  NOMA={n:.4f}  OMA={o:.4f}")
    return np.array(rsma_r), np.array(noma_r), np.array(oma_r)

# ════════════════════════════════════════════════════════════════════════════════
#  SCENARIO 2 — MMF vs I_th  (Cognitive Radio evaluation)
#  Fixed: K=2, SNR=SNR_DB_FIXED
# ════════════════════════════════════════════════════════════════════════════════
def sim_vs_ith(K=2, n_real=N_REAL):
    print(f"\n[2/3] Max-Min Fairness vs I_th  (K={K}, SNR={SNR_DB_FIXED} dB) ...")
    Pt = 10 ** (SNR_DB_FIXED / 10)
    rsma_r, noma_r, oma_r = [], [], []
    for I_th_val in ITH_RANGE:
        r, n, o = _mc_mmf(K, Pt, I_th_val, n_real)
        rsma_r.append(r)
        noma_r.append(n)
        oma_r.append(o)
        print(f"   I_th={I_th_val:.3f} W | RSMA={r:.4f}  NOMA={n:.4f}  OMA={o:.4f}")
    return np.array(rsma_r), np.array(noma_r), np.array(oma_r)

# ════════════════════════════════════════════════════════════════════════════════
#  SCENARIO 3 — MMF vs K  (Number of Users)
#  Fixed: SNR=SNR_DB_FIXED, I_th from config
# ════════════════════════════════════════════════════════════════════════════════
def sim_vs_K(n_real=N_REAL):
    print(f"\n[3/3] Max-Min Fairness vs K  (SNR={SNR_DB_FIXED} dB, I_th={P.I_th:.2f} W) ...")
    Pt = 10 ** (SNR_DB_FIXED / 10)
    rsma_r, noma_r, oma_r = [], [], []
    for K in K_RANGE:
        r, n, o = _mc_mmf(K, Pt, P.I_th, n_real)
        rsma_r.append(r)
        noma_r.append(n)
        oma_r.append(o)
        print(f"   K={K} | RSMA={r:.4f}  NOMA={n:.4f}  OMA={o:.4f}")
    return np.array(rsma_r), np.array(noma_r), np.array(oma_r)

# ════════════════════════════════════════════════════════════════════════════════
#  SCENARIO 4 helpers — SVC PSNR Mapping  (reuses Sc1 rate results)
# ════════════════════════════════════════════════════════════════════════════════
def map_to_psnr(rate_arr):
    """Apply SVC staircase mapping element-wise to a rate array."""
    return np.array([rate_to_psnr(float(r)) for r in rate_arr])

# ════════════════════════════════════════════════════════════════════════════════
#  SCENARIO 8 — Jain's Fairness Index
# ════════════════════════════════════════════════════════════════════════════════
def jains_index(rates):
    """JFI = (Σ R_k)² / (K · Σ R_k²)  ∈ [1/K, 1]; 1 = perfect fairness."""
    rates = np.maximum(np.asarray(rates, dtype=float), 1e-15)
    return float(np.sum(rates) ** 2 / (len(rates) * np.sum(rates ** 2)))

def _opt_rates_rsma(h2_su, h2_pu, g2_su, Pt, I_th):
    """RSMA MMF optimisation → per-user rate vector (K,)."""
    K      = len(h2_su)
    split  = int(np.argmax(h2_su))
    others = [k for k in range(K) if k != split]
    h2_s   = np.array([h2_su[split]] + [h2_su[k] for k in others] + [h2_su[split]])
    g2_s   = np.array([g2_su[split]] + [g2_su[k] for k in others] + [g2_su[split]])
    n_s    = K + 1
    order  = list(range(n_s))

    def _rates(p):
        sinr     = compute_sinr_sic(p, h2_s, order, P.Pp_max, h2_pu)
        rs       = shannon_rate(sinr)
        R        = np.zeros(K)
        R[split] = rs[0] + rs[n_s - 1]
        for i, k in enumerate(others):
            R[k] = rs[i + 1]
        return R

    A_power = []
    b_power = []

    row = np.zeros(n_s)
    row[0] = 1.0
    row[n_s - 1] = 1.0
    A_power.append(row)
    b_power.append(Pt)

    for i in range(1, K):
        row = np.zeros(n_s)
        row[i] = 1.0
        A_power.append(row)
        b_power.append(Pt)

    A_power.append(g2_s)
    b_power.append(I_th)

    best_p, _ = _sca_mmf_optimize(_rates, n_s, A_power, b_power, Pt,
                                  init_seed_offset=0)
    return _rates(best_p) if best_p is not None else np.zeros(K)

def _opt_rates_noma(h2_su, h2_pu, g2_su, Pt, I_th):
    """NOMA MMF optimisation → per-user rate vector (K,)."""
    K      = len(h2_su)
    order  = list(np.argsort(h2_su))

    def _rates(p):
        return shannon_rate(compute_sinr_sic(p, h2_su, order, P.Pp_max, h2_pu))

    A_power = []
    b_power = []
    for k in range(K):
        row = np.zeros(K)
        row[k] = 1.0
        A_power.append(row)
        b_power.append(Pt)
    A_power.append(g2_su)
    b_power.append(I_th)

    best_p, _ = _sca_mmf_optimize(_rates, K, A_power, b_power, Pt,
                                  init_seed_offset=100)
    return _rates(best_p) if best_p is not None else np.zeros(K)

def _mc_jfi(K, Pt, I_th, n_real):
    """Monte-Carlo Jain's FI for RSMA and NOMA."""
    rsma_j, noma_j = [], []
    for r in range(n_real):
        h2_su, h2_pu, g2_su = generate_channels(K, seed=r * 1000 + int(Pt * 100))
        try:
            rsma_j.append(jains_index(_opt_rates_rsma(h2_su, h2_pu, g2_su, Pt, I_th)))
        except Exception:
            rsma_j.append(np.nan)
        try:
            noma_j.append(jains_index(_opt_rates_noma(h2_su, h2_pu, g2_su, Pt, I_th)))
        except Exception:
            noma_j.append(np.nan)
    return np.nanmean(rsma_j), np.nanmean(noma_j)

def sim_jfi_vs_snr(K=2, n_real=N_REAL):
    print(f"\n[Sc8-a] Jain's FI vs SNR  (K={K}) ...")
    rsma_j, noma_j = [], []
    for snr_dB in P.snr_dB_range:
        Pt   = 10 ** (snr_dB / 10)
        I_th = I_TH_RATIO * Pt
        r, n = _mc_jfi(K, Pt, I_th, n_real)
        rsma_j.append(r); noma_j.append(n)
        print(f"   SNR={snr_dB:4.0f} dB | JFI_RSMA={r:.4f}  JFI_NOMA={n:.4f}")
    return np.array(rsma_j), np.array(noma_j)

def sim_jfi_vs_K(n_real=N_REAL):
    print(f"\n[Sc8-b] Jain's FI vs K  (SNR={SNR_DB_FIXED} dB) ...")
    Pt = 10 ** (SNR_DB_FIXED / 10)
    rsma_j, noma_j = [], []
    for K in K_RANGE:
        r, n = _mc_jfi(K, Pt, P.I_th, n_real)
        rsma_j.append(r); noma_j.append(n)
        print(f"   K={K} | JFI_RSMA={r:.4f}  JFI_NOMA={n:.4f}")
    return np.array(rsma_j), np.array(noma_j)

# ════════════════════════════════════════════════════════════════════════════════
#  SCENARIO 11 — Equal Power Allocation Baseline
# ════════════════════════════════════════════════════════════════════════════════
def _eq_power_rsma(h2_su, h2_pu, g2_su, Pt, I_th):
    """RSMA MMF with uniform power across K+1 streams (no optimisation)."""
    K      = len(h2_su)
    split  = int(np.argmax(h2_su))
    others = [k for k in range(K) if k != split]
    h2_s   = np.array([h2_su[split]] + [h2_su[k] for k in others] + [h2_su[split]])
    g2_s   = np.array([g2_su[split]] + [g2_su[k] for k in others] + [g2_su[split]])
    n_s    = K + 1
    p      = np.full(n_s, Pt / n_s)
    interf = np.dot(p, g2_s)
    if interf > I_th:
        p *= (I_th / interf) * 0.99
    order    = list(range(n_s))
    sinr     = compute_sinr_sic(p, h2_s, order, P.Pp_max, h2_pu)
    rs       = shannon_rate(sinr)
    R        = np.zeros(K)
    R[split] = rs[0] + rs[n_s - 1]
    for i, k in enumerate(others):
        R[k] = rs[i + 1]
    return max(float(np.min(R)), 0.0)

def _eq_power_noma(h2_su, h2_pu, g2_su, Pt, I_th):
    """NOMA MMF with uniform power per user (no optimisation)."""
    K      = len(h2_su)
    order  = list(np.argsort(h2_su))
    p      = np.full(K, Pt / K)
    interf = np.dot(p, g2_su)
    if interf > I_th:
        p *= (I_th / interf) * 0.99
    sinr  = compute_sinr_sic(p, h2_su, order, P.Pp_max, h2_pu)
    return max(float(np.min(shannon_rate(sinr))), 0.0)

def _mc_equal_power(K, Pt, I_th, n_real):
    """Monte-Carlo for equal-power baseline."""
    rsma_eq, noma_eq = [], []
    for r in range(n_real):
        h2_su, h2_pu, g2_su = generate_channels(K, seed=r * 1000 + int(Pt * 100))
        try:
            rsma_eq.append(_eq_power_rsma(h2_su, h2_pu, g2_su, Pt, I_th))
        except Exception:
            rsma_eq.append(np.nan)
        try:
            noma_eq.append(_eq_power_noma(h2_su, h2_pu, g2_su, Pt, I_th))
        except Exception:
            noma_eq.append(np.nan)
    return np.nanmean(rsma_eq), np.nanmean(noma_eq)

def sim_equal_vs_snr(K=2, n_real=N_REAL):
    print(f"\n[Sc11] Equal-Power Baseline vs SNR  (K={K}) ...")
    rsma_eq, noma_eq = [], []
    for snr_dB in P.snr_dB_range:
        Pt   = 10 ** (snr_dB / 10)
        I_th = I_TH_RATIO * Pt
        r, n = _mc_equal_power(K, Pt, I_th, n_real)
        rsma_eq.append(r); noma_eq.append(n)
        print(f"   SNR={snr_dB:4.0f} dB | EQ_RSMA={r:.4f}  EQ_NOMA={n:.4f}")
    return np.array(rsma_eq), np.array(noma_eq)

# ════════════════════════════════════════════════════════════════════════════════
#  PLOTTING HELPERS
# ════════════════════════════════════════════════════════════════════════════════
STYLE = {
    'rsma': dict(color='#1a6fbd', marker='o', linestyle='-',  linewidth=2.5, markersize=7),
    'noma': dict(color='#e05c00', marker='s', linestyle='--', linewidth=2.5, markersize=7),
    'oma':  dict(color='#6b9f4a', marker='^', linestyle=':', linewidth=2.0, markersize=6),
}

def _style_ax(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=8)
    ax.grid(True, alpha=0.35, linestyle='--')
    ax.legend(fontsize=11, framealpha=0.92, loc='best')
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=11)

# ════════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  CR-RSMA vs CR-NOMA vs OMA Comparison")
    print("=" * 70)

    # ──────────────────────────────────────────────────────────────────────────
    # Scenarios 1–3  (original)
    # ──────────────────────────────────────────────────────────────────────────
    rsma_snr, noma_snr, oma_snr = sim_vs_snr(K=2, n_real=N_REAL)
    rsma_ith, noma_ith, oma_ith = sim_vs_ith(K=2, n_real=N_REAL)
    rsma_K,   noma_K,   oma_K   = sim_vs_K(n_real=N_REAL)

    # Save all results
    np.savez('sim_results.npz',
             snr_range=P.snr_dB_range,
             ith_range=ITH_RANGE,
             K_range=np.array(K_RANGE),
             rsma_snr=rsma_snr,     noma_snr=noma_snr,     oma_snr=oma_snr,
             rsma_ith=rsma_ith,     noma_ith=noma_ith,     oma_ith=oma_ith,
             rsma_K=rsma_K,         noma_K=noma_K,         oma_K=oma_K)
    print("\n✓ Results saved to sim_results.npz")

    # ──────────────────────────────────────────────────────────────────────────
    # FIGURE 1 — Max-Min Fairness vs SNR
    # ──────────────────────────────────────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(9, 5.8))
    ax1.plot(P.snr_dB_range, rsma_snr, label='CR-RSMA', **STYLE['rsma'])
    ax1.plot(P.snr_dB_range, noma_snr, label='CR-NOMA', **STYLE['noma'])
    ax1.plot(P.snr_dB_range, oma_snr, label='OMA', **STYLE['oma'])

    _style_ax(ax1,
              xlabel='SNR (dB)',
              ylabel='Max-min fairness (bps/Hz)',
              title='Performance Comparison vs SNR')
    plt.tight_layout()
    plt.savefig('fig1_mmf_vs_snr.png', dpi=P.dpi, bbox_inches='tight')
    print("  ✓ fig1_mmf_vs_snr.png")
    plt.close(fig1)

    # ──────────────────────────────────────────────────────────────────────────
    # FIGURE 2 — Max-Min Fairness vs Interference Threshold
    # ──────────────────────────────────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(9, 5.8))
    ax2.plot(ITH_RANGE, rsma_ith, label='CR-RSMA', **STYLE['rsma'])
    ax2.plot(ITH_RANGE, noma_ith, label='CR-NOMA', **STYLE['noma'])
    ax2.plot(ITH_RANGE, oma_ith, label='OMA', **STYLE['oma'])

    _style_ax(ax2,
              xlabel='Interference Threshold $I_{th}$ (W)',
              ylabel='Max-min fairness (bps/Hz)',
              title='Cognitive Radio Evaluation vs Interference Limits')
    plt.tight_layout()
    plt.savefig('fig2_mmf_vs_ith.png', dpi=P.dpi, bbox_inches='tight')
    print("  ✓ fig2_mmf_vs_ith.png")
    plt.close(fig2)

    # ──────────────────────────────────────────────────────────────────────────
    # FIGURE 3 — Max-Min Fairness vs Number of Users
    # ──────────────────────────────────────────────────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(9, 5.8))
    ax3.plot(K_RANGE, rsma_K, label='CR-RSMA', **STYLE['rsma'])
    ax3.plot(K_RANGE, noma_K, label='CR-NOMA', **STYLE['noma'])
    ax3.plot(K_RANGE, oma_K, label='OMA', **STYLE['oma'])

    _style_ax(ax3,
              xlabel='Number of Secondary Users $K$',
              ylabel='Max-min fairness (bps/Hz)',
              title='Scalability vs Number of Users')
    ax3.set_xticks(K_RANGE)
    plt.tight_layout()
    plt.savefig('fig3_mmf_vs_K.png', dpi=P.dpi, bbox_inches='tight')
    print("  ✓ fig3_mmf_vs_K.png")
    plt.close(fig3)

    # ──────────────────────────────────────────────────────────────────────────
    # Summary
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Simulation complete.")
    print("  Output files:")
    print("    fig1_mmf_vs_snr.png")
    print("    fig2_mmf_vs_ith.png")
    print("    fig3_mmf_vs_K.png")
    print("    sim_results.npz")
    print("=" * 70)
