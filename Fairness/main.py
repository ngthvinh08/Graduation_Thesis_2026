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
from scipy.optimize import minimize
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
K_RANGE        = [2, 3, 4, 5, 6]

# Monte Carlo realizations per point
N_REAL         = 200

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

    def neg_mmf(p):
        return -np.min(user_rates(p))

    # Constraints
    cons = []
    # split user total power ≤ Pt
    cons.append({'type': 'ineq', 'fun': lambda p: Pt - p[0] - p[n_s - 1]})
    # each non-split user ≤ Pt
    for i in range(1, K):
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: Pt - p[i]})
    # interference constraint  Σ p_s * g2_s ≤ I_th
    cons.append({'type': 'ineq', 'fun': lambda p: I_th - np.dot(p, g2_s)})

    bounds = [(1e-5, Pt)] * n_s

    # Multi-start
    best_val, best_p = np.inf, None
    for trial in range(3):
        rng  = np.random.default_rng(trial)
        p0   = rng.uniform(Pt * 0.1, Pt * 0.9, n_s)
        p0[0]      = Pt * 0.5
        p0[n_s-1]  = Pt * 0.5
        res = minimize(neg_mmf, p0, method='SLSQP',
                       bounds=bounds, constraints=cons,
                       options={'maxiter': 200, 'ftol': 1e-6})
        if res.fun < best_val:
            best_val = res.fun
            best_p   = res.x

    mmf_rate = -best_val
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

    def neg_mmf(p):
        return -np.min(user_rates(p))

    cons = []
    for k in range(K):
        cons.append({'type': 'ineq', 'fun': lambda p, k=k: Pt - p[k]})
    cons.append({'type': 'ineq', 'fun': lambda p: I_th - np.dot(p, g2_su)})

    bounds = [(1e-5, Pt)] * K

    best_val, best_p = np.inf, None
    for trial in range(3):
        rng = np.random.default_rng(trial + 100)
        p0  = rng.uniform(Pt * 0.1, Pt * 0.9, K)
        res = minimize(neg_mmf, p0, method='SLSQP',
                       bounds=bounds, constraints=cons,
                       options={'maxiter': 200, 'ftol': 1e-6})
        if res.fun < best_val:
            best_val = res.fun
            best_p   = res.x

    mmf_rate = -best_val
    return max(mmf_rate, 0.0)

# ════════════════════════════════════════════════════════════════════════════════
#  MONTE CARLO HELPERS
# ════════════════════════════════════════════════════════════════════════════════
def _mc_mmf(K, Pt, I_th, n_real):
    """Returns (rsma_mmf_mean, noma_mmf_mean) averaged over n_real realisations."""
    rsma_vals, noma_vals = [], []
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
    return np.nanmean(rsma_vals), np.nanmean(noma_vals)

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
    print(f"\n[1/3] MMF vs SNR  (K={K}, I_th scales with Pt, ratio={I_TH_RATIO:.4f}) ...")
    rsma_r, noma_r = [], []
    for snr_dB in P.snr_dB_range:
        Pt    = 10 ** (snr_dB / 10)
        I_th  = I_TH_RATIO * Pt      # scale with transmit power
        r, n  = _mc_mmf(K, Pt, I_th, n_real)
        rsma_r.append(r)
        noma_r.append(n)
        print(f"   SNR={snr_dB:4.0f} dB | Pt={Pt:.3f} W | I_th={I_th:.4f} W | RSMA={r:.4f}  NOMA={n:.4f}")
    return np.array(rsma_r), np.array(noma_r)

# ════════════════════════════════════════════════════════════════════════════════
#  SCENARIO 2 — MMF vs I_th  (Cognitive Radio evaluation)
#  Fixed: K=2, SNR=SNR_DB_FIXED
# ════════════════════════════════════════════════════════════════════════════════
def sim_vs_ith(K=2, n_real=N_REAL):
    print(f"\n[2/3] MMF vs I_th  (K={K}, SNR={SNR_DB_FIXED} dB) ...")
    Pt = 10 ** (SNR_DB_FIXED / 10)
    rsma_r, noma_r = [], []
    for I_th_val in ITH_RANGE:
        r, n = _mc_mmf(K, Pt, I_th_val, n_real)
        rsma_r.append(r)
        noma_r.append(n)
        print(f"   I_th={I_th_val:.3f} W | RSMA={r:.4f}  NOMA={n:.4f}")
    return np.array(rsma_r), np.array(noma_r)

# ════════════════════════════════════════════════════════════════════════════════
#  SCENARIO 3 — MMF vs K  (Number of Users)
#  Fixed: SNR=SNR_DB_FIXED, I_th from config
# ════════════════════════════════════════════════════════════════════════════════
def sim_vs_K(n_real=N_REAL):
    print(f"\n[3/3] MMF vs K  (SNR={SNR_DB_FIXED} dB, I_th={P.I_th:.2f} W) ...")
    Pt = 10 ** (SNR_DB_FIXED / 10)
    rsma_r, noma_r = [], []
    for K in K_RANGE:
        r, n = _mc_mmf(K, Pt, P.I_th, n_real)
        rsma_r.append(r)
        noma_r.append(n)
        print(f"   K={K} | RSMA={r:.4f}  NOMA={n:.4f}")
    return np.array(rsma_r), np.array(noma_r)

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

    cons = [{'type': 'ineq', 'fun': lambda p: Pt - p[0] - p[n_s - 1]}]
    for i in range(1, K):
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: Pt - p[i]})
    cons.append({'type': 'ineq', 'fun': lambda p: I_th - np.dot(p, g2_s)})
    bounds = [(1e-5, Pt)] * n_s

    best_val, best_p = np.inf, None
    for trial in range(3):
        rng = np.random.default_rng(trial)
        p0  = rng.uniform(Pt * 0.1, Pt * 0.9, n_s)
        p0[0] = p0[n_s - 1] = Pt * 0.5
        res = minimize(lambda p: -np.min(_rates(p)), p0, method='SLSQP',
                       bounds=bounds, constraints=cons,
                       options={'maxiter': 200, 'ftol': 1e-6})
        if res.fun < best_val:
            best_val, best_p = res.fun, res.x
    return _rates(best_p) if best_p is not None else np.zeros(K)

def _opt_rates_noma(h2_su, h2_pu, g2_su, Pt, I_th):
    """NOMA MMF optimisation → per-user rate vector (K,)."""
    K      = len(h2_su)
    order  = list(np.argsort(h2_su))

    def _rates(p):
        return shannon_rate(compute_sinr_sic(p, h2_su, order, P.Pp_max, h2_pu))

    cons   = [{'type': 'ineq', 'fun': lambda p, k=k: Pt - p[k]} for k in range(K)]
    cons.append({'type': 'ineq', 'fun': lambda p: I_th - np.dot(p, g2_su)})
    bounds = [(1e-5, Pt)] * K

    best_val, best_p = np.inf, None
    for trial in range(3):
        rng = np.random.default_rng(trial + 100)
        p0  = rng.uniform(Pt * 0.1, Pt * 0.9, K)
        res = minimize(lambda p: -np.min(_rates(p)), p0, method='SLSQP',
                       bounds=bounds, constraints=cons,
                       options={'maxiter': 200, 'ftol': 1e-6})
        if res.fun < best_val:
            best_val, best_p = res.fun, res.x
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
    print("=" * 70)
    print("  CR-RSMA vs CR-NOMA  |  MMF Simulation  |  SVC 4-Layer")
    print(f"  Bandwidth : {P.B/1e3:.0f} kHz")
    print(f"  η (path)  : {P.eta}")
    print(f"  Pp_max    : {P.Pp_max} W   |  Ps_max : {P.Ps_max} W")
    print(f"  w_p = {P.w_p}  |  w_s = {P.w_s}  (fairness weights)")
    print(f"  SNR range : {P.snr_dB_range[0]}–{P.snr_dB_range[-1]} dB, step {int(P.snr_dB_range[1]-P.snr_dB_range[0])} dB")
    print("=" * 70)

    # ──────────────────────────────────────────────────────────────────────────
    # Scenarios 1–3  (original)
    # ──────────────────────────────────────────────────────────────────────────
    rsma_snr, noma_snr = sim_vs_snr(K=2, n_real=N_REAL)
    rsma_ith, noma_ith = sim_vs_ith(K=2, n_real=N_REAL)
    rsma_K,   noma_K   = sim_vs_K(n_real=N_REAL)

    # ──────────────────────────────────────────────────────────────────────────
    # Scenario 4  — PSNR mapping (reuses Sc1 rate results, no extra Monte Carlo)
    # ──────────────────────────────────────────────────────────────────────────
    rsma_psnr = map_to_psnr(rsma_snr)
    noma_psnr = map_to_psnr(noma_snr)

    # ──────────────────────────────────────────────────────────────────────────
    # Scenario 8  — Jain's Fairness Index
    # ──────────────────────────────────────────────────────────────────────────
    rsma_jfi_snr, noma_jfi_snr = sim_jfi_vs_snr(K=2, n_real=N_REAL)
    rsma_jfi_K,   noma_jfi_K   = sim_jfi_vs_K(n_real=N_REAL)

    # ──────────────────────────────────────────────────────────────────────────
    # Scenario 11 — Equal-Power Baseline
    # ──────────────────────────────────────────────────────────────────────────
    rsma_eq, noma_eq = sim_equal_vs_snr(K=2, n_real=N_REAL)

    # Save all results
    np.savez('sim_results.npz',
             snr_range=P.snr_dB_range,
             ith_range=ITH_RANGE,
             K_range=np.array(K_RANGE),
             rsma_snr=rsma_snr,     noma_snr=noma_snr,
             rsma_ith=rsma_ith,     noma_ith=noma_ith,
             rsma_K=rsma_K,         noma_K=noma_K,
             rsma_psnr=rsma_psnr,   noma_psnr=noma_psnr,
             rsma_jfi_snr=rsma_jfi_snr, noma_jfi_snr=noma_jfi_snr,
             rsma_jfi_K=rsma_jfi_K,     noma_jfi_K=noma_jfi_K,
             rsma_eq=rsma_eq,       noma_eq=noma_eq)
    print("\n✓ Results saved to sim_results.npz")

    # ──────────────────────────────────────────────────────────────────────────
    # FIGURE 1 — MMF vs SNR
    # ──────────────────────────────────────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(8, 5.5))
    ax1.plot(P.snr_dB_range, rsma_snr, label='CR-RSMA', **STYLE['rsma'])
    ax1.plot(P.snr_dB_range, noma_snr, label='CR-NOMA', **STYLE['noma'])

    # Shade the gain region
    ax1.fill_between(P.snr_dB_range, noma_snr, rsma_snr,
                     alpha=0.12, color='#1a6fbd', label='RSMA gain')

    _style_ax(ax1,
              xlabel='SNR (dB)',
              ylabel='MMF  (min user rate, bps/Hz)',
              title='CR-RSMA vs CR-NOMA: MMF vs SNR\n'
                    f'(K=2, I_th={P.I_th} W, w_p=w_s={P.w_p})')
    plt.tight_layout()
    plt.savefig('fig1_mmf_vs_snr.png', dpi=P.dpi, bbox_inches='tight')
    print("  ✓ fig1_mmf_vs_snr.png")
    plt.close(fig1)

    # ──────────────────────────────────────────────────────────────────────────
    # FIGURE 2 — MMF vs I_th  (Cognitive Radio evaluation)
    # ──────────────────────────────────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(8, 5.5))
    ax2.plot(ITH_RANGE, rsma_ith, label='CR-RSMA', **STYLE['rsma'])
    ax2.plot(ITH_RANGE, noma_ith, label='CR-NOMA', **STYLE['noma'])
    ax2.fill_between(ITH_RANGE, noma_ith, rsma_ith,
                     alpha=0.12, color='#1a6fbd', label='RSMA gain')

    _style_ax(ax2,
              xlabel='Interference Threshold  $I_{th}$ (W)',
              ylabel='MMF  (min user rate, bps/Hz)',
              title='Cognitive Radio Evaluation: MMF vs $I_{th}$\n'
                    f'(K=2, SNR={SNR_DB_FIXED} dB, RSMA tận dụng phổ tốt hơn NOMA)')
    plt.tight_layout()
    plt.savefig('fig2_mmf_vs_ith.png', dpi=P.dpi, bbox_inches='tight')
    print("  ✓ fig2_mmf_vs_ith.png")
    plt.close(fig2)

    # ──────────────────────────────────────────────────────────────────────────
    # FIGURE 3 — MMF vs K  (Number of Users)
    # ──────────────────────────────────────────────────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(8, 5.5))
    ax3.plot(K_RANGE, rsma_K, label='CR-RSMA', **STYLE['rsma'])
    ax3.plot(K_RANGE, noma_K, label='CR-NOMA', **STYLE['noma'])
    ax3.fill_between(K_RANGE, noma_K, rsma_K,
                     alpha=0.12, color='#1a6fbd', label='RSMA gain')

    _style_ax(ax3,
              xlabel='Number of Secondary Users  $K$',
              ylabel='MMF  (min user rate, bps/Hz)',
              title='Scalability: MMF vs Number of Users K\n'
                    f'(SNR={SNR_DB_FIXED} dB, I_th={P.I_th} W)')
    ax3.set_xticks(K_RANGE)
    plt.tight_layout()
    plt.savefig('fig3_mmf_vs_K.png', dpi=P.dpi, bbox_inches='tight')
    print("  ✓ fig3_mmf_vs_K.png")
    plt.close(fig3)

    # ──────────────────────────────────────────────────────────────────────────
    # FIGURE 4 — Combined 1×3 panel (thesis-ready)
    # ──────────────────────────────────────────────────────────────────────────
    fig4 = plt.figure(figsize=(17, 5))
    gs   = gridspec.GridSpec(1, 3, figure=fig4, wspace=0.38)

    # Panel (a) — vs SNR
    a1 = fig4.add_subplot(gs[0, 0])
    a1.plot(P.snr_dB_range, rsma_snr, label='CR-RSMA', **STYLE['rsma'])
    a1.plot(P.snr_dB_range, noma_snr, label='CR-NOMA', **STYLE['noma'])
    a1.fill_between(P.snr_dB_range, noma_snr, rsma_snr, alpha=0.12, color='#1a6fbd')
    _style_ax(a1, 'SNR (dB)', 'MMF (bps/Hz)',
              f'(a) vs SNR  (K=2, $I_{{th}}$={P.I_th}W)')

    # Panel (b) — vs I_th
    a2 = fig4.add_subplot(gs[0, 1])
    a2.plot(ITH_RANGE, rsma_ith, label='CR-RSMA', **STYLE['rsma'])
    a2.plot(ITH_RANGE, noma_ith, label='CR-NOMA', **STYLE['noma'])
    a2.fill_between(ITH_RANGE, noma_ith, rsma_ith, alpha=0.12, color='#1a6fbd')
    _style_ax(a2, '$I_{th}$ (W)', 'MMF (bps/Hz)',
              f'(b) vs $I_{{th}}$  (K=2, SNR={SNR_DB_FIXED}dB)')

    # Panel (c) — vs K
    a3 = fig4.add_subplot(gs[0, 2])
    a3.plot(K_RANGE, rsma_K, label='CR-RSMA', **STYLE['rsma'])
    a3.plot(K_RANGE, noma_K, label='CR-NOMA', **STYLE['noma'])
    a3.fill_between(K_RANGE, noma_K, rsma_K, alpha=0.12, color='#1a6fbd')
    _style_ax(a3, 'Number of Users K', 'MMF (bps/Hz)',
              f'(c) vs K  (SNR={SNR_DB_FIXED}dB)')
    a3.set_xticks(K_RANGE)

    fig4.suptitle(
        'CR-RSMA vs CR-NOMA  —  Max–Min Fairness (MMF) Performance\n'
        f'SVC 4-Layer | QCIF 30fps | B={int(P.B/1e3)}kHz | η={P.eta} | '
        f'$P_{{p,max}}=P_{{s,max}}={P.Pp_max}$W | $w_p=w_s={P.w_p}$',
        fontsize=12, fontweight='bold', y=1.02)

    plt.savefig('fig4_combined.png', dpi=P.dpi, bbox_inches='tight')
    print("  ✓ fig4_combined.png")
    plt.close(fig4)

    # ──────────────────────────────────────────────────────────────────────────
    # Summary
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Simulation complete.")
    print("  Output files:")
    print("    fig1_mmf_vs_snr.png")
    print("    fig2_mmf_vs_ith.png")
    print("    fig3_mmf_vs_K.png")
    print("    fig4_combined.png")
    print("    sim_results.npz")
    print("=" * 70)

    # ══════════════════════════════════════════════════════════════════════════
    # NEW FIGURES  (Scenarios 4, 8, 11)
    # ══════════════════════════════════════════════════════════════════════════

    # ──────────────────────────────────────────────────────────────────────────
    # FIGURE 5 — Sc4: MMF Rate → PSNR (QoE Mapping)
    # ──────────────────────────────────────────────────────────────────────────
    fig5, ax5 = plt.subplots(figsize=(8, 5.5))
    ax5.plot(P.snr_dB_range, rsma_psnr, label='CR-RSMA', **STYLE['rsma'])
    ax5.plot(P.snr_dB_range, noma_psnr, label='CR-NOMA', **STYLE['noma'])
    ax5.fill_between(P.snr_dB_range, noma_psnr, rsma_psnr,
                     alpha=0.12, color='#1a6fbd', label='RSMA gain')

    # Horizontal SVC layer reference lines
    layer_labels = ['BL (29.87 dB)', 'EL1 (33.17 dB)',
                    'EL2 (36.98 dB)', 'EL3 (41.19 dB)']
    for psnr_val, lbl in zip(LAYER_PSNR, layer_labels):
        ax5.axhline(psnr_val, color='grey', linewidth=0.9,
                    linestyle=':', alpha=0.7)
        ax5.text(P.snr_dB_range[-1] + 0.3, psnr_val, lbl,
                 fontsize=8, va='center', color='grey')

    ax5.set_xlim(P.snr_dB_range[0], P.snr_dB_range[-1] + 3)
    ax5.set_ylabel('PSNR (dB)', fontsize=12, fontweight='bold')
    ax5.set_xlabel('SNR (dB)', fontsize=12, fontweight='bold')
    ax5.set_title('Sc4 — QoE Mapping: MMF Rate → SVC PSNR\n'
                  f'(K=2, I_th scales with Pt, ratio={I_TH_RATIO:.4f})',
                  fontsize=13, fontweight='bold', pad=8)
    ax5.grid(True, alpha=0.35, linestyle='--')
    ax5.legend(fontsize=11, framealpha=0.92, loc='upper left')
    ax5.tick_params(labelsize=11)
    plt.tight_layout()
    plt.savefig('fig5_psnr_qoe.png', dpi=P.dpi, bbox_inches='tight')
    print("  ✓ fig5_psnr_qoe.png")
    plt.close(fig5)

    # ──────────────────────────────────────────────────────────────────────────
    # FIGURE 6 — Sc8: Jain's Fairness Index  (1×2 panel)
    # ──────────────────────────────────────────────────────────────────────────
    fig6, (b1, b2) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel (a) — JFI vs SNR
    b1.plot(P.snr_dB_range, rsma_jfi_snr, label='CR-RSMA', **STYLE['rsma'])
    b1.plot(P.snr_dB_range, noma_jfi_snr, label='CR-NOMA', **STYLE['noma'])
    b1.fill_between(P.snr_dB_range, noma_jfi_snr, rsma_jfi_snr,
                    alpha=0.12, color='#1a6fbd')
    b1.axhline(1.0, color='green', linewidth=1.0, linestyle=':', alpha=0.7,
               label='Perfect fairness (JFI=1)')
    _style_ax(b1,
              xlabel='SNR (dB)',
              ylabel="Jain's Fairness Index",
              title="(a) JFI vs SNR  (K=2)")
    b1.set_ylim(0, 1.05)

    # Panel (b) — JFI vs K
    b2.plot(K_RANGE, rsma_jfi_K, label='CR-RSMA', **STYLE['rsma'])
    b2.plot(K_RANGE, noma_jfi_K, label='CR-NOMA', **STYLE['noma'])
    b2.fill_between(K_RANGE, noma_jfi_K, rsma_jfi_K,
                    alpha=0.12, color='#1a6fbd')
    b2.axhline(1.0, color='green', linewidth=1.0, linestyle=':', alpha=0.7,
               label='Perfect fairness (JFI=1)')
    _style_ax(b2,
              xlabel='Number of Secondary Users  $K$',
              ylabel="Jain's Fairness Index",
              title=f"(b) JFI vs K  (SNR={SNR_DB_FIXED} dB)")
    b2.set_ylim(0, 1.05)
    b2.set_xticks(K_RANGE)

    fig6.suptitle("Sc8 — Jain's Fairness Index: CR-RSMA vs CR-NOMA",
                  fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('fig6_jains_fairness.png', dpi=P.dpi, bbox_inches='tight')
    print("  ✓ fig6_jains_fairness.png")
    plt.close(fig6)

    # ──────────────────────────────────────────────────────────────────────────
    # FIGURE 7 — Sc11: Optimization Gain vs Equal-Power Baseline
    # ──────────────────────────────────────────────────────────────────────────
    STYLE_EQ_R = dict(color='#6aaed6', marker='^', linestyle='--',
                      linewidth=1.8, markersize=6, alpha=0.85)
    STYLE_EQ_N = dict(color='#fda56e', marker='v', linestyle='--',
                      linewidth=1.8, markersize=6, alpha=0.85)

    fig7, ax7 = plt.subplots(figsize=(9, 5.5))
    ax7.plot(P.snr_dB_range, rsma_snr, label='CR-RSMA (optimized)', **STYLE['rsma'])
    ax7.plot(P.snr_dB_range, noma_snr, label='CR-NOMA (optimized)', **STYLE['noma'])
    ax7.plot(P.snr_dB_range, rsma_eq,  label='CR-RSMA (equal power)', **STYLE_EQ_R)
    ax7.plot(P.snr_dB_range, noma_eq,  label='CR-NOMA (equal power)', **STYLE_EQ_N)

    # Shade optimization gains
    ax7.fill_between(P.snr_dB_range, rsma_eq, rsma_snr,
                     alpha=0.13, color='#1a6fbd', label='RSMA opt. gain')
    ax7.fill_between(P.snr_dB_range, noma_eq, noma_snr,
                     alpha=0.13, color='#e05c00', label='NOMA opt. gain')

    _style_ax(ax7,
              xlabel='SNR (dB)',
              ylabel='MMF  (min user rate, bps/Hz)',
              title='Sc11 — Optimization Gain vs Equal-Power Baseline\n'
                    f'(K=2, I_th scales with Pt)')
    ax7.legend(fontsize=10, framealpha=0.92, loc='upper left', ncol=2)
    plt.tight_layout()
    plt.savefig('fig7_equal_power_baseline.png', dpi=P.dpi, bbox_inches='tight')
    print("  ✓ fig7_equal_power_baseline.png")
    plt.close(fig7)

    # ──────────────────────────────────────────────────────────────────────────
    # Final summary
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  All simulations complete.")
    print("  Output files:")
    print("    fig1_mmf_vs_snr.png          (Sc1: MMF vs SNR)")
    print("    fig2_mmf_vs_ith.png          (Sc2: MMF vs I_th)")
    print("    fig3_mmf_vs_K.png            (Sc3: MMF vs K)")
    print("    fig4_combined.png            (Sc1–3 combined panel)")
    print("    fig5_psnr_qoe.png            (Sc4: PSNR QoE mapping)")
    print("    fig6_jains_fairness.png      (Sc8: Jain's Fairness Index)")
    print("    fig7_equal_power_baseline.png(Sc11: vs Equal-Power baseline)")
    print("    sim_results.npz              (all numerical results)")
    print("=" * 70)