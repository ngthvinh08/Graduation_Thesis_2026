"""
optimizer.py — Super-Fast SCA Optimization with Algebraic RDO
Algebraic Reduction: D = C_rdo / R_th^2
Eliminates all expensive log/MB loops inside the optimizer.
"""
import numpy as np
from scipy.optimize import minimize
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import *
from .qoe_fairness_model import (compute_qoe, jain_fairness, max_min_fairness, objective_wsum, objective_maxmin)

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def compute_rates_rsma(P_s1c, P_s1p, P_s2c, P_s2p, P_pu, g_s1, g_s2, g_p):
    """Tính tốc độ uplink CR-RSMA (Common -> PU -> Private) — Vectorized."""
    sensing_eff = 0.92 
    gs1_safe, gs2_safe, gp_safe = g_s1 + 1e-10, g_s2 + 1e-10, g_p + 1e-10
    
    # 1. Decode SU1-Common (W1,1)
    # Interference: SU1-Private, SU2-Common, SU2-Private, PU
    gamma_s1c = (P_s1c * gs1_safe) / (P_s1p * gs1_safe + P_s2c * gs2_safe + P_s2p * gs2_safe + P_pu * gp_safe + SIGMA2)
    R_s1c = B * np.log2(1 + gamma_s1c) * sensing_eff
    
    # 2. Decode SU2-Common (W2,1)
    # Interference: SU1-Private, SU2-Private, PU
    gamma_s2c = (P_s2c * gs2_safe) / (P_s1p * gs1_safe + P_s2p * gs2_safe + P_pu * gp_safe + SIGMA2)
    R_s2c = B * np.log2(1 + gamma_s2c) * sensing_eff
    
    # 3. Decode PU (WK)
    # Interference: SU1-Private, SU2-Private
    gamma_pu = (P_pu * gp_safe) / (P_s1p * gs1_safe + P_s2p * gs2_safe + SIGMA2)
    R_pu = B * np.log2(1 + gamma_pu) * sensing_eff
    
    # 4. Decode SU1-Private (W1,2)
    # Interference: SU2-Private
    gamma_s1p = (P_s1p * gs1_safe) / (P_s2p * gs2_safe + SIGMA2)
    R_s1p = B * np.log2(1 + gamma_s1p) * sensing_eff
    
    # 5. Decode SU2-Private (W2,2)
    # Interference: Noise
    gamma_s2p = (P_s2p * gs2_safe) / SIGMA2
    R_s2p = B * np.log2(1 + gamma_s2p) * sensing_eff
    
    return R_pu, R_s1c + R_s1p, R_s2c + R_s2p


def compute_rates_noma(P_s1c, P_s1p, P_s2c, P_s2p, P_pu, g_s1, g_s2, g_p):
    """Tính tốc độ uplink CR-NOMA với SIC thích ứng theo kênh tức thời."""
    sensing_eff = 0.92
    gs1_safe, gs2_safe, gp_safe = g_s1 + 1e-10, g_s2 + 1e-10, g_p + 1e-10

    # Với NOMA, toàn bộ công suất SU được gộp vào một luồng duy nhất.
    P_s1 = P_s1c + P_s1p
    P_s2 = P_s2c + P_s2p

    # PU vẫn được ưu tiên theo CR, nên chỉ chịu nhiễu từ hai SU.
    gamma_pu = (P_pu * gp_safe) / (P_s1 * gs1_safe + P_s2 * gs2_safe + SIGMA2)
    R_pu = B * np.log2(1 + gamma_pu) * sensing_eff

    # SIC order: user có received power mạnh hơn được decode sau và không bị user còn lại gây nhiễu.
    recv_s1 = P_s1 * gs1_safe
    recv_s2 = P_s2 * gs2_safe
    s1_stronger = recv_s1 >= recv_s2

    gamma_s1_weak = recv_s1 / (recv_s2 + P_pu * gp_safe + SIGMA2)
    gamma_s2_weak = recv_s2 / (recv_s1 + P_pu * gp_safe + SIGMA2)
    gamma_s1_strong = recv_s1 / (P_pu * gp_safe + SIGMA2)
    gamma_s2_strong = recv_s2 / (P_pu * gp_safe + SIGMA2)

    gamma_s1 = np.where(s1_stronger, gamma_s1_strong, gamma_s1_weak)
    gamma_s2 = np.where(s1_stronger, gamma_s2_weak, gamma_s2_strong)

    R_s1 = B * np.log2(1 + gamma_s1) * sensing_eff
    R_s2 = B * np.log2(1 + gamma_s2) * sensing_eff
    return R_pu, R_s1, R_s2


def compute_rates(P_s1c, P_s1p, P_s2c, P_s2p, P_pu, g_s1, g_s2, g_p, scheme='rsma'):
    if scheme == 'rsma':
        return compute_rates_rsma(P_s1c, P_s1p, P_s2c, P_s2p, P_pu, g_s1, g_s2, g_p)
    if scheme == 'noma':
        return compute_rates_noma(P_s1c, P_s1p, P_s2c, P_s2p, P_pu, g_s1, g_s2, g_p)
    raise ValueError(f"Unsupported access scheme: {scheme}")

def compute_svc_layers(R_tot):
    R_tot = np.atleast_1d(R_tot)
    k = np.zeros_like(R_tot, dtype=int)
    k = np.where(R_tot < R_LAYER[0], 0, k)
    k = np.where((R_tot >= R_LAYER[0]) & (R_tot < np.sum(R_LAYER[:2])), 1, k)
    k = np.where((R_tot >= np.sum(R_LAYER[:2])) & (R_tot < np.sum(R_LAYER[:3])), 2, k)
    k = np.where(R_tot >= np.sum(R_LAYER[:3]), 3, k)
    return k.astype(int)

def precalculate_rdo_constants(alpha_frames, gamma_frames):
    alpha_mat = np.array(alpha_frames)
    gamma_mat = np.array(gamma_frames)
    m_blocks = np.prod(alpha_mat.shape[1:])
    term = (alpha_mat**(2/3)) * (gamma_mat**(1/3))
    sum_term = np.sum(term, axis=tuple(range(1, term.ndim)))
    return (sum_term ** 3) / m_blocks

def compute_psnr_from_D(D):
    return 10 * np.log10(255**2 / np.maximum(D, 1e-10))

# ============================================================
# MAIN EVALUATION PIPELINE
# ============================================================
def evaluate_all(P_s1c, P_s1p, P_s2c, P_s2p, P_pu, g_s1, g_s2, g_p, c_rdo_arr, mode, scheme='rsma'):
    R_p, R_s1, R_s2 = compute_rates(P_s1c, P_s1p, P_s2c, P_s2p, P_pu, g_s1, g_s2, g_p, scheme=scheme)
    
    k_p, k_s1, k_s2 = compute_svc_layers(R_p), compute_svc_layers(R_s1), compute_svc_layers(R_s2)

    D_p = c_rdo_arr / (np.maximum(R_p, 1e2)**2)
    D_s1 = c_rdo_arr / (np.maximum(R_s1, 1e2)**2)
    D_s2 = c_rdo_arr / (np.maximum(R_s2, 1e2)**2)

    PSNR_p, PSNR_s1, PSNR_s2 = compute_psnr_from_D(D_p), compute_psnr_from_D(D_s1), compute_psnr_from_D(D_s2)
    PSNR_p[k_p == 0], PSNR_s1[k_s1 == 0], PSNR_s2[k_s2 == 0] = 0.0, 0.0, 0.0

    # P_tot for UL: User consumes its own Transmit power
    P_tot_p = P_pu + P_ENC + P_FLY
    P_tot_s1 = P_s1c + P_s1p + P_ENC + P_FLY
    P_tot_s2 = P_s2c + P_s2p + P_ENC + P_FLY

    QoE_p = compute_qoe(PSNR_p, P_tot_p, k_layers=k_p)
    QoE_s1 = compute_qoe(PSNR_s1, P_tot_s1, k_layers=k_s1)
    QoE_s2 = compute_qoe(PSNR_s2, P_tot_s2, k_layers=k_s2)

    F = jain_fairness(QoE_s1, QoE_s2)
    obj = objective_wsum(QoE_p, QoE_s1, QoE_s2) if mode == 'wsum' else objective_maxmin(QoE_p, QoE_s1, QoE_s2)

    return {
        'obj': obj, 'QoE_p': QoE_p, 'QoE_s1': QoE_s1, 'QoE_s2': QoE_s2,
        'PSNR_p': PSNR_p, 'PSNR_s1': PSNR_s1, 'PSNR_s2': PSNR_s2,
        'k_p': k_p, 'k_s1': k_s1, 'k_s2': k_s2, 'F': F
    }

# ============================================================
# OPTIMIZATION LOOP
# ============================================================
def update_resource_allocation(P_s1c, P_s1p, P_s2c, P_s2p, P_pu, g_s1, g_s2, g_p, c_rdo_arr, mode, scheme='rsma'):
    n_slots = len(g_p)
    
    def neg_obj(x):
        P_s1c_n = np.clip(x[0*n_slots:1*n_slots], 0, P_S_MAX)
        P_s1p_n = np.clip(x[1*n_slots:2*n_slots], 0, P_S_MAX)
        P_s2c_n = np.clip(x[2*n_slots:3*n_slots], 0, P_S_MAX)
        P_s2p_n = np.clip(x[3*n_slots:4*n_slots], 0, P_S_MAX)
        P_pu_n  = np.clip(x[4*n_slots:5*n_slots], 0, P_P_MAX)
        
        res = evaluate_all(P_s1c_n, P_s1p_n, P_s2c_n, P_s2p_n, P_pu_n, g_s1, g_s2, g_p, c_rdo_arr, mode, scheme=scheme)
        return -res['obj']

    cons = [
        {'type': 'ineq', 'fun': lambda x: P_S_MAX - (x[0*n_slots:1*n_slots] + x[1*n_slots:2*n_slots])}, 
        {'type': 'ineq', 'fun': lambda x: P_S_MAX - (x[2*n_slots:3*n_slots] + x[3*n_slots:4*n_slots])}, 
        {'type': 'ineq', 'fun': lambda x: P_P_MAX - x[4*n_slots:5*n_slots]} 
    ]
    
    x0 = np.concatenate([P_s1c, P_s1p, P_s2c, P_s2p, P_pu])
    res = minimize(neg_obj, x0, method='SLSQP', constraints=cons, options={'maxiter': 20, 'ftol': 1e-4})
    
    x = res.x if res.success else x0
    return (np.clip(x[0*n_slots:1*n_slots], 1e-6, P_S_MAX),
            np.clip(x[1*n_slots:2*n_slots], 1e-6, P_S_MAX),
            np.clip(x[2*n_slots:3*n_slots], 1e-6, P_S_MAX),
            np.clip(x[3*n_slots:4*n_slots], 1e-6, P_S_MAX),
            np.clip(x[4*n_slots:5*n_slots], 1e-6, P_P_MAX), res.success)

def run_sca(g_p, g_s1, g_s2, alpha_frames=None, gamma_frames=None, mode='wsum', scheme='rsma'):
    n_slots = len(g_p)
    if alpha_frames is None: alpha_frames = [ALPHA_RD] * n_slots
    if gamma_frames is None: gamma_frames = [GAMMA_RD] * n_slots

    print(f"  [PRE] Pre-calculating algebraic RDO constants for {n_slots} slots ({scheme.upper()})...")
    c_rdo_arr = precalculate_rdo_constants(alpha_frames, gamma_frames)

    P_s1c, P_s1p = np.ones(n_slots)*0.2, np.ones(n_slots)*0.3
    P_s2c, P_s2p = np.ones(n_slots)*0.2, np.ones(n_slots)*0.3
    P_pu = np.ones(n_slots)*0.5
    
    best_res, obj_hist, fair_hist = {}, [], []
    
    for it in range(MAX_ITER):
        res = evaluate_all(P_s1c, P_s1p, P_s2c, P_s2p, P_pu, g_s1, g_s2, g_p, c_rdo_arr, mode, scheme=scheme)
        obj_hist.append(res['obj'])
        fair_hist.append(res['F'].mean())
        print(f"  Iter {it+1:2d}: Obj={res['obj']:10.3f}, PSNR_p={res['PSNR_p'].mean():5.1f}, Fair={res['F'].mean():.3f}")
        
        P_s1c, P_s1p, P_s2c, P_s2p, P_pu, success = update_resource_allocation(
            P_s1c, P_s1p, P_s2c, P_s2p, P_pu, g_s1, g_s2, g_p, c_rdo_arr, mode, scheme=scheme
        )
        
        if it > 0 and abs(obj_hist[-1] - obj_hist[-2]) < TOL:
            print(f"  [CONVERGED] in {it+1} iterations")
            break
        best_res = res

    return {**best_res, 'obj_hist': obj_hist, 'fair_hist': fair_hist, 
            'P_s1c': P_s1c, 'P_s1p': P_s1p, 'P_s2c': P_s2c, 'P_s2p': P_s2p, 'P_pu': P_pu}
