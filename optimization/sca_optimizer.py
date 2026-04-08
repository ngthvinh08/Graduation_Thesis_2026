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
def compute_rates(P_c, P_p, P_s1, P_s2, g_p, g_s1, g_s2):
    """Tính tốc độ RSMA chuẩn (Common + Private) — Vectorized."""
    SIGMA2 = 1e-3
    sensing_eff = 0.92 
    gp_safe, gs1_safe, gs2_safe = g_p + 1e-10, g_s1 + 1e-10, g_s2 + 1e-10
    
    gamma_pc = P_c * gp_safe / ((P_p + P_s1 + P_s2) * gp_safe + SIGMA2)
    gamma_s1c = P_c * gs1_safe / ((P_p + P_s1 + P_s2) * gs1_safe + SIGMA2)
    gamma_s2c = P_c * gs2_safe / ((P_p + P_s1 + P_s2) * gs2_safe + SIGMA2)
    
    gamma_c_min = np.minimum(np.minimum(gamma_pc, gamma_s1c), gamma_s2c)
    R_c = B * np.log2(1 + gamma_c_min) * sensing_eff
    C_p, C_s1, C_s2 = R_c * 0.4, R_c * 0.3, R_c * 0.3
    
    R_pp = B * np.log2(1 + P_p * gp_safe / ((P_s1 + P_s2) * gp_safe + SIGMA2)) * sensing_eff
    R_s1p = B * np.log2(1 + P_s1 * gs1_safe / ((P_p + P_s2) * gs1_safe + SIGMA2)) * sensing_eff
    R_s2p = B * np.log2(1 + P_s2 * gs2_safe / ((P_p + P_s1) * gs2_safe + SIGMA2)) * sensing_eff
    
    return C_p + R_pp, C_s1 + R_s1p, C_s2 + R_s2p, R_c

def compute_svc_layers(R_tot):
    R_tot = np.atleast_1d(R_tot)
    k = np.zeros_like(R_tot, dtype=int)
    k = np.where(R_tot < R_LAYER[0], 0, k)
    k = np.where((R_tot >= R_LAYER[0]) & (R_tot < np.sum(R_LAYER[:2])), 1, k)
    k = np.where((R_tot >= np.sum(R_LAYER[:2])) & (R_tot < np.sum(R_LAYER[:3])), 2, k)
    k = np.where(R_tot >= np.sum(R_LAYER[:3]), 3, k)
    return k.astype(int)

def precalculate_rdo_constants(alpha_frames, gamma_frames):
    """
    Tính hằng số C_rdo cho mỗi khung hình:
    C_rdo = (Σ α^(2/3) * γ^(1/3))^3 / M
    """
    alpha_mat = np.array(alpha_frames)
    gamma_mat = np.array(gamma_frames)
    m_blocks = np.prod(alpha_mat.shape[1:])
    
    term = (alpha_mat**(2/3)) * (gamma_mat**(1/3))
    sum_term = np.sum(term, axis=tuple(range(1, term.ndim)))
    c_rdo = (sum_term ** 3) / m_blocks
    return c_rdo

def compute_psnr_from_D(D):
    return 10 * np.log10(255**2 / np.maximum(D, 1e-10))

# ============================================================
# MAIN EVALUATION PIPELINE — Super Fast
# ============================================================
def evaluate_all(P_c, P_p, P_s1, P_s2, g_p, g_s1, g_s2, c_rdo_arr, mode):
    n_slots = len(g_p)
    R_p, R_s1, R_s2, _ = compute_rates(P_c, P_p, P_s1, P_s2, g_p, g_s1, g_s2)
    
    k_p = compute_svc_layers(R_p)
    k_s1 = compute_svc_layers(R_s1)
    k_s2 = compute_svc_layers(R_s2)

    # Algebraic RDO: D = C_rdo / R_th^2
    D_p = c_rdo_arr / (np.maximum(R_p, 1e2)**2)
    D_s1 = c_rdo_arr / (np.maximum(R_s1, 1e2)**2)
    D_s2 = c_rdo_arr / (np.maximum(R_s2, 1e2)**2)

    PSNR_p, PSNR_s1, PSNR_s2 = compute_psnr_from_D(D_p), compute_psnr_from_D(D_s1), compute_psnr_from_D(D_s2)
    PSNR_p[k_p == 0], PSNR_s1[k_s1 == 0], PSNR_s2[k_s2 == 0] = 0.0, 0.0, 0.0

    # QoE & Fairness
    P_tot_p = P_p + P_c*0.4 + P_ENC + P_FLY
    P_tot_s1 = P_s1 + P_c*0.3 + P_ENC + P_FLY
    P_tot_s2 = P_s2 + P_c*0.3 + P_ENC + P_FLY

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
def update_resource_allocation(P_c, P_p, P_s1, P_s2, g_p, g_s1, g_s2, c_rdo_arr, mode):
    n_slots = len(g_p)
    P_MAX = P_P_MAX + P_S_MAX
    
    def neg_obj(x):
        x_clipped = np.clip(x, 0, P_MAX)
        Pc, Pp, Ps1, Ps2 = x_clipped[0*n_slots:1*n_slots], x_clipped[1*n_slots:2*n_slots], x_clipped[2*n_slots:3*n_slots], x_clipped[3*n_slots:4*n_slots]
        res = evaluate_all(Pc, Pp, Ps1, Ps2, g_p, g_s1, g_s2, c_rdo_arr, mode)
        return -res['obj']

    cons = [
        {'type': 'ineq', 'fun': lambda x: P_MAX - (x[0*n_slots:1*n_slots] + x[1*n_slots:2*n_slots] + x[2*n_slots:3*n_slots] + x[3*n_slots:4*n_slots])},
        {'type': 'ineq', 'fun': lambda x: P_MAX * n_slots * 0.8 - np.sum(x)}
    ]
    
    x0 = np.concatenate([P_c, P_p, P_s1, P_s2])
    res = minimize(neg_obj, x0, method='SLSQP', constraints=cons, options={'maxiter': 20, 'ftol': 1e-4})
    
    x = res.x if res.success else x0
    return (np.clip(x[0*n_slots:1*n_slots], 1e-6, P_MAX),
            np.clip(x[1*n_slots:2*n_slots], 1e-6, P_MAX),
            np.clip(x[2*n_slots:3*n_slots], 1e-6, P_MAX),
            np.clip(x[3*n_slots:4*n_slots], 1e-6, P_MAX), res.success)

def run_sca(g_p, g_s1, g_s2, alpha_frames=None, gamma_frames=None, mode='wsum'):
    n_slots = len(g_p)
    if alpha_frames is None: alpha_frames = [ALPHA_RD] * n_slots
    if gamma_frames is None: gamma_frames = [GAMMA_RD] * n_slots

    # Pre-calculate RDO Constants (DO THIS ONCE)
    print(f"  [PRE] Pre-calculating algebraic RDO constants for {n_slots} slots...")
    c_rdo_arr = precalculate_rdo_constants(alpha_frames, gamma_frames)

    P_c, P_p, P_s1, P_s2 = np.ones(n_slots)*0.2, np.ones(n_slots)*0.4, np.ones(n_slots)*0.1, np.ones(n_slots)*0.1
    best_res, obj_hist, fair_hist = {}, [], []
    
    for it in range(MAX_ITER):
        res = evaluate_all(P_c, P_p, P_s1, P_s2, g_p, g_s1, g_s2, c_rdo_arr, mode)
        obj_hist.append(res['obj'])
        fair_hist.append(res['F'].mean())
        print(f"  Iter {it+1:2d}: Obj={res['obj']:10.3f}, PSNR_p={res['PSNR_p'].mean():5.1f}, Fair={res['F'].mean():.3f}")
        
        P_c, P_p, P_s1, P_s2, success = update_resource_allocation(P_c, P_p, P_s1, P_s2, g_p, g_s1, g_s2, c_rdo_arr, mode)
        
        if it > 0 and abs(obj_hist[-1] - obj_hist[-2]) < TOL:
            print(f"  [CONVERGED] in {it+1} iterations")
            break
        best_res = res

    return {**best_res, 'obj_hist': obj_hist, 'fair_hist': fair_hist, 'P_c': P_c, 'P_p': P_p, 'P_s1': P_s1, 'P_s2': P_s2}