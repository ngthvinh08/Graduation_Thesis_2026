"""
qoe.py — Hàm QoE và Fairness

QoE_u[n] = a*log(1+PSNR[n]) - b*ΔQ[n] - c*T_delay[n] - d*P_tot[n]

Fairness:
  1. Weighted Jain's index
  2. Weighted Max-Min
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import *


def compute_qoe(PSNR, P_tot, T_delay=None, k_layers=None):
    """
    Tính QoE cho một user qua tất cả slots
    QoE[n] = a*log(1+PSNR[n]) - b*ΔQ[n] - c*T_delay[n] - d*P_tot[n] - e*Stalling[n]
    """
    n_slots = len(PSNR)
    QoE = np.zeros(n_slots)

    for n in range(n_slots):
        # Thành phần 1: chất lượng video (log-scale)
        quality = A_U * np.log(1 + PSNR[n] + 1e-6)

        # Thành phần 2: biến động chất lượng ΔQ (Flickering penalty)
        delta_Q = abs(PSNR[n] - PSNR[n-1]) if n > 0 else 0.0
        flickering = B_U * delta_Q

        # Thành phần 3: delay
        delay = C_U * T_delay[n] if T_delay is not None else 0.0

        # Thành phần 4: công suất tổng
        power = D_U * P_tot[n]
        
        # Thành phần 5: Stalling penalty (New)
        # Stalling xảy ra khi PSNR cực thấp hoặc k_layers = 0
        stalling = 0.0
        if PSNR[n] < 20.0 or (k_layers is not None and k_layers[n] == 0):
            stalling = E_U  # Trọng số phạt từ config

        QoE[n] = quality - flickering - delay - power - stalling

    return QoE


def psnr_to_mos(psnr):
    """
    Ánh xạ PSNR (dB) sang thang điểm MOS (1-5) theo chuẩn ITU-R BT.500.
    Xấp xỉ: MOS = 1 + 0.1 * (psnr - 20) nếu psnr > 20, else 1.
    """
    mos = 1.0 + 0.1 * (np.array(psnr) - 20)
    return np.clip(mos, 1.0, 5.0)


def jain_fairness(QoE_p, QoE_s):
    """
    Weighted Jain's fairness index
    F[n] = (η_p*QoE_p + η_s*QoE_s)² / [(η_p+η_s)(η_p*QoE_p² + η_s*QoE_s²)]
    """
    eps = 1e-10
    n_slots = len(QoE_p)
    F   = np.zeros(n_slots)

    for n in range(n_slots):
        num  = (ETA_P * QoE_p[n] + ETA_S * QoE_s[n])**2
        den  = (ETA_P + ETA_S) * (
            ETA_P * QoE_p[n]**2 + ETA_S * QoE_s[n]**2 + eps
        )
        F[n] = num / den

    return F

# Max-Min Fairness: t[n] = min(QoE_p, κ * QoE_s)
# → κ = 0.8 nghĩa là: QoE của SU bị "giảm giá" 20%
# → Hệ thống sẽ luôn ưu tiên bảo vệ QoE của PU trước
def max_min_fairness(QoE_p, QoE_s):
    """
    Weighted Max-Min fairness
    t[n] = min(QoE_p[n], κ*QoE_s[n])
    """
    return np.minimum(QoE_p, KAPPA * QoE_s)


def objective_wsum(QoE_p, QoE_s1, QoE_s2=None):
    """
    Bài toán 1: Weighted-Sum + Fairness regularization
    1 PU + 2 SU case: max Σ_n [ω_p*QoE_p + ω_s*(QoE_s1+QoE_s2)/2 + ρ*F(s1,s2)]
    """
    # For 2 SUs: fairness is between s1 and s2
    F = jain_fairness(QoE_s1, QoE_s2) if QoE_s2 is not None else jain_fairness(QoE_p, QoE_s1)
    QoE_s = (QoE_s1 + QoE_s2) / 2 if QoE_s2 is not None else QoE_s1
    return np.sum(OMEGA_P * QoE_p + OMEGA_S * QoE_s + RHO * F)


def objective_maxmin(QoE_p, QoE_s1, QoE_s2=None):
    """
    Bài toán 2: Weighted Max-Min QoE Fairness
    1 PU + 2 SU case: max Σ_n min(QoE_p, κ*min(QoE_s1, QoE_s2))
    """
    if QoE_s2 is not None:
        # 2 SU case: take minimum of s1 and s2
        QoE_s_min = np.minimum(QoE_s1, QoE_s2)
        t = max_min_fairness(QoE_p, QoE_s_min)
    else:
        # 1 SU case
        t = max_min_fairness(QoE_p, QoE_s1)
    return np.sum(t)


if __name__ == '__main__':
    PSNR_p = np.random.uniform(30, 40, N_SLOTS)
    PSNR_s = np.random.uniform(25, 35, N_SLOTS)
    P_p    = np.ones(N_SLOTS) * 0.5
    P_s    = np.ones(N_SLOTS) * 0.4

    QoE_p = compute_qoe(PSNR_p, P_p)
    QoE_s = compute_qoe(PSNR_s, P_s)
    F     = jain_fairness(QoE_p, QoE_s)

    print(f"QoE_p mean  = {QoE_p.mean():.3f}")
    print(f"QoE_s mean  = {QoE_s.mean():.3f}")
    print(f"Fairness    = {F.mean():.3f}")
    print(f"Obj W-Sum   = {objective_wsum(QoE_p, QoE_s):.3f}")
    print(f"Obj Max-Min = {objective_maxmin(QoE_p, QoE_s):.3f}")