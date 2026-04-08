"""
channel.py — Mô hình kênh UAV
1 PU + 2 SU

g_u[n] = β0 / ||q_u[n] - q_b||^α
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import *


def channel_gain(q_u, q_ref=Q_B, k_factor=0):
    """
    Tính channel gain bao gồm large-scale path loss và small-scale fading.
    
    Args:
        q_u: Vị trí UAV
        q_ref: Vị trí BS (với BS là trạm gốc)
        k_factor: Rician K-factor (k=0 là Rayleigh, k>0 là Rician)
    """
    dist = max(np.linalg.norm(q_u - q_ref), 1.0)
    g_large = BETA0 / (dist ** ALPHA_PL)
    
    # Small-scale fading
    if k_factor > 0:
        # Rician fading
        s = np.sqrt(k_factor / (k_factor + 1))
        sigma = np.sqrt(1 / (2 * (k_factor + 1)))
        # scipy rice: b = s/sigma = sqrt(2*K)
        from scipy.stats import rice
        fading = rice.rvs(np.sqrt(2 * k_factor), scale=sigma)
    else:
        # Rayleigh fading (K=0)
        # scipy rice with b=0 is Rayleigh with sigma = scale
        from scipy.stats import rice
        fading = rice.rvs(0, scale=np.sqrt(1/2))
        
    return g_large * (fading ** 2)


def generate_trajectories():
    """
    1 PU + 2 SU với quỹ đạo khác nhau
    → tạo ra sự bất đối xứng kênh → cần fairness
    """
    # PU: bay thẳng
    q_p = np.array([
        np.array([10.0 + n*0.5, 10.0, 20.0])
        for n in range(N_SLOTS)
    ])
    
    # SU1: vòng tròn bán kính 15m (gần BS hơn)
    q_s1 = np.array([
        np.array([15*np.cos(2*np.pi*n/N_SLOTS),
                  15*np.sin(2*np.pi*n/N_SLOTS), 20.0])
        for n in range(N_SLOTS)
    ])
    
    # SU2: vòng tròn bán kính 25m lệch pha (xa BS hơn)
    q_s2 = np.array([
        np.array([25*np.cos(2*np.pi*n/N_SLOTS + np.pi),
                  25*np.sin(2*np.pi*n/N_SLOTS + np.pi), 20.0])
        for n in range(N_SLOTS)
    ])
    return q_p, q_s1, q_s2


def compute_channels_gain(q_p, q_s1, q_s2):
    # PU (Primary User): Thường có LoS tốt hơn -> Rician (K=5)
    g_p  = np.array([channel_gain(q_p[n],  k_factor=5.0) for n in range(N_SLOTS)])
    
    # SU (Secondary Users): Năng lượng thấp, nhiều vật cản -> Rayleigh (K=0)
    g_s1 = np.array([channel_gain(q_s1[n], k_factor=0.0) for n in range(N_SLOTS)])
    g_s2 = np.array([channel_gain(q_s2[n], k_factor=0.0) for n in range(N_SLOTS)])
    
    return g_p, g_s1, g_s2


if __name__ == '__main__':
    q_p, q_s1, q_s2 = generate_trajectories()
    g_p, g_s1, g_s2 = compute_channels_gain(q_p, q_s1, q_s2)
    print(f"g_p  mean = {g_p.mean():.4f}")
    print(f"g_s1 mean = {g_s1.mean():.4f}")
    print(f"g_s2 mean = {g_s2.mean():.4f}")