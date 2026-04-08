"""
config.py — Thông số hệ thống
CR-RSMA + UAV + QoE Fairness
"""
import numpy as np

# ============================================================
# THÔNG SỐ MẠNG
# ============================================================
N_SLOTS  = 100       # Số time slots
B        = 1e6      # Bandwidth (Hz)
SIGMA2   = 1e-3     # Noise power (W)

BETA0    = 2.0      # Gain kênh tại khoảng cách 1m từ BS
ALPHA_PL = 2.0      # Tốc độ suy giảm tín hiệu theo khoảng cách

# ============================================================
# THÔNG SỐ PU
# ============================================================
P_P_MAX  = 1.0      # Công suất tối đa PU (W)
Q_B      = np.array([0.0, 0.0, 0.0])  # Vị trí BS

# ============================================================
# THÔNG SỐ SU
# ============================================================
K_SU     = 2        # Số lượng SU
P_S_MAX  = 1.0      # Công suất tối đa mỗi SU (W)
V_S_MAX  = 20.0     # Tốc độ bay tối đa (m/s)

# ============================================================
# THÔNG SỐ SVC
# ============================================================
L_MAX    = 3        # Số lớp tối đa (BL + EL1 + EL2)
R_LAYER  = np.array([200e3, 400e3, 800e3])  # Bitrate/lớp (bps)

# RDO parameters — R(QP) = α*2^(-(QP-12)/6), D(QP) = γ*2^((QP-12)/3)
# Calibrated for realistic PSNR (30-45 dB) and Bitrate (200-1500 kbps)
ALPHA_RD = np.array([3500.0, 2500.0, 1500.0])  # Adjusted for kbps range
GAMMA_RD = np.array([1.0, 1.5, 2.0])           # Adjusted for 35-45 dB range
M_BLOCKS = 300  # 320/16 * 240/16 = 300 macroblocks

# ============================================================
# THÔNG SỐ QoE
# ============================================================
A_U = 1.0   # Trọng số log(1+PSNR)
B_U = 0.8   # Trọng số biến động ΔQ (Tăng để phạt flickering)
C_U = 0.5   # Trọng số delay (Tăng tính quan trọng)
D_U = 0.1   # Trọng số công suất (Giảm nhẹ ưu tiên chất lượng)
E_U = 1.5   # Trọng số Stalling penalty (New)

# ============================================================
# THÔNG SỐ FAIRNESS
# ============================================================
KAPPA   = 0.8   # Hệ số ưu tiên PU (0 < κ < 1)
ETA_P   = 2.0   # Trọng số PU trong Jain's index
ETA_S   = 1.0   # Trọng số SU trong Jain's index
RHO     = 0.5   # Trọng số fairness regularization
OMEGA_P = 0.6   # Trọng số QoE_p
OMEGA_S = 0.4   # Trọng số QoE_s

# ============================================================
# THÔNG SỐ OPTIMIZER
# ============================================================
TOL      = 1e-4

# ============================================================
# THÔNG SỐ UAV POWER
# ============================================================
P_FLY   = 0.10   # Công suất bay (W)
P_ENC   = 0.05   # Công suất mã hóa video (W)
DELTA_T = 1.0    # Độ dài time slot (s)
H_UAV   = 20.0   # Độ cao bay cố định (m)

# ============================================================
# TARGET BITRATE
# ============================================================
R_TH_PU = 500e3   # Target bitrate PU (bps)
R_TH_SU = 400e3   # Target bitrate mỗi SU (bps)

# ============================================================
# SVC CODING MODES
# ============================================================
PHI_MODE = {
    'intra':         1.0,
    'inter':         0.6,
    'intra_refresh': 0.8,
    'skip':          0.02
}
LAYER_MODES = ['intra', 'inter', 'inter']
MAX_ITER = 30