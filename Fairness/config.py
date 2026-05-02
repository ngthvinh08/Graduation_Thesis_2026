"""
Configuration Parameters for CR-RSMA MMF-QoE Simulation

System  : 1 PU + 1 SU (Underlay Cognitive Radio)
Video   : QCIF (176×144), 30 fps, GOP = 8
SVC     : 4 layers — Base Layer (BL) + EL1 + EL2 + EL3
QP      : 40 → 34 → 28 → 22  (quality tăng dần)
SNR     : 0 – 30 dB, bước 2 dB
Fairness: w_p = w_s = 0.5
"""

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# 3.1 – 3.2  Video & Operating Points
# ═══════════════════════════════════════════════════════════════════════════════
class SVCParams:
    """
    SVC 4-layer configuration
    Sections 3.1 (video), 3.2 (operating points), 3.3 (layer bitrates)
    """

    # ── 3.1  Thông số video cơ bản ─────────────────────────────────────────
    resolution  = (176, 144)    # QCIF
    fps         = 30            # frames per second
    gop_size    = 8             # Group of Pictures (GOP) size
    qp_levels   = [40, 34, 28, 22]   # QP thấp → chất lượng cao

    # ── 3.2  Operating points (QP → bitrate tích lũy & PSNR) ───────────────
    #              QP = 40       QP = 34      QP = 28      QP = 22
    cumulative_bitrate_kbps = [71.3776, 142.9856, 287.8160, 544.2432]   # kbps
    psnr_dB                 = [29.8718,  33.1740,  36.9828,  41.1876]   # dB

    # ── 3.3  Cấu trúc SVC 4 lớp — bitrate tăng thêm của từng lớp ──────────
    #   BL  : 71.3776 kbps
    #   EL1 : 71.6080 kbps   (cộng dồn → 142.9856)
    #   EL2 : 144.8304 kbps  (cộng dồn → 287.8160)
    #   EL3 : 256.4272 kbps  (cộng dồn → 544.2432)
    layer_bitrate_kbps = [
         71.3776,   # Base Layer       (BL)
         71.6080,   # Enhancement Layer 1 (EL1)
        144.8304,   # Enhancement Layer 2 (EL2)
        256.4272,   # Enhancement Layer 3 (EL3)
    ]
    n_layers = 4   # BL + EL1 + EL2 + EL3

    # PSNR tương ứng khi giải mã đến lớp thứ k (k = 1…4)
    layer_psnr_dB = [29.8718, 33.1740, 36.9828, 41.1876]


# ═══════════════════════════════════════════════════════════════════════════════
# 3.4  Thông số tầng vật lý & tối ưu hóa
# ═══════════════════════════════════════════════════════════════════════════════
class SystemParams:
    """Physical layer parameters (Section 3.4)"""

    # ── Băng thông ─────────────────────────────────────────────────────────
    B    = 140e3        # system bandwidth (Hz) = 140 kHz

    # ── AMC (Adaptive Modulation & Coding) ─────────────────────────────────
    c1   = 0.905        # AMC coefficient c1
    c2   = 1.34         # AMC coefficient c2

    # ── Hệ số suy hao đường truyền ─────────────────────────────────────────
    eta  = 2.0          # path-loss exponent η

    # ── Ngân sách công suất ────────────────────────────────────────────────
    Pp_max  = 1.0       # PU max power  (W)
    Ps_max  = 1.0       # SU max total power (W),  Ps_c + Ps_p ≤ Ps_max

    # ── Cấu hình công suất không tối ưu (tham chiếu) ──────────────────────
    # (Ps_c, Pp, Ps_p) = (0.75, 1.0, 0.25) W
    Ps_c_ref = 0.75     # SU common stream power (W)
    Pp_ref   = 1.0      # PU power (W)
    Ps_p_ref = 0.25     # SU private stream power (W)

    # ── Nhiễu ──────────────────────────────────────────────────────────────
    sigma2   = 1.0      # noise variance (normalized)

    # ── Ngưỡng nhiễu (ràng buộc Underlay CR) ───────────────────────────────
    I_th     = 0.1      # max interference power at PU receiver (W)


# ═══════════════════════════════════════════════════════════════════════════════
# 3.4  QoE & Fairness weights
# ═══════════════════════════════════════════════════════════════════════════════
class QoEParams:
    """QoE fairness weights (Section 3.4)"""

    # Trọng số QoE — cân bằng giữa PU và SU
    w_p = 0.5   # weight of Primary User QoE
    w_s = 0.5   # weight of Secondary User QoE


# ═══════════════════════════════════════════════════════════════════════════════
# Simulation sweep parameters
# ═══════════════════════════════════════════════════════════════════════════════
class SimulationParams:
    """Simulation sweep parameters (Section 3.4)"""

    # ── SNR sweep: 0 – 30 dB, bước 2 dB ───────────────────────────────────
    snr_dB_range = np.arange(0, 31, 2)     # [0, 2, 4, …, 30]  →  16 điểm


# ═══════════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════════
class PlotParams:
    """Visualization parameters"""

    figsize      = (8, 6)
    dpi          = 300
    fontsize_title  = 13
    fontsize_label  = 12
    fontsize_legend = 10
    grid_alpha      = 0.3

    colors = {
        'rsma': '#1f77b4',   # blue
        'noma': '#ff7f0e',   # orange
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Master Config — gộp tất cả vào một namespace
# ═══════════════════════════════════════════════════════════════════════════════
class Config:
    """Master configuration object"""

    def __init__(self):
        # ── SVC ────────────────────────────────────────────────────────────
        self.resolution              = SVCParams.resolution
        self.fps                     = SVCParams.fps
        self.gop_size                = SVCParams.gop_size
        self.qp_levels               = SVCParams.qp_levels
        self.cumulative_bitrate_kbps = SVCParams.cumulative_bitrate_kbps
        self.psnr_dB                 = SVCParams.psnr_dB
        self.layer_bitrate_kbps      = SVCParams.layer_bitrate_kbps
        self.n_layers                = SVCParams.n_layers
        self.layer_psnr_dB           = SVCParams.layer_psnr_dB

        # ── Physical Layer ─────────────────────────────────────────────────
        self.B           = SystemParams.B
        self.c1          = SystemParams.c1
        self.c2          = SystemParams.c2
        self.eta         = SystemParams.eta
        self.Pp_max      = SystemParams.Pp_max
        self.Ps_max      = SystemParams.Ps_max
        self.Ps_c_ref    = SystemParams.Ps_c_ref
        self.Pp_ref      = SystemParams.Pp_ref
        self.Ps_p_ref    = SystemParams.Ps_p_ref
        self.sigma2      = SystemParams.sigma2
        self.I_th        = SystemParams.I_th

        # ── QoE / Fairness ─────────────────────────────────────────────────
        self.w_p = QoEParams.w_p
        self.w_s = QoEParams.w_s

        # ── Simulation ─────────────────────────────────────────────────────
        self.snr_dB_range = SimulationParams.snr_dB_range

        # ── Plot ───────────────────────────────────────────────────────────
        self.figsize         = PlotParams.figsize
        self.dpi             = PlotParams.dpi
        self.colors          = PlotParams.colors


# ───────────────────────────────────────────────────────────────────────────────
# Global config instance
# ───────────────────────────────────────────────────────────────────────────────
P = Config()
