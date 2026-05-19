"""
batch_analysis.py — Phân tích Trade-off đa tham số
Vẽ biểu đồ PSNR vs Power và Fairness vs Kappa (Academic Style)
"""
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# Thêm đường dẫn gốc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from optimization.uav_channel_model import compute_channels_gain, generate_trajectories
from optimization.sca_optimizer import run_sca
from visualization.visualize import plot_tradeoff, plot_scheme_results

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

def run_batch_power():
    """Phân tích PSNR trung bình theo Công suất phát (P_max)"""
    print("\n" + "="*50)
    print("ANALYSIS A: PSNR vs. TRANSMIT POWER")
    print("="*50)
    
    # Setup kịch bản tĩnh
    np.random.seed(42)
    q_p, q_s1, q_s2 = generate_trajectories()
    g_p, g_s1, g_s2 = compute_channels_gain(q_p, q_s1, q_s2)
    
    p_range = np.linspace(0.1, 2.0, 10)  # Từ 0.1W đến 2W
    results = {
        'PSNR_PU (W-Sum)': [],
        'PSNR_PU (Max-Min)': [],
        'PSNR_SU (W-Sum)': [],
        'PSNR_SU (Max-Min)': []
    }

    import optimization.sca_optimizer as sca
    original_p_s = sca.P_S_MAX
    original_p_p = sca.P_P_MAX

    for p in p_range:
        print(f"  Testing P_max = {p:.2f} W...")
        # Inject tham số
        sca.P_S_MAX = p
        sca.P_P_MAX = p
        
        # Chạy W-Sum
        res_w = run_sca(g_p, g_s1, g_s2, mode='wsum', scheme='rsma')
        # Chạy Max-Min
        res_m = run_sca(g_p, g_s1, g_s2, mode='maxmin', scheme='rsma')
        
        results['PSNR_PU (W-Sum)'].append(res_w['PSNR_p'].mean())
        results['PSNR_PU (Max-Min)'].append(res_m['PSNR_p'].mean())
        results['PSNR_SU (W-Sum)'].append((res_w['PSNR_s1'].mean() + res_w['PSNR_s2'].mean())/2)
        results['PSNR_SU (Max-Min)'].append((res_m['PSNR_s1'].mean() + res_m['PSNR_s2'].mean())/2)

    # Khôi phục
    sca.P_S_MAX = original_p_s
    sca.P_P_MAX = original_p_p

    plot_tradeoff(
        p_range, results, 
        'Transmit Power Limit P_max (W)', 
        'Average PSNR (dB)',
        'Impact of Power Budget on Video Quality',
        os.path.join(RESULTS_DIR, 'tradeoff_psnr_vs_power.png')
    )

def run_batch_kappa():
    """Phân tích Fairness theo hệ số Kappa"""
    print("\n" + "="*50)
    print("ANALYSIS B: FAIRNESS vs. KAPPA")
    print("="*50)
    
    np.random.seed(42)
    q_p, q_s1, q_s2 = generate_trajectories()
    g_p, g_s1, g_s2 = compute_channels_gain(q_p, q_s1, q_s2)
    
    k_range = np.linspace(0.1, 1.2, 10)
    results = {
        'Fairness (Max-Min)': [],
        'Fairness (W-Sum)': []
    }

    import optimization.sca_optimizer as sca
    import optimization.qoe_fairness_model as qoe_model
    original_kappa_sca = sca.KAPPA
    original_kappa_qoe = qoe_model.KAPPA

    for k in k_range:
        print(f"  Testing KAPPA = {k:.2f}...")
        sca.KAPPA = k
        qoe_model.KAPPA = k
        
        res_w = run_sca(g_p, g_s1, g_s2, mode='wsum', scheme='rsma')
        res_m = run_sca(g_p, g_s1, g_s2, mode='maxmin', scheme='rsma')
        
        results['Fairness (W-Sum)'].append(np.mean(res_w['fair_hist']))
        results['Fairness (Max-Min)'].append(np.mean(res_m['fair_hist']))

    sca.KAPPA = original_kappa_sca
    qoe_model.KAPPA = original_kappa_qoe

    plot_tradeoff(
        k_range, results, 
        'Fairness Weight Parameter (Kappa)', 
        "Jain's Fairness Index",
        'Fairness Sensitivity Analysis',
        os.path.join(RESULTS_DIR, 'tradeoff_fairness_vs_kappa.png')
    )


def run_batch_scheme_comparison():
    """So sánh trực tiếp CR-RSMA và CR-NOMA trên cùng budget công suất."""
    print("\n" + "="*50)
    print("ANALYSIS C: CR-RSMA vs. CR-NOMA")
    print("="*50)

    np.random.seed(42)
    q_p, q_s1, q_s2 = generate_trajectories()
    g_p, g_s1, g_s2 = compute_channels_gain(q_p, q_s1, q_s2)

    p_range = np.linspace(0.1, 2.0, 10)
    qoe_results = {
        'Avg QoE SU (RSMA)': [],
        'Avg QoE SU (NOMA)': []
    }
    fairness_results = {
        'Fairness (RSMA)': [],
        'Fairness (NOMA)': []
    }

    import optimization.sca_optimizer as sca
    original_p_s = sca.P_S_MAX
    original_p_p = sca.P_P_MAX

    for p in p_range:
        print(f"  Comparing schemes at P_max = {p:.2f} W...")
        sca.P_S_MAX = p
        sca.P_P_MAX = p

        res_rsma = run_sca(g_p, g_s1, g_s2, mode='wsum', scheme='rsma')
        res_noma = run_sca(g_p, g_s1, g_s2, mode='wsum', scheme='noma')

        qoe_results['Avg QoE SU (RSMA)'].append((res_rsma['QoE_s1'].mean() + res_rsma['QoE_s2'].mean()) / 2)
        qoe_results['Avg QoE SU (NOMA)'].append((res_noma['QoE_s1'].mean() + res_noma['QoE_s2'].mean()) / 2)
        fairness_results['Fairness (RSMA)'].append(np.mean(res_rsma['fair_hist']))
        fairness_results['Fairness (NOMA)'].append(np.mean(res_noma['fair_hist']))

    sca.P_S_MAX = original_p_s
    sca.P_P_MAX = original_p_p

    plot_tradeoff(
        p_range, qoe_results,
        'Transmit Power Limit P_max (W)',
        'Average SU QoE',
        'CR-RSMA vs. CR-NOMA: SU QoE under Shared Power Budget',
        os.path.join(RESULTS_DIR, 'tradeoff_rsma_vs_noma_qoe.png')
    )
    plot_tradeoff(
        p_range, fairness_results,
        'Transmit Power Limit P_max (W)',
        "Jain's Fairness Index",
        'CR-RSMA vs. CR-NOMA: SU Fairness under Shared Power Budget',
        os.path.join(RESULTS_DIR, 'tradeoff_rsma_vs_noma_fairness.png')
    )

    # Hình tổng hợp tại một điểm vận hành điển hình để dùng trong báo cáo/luận văn.
    ref_p = 1.0
    print(f"  Generating scheme snapshot at reference P_max = {ref_p:.2f} W...")
    sca.P_S_MAX = ref_p
    sca.P_P_MAX = ref_p
    res_rsma_ref = run_sca(g_p, g_s1, g_s2, mode='wsum', scheme='rsma')
    res_noma_ref = run_sca(g_p, g_s1, g_s2, mode='wsum', scheme='noma')
    plot_scheme_results(
        res_rsma_ref,
        res_noma_ref,
        'CR-RSMA vs. CR-NOMA at Reference Power Budget',
        os.path.join(RESULTS_DIR, 'scheme_snapshot_rsma_vs_noma.png')
    )

    sca.P_S_MAX = original_p_s
    sca.P_P_MAX = original_p_p

if __name__ == "__main__":
    run_batch_power()
    run_batch_kappa()
    run_batch_scheme_comparison()
    print("\n[DONE] Batch analysis completed. Check results/ folder.")
