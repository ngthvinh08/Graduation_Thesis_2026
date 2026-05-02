"""
visualize.py — Vẽ kết quả
Input: dict kết quả từ optimizer.run_sca()
Output: file PNG chứa 7 biểu đồ
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.transforms import Bbox
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import N_SLOTS


def style_ax(ax):
    """Áp dụng style học thuật (Academic Style)"""
    ax.set_facecolor('white')
    ax.tick_params(colors='black', labelsize=10)
    ax.grid(True, color='gray', linestyle='--', alpha=0.3)
    for sp in ax.spines.values():
        sp.set_color('black')
        sp.set_linewidth(1.0)
    ax.set_title(ax.get_title(), color='black', fontsize=12, fontweight='bold')

def plot_results(res1, res2, save_path='results.png'):
    """
    Vẽ kết quả so sánh 2 bài toán (Academic Style):
    res1: Weighted-Sum Fairness
    res2: Max-Min Fairness
    """
    slots = np.arange(N_SLOTS)
    C1, C2, C3 = '#1B4F72', '#B03A2E', '#1E8449'
    
    fig = plt.figure(figsize=(20, 16), facecolor='white')
    fig.suptitle(
        'Uplink RSMA + UAV: QoE Fairness Optimization Analysis\n'
        'Weighted-Sum (Efficiency) vs Weighted Max-Min (Fairness)',
        fontsize=16, color='black', fontweight='bold', y=0.98
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)

    # ---- Plot 1: Convergence ----
    ax = fig.add_subplot(gs[0, :2])
    ax.plot(res1['obj_hist'], 'o-', color=C1, lw=1.5, ms=4, label='W-Sum')
    ax.plot(res2['obj_hist'], 's-', color=C2, lw=1.5, ms=4, label='Max-Min')
    ax.set_title('Algorithm Convergence Profile')
    ax.set_xlabel('Iteration Index')
    ax.set_ylabel('Objective Value')
    style_ax(ax)
    ax.legend(loc='best', fontsize=10)

    # ---- Plot 2: Fairness ----
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(res1['fair_hist'], 'o-', color=C1, lw=1.5, ms=4, label='W-Sum')
    ax.plot(res2['fair_hist'], 's-', color=C2, lw=1.5, ms=4, label='Max-Min')
    ax.axhline(0.9, color='darkorange', ls='--', alpha=0.8, label='Target Fair')
    ax.set_title("Jain's Fairness Index Progress")
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Fairness F')
    ax.set_ylim(0, 1.05)
    style_ax(ax)
    ax.legend(loc='lower right', fontsize=9)

    # ---- Plot 3: PSNR PU ----
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(slots, res1['PSNR_p'], '-',  color=C1, lw=1.5, marker='o', markevery=10, label='W-Sum')
    ax.plot(slots, res2['PSNR_p'], '--', color=C2, lw=1.5, marker='s', markevery=10, label='Max-Min')
    ax.set_title('Primary User (PU) PSNR (dB)')
    ax.set_xlabel('Time Slot (n)')
    ax.set_ylabel('PSNR (dB)')
    style_ax(ax)
    ax.legend(loc='best', fontsize=9)

    # ---- Plot 4: PSNR SUs ----
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(slots, res1['PSNR_s1'], '-',  color=C1, lw=1.2, label='SU1 (W-Sum)')
    ax.plot(slots, res2['PSNR_s1'], '--', color=C2, lw=1.2, label='SU1 (Max-Min)')
    ax.plot(slots, res1['PSNR_s2'], '-',  color=C3, lw=1.2, label='SU2 (W-Sum)')
    ax.plot(slots, res2['PSNR_s2'], ':',  color='purple', lw=1.2, label='SU2 (Max-Min)')
    ax.set_title('Secondary Users (SU) PSNR')
    ax.set_xlabel('Time Slot')
    ax.set_ylabel('PSNR (dB)')
    style_ax(ax)
    ax.legend(loc='best', fontsize=8, ncol=2)

    # ---- Plot 5: SVC Layers ----
    ax = fig.add_subplot(gs[1, 2])
    ax.step(slots, res1['k_p'], where='post', color=C1, lw=2, label='PU')
    ax.step(slots, res1['k_s1'], where='post', color=C3, lw=1.5, ls='--', label='SU1')
    ax.step(slots, res1['k_s2'], where='post', color='darkorange', lw=1.5, ls=':', label='SU2')
    ax.set_title('SVC Layer Allocation (L_k*)')
    ax.set_xlabel('Time Slot')
    ax.set_ylabel('Layer Count')
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(['None', 'BL', 'BL+EL1', 'BL+EL1+EL2'])
    style_ax(ax)
    ax.legend(loc='best', fontsize=8)

    # ---- Plot 6: Power Allocation ----
    ax = fig.add_subplot(gs[2, :2])
    ax.plot(slots, res1['P_s1c'], '-',  color=C1, lw=1.5, label='P_s1,c (Common)')
    ax.plot(slots, res1['P_s1p'], '--', color=C1, lw=1.2, alpha=0.6, label='P_s1,p (Private)')
    ax.plot(slots, res1['P_s2c'], '-',  color=C3, lw=1.5, label='P_s2,c (Common)')
    ax.plot(slots, res1['P_s2p'], '--', color=C3, lw=1.2, alpha=0.6, label='P_s2,p (Private)')
    ax.plot(slots, res1['P_pu'],  '-',  color=C2, lw=2, label='P_pu (Primary)')
    ax.set_title('Resource Scaling: Power Allocation (W-Sum Mode)')
    ax.set_xlabel('Time Slot')
    ax.set_ylabel('Power (Watts)')
    style_ax(ax)
    ax.legend(loc='upper right', ncol=3, fontsize=9)

    # ---- Plot 7: Statistics Table ----
    ax = fig.add_subplot(gs[2, 2])
    ax.axis('off')
    rows = ['Avg PSNR PU (dB)', 'Avg PSNR SU1 (dB)', 'Avg PSNR SU2 (dB)',
            'Avg QoE PU', 'Avg QoE SU1', 'Avg QoE SU2',
            'Final Fairness', 'Final Objective']
    data = [
        [f"{res1['PSNR_p'].mean():.2f}", f"{res2['PSNR_p'].mean():.2f}"],
        [f"{res1['PSNR_s1'].mean():.2f}", f"{res2['PSNR_s1'].mean():.2f}"],
        [f"{res1['PSNR_s2'].mean():.2f}", f"{res2['PSNR_s2'].mean():.2f}"],
        [f"{res1['QoE_p'].mean():.3f}",  f"{res2['QoE_p'].mean():.3f}"],
        [f"{res1['QoE_s1'].mean():.3f}", f"{res2['QoE_s1'].mean():.3f}"],
        [f"{res1['QoE_s2'].mean():.3f}", f"{res2['QoE_s2'].mean():.3f}"],
        [f"{np.mean(res1['fair_hist']):.3f}", f"{np.mean(res2['fair_hist']):.3f}"],
        [f"{res1['obj_hist'][-1]:.2f}",  f"{res2['obj_hist'][-1]:.2f}"],
    ]
    t = ax.table(cellText=data, rowLabels=rows, colLabels=['W-Sum', 'Max-Min'],
                 cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    t.auto_set_font_size(False)
    t.set_fontsize(10)
    for (r, c), cell in t.get_celld().items():
        if r == 0: cell.set_facecolor('#D5D8DC')
        elif c == -1: cell.set_facecolor('#F2F3F4')
        cell.set_edgecolor('black')
    ax.set_title('Performance Comparison', color='black', fontsize=12, fontweight='bold')

    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Results saved to {save_path}")

def plot_tradeoff(x_data, y_series, x_label, y_label, title, save_path):
    """Vẽ biểu đồ Trade-off chuyên nghiệp (Academic Style)"""
    markers = ['o', 's', 'D', '^', 'v', 'p']
    linestyles = ['-', '--', ':', '-.']
    colors = ['#1B4F72', '#B03A2E', '#1E8449', '#D35400', '#8E44AD']

    plt.figure(figsize=(10, 8), facecolor='white')
    ax = plt.gca()
    
    for i, (label, data) in enumerate(y_series.items()):
        plt.plot(x_data, data, 
                 marker=markers[i % len(markers)], 
                 linestyle=linestyles[i % len(linestyles)],
                 color=colors[i % len(colors)],
                 label=label, lw=2, ms=7)

    plt.xlabel(x_label, fontsize=12, fontweight='bold')
    plt.ylabel(y_label, fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    
    style_ax(ax)
    plt.legend(loc='best', fontsize=11, frameon=True, shadow=True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, facecolor='white')
    plt.close()
    print(f"✓ Trade-off plot saved to {save_path}")
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_scheme_results(res_rsma, res_noma, title, save_path):
    """So sánh trực tiếp CR-RSMA và CR-NOMA trên các chỉ số chính."""
    slots = np.arange(N_SLOTS)
    c_rsma, c_noma = '#1B4F72', '#B03A2E'
    c_aux1, c_aux2 = '#1E8449', '#D35400'

    fig = plt.figure(figsize=(18, 12), facecolor='white')
    fig.suptitle(title, fontsize=16, color='black', fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25)

    # ---- Plot 1: Convergence ----
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(res_rsma['obj_hist'], 'o-', color=c_rsma, lw=1.5, ms=4, label='CR-RSMA')
    ax.plot(res_noma['obj_hist'], 's-', color=c_noma, lw=1.5, ms=4, label='CR-NOMA')
    ax.set_title('Objective Convergence')
    ax.set_xlabel('Iteration Index')
    ax.set_ylabel('Objective Value')
    style_ax(ax)
    ax.legend(loc='best', fontsize=10)

    # ---- Plot 2: Fairness ----
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(res_rsma['fair_hist'], 'o-', color=c_rsma, lw=1.5, ms=4, label='CR-RSMA')
    ax.plot(res_noma['fair_hist'], 's-', color=c_noma, lw=1.5, ms=4, label='CR-NOMA')
    ax.set_title("SU Jain Fairness Progress")
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Fairness F')
    ax.set_ylim(0, 1.05)
    style_ax(ax)
    ax.legend(loc='lower right', fontsize=10)

    # ---- Plot 3: QoE of SUs ----
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(slots, res_rsma['QoE_s1'], '-', color=c_rsma, lw=1.2, label='SU1 (RSMA)')
    ax.plot(slots, res_rsma['QoE_s2'], '--', color=c_aux1, lw=1.2, label='SU2 (RSMA)')
    ax.plot(slots, res_noma['QoE_s1'], '-', color=c_noma, lw=1.2, label='SU1 (NOMA)')
    ax.plot(slots, res_noma['QoE_s2'], '--', color=c_aux2, lw=1.2, label='SU2 (NOMA)')
    ax.set_title('Secondary User QoE')
    ax.set_xlabel('Time Slot')
    ax.set_ylabel('QoE Score')
    style_ax(ax)
    ax.legend(loc='best', fontsize=8, ncol=2)

    # ---- Plot 4: Summary Table ----
    ax = fig.add_subplot(gs[1, 1])
    ax.axis('off')
    rows = [
        'Avg PSNR PU (dB)', 'Avg PSNR SU (dB)', 'Avg QoE PU', 'Avg QoE SU',
        'Final Fairness', 'Final Objective'
    ]
    avg_psnr_su_rsma = (res_rsma['PSNR_s1'].mean() + res_rsma['PSNR_s2'].mean()) / 2
    avg_psnr_su_noma = (res_noma['PSNR_s1'].mean() + res_noma['PSNR_s2'].mean()) / 2
    avg_qoe_su_rsma = (res_rsma['QoE_s1'].mean() + res_rsma['QoE_s2'].mean()) / 2
    avg_qoe_su_noma = (res_noma['QoE_s1'].mean() + res_noma['QoE_s2'].mean()) / 2
    data = [
        [f"{res_rsma['PSNR_p'].mean():.2f}", f"{res_noma['PSNR_p'].mean():.2f}"],
        [f"{avg_psnr_su_rsma:.2f}", f"{avg_psnr_su_noma:.2f}"],
        [f"{res_rsma['QoE_p'].mean():.3f}", f"{res_noma['QoE_p'].mean():.3f}"],
        [f"{avg_qoe_su_rsma:.3f}", f"{avg_qoe_su_noma:.3f}"],
        [f"{np.mean(res_rsma['fair_hist']):.3f}", f"{np.mean(res_noma['fair_hist']):.3f}"],
        [f"{res_rsma['obj_hist'][-1]:.2f}", f"{res_noma['obj_hist'][-1]:.2f}"],
    ]
    t = ax.table(
        cellText=data,
        rowLabels=rows,
        colLabels=['CR-RSMA', 'CR-NOMA'],
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    t.auto_set_font_size(False)
    t.set_fontsize(10)
    for (r, c), cell in t.get_celld().items():
        if r == 0:
            cell.set_facecolor('#D5D8DC')
        elif c == -1:
            cell.set_facecolor('#F2F3F4')
        cell.set_edgecolor('black')
    ax.set_title('Scheme-Level Comparison', color='black', fontsize=12, fontweight='bold')

    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Scheme comparison saved to {save_path}")


if __name__ == '__main__':
    print("visualize.py: import và dùng plot_results(res1, res2)")
