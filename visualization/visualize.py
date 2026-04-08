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
    """Áp dụng style chung cho tất cả axes"""
    ax.set_facecolor('#161b22')
    ax.tick_params(colors='white')
    ax.grid(True, color='#30363d', alpha=0.5)
    for sp in ax.spines.values():
        sp.set_color('#30363d')


def plot_results(res1, res2, save_path='results.png'):
    """
    Vẽ kết quả so sánh 2 bài toán:
    res1: Weighted-Sum Fairness
    res2: Max-Min Fairness
    """
    slots = np.arange(N_SLOTS)
    C1, C2, C3 = '#74c0fc', '#ff6b6b', '#69db7c'

    fig = plt.figure(figsize=(22, 18), facecolor='#0d1117')
    fig.suptitle(
        'CR-RSMA + UAV: QoE Fairness Optimization\n'
        'Weighted-Sum Fairness vs Weighted Max-Min Fairness',
        fontsize=15, color='white', fontweight='bold', y=0.98
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ---- Plot 1: Convergence ----
    ax = fig.add_subplot(gs[0, :2])
    style_ax(ax)
    ax.plot(res1['obj_hist'], 'o-', color=C1, lw=2,
            ms=5, label='Weighted-Sum')
    ax.plot(res2['obj_hist'], 's-', color=C2, lw=2,
            ms=5, label='Max-Min')
    ax.set_title('SCA Convergence — Objective Function',
                 color='white', fontsize=12)
    ax.set_xlabel('Iteration', color='white')
    ax.set_ylabel('Objective value', color='white')
    ax.legend(facecolor='#161b22', labelcolor='white')

    # ---- Plot 2: Fairness ----
    ax = fig.add_subplot(gs[0, 2])
    style_ax(ax)
    ax.plot(res1['fair_hist'], 'o-', color=C1, lw=2,
            ms=5, label='Weighted-Sum')
    ax.plot(res2['fair_hist'], 's-', color=C2, lw=2,
            ms=5, label='Max-Min')
    ax.axhline(0.9, color=C3, ls='--', alpha=0.7,
               label='Threshold=0.9')
    ax.set_title("Jain's Fairness Index",
                 color='white', fontsize=12)
    ax.set_xlabel('Iteration', color='white')
    ax.set_ylabel('Fairness F', color='white')
    ax.set_ylim(0, 1.1)
    ax.legend(facecolor='#161b22', labelcolor='white', fontsize=8)

    # ---- Plot 3: PSNR PU ----
    ax = fig.add_subplot(gs[1, 0])
    style_ax(ax)
    ax.plot(slots, res1['PSNR_p'], '-',  color=C1, lw=2, label='W-Sum')
    ax.plot(slots, res2['PSNR_p'], '--', color=C2, lw=2, label='Max-Min')
    ax.set_title('PSNR PU theo Time Slots (dB)',
                 color='white', fontsize=12)
    ax.set_xlabel('Time slot', color='white')
    ax.set_ylabel('PSNR (dB)', color='white')
    ax.legend(facecolor='#161b22', labelcolor='white')

    # ---- Plot 4: PSNR SU1 & SU2 ----
    ax = fig.add_subplot(gs[1, 1])
    style_ax(ax)
    ax.plot(slots, res1['PSNR_s1'], '-',  color=C1, lw=2, label='SU1 (W-Sum)')
    ax.plot(slots, res2['PSNR_s1'], '--', color=C2, lw=2, label='SU1 (Max-Min)')
    ax.plot(slots, res1['PSNR_s2'], '-',  color=C3, lw=2, label='SU2 (W-Sum)')
    ax.plot(slots, res2['PSNR_s2'], '--', color='#69db7c', lw=2, alpha=0.6, label='SU2 (Max-Min)')
    ax.set_title('PSNR SU1 & SU2 theo Time Slots (dB)',
                 color='white', fontsize=12)
    ax.set_xlabel('Time slot', color='white')
    ax.set_ylabel('PSNR (dB)', color='white')
    ax.legend(facecolor='#161b22', labelcolor='white', fontsize=8)

    # ---- Plot 5: SVC Layers ----
    ax = fig.add_subplot(gs[1, 2])
    style_ax(ax)
    ax.step(slots, res1['k_p'], where='post', color=C1,
            lw=2, label='PU layers')
    ax.step(slots, res1['k_s1'], where='post', color=C3,
            lw=2, ls='--', label='SU1 (W-Sum)')
    ax.step(slots, res1['k_s2'], where='post', color='#ffa94d',
            lw=2, ls=':', label='SU2 (W-Sum)')
    ax.step(slots, res2['k_s1'], where='post', color=C2,
            lw=2, alpha=0.6, label='SU1 (Max-Min)')
    ax.set_title('SVC Layers L_k*', color='white', fontsize=12)
    ax.set_xlabel('Time slot', color='white')
    ax.set_ylabel('Số lớp', color='white')
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(['0', 'BL', 'BL+EL1', 'BL+EL1+EL2'],
                       color='white', fontsize=8)
    ax.legend(facecolor='#161b22', labelcolor='white', fontsize=7)

    # ---- Plot 6: Power Allocation ----
    ax = fig.add_subplot(gs[2, :2])
    style_ax(ax)
    ax.plot(slots, res1['P_c'],   '-',  color=C1,
            lw=2, label='P_c (Common)')
    ax.plot(slots, res1['P_p'],   '--', color='#ffa94d',
            lw=2, label='P_p (Private PU)')
    ax.plot(slots, res1['P_s1'],  '-',  color=C3,
            lw=2, label='P_s1 (Private SU1)')
    ax.plot(slots, res1['P_s2'],  '-',  color='#ff6b6b',
            lw=2, label='P_s2 (Private SU2)')
    ax.set_title('Power Allocation — Weighted-Sum (W)',
                 color='white', fontsize=12)
    ax.set_xlabel('Time slot', color='white')
    ax.set_ylabel('Power (W)', color='white')
    ax.legend(facecolor='#161b22', labelcolor='white', fontsize=8)

    # ---- Plot 7: Bảng so sánh ----
    ax = fig.add_subplot(gs[2, 2])
    ax.set_facecolor('#161b22')
    ax.axis('off')

    rows = ['PSNR_PU (dB)', 'PSNR_SU1 (dB)', 'PSNR_SU2 (dB)',
            'QoE_PU', 'QoE_SU1', 'QoE_SU2',
            'Fairness', 'Objective']
    data = [
        [f"{res1['PSNR_p'].mean():.1f}", f"{res2['PSNR_p'].mean():.1f}"],
        [f"{res1['PSNR_s1'].mean():.1f}", f"{res2['PSNR_s1'].mean():.1f}"],
        [f"{res1['PSNR_s2'].mean():.1f}", f"{res2['PSNR_s2'].mean():.1f}"],
        [f"{res1['QoE_p'].mean():.3f}",  f"{res2['QoE_p'].mean():.3f}"],
        [f"{res1['QoE_s1'].mean():.3f}", f"{res2['QoE_s1'].mean():.3f}"],
        [f"{res1['QoE_s2'].mean():.3f}", f"{res2['QoE_s2'].mean():.3f}"],
        [f"{np.mean(res1['fair_hist']):.3f}",
         f"{np.mean(res2['fair_hist']):.3f}"],
        [f"{res1['obj_hist'][-1]:.3f}",  f"{res2['obj_hist'][-1]:.3f}"],
    ]
    t = ax.table(cellText=data, rowLabels=rows,
                 colLabels=['W-Sum', 'Max-Min'],
                 cellLoc='center', loc='center',
                 bbox=Bbox.from_bounds(0, 0, 1, 1))
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    for (r, c), cell in t.get_celld().items():
        cell.set_facecolor('#388bfd' if r == 0 else
                           '#21262d' if c == -1 else '#161b22')
        cell.set_text_props(color='white',
                            fontweight='bold' if r == 0 else 'normal')
        cell.set_edgecolor('#30363d')
    ax.set_title('So sánh kết quả', color='white', fontsize=12)

    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor='#0d1117', edgecolor='none')
    plt.close()
    print(f"✓ Saved: {save_path}")


if __name__ == '__main__':
    print("visualize.py: import và dùng plot_results(res1, res2)")