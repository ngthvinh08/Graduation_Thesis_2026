"""
main.py — Main Entry Point with Real Video Input

Quy trình:
  1. Đọc video từ file (hoặc tạo test video)
  2. Xử lý từng frame với SVC encoder
  3. Chạy SCA optimization cho bitrate/power allocation
  4. Tính toán QoE & Fairness
  5. Visualize results

"""
import os
import sys
import numpy as np
from pathlib import Path

from config import *
from video.video_input import VideoReader, create_test_video
from video.svc_encoder import SVCEncoder, RDOOptimizer
from video.frame_processor import MacroblockAnalyzer
from optimization.uav_channel_model import compute_channels_gain, generate_trajectories
from optimization.sca_optimizer import run_sca
from optimization.qoe_fairness_model import compute_qoe
from visualization.visualize import plot_results, plot_scheme_results

# Thư mục lưu kết quả
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def create_or_get_video(video_path: str = None, create_if_missing: bool = False):
    """
    Tạo hoặc lấy video từ file.
    
    Args:
        video_path: Đường dẫn video (nếu None, dùng test video)
        create_if_missing: Tạo test video nếu file không tồn tại
    
    Returns:
        VideoReader object hoặc None
    """
    if video_path is None or not os.path.exists(video_path):
        if video_path is None:
            video_path = os.path.join(RESULTS_DIR, 'test_video.mp4')
        
        if create_if_missing:
            print(f"Creating test video: {video_path}")
            create_test_video(video_path, width=320, height=240, 
                            num_frames=N_SLOTS, fps=30)
            return VideoReader(video_path)
    else:
        return VideoReader(video_path)
    
    return None


def encode_video_frames(video_reader: VideoReader, num_frames: int = None,
                       target_bitrate_per_layer: np.ndarray = None) -> dict:
    """
    Mã hóa video frames bằng SVC encoder.
    
    Args:
        video_reader: VideoReader object
        num_frames: Số frame cần encode (default = tất cả)
        target_bitrate_per_layer: Target bitrate mỗi layer (optional)
    
    Returns:
        encoding_results: Dict chứa encoding info cho từng frame
    """
    if num_frames is None:
        num_frames = min(N_SLOTS, video_reader.total_frames)
    
    encoder = SVCEncoder(num_layers=L_MAX)
    rdo_optimizer = RDOOptimizer(encoder)
    
    # Đọc frames
    print(f"\nReading {num_frames} frames from video...")
    frames = []
    for i in range(num_frames):
        ret, frame = video_reader.read_frame()
        if not ret:
            break
        frames.append(frame)
    
    if not frames:
        print("[ERROR] Không đọc được frame nào từ video.")
        return {'frames': [], 'encoding_results': [], 'bitrate_per_layer': np.array([]), 'psnr_per_layer': np.array([])}

    print(f"[OK] Read {len(frames)} frames (size: {frames[0].shape})")
    
    # Mã hóa frames
    print(f"\nEncoding {len(frames)} frames...")
    encoding_results = []
    
    for i, frame in enumerate(frames):
        if i % max(1, len(frames)//10) == 0:
            print(f"  Frame {i}/{len(frames)}")
        
        frame_prev = frames[i-1] if i > 0 else None
        
        # Encode frame này
        result = encoder.encode_frame(frame, frame_prev=frame_prev)
        encoding_results.append(result)
    
    # Aggregate results
    bitrate_per_layer = np.array([r['bitrate_per_layer'] for r in encoding_results])
    psnr_per_layer = np.array([r['psnr_per_layer'] for r in encoding_results])
    
    print(f"[OK] Encoding complete")
    print(f"  Avg bitrate/layer: {np.mean(bitrate_per_layer, axis=0) / 1e3}")
    print(f"  Avg PSNR/layer: {np.mean(psnr_per_layer, axis=0)}")
    
    return {
        'frames': frames,
        'encoding_results': encoding_results,
        'bitrate_per_layer': bitrate_per_layer,
        'psnr_per_layer': psnr_per_layer,
        'alpha_frames': [r['alpha_per_layer'][L_MAX-1] for r in encoding_results],
        'gamma_frames': [r['gamma_per_layer'][L_MAX-1] for r in encoding_results],
        'avg_bitrate': np.mean(bitrate_per_layer.sum(axis=1)),
        'avg_psnr': np.mean(psnr_per_layer),
    }


def compute_video_qoe(encoding_results: dict, g_p: np.ndarray, 
                    g_s1: np.ndarray, g_s2: np.ndarray) -> dict:
    """
    Tính QoE cho video streaming.
    
    Input từ SVC encoder:
      - Bitrate per layer
      - PSNR per layer
    
    Output: QoE metrics cho PU và SUs
    """
    from optimization.qoe_fairness_model import compute_qoe
    
    num_frames = len(encoding_results['encoding_results'])
    
    # Aggregate bitrate/PSNR từng frame thành N_SLOTS
    # (Combine multiple frames thành một slot nếu cần)
    psnr_p = np.zeros(num_frames)
    psnr_s1 = np.zeros(num_frames)
    psnr_s2 = np.zeros(num_frames)
    
    # Simulate: PU nhận full quality, SUs nhận degraded quality
    for n in range(num_frames):
        # PU: nhận base layer
        psnr_p[n] = encoding_results['psnr_per_layer'][n, 0]
        
        # SU1, SU2: nhận base + EL1
        psnr_s1[n] = np.mean(encoding_results['psnr_per_layer'][n, :2])
        psnr_s2[n] = np.mean(encoding_results['psnr_per_layer'][n, :2])
    
    # Normalize PSNR to [0, 51]
    psnr_p = np.clip(psnr_p, 0, 51)
    psnr_s1 = np.clip(psnr_s1, 0, 51)
    psnr_s2 = np.clip(psnr_s2, 0, 51)
    
    # Estimate delay (simple model)
    delay_p = 0.03 * np.ones(num_frames)  # 30ms
    delay_s1 = 0.05 * np.ones(num_frames)  # 50ms
    delay_s2 = 0.05 * np.ones(num_frames)  # 50ms
    
    # Power consumption (from UAV model)
    power_p = P_P_MAX / 2 * np.ones(num_frames)
    power_s1 = P_S_MAX / 2 * np.ones(num_frames)
    power_s2 = P_S_MAX / 2 * np.ones(num_frames)
    
    # Compute QoE
    QoE_p = compute_qoe(psnr_p, power_p, T_delay=delay_p)
    QoE_s1 = compute_qoe(psnr_s1, power_s1, T_delay=delay_s1)
    QoE_s2 = compute_qoe(psnr_s2, power_s2, T_delay=delay_s2)
    
    return {
        'PSNR_p': psnr_p,
        'PSNR_s1': psnr_s1,
        'PSNR_s2': psnr_s2,
        'QoE_p': QoE_p,
        'QoE_s1': QoE_s1,
        'QoE_s2': QoE_s2,
        'delay_p': delay_p,
        'delay_s1': delay_s1,
        'delay_s2': delay_s2,
    }


def main():
    """Main entry point."""
    print("=" * 70)
    print("CR-RSMA + UAV + QoE Fairness | Video Input")
    print("=" * 70)
    print(f"N_SLOTS = {N_SLOTS} | L_MAX = {L_MAX} SVC layers")
    print(f"K_SU = {K_SU} | P_P_MAX = {P_P_MAX}W | P_S_MAX = {P_S_MAX}W")
    print(f"Fairness: SU1 vs SU2 | KAPPA = {KAPPA}")
    
    # ========== PHASE 1: Video Processing ==========
    print("\n" + "=" * 70)
    print("PHASE 1: VIDEO PROCESSING")
    print("=" * 70)
    
    # Get video (Sử dụng đường dẫn tương đối để tăng tính di động)
    video_file = os.path.join(BASE_DIR, 'video', 'sample-5s.mp4')
    video_reader = create_or_get_video(
        video_path=video_file,
        create_if_missing=True
    )
    
    if video_reader is None:
        print("[ERROR] Không thể khởi tạo VideoReader. Thoát...")
        return

    # Encode video
    encoding_data = encode_video_frames(video_reader, num_frames=N_SLOTS)
    
    if not encoding_data['encoding_results']:
        print("[ERROR] Không có dữ liệu encoding. Thoát...")
        video_reader.close()
        return

    # Close video
    video_reader.close()
    
    # ========== PHASE 2: Channel Setup ==========
    print("\n" + "=" * 70)
    print("PHASE 2: CHANNEL SETUP")
    print("=" * 70)
    
    np.random.seed(42)  # For reproducibility
    q_p, q_s1, q_s2 = generate_trajectories()
    g_p, g_s1, g_s2 = compute_channels_gain(q_p, q_s1, q_s2)
    
    print(f"\nChannel gains (mean):")
    print(f"  g_p  = {g_p.mean():.4f}")
    print(f"  g_s1 = {g_s1.mean():.4f}  (SU1)")
    print(f"  g_s2 = {g_s2.mean():.4f}  (SU2)")
    
    # ========== PHASE 3: QoE & Video Metrics ==========
    print("\n" + "=" * 70)
    print("PHASE 3: QoE COMPUTATION")
    print("=" * 70)
    
    qoe_data = compute_video_qoe(encoding_data, g_p, g_s1, g_s2)
    
    alpha_frames = encoding_data['alpha_frames']
    gamma_frames = encoding_data['gamma_frames']
    
    print(f"\nQoE Summary (from video):")
    print(f"  PSNR_p: {qoe_data['PSNR_p'].mean():.2f} dB")
    print(f"  PSNR_s1: {qoe_data['PSNR_s1'].mean():.2f} dB")
    print(f"  PSNR_s2: {qoe_data['PSNR_s2'].mean():.2f} dB")
    print(f"  QoE_p: {qoe_data['QoE_p'].mean():.2f}")
    print(f"  QoE_s1: {qoe_data['QoE_s1'].mean():.2f}")
    print(f"  QoE_s2: {qoe_data['QoE_s2'].mean():.2f}")
    
    # ========== PHASE 4: SCA Optimization ==========
    print("\n" + "=" * 70)
    print("PHASE 4: SCA OPTIMIZATION")
    print("=" * 70)
    
    # Create pseudo-PSNRs for SCA (use video PSNRs)
    psnr_p_sca = qoe_data['PSNR_p']
    psnr_s1_sca = qoe_data['PSNR_s1']
    psnr_s2_sca = qoe_data['PSNR_s2']
    
    res_wsum = run_sca(g_p, g_s1, g_s2, alpha_frames, gamma_frames, mode='wsum', scheme='rsma')
    res_maxmin = run_sca(g_p, g_s1, g_s2, alpha_frames, gamma_frames, mode='maxmin', scheme='rsma')
    res_rsma_scheme = run_sca(g_p, g_s1, g_s2, alpha_frames, gamma_frames, mode='wsum', scheme='rsma')
    res_noma_scheme = run_sca(g_p, g_s1, g_s2, alpha_frames, gamma_frames, mode='wsum', scheme='noma')
    
    # ========== PHASE 5: Results Summary ==========
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<25} {'W-Sum':>15} {'Max-Min':>15}")
    print("-" * 55)
    
    metrics = [
        ('PSNR_PU (dB)', res_wsum['PSNR_p'].mean(), res_maxmin['PSNR_p'].mean()),
        ('PSNR_SU1 (dB)', res_wsum['PSNR_s1'].mean(), res_maxmin['PSNR_s1'].mean()),
        ('PSNR_SU2 (dB)', res_wsum['PSNR_s2'].mean(), res_maxmin['PSNR_s2'].mean()),
        ('QoE_SU1', res_wsum['QoE_s1'].mean(), res_maxmin['QoE_s1'].mean()),
        ('QoE_SU2', res_wsum['QoE_s2'].mean(), res_maxmin['QoE_s2'].mean()),
        ('Fairness(SU)', np.mean(res_wsum['fair_hist']), np.mean(res_maxmin['fair_hist'])),
        ('Objective', res_wsum['obj_hist'][-1], res_maxmin['obj_hist'][-1]),
    ]
    
    for name, v1, v2 in metrics:
        print(f"{name:<25} {v1:>15.3f} {v2:>15.3f}")

    print("\n" + "=" * 70)
    print("SCHEME COMPARISON (W-SUM OBJECTIVE)")
    print("=" * 70)
    print(f"{'Metric':<25} {'CR-RSMA':>15} {'CR-NOMA':>15}")
    print("-" * 55)

    scheme_metrics = [
        ('PSNR_PU (dB)', res_rsma_scheme['PSNR_p'].mean(), res_noma_scheme['PSNR_p'].mean()),
        ('PSNR_SU_avg (dB)',
         (res_rsma_scheme['PSNR_s1'].mean() + res_rsma_scheme['PSNR_s2'].mean()) / 2,
         (res_noma_scheme['PSNR_s1'].mean() + res_noma_scheme['PSNR_s2'].mean()) / 2),
        ('QoE_PU', res_rsma_scheme['QoE_p'].mean(), res_noma_scheme['QoE_p'].mean()),
        ('QoE_SU_avg',
         (res_rsma_scheme['QoE_s1'].mean() + res_rsma_scheme['QoE_s2'].mean()) / 2,
         (res_noma_scheme['QoE_s1'].mean() + res_noma_scheme['QoE_s2'].mean()) / 2),
        ('Fairness(SU)', np.mean(res_rsma_scheme['fair_hist']), np.mean(res_noma_scheme['fair_hist'])),
        ('Objective', res_rsma_scheme['obj_hist'][-1], res_noma_scheme['obj_hist'][-1]),
    ]

    for name, v1, v2 in scheme_metrics:
        print(f"{name:<25} {v1:>15.3f} {v2:>15.3f}")
    
    # ========== PHASE 6: Visualization ==========
    print("\n[Generating plots...]")
    save_path = os.path.join(RESULTS_DIR, 'cr_rsma_video_1pu_2su.png')
    plot_results(res_wsum, res_maxmin, save_path=save_path)
    scheme_save_path = os.path.join(RESULTS_DIR, 'cr_rsma_vs_noma_video_1pu_2su.png')
    plot_scheme_results(
        res_rsma_scheme,
        res_noma_scheme,
        'CR-RSMA vs. CR-NOMA for Video Uplink QoE',
        scheme_save_path
    )
    
    # Save video encoding stats
    video_stats_path = os.path.join(RESULTS_DIR, 'video_encoding_stats.txt')
    with open(video_stats_path, 'w') as f:
        f.write("VIDEO ENCODING STATISTICS\n")
        f.write("=" * 50 + "\n")
        f.write(f"Frames processed: {len(encoding_data['frames'])}\n")
        f.write(f"Average bitrate: {encoding_data['avg_bitrate']/1e3:.1f} kbps\n")
        f.write(f"Average PSNR: {encoding_data['avg_psnr']:.2f} dB\n")
        f.write(f"\nBitrate per layer (avg):\n")
        for l in range(L_MAX):
            br = np.mean(encoding_data['bitrate_per_layer'][:, l])
            f.write(f"  Layer {l}: {br/1e3:.1f} kbps\n")
        f.write(f"\nPSNR per layer (avg):\n")
        for l in range(L_MAX):
            psnr = np.mean(encoding_data['psnr_per_layer'][:, l])
            f.write(f"  Layer {l}: {psnr:.2f} dB\n")
    
    # ========== PHASE 7: Summary Table ==========
    summary_md = f"""
### Kết quả mô phỏng (KAPPA = {KAPPA}):

| Chỉ số | Weighted-Sum (Hiệu suất) | Max-Min (Công bằng) |
| :--- | :--- | :--- |
| **PSNR PU** | {res_wsum['PSNR_p'].mean():.3f} dB | {res_maxmin['PSNR_p'].mean():.3f} dB |
| **PSNR SU1** | {res_wsum['PSNR_s1'].mean():.3f} dB | {res_maxmin['PSNR_s1'].mean():.3f} dB |
| **PSNR SU2** | {res_wsum['PSNR_s2'].mean():.3f} dB | {res_maxmin['PSNR_s2'].mean():.3f} dB |
| **Jain's Fairness Index** | {np.mean(res_wsum['fair_hist']):.3f} | {np.mean(res_maxmin['fair_hist']):.3f} |

### So sánh cơ chế truy nhập (W-Sum):

| Chỉ số | CR-RSMA | CR-NOMA |
| :--- | :--- | :--- |
| **PSNR PU** | {res_rsma_scheme['PSNR_p'].mean():.3f} dB | {res_noma_scheme['PSNR_p'].mean():.3f} dB |
| **PSNR SU trung bình** | {((res_rsma_scheme['PSNR_s1'].mean() + res_rsma_scheme['PSNR_s2'].mean()) / 2):.3f} dB | {((res_noma_scheme['PSNR_s1'].mean() + res_noma_scheme['PSNR_s2'].mean()) / 2):.3f} dB |
| **QoE PU** | {res_rsma_scheme['QoE_p'].mean():.3f} | {res_noma_scheme['QoE_p'].mean():.3f} |
| **QoE SU trung bình** | {((res_rsma_scheme['QoE_s1'].mean() + res_rsma_scheme['QoE_s2'].mean()) / 2):.3f} | {((res_noma_scheme['QoE_s1'].mean() + res_noma_scheme['QoE_s2'].mean()) / 2):.3f} |
| **Jain's Fairness Index** | {np.mean(res_rsma_scheme['fair_hist']):.3f} | {np.mean(res_noma_scheme['fair_hist']):.3f} |
"""
    print(summary_md)
    with open(os.path.join(RESULTS_DIR, 'last_summary.md'), 'w', encoding='utf-8') as f:
        f.write(summary_md)

    print(f"\n✓ Hoàn thành!")
    print(f"  Plot saved: {save_path}")
    print(f"  Scheme plot: {scheme_save_path}")
    print(f"  Video stats: {video_stats_path}")
    print(f"  Summary MD: {os.path.join(RESULTS_DIR, 'last_summary.md')}")


if __name__ == '__main__':
    main()
