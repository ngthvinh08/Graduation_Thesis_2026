"""
svc_encoder.py — SVC (Scalable Video Coding) Encoder with RDO

Chức năng:
  1. Encode frame thành multi-layer SVC
  2. Tính bitrate & PSNR cho từng layer
  3. Tính bitrate-distortion tradeoff (RDO)
  4. Hỗ trợ dynamic QP assignment

SVC Layer structure:
  - Base Layer (L=0): Low resolution, mandatory
  - Enhancement Layer 1,2: Higher resolution/quality (optional)
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from .frame_processor import (
    MacroblockAnalyzer, compute_psnr, compute_rdo_parameters,
    predict_bitrate, predict_distortion, qp_to_psnr_estimate
)
from config import L_MAX, R_LAYER, ALPHA_RD, GAMMA_RD


class SVCEncoder:
    """
    SVC Encoder: Mã hóa video thành multiple layers.
    
    Attributes:
        num_layers: Số lớp SVC (default = 3: BL + 2 ELs)
        layer_bitrates: Target bitrate cho mỗi layer
        analyzer: MacroblockAnalyzer instance
    """
    
    def __init__(self, num_layers: int = L_MAX):
        """
        Khởi tạo SVCEncoder.
        
        Args:
            num_layers: Số lượng layer (default = 3)
        """
        self.num_layers = num_layers
        self.layer_bitrates = R_LAYER[:num_layers].copy()
        self.analyzer = MacroblockAnalyzer()
        print(self.analyzer)

        print(f"[OK] SVCEncoder initialized: {num_layers} layers")
        for l in range(num_layers):
            print(f"  Layer {l}: {self.layer_bitrates[l]/1e3:.0f} kbps (target)")
    
    def encode_frame(self, frame: np.ndarray, frame_prev: np.ndarray = None,
                     QP_per_layer: np.ndarray = None) -> dict:
        """
        Encode một frame thành SVC layers.
        
        Args:
            frame: Input frame (H, W, 3) RGB
            frame_prev: Previous frame (H, W, 3) để tính motion (optional)
            QP_per_layer: QP parameters (num_layers,) hoặc None (auto)
        
        Returns:
            result: Dict chứa
                - bitrate_per_layer: [BR_L0, BR_L1, BR_L2]
                - psnr_per_layer: [PSNR_L0, PSNR_L1, PSNR_L2]
                - encoded_layers: Encoded bitstream (mô phỏng)
                - QP_per_layer: QP sử dụng
                - texture_map: Texture complexity map
                - motion_map: Motion map (nếu có frame_prev)
        """
        # astype là để chuyển frame sang uint8
        frame = frame.astype(np.uint8)
        
        # Phân tích macroblocks
        mbs, props = self.analyzer.get_macroblocks(frame)
        texture = props['texture']
        
        # Tính motion nếu có frame trước
        motion = None
        if frame_prev is not None:
            motion = self.analyzer.estimate_motion(frame_prev, frame)
        
        # Nếu không có QP, tìm QP tối ưu cho mỗi layer
        if QP_per_layer is None:
            QP_per_layer = self._find_optimal_qp(texture, motion)
        
        # Encode từng layer
        bitrate_per_layer = []
        psnr_per_layer = []
        alpha_per_layer = []
        gamma_per_layer = []
        
        for layer in range(self.num_layers):
            # RDO parameters cho layer này
            alpha, gamma = compute_rdo_parameters(texture, motion, layer=layer)
            alpha_per_layer.append(alpha)
            gamma_per_layer.append(gamma)
            
            # QP cho layer này (có thể là scalar hoặc array)
            if np.isscalar(QP_per_layer[layer]):
                QP_layer = np.ones_like(texture) * QP_per_layer[layer]
            else:
                QP_layer = QP_per_layer[layer]
            
            # Predict bitrate & distortion
            bitrate = predict_bitrate(QP_layer, alpha, layer=layer)
            
            # PSNR estimation: tính từ từng MB rồi average
            psnr_per_mb = np.zeros_like(texture, dtype=np.float32)
            for i in range(texture.shape[0]):
                for j in range(texture.shape[1]):
                    psnr_per_mb[i, j] = qp_to_psnr_estimate(
                        QP_layer[i, j], gamma[i, j]
                    )
            psnr = np.mean(psnr_per_mb)
            
            bitrate_per_layer.append(bitrate)
            psnr_per_layer.append(psnr)
        
        result = {
            'bitrate_per_layer': np.array(bitrate_per_layer),
            'psnr_per_layer': np.array(psnr_per_layer),
            'alpha_per_layer': alpha_per_layer,
            'gamma_per_layer': gamma_per_layer,
            'QP_per_layer': QP_per_layer,
            'texture_map': texture,
            'motion_map': motion,
            'total_bitrate': np.sum(bitrate_per_layer),
            'avg_psnr': np.mean(psnr_per_layer),
        }
        
        return result
    
    def _find_optimal_qp(self, texture: np.ndarray, motion: np.ndarray = None,
                         max_iterations: int = 10) -> np.ndarray:
        """
        Tìm QP tối ưu cho mỗi layer để đạt target bitrate.
        
        Sử dụng binary search.
        
        Args:
            texture: Texture map
            motion: Motion map (optional)
            max_iterations: Số iteration của binary search
        
        Returns:
            QP_per_layer: Optimal QP (num_layers,)
        """
        QP_per_layer = np.zeros(self.num_layers)
        
        for layer in range(self.num_layers):
            target_br = self.layer_bitrates[layer]
            
            # Binary search: QP trong range [10, 51]
            qp_low, qp_high = 10, 51
            
            for _ in range(max_iterations):
                qp_mid = (qp_low + qp_high) / 2
                
                # Tính bitrate với QP này
                alpha, gamma = compute_rdo_parameters(texture, motion, layer=layer)
                QP_test = np.ones_like(texture) * qp_mid
                bitrate = predict_bitrate(QP_test, alpha, layer=layer)
                
                if bitrate > target_br:
                    # Bitrate quá cao → tăng QP (giảm bitrate)
                    qp_low = qp_mid
                else:
                    # Bitrate quá thấp → giảm QP (tăng bitrate)
                    qp_high = qp_mid
            
            QP_per_layer[layer] = (qp_low + qp_high) / 2
        
        return QP_per_layer
    
    def encode_stream(self, frames: list, frame_prev_list: list = None) -> dict:
        """
        Encode một sequence frames.
        
        Args:
            frames: List [frame1, frame2, ..., frame_n]
            frame_prev_list: List frame trước (optional, để tính motion)
        
        Returns:
            stream_result: Dict chứa
                - frame_results: List result từng frame
                - total_bitrate: Tổng bitrate (bps)
                - avg_psnr_per_layer: PSNR trung bình/layer
        """
        frame_results = []
        bitrate_per_layer_list = []
        psnr_per_layer_list = []
        
        for i, frame in enumerate(frames):
            frame_prev = None
            if frame_prev_list is not None and i > 0:
                frame_prev = frame_prev_list[i-1]
            elif i > 0:
                frame_prev = frames[i-1]
            
            result = self.encode_frame(frame, frame_prev=frame_prev)
            frame_results.append(result)
            
            bitrate_per_layer_list.append(result['bitrate_per_layer'])
            psnr_per_layer_list.append(result['psnr_per_layer'])
        
        # Aggregate results
        bitrate_per_layer_list = np.array(bitrate_per_layer_list)
        psnr_per_layer_list = np.array(psnr_per_layer_list)
        
        stream_result = {
            'frame_results': frame_results,
            'total_bitrate': np.sum(bitrate_per_layer_list),
            'total_frames': len(frames),
            'bitrate_per_frame': bitrate_per_layer_list.sum(axis=1),
            'avg_bitrate': np.mean(bitrate_per_layer_list.sum(axis=1)),
            'avg_psnr_per_layer': np.mean(psnr_per_layer_list, axis=0),
            'avg_psnr': np.mean(psnr_per_layer_list),
        }
        
        return stream_result


class RDOOptimizer:
    """
    RDO (Rate-Distortion Optimization) Optimizer.
    
    Tìm QP tối ưu để tối đa hóa quality (minimize distortion)
    trong ràng buộc bitrate budget.
    """
    
    def __init__(self, encoder: SVCEncoder):
        """
        Khởi tạo RDOOptimizer.
        
        Args:
            encoder: SVCEncoder instance
        """
        self.encoder = encoder
    
    def optimize_qp_for_bitrate_target(self, frame: np.ndarray,
                                       target_bitrate: float,
                                       frame_prev: np.ndarray = None) -> dict:
        """
        Optimize QP để đạt target bitrate với max quality.
        
        Args:
            frame: Input frame
            target_bitrate: Target bitrate (bps)
            frame_prev: Previous frame (optional)
        
        Returns:
            result: Encoding result + optimization info
        """
        # Phân tích frame
        mbs, props = self.encoder.analyzer.get_macroblocks(frame)
        texture = props['texture']
        
        motion = None
        if frame_prev is not None:
            motion = self.encoder.analyzer.estimate_motion(frame_prev, frame)
        
        # Binary search để tìm QP scaling factor
        scale_low, scale_high = 0.5, 2.0
        best_qp = None
        best_result = None
        
        for _ in range(10):  # iterations
            scale_mid = (scale_low + scale_high) / 2
            
            # Tìm QP base theo target bitrate
            QP_base = 28.0  # Default QP
            QP_per_layer = np.ones(self.encoder.num_layers) * QP_base * scale_mid
            
            # Clamp QP to [10, 51]
            QP_per_layer = np.clip(QP_per_layer, 10, 51)
            
            # Encode với QP này
            result = self.encoder.encode_frame(frame, frame_prev=frame_prev,
                                              QP_per_layer=QP_per_layer)
            
            actual_bitrate = result['total_bitrate']
            
            if actual_bitrate < target_bitrate:
                # Bitrate thấp hơn target → giảm QP (tăng quality)
                scale_high = scale_mid
            else:
                # Bitrate cao hơn target → tăng QP (giảm quality)
                scale_low = scale_mid
            
            best_result = result
        
        best_result['target_bitrate'] = target_bitrate
        best_result['bitrate_error'] = abs(best_result['total_bitrate'] - target_bitrate)
        
        return best_result


if __name__ == '__main__':
    # Test
    print("Testing SVC Encoder...")
    
    # Create synthetic test frames
    frame1 = np.random.randint(50, 200, size=(320, 240, 3), dtype=np.uint8)
    frame2 = frame1.copy()
    frame2[50:100, 50:100] += 30  # Add motion
    
    # Initialize encoder
    encoder = SVCEncoder(num_layers=3)
    
    # Encode single frame
    print("\nEncoding frame 1...")
    result = encoder.encode_frame(frame1)
    print(f"Bitrate per layer: {(result['bitrate_per_layer'] / 1e3).tolist()}")
    print(f"PSNR per layer: {result['psnr_per_layer']}")
    print(f"Total bitrate: {result['total_bitrate']/1e3:.1f} kbps")
    print(f"Avg PSNR: {result['avg_psnr']:.2f} dB")
    
    # Encode sequence with motion
    print("\nEncoding sequence (2 frames)...")
    result_seq = encoder.encode_stream([frame1, frame2])
    print(f"Total bitrate: {result_seq['total_bitrate']/1e3:.1f} kbps")
    print(f"Avg bitrate/frame: {result_seq['avg_bitrate']/1e3:.1f} kbps")
    print(f"Avg PSNR: {result_seq['avg_psnr']:.2f} dB")
    print(f"Avg PSNR per layer: {result_seq['avg_psnr_per_layer']}")
    
    # RDO optimization
    print("\nRDO Optimization...")
    rdo = RDOOptimizer(encoder)
    target_br = 500e3  # 500 kbps
    result_rdo = rdo.optimize_qp_for_bitrate_target(frame1, target_br)
    print(f"Target: {target_br/1e3:.0f} kbps")
    print(f"Actual: {result_rdo['total_bitrate']/1e3:.1f} kbps")
    print(f"Error: {result_rdo['bitrate_error']/1e3:.1f} kbps")
    print(f"PSNR: {result_rdo['avg_psnr']:.2f} dB")
