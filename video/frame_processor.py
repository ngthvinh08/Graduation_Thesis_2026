"""
frame_processor.py — Frame Processing Module

Chức năng:
  1. Chia frame thành macroblock (16×16)
  2. Tính motion/texture properties của mỗi macroblock
  3. Tính PSNR giữa hai frame
  4. Tính bitrate từ QP và layer info

Sử dụng: NumPy, SciPy, scikit-image
"""
import numpy as np
from scipy import signal
from skimage.metrics import peak_signal_noise_ratio as skimage_psnr
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import M_BLOCKS, ALPHA_RD, GAMMA_RD, L_MAX, R_LAYER


class MacroblockAnalyzer:
    """
    Phân tích các macroblock (16×16 pixels) từ frame.
    
    Mỗi macroblock có:
    - Motion level (từ optical flow hoặc frame difference)
    - Texture complexity (từ gradient magnitude)
    - Predicted bitrate & distortion (từ RDO model)
    """
    
    MB_SIZE = 16  # 16×16 pixels per macroblock
    
    def __init__(self):
        """Khởi tạo MacroblockAnalyzer."""
        pass
    
    def get_macroblocks(self, frame: np.ndarray) -> tuple:
        """
        Chia frame thành macroblocks và trả về properties.
        
        Args:
            frame: Frame (H, W, 3) hoặc (H, W) nếu là YUV component
        
        Returns:
            mbs: List các macroblock (H//16, W//16, 16, 16)
            props: Dict chứa properties của từng MB
        """
        if frame.ndim == 3:
            # Nếu RGB, chuyển thành grayscale
            frame_gray = np.mean(frame, axis=2).astype(np.uint8)
        else:
            frame_gray = frame.astype(np.uint8)
        
        H, W = frame_gray.shape
        num_mbs_h = H // self.MB_SIZE
        num_mbs_w = W // self.MB_SIZE
        
        # Vòng for này để chia frame thành macroblock
        # Chạy num_mbs_h × num_mbs_w lần (VD: 15×20 = 300 với frame 320×240)
        mbs = []
        for i in range(num_mbs_h):
            for j in range(num_mbs_w):
                y_start = i * self.MB_SIZE
                y_end = y_start + self.MB_SIZE
                x_start = j * self.MB_SIZE
                x_end = x_start + self.MB_SIZE
                
                mb = frame_gray[y_start:y_end, x_start:x_end]
                mbs.append(mb)
        
        mbs = np.array(mbs).reshape(num_mbs_h, num_mbs_w, self.MB_SIZE, self.MB_SIZE)
        
        # Đây là dict chứa properties của từng MB
        props = {
            'num_mbs_h': num_mbs_h,
            'num_mbs_w': num_mbs_w,
            'texture': self._compute_texture(mbs),
            'gradient': self._compute_gradient(mbs),
        }
        
        return mbs, props
    
    def _compute_texture(self, mbs: np.ndarray) -> np.ndarray:
        """
        Tính texture complexity của mỗi MB (dựa trên variance).
        
        Args:
            mbs: Shape (num_mbs_h, num_mbs_w, 16, 16)
        
        Returns:
            texture: Shape (num_mbs_h, num_mbs_w), range [0, 255]
        """
        num_mbs_h, num_mbs_w = mbs.shape[:2]
        texture = np.zeros((num_mbs_h, num_mbs_w))
        
        for i in range(num_mbs_h):
            for j in range(num_mbs_w):
                mb = mbs[i, j].astype(np.float32)
                texture[i, j] = np.var(mb)  # Variance as texture measure
        
        # Normalize to [0, 255]
        texture_max = np.max(texture) if np.max(texture) > 0 else 1
        texture = (texture / texture_max * 255).astype(np.uint8)
        
        return texture
    
    def _compute_gradient(self, mbs: np.ndarray) -> np.ndarray:
        """
        Tính gradient magnitude của mỗi MB.
        
        Args:
            mbs: Shape (num_mbs_h, num_mbs_w, 16, 16)
        
        Returns:
            gradient: Shape (num_mbs_h, num_mbs_w), range [0, 255]
        """
        num_mbs_h, num_mbs_w = mbs.shape[:2]
        gradient = np.zeros((num_mbs_h, num_mbs_w))
        
        # Sobel kernel
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        
        for i in range(num_mbs_h):
            for j in range(num_mbs_w):
                mb = mbs[i, j].astype(np.float32)
                
                grad_x = signal.convolve2d(mb, sobel_x, mode='same')
                grad_y = signal.convolve2d(mb, sobel_y, mode='same')
                
                grad_mag = np.sqrt(grad_x**2 + grad_y**2)
                gradient[i, j] = np.mean(grad_mag)
        
        # Normalize to [0, 255]
        grad_max = np.max(gradient) if np.max(gradient) > 0 else 1
        gradient = (gradient / grad_max * 255).astype(np.uint8)
        
        return gradient
    
    def estimate_motion(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """
        Ước lượng motion level giữa hai frame (frame difference).
        
        Args:
            frame1: Frame t (H, W)
            frame2: Frame t+1 (H, W)
        
        Returns:
            motion: Motion level cho mỗi MB (num_mbs_h, num_mbs_w)
        """
        if frame1.ndim == 3:
            frame1 = np.mean(frame1, axis=2)
        if frame2.ndim == 3:
            frame2 = np.mean(frame2, axis=2)
        
        frame1 = frame1.astype(np.float32)
        frame2 = frame2.astype(np.float32)
        
        # Frame difference
        diff = np.abs(frame2 - frame1)
        
        # Aggregate to MBs
        num_mbs_h = frame1.shape[0] // self.MB_SIZE
        num_mbs_w = frame1.shape[1] // self.MB_SIZE
        
        motion = np.zeros((num_mbs_h, num_mbs_w))
        for i in range(num_mbs_h):
            for j in range(num_mbs_w):
                y_start = i * self.MB_SIZE
                y_end = y_start + self.MB_SIZE
                x_start = j * self.MB_SIZE
                x_end = x_start + self.MB_SIZE
                
                motion[i, j] = np.mean(diff[y_start:y_end, x_start:x_end])
        
        # Normalize to [0, 255]
        motion_max = np.max(motion) if np.max(motion) > 0 else 1
        motion = (motion / motion_max * 255).astype(np.uint8)
        
        return motion


def compute_psnr(frame_original: np.ndarray, frame_reconstructed: np.ndarray) -> float:
    """
    Tính PSNR (Peak Signal-to-Noise Ratio).
    
    PSNR = 10 * log10(MAX² / MSE)
    
    Args:
        frame_original: Original frame (H, W, 3 hoặc H, W)
        frame_reconstructed: Reconstructed frame (H, W, 3 hoặc H, W)
    
    Returns:
        psnr: PSNR value (dB)
    """
    # Ensure both frames are the same shape
    assert frame_original.shape == frame_reconstructed.shape, \
        f"Shape mismatch: {frame_original.shape} vs {frame_reconstructed.shape}"
    
    # Use scikit-image's PSNR
    psnr = skimage_psnr(frame_original, frame_reconstructed, data_range=255)
    return psnr


def compute_rdo_parameters(texture: np.ndarray, motion: np.ndarray = None,
                           layer: int = 0) -> tuple:
    """
    Tính RDO parameters cho mỗi MB dựa trên texture/motion.
    
    RDO model: R(QP) = α*2^(-(QP-12)/6), D(QP) = γ*2^((QP-12)/3)
    
    Args:
        texture: Texture map (num_mbs_h, num_mbs_w), range [0, 255]
        motion: Motion map (num_mbs_h, num_mbs_w), range [0, 255] (optional)
        layer: SVC layer index (0=base, 1=EL1, 2=EL2)
    
    Returns:
        alpha: Rate parameter per MB
        gamma: Distortion parameter per MB
    """
    num_mbs_h, num_mbs_w = texture.shape
    
    # Base RDO parameters từ config
    alpha_base = ALPHA_RD[layer]
    gamma_base = GAMMA_RD[layer]
    
    # Adjust dựa trên texture complexity
    texture_normalized = texture.astype(np.float32) / 255.0
    
    if motion is not None:
        motion_normalized = motion.astype(np.float32) / 255.0
        complexity = 0.6 * texture_normalized + 0.4 * motion_normalized
    else:
        complexity = texture_normalized
    
    # Regions với complexity cao → cần α lớn hơn (capture more details)
    # Complexity scale: 0.8 ~ 1.2 (Adaptive QP effect)
    complexity_scale = 0.8 + 0.4 * complexity
    
    # Adaptive QP: High complexity MBs get slightly higher QP (simulated via gamma)
    alpha = alpha_base * complexity_scale
    gamma = gamma_base * (1.0 / complexity_scale)  # Better quality for smooth areas
    
    return alpha, gamma


def predict_bitrate(QP_array: np.ndarray, alpha: np.ndarray, 
                    layer: int = 0) -> float:
    """
    Dự đoán bitrate từ QP array và RDO parameters.
    
    R(QP) = α*2^(-(QP-12)/6)
    
    Args:
        QP_array: QP cho mỗi MB (num_mbs_h, num_mbs_w)
        alpha: Rate parameter (num_mbs_h, num_mbs_w)
        layer: SVC layer
    
    Returns:
        bitrate: Total bitrate (bits)
    """
    # Exponent term
    exp_term = -np.clip(QP_array - 12, -20, 40) / 6.0
    
    # Rate per MB (bits)
    rate_mb = alpha * np.power(2.0, exp_term)
    
    # Total rate (aggregate all MBs)
    bitrate = np.sum(rate_mb)
    
    return bitrate


def predict_distortion(QP_array: np.ndarray, gamma: np.ndarray,
                       layer: int = 0) -> float:
    """
    Dự đoán distortion (inverse of quality) từ QP.
    
    D(QP) = γ*2^((QP-12)/3)
    
    Args:
        QP_array: QP cho mỗi MB
        gamma: Distortion parameter
        layer: SVC layer
    
    Returns:
        avg_distortion: Average distortion
    """
    exp_term = np.clip(QP_array - 12, -20, 40) / 3.0
    
    distortion_mb = gamma * np.power(2.0, exp_term)
    
    avg_distortion = np.mean(distortion_mb)
    
    return avg_distortion


def qp_to_psnr_estimate(QP: float, gamma: float) -> float:
    """
    Ước lượng PSNR từ QP và gamma parameter.
    
    PSNR ≈ 51 - 10*log10(D(QP))
    
    Args:
        QP: Quantization parameter
        gamma: Distortion parameter
    
    Returns:
        psnr: Estimated PSNR (dB)
    """
    D = gamma * np.power(2.0, (QP - 12) / 3.0)
    
    # Simulate Loop Filter (Deblocking) loss: 
    # High QP leads to more blocking artifacts, which loop filter smoothes but adds slight blur.
    # Blur reduces PSNR slightly (0.5 - 1.5 dB)
    loop_filter_loss = 0.5 + 1.0 * (QP / 51.0)
    
    psnr = 51 - 10 * np.log10(D + 1e-10) - loop_filter_loss
    return psnr


if __name__ == '__main__':
    # Test
    print("Testing frame processor...")
    
    # Create synthetic frame
    frame = np.random.randint(0, 256, size=(320, 240, 3), dtype=np.uint8)
    frame[:100, :100] = 200  # Uniform region
    
    # Analyze macroblocks
    analyzer = MacroblockAnalyzer()
    mbs, props = analyzer.get_macroblocks(frame)
    
    print(f"Macroblocks: {props['num_mbs_h']}×{props['num_mbs_w']}")
    print(f"Texture map: {props['texture'].shape}")
    print(f"Gradient map: {props['gradient'].shape}")
    
    # Test PSNR
    frame2 = frame.copy()
    frame2[10:20, 10:20] = 100  # Add some noise
    
    psnr = compute_psnr(frame, frame2)
    print(f"\nPSNR: {psnr:.2f} dB")
    
    # Test RDO
    alpha, gamma = compute_rdo_parameters(props['texture'], layer=0)
    print(f"\nAlpha shape: {alpha.shape}")
    print(f"Gamma shape: {gamma.shape}")
    
    # Test bitrate prediction
    QP_array = np.ones((props['num_mbs_h'], props['num_mbs_w'])) * 28
    bitrate = predict_bitrate(QP_array, alpha[0])
    print(f"Predicted bitrate (QP=28): {bitrate:.0f} bits")
    
    # Test PSNR estimation
    psnr_est = qp_to_psnr_estimate(28, gamma[0].mean())
    print(f"Estimated PSNR (QP=28): {psnr_est:.2f} dB")
