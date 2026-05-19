"""
video_input.py — Video Input Module

Chức năng:
  1. Đọc video từ file (MP4, AVI, MOV, etc.)
  2. Tách từng frame từ video
  3. Chuyển đổi RGB → YUV
  4. Hỗ trợ synthetic test video

Sử dụng: OpenCV + NumPy
"""
import os
import cv2
import numpy as np
from pathlib import Path


class VideoReader:
    """
    Đọc video và tách thành các frame.
    
    Attributes:
        filepath: Đường dẫn file video
        cap: OpenCV VideoCapture object
        fps: Frame rate (frames per second)
        total_frames: Tổng số frame
        width, height: Độ phân giải video
        current_frame_idx: Index của frame hiện tại
    """
    
    def __init__(self, filepath: str):
        """
        Khởi tạo VideoReader.
        
        Args:
            filepath: Đường dẫn file video
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Video file not found: {filepath}")
        
        self.filepath = filepath
        self.cap = cv2.VideoCapture(filepath)
        
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {filepath}")
        
        # Lấy thông tin video
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.current_frame_idx = 0
        
        print(f"[OK] Video loaded: {Path(filepath).name}")
        print(f"  Resolution: {self.width}×{self.height}")
        print(f"  FPS: {self.fps:.2f}")
        print(f"  Total frames: {self.total_frames}")
    
    def read_frame(self) -> tuple:
        """
        Đọc frame tiếp theo.
        
        Returns:
            (success: bool, frame_rgb: ndarray)
                - success: True nếu đọc thành công
                - frame_rgb: Frame dưới dạng RGB (H, W, 3)
        """
        ret, frame_bgr = self.cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            self.current_frame_idx += 1
            return ret, frame_rgb
        return ret, None
    
    def read_n_frames(self, n: int) -> list:
        """
        Đọc n frame tiếp theo.
        
        Args:
            n: Số lượng frame cần đọc
        
        Returns:
            List [frame1, frame2, ..., framen] (RGB format)
        """
        frames = []
        for _ in range(n):
            ret, frame = self.read_frame()
            if not ret:
                break
            frames.append(frame)
        return frames
    
    def reset(self):
        """Reset về frame đầu tiên."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.current_frame_idx = 0
    
    def seek_frame(self, frame_idx: int):
        """
        Nhảy đến frame cụ thể.
        
        Args:
            frame_idx: Index của frame (0-based)
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        self.current_frame_idx = frame_idx
    
    def close(self):
        """Đóng video file."""
        if self.cap is not None:
            self.cap.release()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
    
    def __del__(self):
        self.close()


def create_test_video(output_path: str, width: int = 320, height: int = 240, 
                      num_frames: int = 100, fps: int = 30):
    """
    Tạo video test tổng hợp (synthetic video).
    
    Gồm các phần:
    - Top-left: Static region (easy to compress)
    - Top-right: Gradient region (medium)
    - Bottom: Moving circles (hard to compress - high motion)
    
    Args:
        output_path: Đường dẫn lưu video
        width, height: Độ phân giải
        num_frames: Số lượng frame
        fps: Frame rate
    """
    # Khởi tạo VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    print(hasattr(cv2, "VideoWriter_fourcc"))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Creating test video: {width}×{height}, {num_frames} frames @ {fps} FPS")
    
    for frame_id in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Top-left: Static white region (low motion)
        cv2.rectangle(frame, (0, 0), (width//2, height//2), (255, 255, 255), -1)
        
        # Top-right: Gradient (medium complexity)
        for x in range(width//2, width):
            intensity = int(255 * (x - width//2) / (width//2))
            cv2.line(frame, (x, 0), (x, height//2), (intensity, intensity, intensity))
        
        # Bottom: Moving circles (high motion = hard to compress)
        y_center = height // 2 + height // 4
        x_offset = int(width * 0.3 * np.sin(2 * np.pi * frame_id / num_frames))
        x_center = width // 2 + x_offset
        cv2.circle(frame, (x_center, y_center), 20, (0, 255, 0), -1)
        
        # Add text with frame number
        cv2.putText(frame, f"Frame {frame_id}", (10, height - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Convert to BGR for video writing
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()
    print(f"[OK] Test video saved: {output_path}")


def rgb_to_yuv(frame_rgb: np.ndarray) -> tuple:
    """
    Chuyển RGB → YUV 4:2:0 (như H.264/H.265).
    
    Args:
        frame_rgb: Frame dưới dạng RGB (H, W, 3), uint8
    
    Returns:
        (Y, U, V): 
            - Y: Luminance (H, W), uint8
            - U: Chrominance U (H//2, W//2), uint8
            - V: Chrominance V (H//2, W//2), uint8
    """
    # RGB → YUV (BT.601 standard)
    # Y  = 0.299*R + 0.587*G + 0.114*B
    # U  = -0.147*R - 0.289*G + 0.436*B + 128
    # V  = 0.615*R - 0.515*G - 0.100*B + 128
    
    frame_float = frame_rgb.astype(np.float32)
    
    Y = (0.299 * frame_float[:,:,0] + 
         0.587 * frame_float[:,:,1] + 
         0.114 * frame_float[:,:,2])
    
    U = (-0.147 * frame_float[:,:,0] - 
         0.289 * frame_float[:,:,1] + 
         0.436 * frame_float[:,:,2] + 128)
    
    V = (0.615 * frame_float[:,:,0] - 
         0.515 * frame_float[:,:,1] - 
         0.100 * frame_float[:,:,2] + 128)
    
    # Clamp to [0, 255]
    Y = np.clip(Y, 0, 255).astype(np.uint8)
    U = np.clip(U, 0, 255).astype(np.uint8)
    V = np.clip(V, 0, 255).astype(np.uint8)
    
    # Downsample U, V (4:2:0 chroma subsampling)
    U = U[::2, ::2]  # H//2, W//2
    V = V[::2, ::2]  # H//2, W//2
    
    return Y, U, V


def yuv_to_rgb(Y: np.ndarray, U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Chuyển YUV (4:2:0) → RGB (ngược lại).
    
    Args:
        Y: Luminance (H, W)
        U: Chrominance U (H//2, W//2)
        V: Chrominance V (H//2, W//2)
    
    Returns:
        frame_rgb: RGB frame (H, W, 3)
    """
    H, W = Y.shape
    
    # Upsample U, V back to (H, W)
    U_full = np.repeat(np.repeat(U, 2, axis=0), 2, axis=1)[:H, :W]
    V_full = np.repeat(np.repeat(V, 2, axis=0), 2, axis=1)[:H, :W]
    
    Y_float = Y.astype(np.float32)
    U_float = U_full.astype(np.float32) - 128.0
    V_float = V_full.astype(np.float32) - 128.0
    
    # YUV → RGB (inverse transformation)
    # R = Y + 1.402*V
    # G = Y - 0.344*U - 0.714*V
    # B = Y + 1.772*U
    
    R = Y_float + 1.402 * V_float
    G = Y_float - 0.344 * U_float - 0.714 * V_float
    B = Y_float + 1.772 * U_float
    
    # Clamp to [0, 255]
    R = np.clip(R, 0, 255).astype(np.uint8)
    G = np.clip(G, 0, 255).astype(np.uint8)
    B = np.clip(B, 0, 255).astype(np.uint8)
    
    frame_rgb = np.stack([R, G, B], axis=-1)
    return frame_rgb


if __name__ == '__main__':
    # Test: Tạo test video
    test_video_path = 'test_video.mp4'
    create_test_video(test_video_path, width=320, height=240, 
                      num_frames=100, fps=30)
    
    # Test: Đọc video
    with VideoReader(test_video_path) as reader:
        print(f"\nReading first 5 frames...")
        frames = reader.read_n_frames(5)
        print(f"✓ Read {len(frames)} frames")
        
        # Test YUV conversion
        if len(frames) > 0:
            frame = frames[0]
            Y, U, V = rgb_to_yuv(frame)
            print(f"\nYUV conversion:")
            print(f"  Y shape: {Y.shape}")
            print(f"  U shape: {U.shape}")
            print(f"  V shape: {V.shape}")
            
            # Convert back
            frame_reconstructed = yuv_to_rgb(Y, U, V)
            print(f"  Reconstructed RGB: {frame_reconstructed.shape}")
