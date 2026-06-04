# CR-RSMA vs CR-NOMA vs OMA: Hệ thống và Mô tả Chi tiết

## 📋 Mục lục
1. [Tổng quan hệ thống](#tổng-quan-hệ-thống)
2. [Kiến trúc hệ thống](#kiến-trúc-hệ-thống)
3. [Các tham số chính](#các-tham-số-chính)
4. [Các kỹ thuật truy cập](#các-kỹ-thuật-truy-cập)
5. [Các scenario simulation](#các-scenario-simulation)
6. [Metrics đánh giá](#metrics-đánh-giá)
7. [Kết quả mong đợi](#kết-quả-mong-đợi)

---

## 🎯 Tổng quan hệ thống

### Vấn đề nghiên cứu
Nghiên cứu **công bằng (fairness) trong Cognitive Radio** bằng cách so sánh ba kỹ thuật truy cập:
- **CR-RSMA** (Rate-Splitting Multiple Access)
- **CR-NOMA** (Non-Orthogonal Multiple Access)
- **OMA** (Orthogonal Multiple Access - baseline)

### Mục tiêu
Tối đa hóa **Max-Min Fairness (MMF)** - đảm bảo user yếu nhất có thể đạt rate cao nhất có thể.

### Ứng dụng
- **Video streaming** (SVC 4-layer)
- **Underlay Cognitive Radio** (PU + SU cùng spectrum)
- **SISO MAC** (Single Input Single Output Multiple Access Channel)

---

## 🏗️ Kiến trúc hệ thống

### Tô pô mạng
```
┌─────────────────┐
│   Primary User  │────────┐
│    (PU) - 1 user│        │
└─────────────────┘        │ Shared Spectrum
                           │
┌─────────────────────────────────────────────────────────┐
│         Secondary Users (SU) - K users                   │
│  (K = 2, 3, 4, 5, 6 users)                              │
└─────────────────────────────────────────────────────────┘

Central Authority (Base Station / Access Point)
- Phân bổ công suất tối ưu
- Lập lịch người dùng
```

### Loại kênh
- **Rayleigh fading** (mô phỏng kênh không dây thực tế)
- **AWGN** (Additive White Gaussian Noise)
- **Underlay constraint**: Interference từ SU đến PU ≤ I_th (Interference Threshold)

---

## ⚙️ Các tham số chính

### Tầng vật lý

| Tham số | Giá trị | Ý nghĩa |
|--------|--------|---------|
| **Bandwidth (B)** | 140 kHz | Băng thông hệ thống |
| **Path loss exponent (η)** | 2.0 | Hệ số suy hao |
| **PU max power (Pp_max)** | 1.0 W | Công suất tối đa của PU |
| **SU max total power (Ps_max)** | 1.0 W | Công suất tối đa của SU |
| **Noise variance (σ²)** | 1.0 | Phương sai nhiễu (chuẩn hóa) |
| **Interference threshold (I_th)** | 0.1 W | Giới hạn nhiễu tại PU receiver |

### Video & SVC (Scalable Video Coding)

| Lớp | Bitrate (kbps) | PSNR (dB) | Chát lượng |
|-----|----------------|-----------|-----------|
| **BL** (Base Layer) | 71.38 | 29.87 | Xấu |
| **EL1** (Enhancement 1) | +71.61 → 142.99 | 33.17 | Tạm được |
| **EL2** (Enhancement 2) | +144.83 → 287.82 | 36.98 | Tốt |
| **EL3** (Enhancement 3) | +256.43 → 544.24 | 41.19 | Rất tốt |

### Fairness Weights (QoE)

| Tham số | Giá trị | Ý nghĩa |
|--------|--------|---------|
| **w_p** (weight PU) | 0.5 | Trọng số cho Primary User |
| **w_s** (weight SU) | 0.5 | Trọng số cho Secondary User |

---

## 🔄 Các kỹ thuật truy cập

### 1️⃣ CR-RSMA (Rate-Splitting Multiple Access)

**Cấu trúc:**
- Một user "split" có **2 stream** (common + private)
- Các user khác có **1 stream** (private)
- **Tổng: K+1 streams** (K = số SU)

**Ví dụ K=2:**
```
Streams: [c1 (split user), p_1 (user 1), c2 (split user)]
Decoding order: c1 → p_1 → c2 (theo thứ tự quy định)
```

**Lợi điểm:**
- ✅ Tối ưu hóa stream splitting → rate cao
- ✅ SIC (Successive Interference Cancellation) → giải mã tốt
- ✅ **Rate highest** trong 3 kỹ thuật

### 2️⃣ CR-NOMA (Non-Orthogonal Multiple Access)

**Cấu trúc:**
- **K streams** cho K users
- Không chia thành common/private

**Decoding order:**
- Weakest user first (ascending by channel gain)
- SIC decoder giải mã theo thứ tự này

**Lợi điểm:**
- ✅ Đơn giản hơn RSMA
- ✅ Vẫn có SIC improvement
- ⚠️ Rate: trung bình (giữa RSMA và OMA)

### 3️⃣ OMA (Orthogonal Multiple Access - Baseline)

**Cấu trúc:**
- **K streams** cho K users
- Mỗi user có orthogonal time/frequency slot
- **Không có interference** giữa các SU (nhưng lãng phí phổ)

**Lợi điểm:**
- ✅ Đơn giản, dễ triển khai
- ⚠️ Lãng phí phổ (chỉ dùng 1/K thời gian/tần số)
- ❌ **Rate thấp nhất** (baseline để so sánh)

---

## 📊 Các scenario simulation

### **Scenario 1: MMF vs SNR** (fig1_mmf_vs_snr.png)

**Điều kiện cố định:**
- K = 2 (2 SU)
- I_th scales với Pt: I_th = I_TH_RATIO × Pt

**Biến đổi:**
- SNR: 0 → 30 dB (bước 2 dB)

**Ý nghĩa:**
- Khi SNR ↑ → công suất Pt ↑ → user được power lớn → rate ↑
- So sánh ba kỹ thuật ở điều kiện SNR khác nhau

**Kết quả mong đợi:**
```
Ranking: RSMA > NOMA > OMA (tất cả tăng dần với SNR)
```

---

### **Scenario 2: MMF vs I_th** (fig2_mmf_vs_ith.png)

**Điều kiện cố định:**
- K = 2 (2 SU)
- SNR = 20 dB (cố định)

**Biến đổi:**
- I_th: 0.05 → 1.0 W (15 giá trị)

**Ý nghĩa:**
- Khi I_th ↓ (constraint chặt) → phải giảm power → rate ↓
- Thể hiện tác động của **CR constraint** lên fairness
- **Scenario quan trọng nhất** cho Cognitive Radio

**Kết quả mong đợi:**
```
Ranking: RSMA > NOMA > OMA (tất cả giảm dần khi I_th ↓)
```

---

### **Scenario 3: MMF vs K** (fig3_mmf_vs_k.png)

**Điều kiện cố định:**
- SNR = 20 dB
- I_th = 0.1 W

**Biến đổi:**
- K: 2, 3, 4, 5, 6 users

**Ý nghĩa:**
- Khi K ↑ (nhiều user) → cần chia power → rate ↓
- Thể hiện **scalability** của các kỹ thuật
- User ít → rate cao; user nhiều → rate thấp

**Kết quả mong đợi:**
```
Ranking: RSMA > NOMA > OMA (tất cả giảm dần khi K ↑)
```

---

## 📈 Metrics đánh giá

### 1. **Max-Min Fairness (MMF)**
$$\text{MMF} = \max_{\mathbf{p}} \min_k R_k(\mathbf{p})$$

- **Định nghĩa**: Rate của user yếu nhất
- **Mục tiêu**: Tối đa hóa
- **Đơn vị**: kbps (sau chuyển đổi)
- **Phạm vi**: 0 → 0.8 kbps (tùy scenario)

### 2. **Shannon Capacity**
$$R_k = \log_2(1 + \text{SINR}_k) \text{ [bps/Hz]}$$
$$R_k = \log_2(1 + \text{SINR}_k) \times \frac{B}{1000} \text{ [kbps]}$$

- **SINR**: Signal-to-Interference-plus-Noise Ratio
- **Không FBL**: Mở rộng (finite blocklength) không được tính

### 3. **Spectral Efficiency**
$$\eta_k = \frac{R_k}{B} \text{ [bits/s/Hz]}$$

- Thể hiện hiệu suất sử dụng phổ
- Cao → tốt (dùng phổ hiệu quả)

---

## ✅ Kết quả mong đợi

### Ranking (lý thuyết)
```
RSMA ≥ NOMA ≥ OMA
```

| Kỹ thuật | Ưu điểm | Nhược điểm |
|---------|--------|-----------|
| **RSMA** ✅ | • Stream splitting tối ưu<br>• SIC decoder<br>• **Rate cao nhất** | • Phức tạp<br>• Tối ưu 3 tham số |
| **NOMA** ⚠️ | • Đơn giản hơn RSMA<br>• SIC decoder<br>• Rate trung bình | • Không tối ưu phân chia |
| **OMA** ❌ | • Đơn giản nhất<br>• Dễ triển khai | • **Lãng phí phổ**<br>• **Rate thấp nhất** |

---

## 🔍 Giải thích dữ liệu

### Tại sao trục Y khác nhau giữa các scenario?

| Scenario | Range trục Y | Lý do |
|----------|-------------|-------|
| **vs SNR** | 0 → 0.8 kbps | SNR ↑ → power ↑ → rate ↑ (max) |
| **vs I_th** | 0 → 0.4 kbps | I_th ↓ → power ↓ → rate ↓ (mid) |
| **vs K** | 0 → 0.15 kbps | K ↑ → chia power → rate ↓ (min) |

### Tại sao chọn kbps?
- **Khớp với SVC layer rates** (71-544 kbps)
- **Trực quan hơn** bps/Hz (0-6 → 0-0.8 kbps)
- **Chuẩn trong video coding**

---

## 🚀 Chạy Simulation

### Cách chạy
```bash
cd fairness/
python main.py
```

### Output
```
✓ fig1_mmf_vs_snr.png      (Scenario 1)
✓ fig2_mmf_vs_ith.png      (Scenario 2)
✓ fig3_mmf_vs_K.png        (Scenario 3)
✓ sim_results.npz          (Dữ liệu số)
```

### Monte Carlo
- **200 realizations** per point
- **Rayleigh fading** channels
- **Random seeds** để tái tạo

---

## 📚 Tài liệu tham khảo

### Khái niệm chính
- **RSMA** (Rate-Splitting): Viacci et al.
- **NOMA**: Saito et al., Li et al.
- **Cognitive Radio**: Haykin
- **Max-Min Fairness**: Jain, Laxmi

### Công nghệ liên quan
- **SVC** (Scalable Video Coding): H.264/AVC SVC
- **SIC** (Successive Interference Cancellation)
- **Shannon Capacity**: Information Theory

---

**Document version:** 1.0  
**Last updated:** June 2, 2026  
**Author:** Graduation Thesis - Fairness in CR Systems
