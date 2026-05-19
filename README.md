# CR-RSMA vs CR-NOMA Max-Min Fairness Simulation

**Thesis Project**: Comparative performance evaluation of **Cognitive Radio Rate-Splitting Multiple Access (CR-RSMA)** and **Cognitive Radio Non-Orthogonal Multiple Access (CR-NOMA)** with focus on fairness-optimal resource allocation for multimedia streaming.

---

## 📋 System Model

### **Network Configuration**
- **Setup**: Underlay Cognitive Radio (Uplink/MAC)
  - 1 Primary User (PU) receiver
  - K Secondary Users (SUs) transmitters (K = 2, 3, 4, 5, 6)
  - Single-input single-output (SISO) channels
  
### **Channel Model**
- Rayleigh fading with path-loss exponent η = 2.0
- Monte Carlo Simulation: 200 fading realizations per condition
- Shannon capacity (no finite block length)

### **Fairness Objective**
**Max-Min Fairness (MMF)**:
$$\text{maximize} \quad \min_k R_k$$

where $R_k$ is the achievable rate (bps/Hz) of user $k$.

**Fairness Metric**: Jain's Fairness Index
$$JFI = \frac{(\sum R_k)^2}{K \cdot \sum R_k^2} \in [1/K, 1]$$

---

## 🔋 Multiple Access Schemes

### **CR-RSMA (Rate-Splitting Multiple Access)**
- **Stream Splitting Strategy**: 
  - Strongest SU's stream split into private + common parts
  - Total K+1 streams with sequential decoding order
  - Decoding cascade: common_part1 → privates (SIC) → common_part2
  
- **Advantage**: Better resource utilization & improved fairness via splitting
- **Optimization**: Allocate transmit power across streams to maximize $\min_k R_k$

### **CR-NOMA (Non-Orthogonal Multiple Access)**
- **Power-Domain NOMA**: K independent streams per user
- **Decoding**: Weakest-to-strongest user successive interference cancellation (SIC)
- **Simpler but**: Limited fairness potential due to SIC sequencing

### **Optimization Constraints**
1. **Transmit Power Budget**:  $\sum p_i \leq P_t$
2. **CR Interference Limit**: $\sum p_i \cdot g_{i,\text{PU}} \leq I_{\text{th}}$ (Protect PU)
3. **Method**: Sequential Least Squares Programming (SLSQP) with multi-start initialization

---

## 🎬 SVC Video Quality Mapping

**Scalable Video Coding (SVC)**: 4-layer structure (H.264 SVC)
- **Format**: QCIF (176×144), 30 fps, Group-of-Pictures (GOP) = 8

| Layer | Bitrate (Cumulative) | PSNR | Description |
|-------|---------------------|------|-------------|
| **BL** (Base)       | 71.38 kbps | 29.87 dB | Minimum decodable stream |
| **BL+EL1**          | 142.99 kbps | 33.17 dB | Low-quality streaming |
| **BL+EL1+EL2**      | 287.82 kbps | 36.98 dB | Medium quality |
| **BL+EL1+EL2+EL3**  | 544.24 kbps | 41.19 dB | High-quality stream |

**Rate-to-PSNR Conversion**: Normalized rate (bps/Hz) automatically mapped to nearest SVC layer quality.

---

## 📊 Simulation Scenarios

| Scenario | Variable | Fixed Parameters | Purpose |
|----------|----------|-----------------|---------|
| **Sc1** | SNR (0–30 dB) | K=2, I_th scales | Performance vs transmit power |
| **Sc2** | I_th (0.05–1.0 W) | K=2, SNR=20 dB | CR constraint tightness impact |
| **Sc3** | K (2–6 users) | SNR=20 dB, I_th=0.1 W | Scalability with user count |
| **Sc4** | SNR → PSNR mapping | K=2 | Video quality vs transmission power |
| **Sc8** | Jain's FI comparison | K=2, SNR=20 dB | Fairness index across SNR/K |
| **Sc11** | vs equal-power baseline | K=2, SNR=20 dB | Value of MMF optimization |

**Monte Carlo Parameters**:
- **Realizations per point**: 200
- **I_th Scaling**: I_th/P_t ratio held constant (I_TH_RATIO = 0.01) across SNR sweep
- **Channel regeneration**: Fading coefficients regenerated per realization

---

## 📁 Project Structure

```
fairness/
├── main.py                  # Main simulation engine & scenario runners
├── config.py                # System parameters & SVC configuration
├── sim_results.npz          # Saved numerical results (NumPy archive)
├── README.md                # This file
└── __pycache__/             # Python cache
```

### **File Descriptions**

**`config.py`** — System & Video Configuration
- `SVCParams`: SVC 4-layer bitrates, PSNR thresholds, video resolution
- System parameters: Bandwidth B=140 kHz, power limits, interference threshold I_th
- Fairness weights: w_p = w_s = 0.5 (equal PU/SU priority)

**`main.py`** — Simulation Engine
- **Core Functions**:
  - `rsma_mmf()`: RSMA power allocation optimization
  - `noma_mmf()`: NOMA power allocation optimization
  - `compute_sinr_sic()`: Post-SIC SINR computation per stream
  - `shannon_rate()`: SINR → bps/Hz capacity conversion
  - `rate_to_psnr()`: Rate → PSNR mapping via SVC layers
  - `jains_index()`: Fairness index calculation
  
- **Simulation Runners**:
  - `sim_vs_snr()`: Scenario 1 (SNR sweep)
  - `sim_vs_ith()`: Scenario 2 (I_th sweep)
  - `sim_vs_K()`: Scenario 3 (User count sweep)
  - `_mc_average()`: Monte Carlo averaging across fading realizations

---

## 🚀 Usage

### **Requirements**
```bash
pip install numpy scipy matplotlib
```

### **Run Full Simulation**
```bash
python main.py
```

This will execute all scenarios and generate:

**Numerical Results**:
- `sim_results.npz` — Dictionary with keys for each scenario's results

**Visualizations**:
- `fig1_mmf_vs_snr.png` — MMF rate vs SNR (Sc1)
- `fig2_mmf_vs_ith.png` — MMF rate vs interference threshold (Sc2)
- `fig3_mmf_vs_K.png` — MMF vs number of users (Sc3)
- `fig4_combined.png` — 1×3 thesis-ready panel (Sc1–3)
- `fig5_psnr_qoe.png` — Video quality (PSNR) vs SNR (Sc4)
- `fig6_jains_fairness.png` — Fairness index comparison (Sc8)
- `fig7_equal_power_baseline.png` — Optimization gain vs equal allocation (Sc11)

### **Customization**

Edit `config.py` to modify:

```python
# Video parameters
resolution = (176, 144)          # QCIF format
fps = 30                         # Frames per second
gop_size = 8                     # Group of Pictures

# Network parameters
B = 140e3                        # Bandwidth (Hz)
Pt_max = 1.0                     # Transmit power (W)
I_th = 0.1                       # Interference threshold (W)

# Simulation parameters
N_REAL = 200                     # Monte Carlo realizations
SNR_RANGE = np.linspace(0, 30, 16)  # SNR sweep (dB)
```

---

## 📈 Key Results

### **Expected Findings**
1. **RSMA Advantage**: RSMA achieves higher minimum rate (MMF) than NOMA across all SNR/K conditions
2. **Fairness Gain**: RSMA maintains higher Jain's FI, especially under tight CR constraints
3. **Scalability**: Both schemes degrade with increasing K, but RSMA degrades more gracefully
4. **QoE Improvement**: Critical rates achieved at lower SNR with RSMA → better video experience earlier

### **Output Interpretation**

- **MMF curves (Sc1–3)**: Higher MMF = fairer resource allocation
- **PSNR curves (Sc4)**: Video quality improvement mapped to SVC layers (29.87 → 41.19 dB)
- **Fairness Index (Sc8)**: Closer to 1.0 = more equitable rate distribution
- **Baseline Gap (Sc11)**: Demonstrates value of optimization vs naive equal-power allocation

---

## 🔗 Dependencies

- **NumPy** — Numerical arrays & Monte Carlo simulations
- **SciPy** — Optimization (SLSQP) for power allocation
- **Matplotlib** — Figure generation & visualization

Install all with:
```bash
pip install numpy scipy matplotlib
```

---

## 📚 References

- **System Model**: Underlay Cognitive Radio MAC with SU power constraints + CR interference limit
- **Optimization**: SEQ-MINORANT approach for non-convex MMF problem
- **Video Codec**: H.264 SVC standard (JM reference encoder bitrates)
- **Fairness**: Jain's Index & max-min fairness principles

---

## ℹ️ Notes

- **Reproducibility**: Fixed random seed (`np.random.seed(42)`) for deterministic fading realization
- **Computation Time**: Full simulation ~5–10 min depending on hardware & realizations
- **I_th Scaling**: Ratio I_th/P_t held constant to ensure monotone increasing performance with SNR
- **SIC Assumption**: Assumes perfect successive interference cancellation (no errors propagation)