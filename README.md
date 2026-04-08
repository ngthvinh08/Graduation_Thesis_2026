# Graduation Thesis 2026: CR-RSMA + UAV + QoE Fairness

**Author:** Nguyen Thanh Vinh  
**Date:** May, 2026

## Overview
This project implements an optimization framework for **Cognitive Radio Rate-Splitting Multiple Access (CR-RSMA)** networks assisted by **Unmanned Aerial Vehicles (UAVs)**. The goal is to maximize the **Quality of Experience (QoE)** and ensure **Fairness** for video streaming users employing **Scalable Video Coding (SVC)**.

The system utilizes **Successive Convex Approximation (SCA)** to optimize power allocation, bitrate, and trajectory parameters under interference and resource constraints.

---

## Key Features

### 1. Scalable Video Coding (SVC)
- Support for multiple video layers: **Base Layer (BL)** and **Enhancement Layers (EL)**.
- Real-time frame processing and encoding analysis (PSNR, bitrate).
- Adaptive layer selection based on channel conditions.

### 2. CR-RSMA & Communication Model
- Implementation of Rate-Splitting Multiple Access (RSMA) for enhanced spectral efficiency.
- Cognitive Radio integration: Primary User (PU) protection and Secondary User (SU) opportunistic access.
- Dynamic channel modeling for UAV mobility (LoS/NLoS components).

### 3. Optimization Suite
- **SCA Optimizer**: Solves non-convex resource allocation problems.
- **Objective Functions**:
  - **W-Sum**: Weighted Sum of QoE for maximum system utility.
  - **Max-Min**: Fair allocation ensuring the minimum user performance is maximized.
- **Fairness Metrics**: Jain’s Fairness Index integration.

### 4. Visualization & Analytics
- Automated plotting of PSNR, bitrates, and QoE over time.
- Comparison of different optimization modes (W-Sum vs. Max-Min).
- Video encoding statistics reporting.

---

## 📁 Project Structure

```text
├── main.py                # Main entry point for simulation
├── config.py              # System parameters and configuration
├── video/                 # Video processing module
│   ├── svc_encoder.py     # SVC layer management
│   ├── video_input.py     # Opencv video reading utils
│   └── frame_processor.py # Macroblock analysis
├── optimization/          # Mathematical models
│   ├── sca_optimizer.py   # Successive Convex Approximation logic
│   ├── uav_channel_model.py # UAV trajectory and gains
│   └── qoe_fairness_model.py # QoE and Fairness calculations
├── visualization/         # Plotting utilities
├── results/               # Output directory for plots and stats
└── documents/             # Project documentation and references
```

---

## Configuration
You can adjust system parameters in `config.py`, including:
- Network bandwidth (`B`) and noise power (`SIGMA2`).
- UAV power constraints (`P_S_MAX`, `P_FLY`).
- SVC layer bitrates (`R_LAYER`).
- QoE weights (`A_U`, `B_U`, `C_U`, etc.).

---

## Results Summary
After execution, check `results/cr_rsma_video_1pu_2su.png` for performance visualizations and `results/video_encoding_stats.txt` for detailed metrics.
