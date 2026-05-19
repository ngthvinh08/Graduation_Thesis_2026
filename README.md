# CR-RSMA vs CR-NOMA Fairness Simulation

Simulation comparing **Cognitive Radio Rate-Splitting Multiple Access (CR-RSMA)** and **Cognitive Radio Non-Orthogonal Multiple Access (CR-NOMA)** for multimedia fairness optimization.

## Project Overview

- **System**: 1 Primary User (PU) + K Secondary Users (SUs) in underlay cognitive radio setup
- **Objective**: Max-Min Fair (MMF) resource allocation maximizing minimum rate: max min_k R_k
- **Video Codec**: Scalable Video Coding (SVC) with 4 layers (Base Layer + 3 Enhancement Layers)
- **Video Format**: QCIF (176×144), 30 fps, GOP size 8
- **Performance Metrics**: Shannon capacity (no finite block length), mapped to PSNR via SVC layers

## Configuration

Configure system parameters in [config.py](config.py):
- **Video layers**: Bitrates per layer (kbps), PSNR thresholds
- **Channel parameters**: SNR range, interference threshold (I_th)
- **Simulation**: Monte Carlo realizations, fairness weights

## Simulation Parameters

Defined in [main.py](main.py):
- **SNR Range**: 0-30 dB (2 dB step)
- **K Range** (# of SUs): 2, 3, 4, 5, 6
- **Interference Threshold (I_th)**: 0.05–1.0 (15 points)
- **Monte Carlo Runs**: 200 realizations per point

## Results

Simulation outputs saved to `sim_results.npz` containing:
1. MMF objective vs SNR
2. MMF objective vs I_th 
3. MMF objective vs number of users (K)

## Key Concepts

- **SVC 4-Layer Structure**:
  - BL (Base Layer): 71.38 kbps, PSNR 29.87 dB
  - EL1, EL2, EL3 (Enhancement Layers): Incremental quality improvement up to PSNR 41.19 dB
  
- **Fairness**: Weighted fairness with w_p = w_s = 0.5 (equal PU/SU weights)

## Usage

Run simulation:
```bash
python main.py