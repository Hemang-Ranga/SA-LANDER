# SA-LANDER: Semantic-Adaptive Bounding for Federated Class-Incremental Learning
**Extension of LANDER (Tran et al., CVPR 2024)**  
**Author:** Hemang Ranga
 
## Overview

SA-LANDER extends the LANDER framework for Federated Class-Incremental Learning (FCIL) by replacing the fixed bounding radius (`r = 0.015`) with an **adaptive per-class bounding radius** tracked via Exponential Moving Average (EMA). This single modification improves average accuracy by **+7.51%** and reduces forgetting by **-12.82%** on CIFAR-100 compared to paper-reported LANDER results.

## Repository Structure

```
├── LANDER_reproduce.ipynb              # Reproduced the result of the original baseline model(with reduced parameters, compute)
├── sa-lander-normalized.ipynb          # Main SA-LANDER notebook (CIFAR-100, 80 rounds)
├── landervssa-landerorganclip.ipynb     # OrganAMNIST medical dataset comparison
├── SA_LANDER_Report.pdf                # Final technical report
├── requirements.txt                     # Python dependencies
├── README.md                            # This file
└── results/
    ├── cifar100_results.png             # CIFAR-100 comparison plots
    └── organ_results.png                # OrganAMNIST comparison plots
```

## Key Results

### CIFAR-100 (Non-IID, beta=0.5, 5 tasks)

| Method       | Avg Accuracy (↑) | Forgetting (↓) |
|-------------|------------------|----------------|
| TARGET       | 33.33%           | 39.23%         |
| LANDER       | 48.23%           | 30.61%         |
| **SA-LANDER**| **55.74%**       | **17.79%**     |

### OrganAMNIST (Same Budget Comparison)

| Method       | Avg Accuracy (↑) |
|-------------|------------------|
| LANDER       | 68.32%           |
| **SA-LANDER**| **70.66%** (+2.34%) |

## Running the Code

### Prerequisites
- Kaggle account with GPU (T4 x2 recommended)
- HuggingFace account and API token (for CLIP model download)

### Setup on Kaggle

1. Upload the notebook (`.ipynb` file) to Kaggle
2. Enable GPU: **Settings → Accelerator → GPU T4 x2**
3. Add HuggingFace token: **Add-ons → Secrets → Add `HF_TOKEN`**
4. Run all cells sequentially

### CIFAR-100 Experiment
- Open `sa-lander-normalized.ipynb`
- Default config: 80 rounds/task, 5 clients, 5 tasks, beta=0.5
- Estimated runtime: ~5-6 hours on T4

### OrganAMNIST Experiment
- Open `landervssa-landerorganclip.ipynb`
- Runs BOTH LANDER and SA-LANDER with identical budget
- Default config: 30 rounds/task, 3 clients, 3 tasks
- Estimated runtime: ~2 hours on T4

## Method Summary

### What SA-LANDER Changes
LANDER uses a fixed bounding radius $r = 0.015$ for all classes:

$$\mathcal{B}(f, e) = \max(0, \|e_y - W(f)\|^2 - 0.015)$$

SA-LANDER replaces this with adaptive per-class radii tracked via Exponential Moving Average (EMA):

$$d_y = \text{mean\_batch}(\|e_y - W(f)\|^2)$$
$$r_y = 0.9r_y + 0.1d_y$$
$$r_y = \text{clamp}(r_y, 0.005, 0.05)$$
$$\mathcal{B}_{\text{adaptive}} = \text{mean}(\max(0, \text{dist}_i - r_{y_i}))$$

### What Stays the Same
- LTE anchors from CLIP (fixed, not adapted)
- NoisyLayerGenerator architecture
- FedAvg aggregation protocol
- All loss weights and hyperparameters
- ResNet-18 backbone

## Datasets

### CIFAR-100
- Automatically downloaded via `torchvision.datasets.CIFAR100`
- 100 classes, 50K train / 10K test images, 32x32 RGB

### OrganAMNIST
- Automatically downloaded via `medmnist.OrganAMNIST`
- 11 organ classes from CT scans, ~58K images, 28x28 grayscale

## Dependencies

See `requirements.txt`. All dependencies are available on Kaggle by default except:
- `transformers` (for CLIP text encoder)
- `medmnist` (for OrganAMNIST, medical notebook only)

## Acknowledgments
This research project was developed as part of the CS6450 coursework at the Indian Institute of Technology Hyderabad (IITH), under the guidance of Prof. Dr. C. Krishna Mohan.

## Citation

```bibtex
@inproceedings{tran2024lander,
  title={Text-Enhanced Data-free Approach for Federated Class-Incremental Learning},
  author={Tran, Minh-Tuan and Le, Trung and Le, Xuan-May and Harandi, Mehrtash and Phung, Dinh},
  booktitle={CVPR},
  pages={23870--23880},
  year={2024}
}
```
