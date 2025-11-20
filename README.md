Federated Domain Adaptation on OfficeHome using ViT + MAPA + PGD
================================================================

This repository implements a federated domain adaptation framework for the OfficeHome dataset combining:
• Vision Transformer (ViT-B/16)
• MAPA-style feature affinity alignment (via cosine similarity matrices)
• PGD adversarial augmentation for robustness
• Federated Averaging (FedAvg) across class-balanced clients

Goal: Train clients on a labeled source domain and transfer structure/knowledge to an unlabeled target domain without directly using target labels.

0. Quick Start Environment (Windows PowerShell)
----------------------------------------------
```powershell
conda create -n fedvit python=3.10 -y
conda activate fedvit
pip install -r requirements.txt
```
Optional acceleration:
```powershell
pip install accelerate
```
Verify GPU (optional):
```powershell
python - <<'PY'
import torch; print('CUDA available:', torch.cuda.is_available())
PY
```

1. Project Structure
--------------------
```
project/
│── train.py           # Main federated training script (OfficeHome ready)
│── requirements.txt   # Dependencies
│── README.md          # This documentation
│── OfficeHome/        # (Place the extracted dataset here)
│     ├── Art/
│     ├── Clipart/
│     ├── Product/
│     └── Real_World/
```
Script default root: `./OfficeHome`. You can override via `--root`.

2. OfficeHome Dataset
---------------------
Download: http://hemanthdv.org/officeHomeDataset.html (4 domains, 65 classes each).

Expected structure example (abbreviated):
```
OfficeHome/
    Art/
        Alarm_Clock/xxx.jpg
        Backpack/yyy.png
        ... (65 classes)
    Clipart/
    Product/
    Real_World/
```
Each domain has identical class names. Source domain images get integer labels; target domain images are marked -1 (unlabeled) inside the loader.

Placement: Extract dataset so that `train.py` sees paths like `./OfficeHome/Art/Alarm_Clock/...`.

3. Training Configuration
-------------------------
Config is passed as CLI args (see `python train.py --help`):
```
--source_domain Art
--target_domain Clipart
--num_clients 3
--epochs 5
--batch_size 16
--lr 1e-4
--pgd_eps 0.007843   (≈2/255)
--pgd_alpha 0.001960 (≈0.5/255)
--pgd_iter 10
--lambda_aff 1.0
```
Select GPU (PowerShell):
```powershell
$env:CUDA_VISIBLE_DEVICES = "0"
```

4. Components Overview
----------------------
• Dataset: `OfficeHomeDataset` loads domain folders; target domain produced with labels = -1.
• Client Split: `class_balanced_split` ensures each client gets roughly equal samples per class (stability in federated updates).
• Model: ViT backbone (`vit_base_patch16_224` pretrained) + linear classifier; feature embeddings used for affinity matrices.
• PGD: Iterative attack crafts adversarial variants of source images; used only during local client training.
• Affinity Alignment: Cosine similarity matrices A (clean), B (adv), C (target) with Frobenius norm alignment encouraging structural consistency and attack invariance.
• FedAvg: After each epoch, client weights averaged—global model broadcast back.
• Evaluation: Source-domain classification accuracy reported (target unlabeled by design).

5. Loss Details
---------------
Total client loss: `CrossEntropy + lambda_aff * AlignmentLoss`
Alignment loss: `||A-B||_F + ||A-C||_F + ||B-C||_F` scaled by batch².

6. Running Training
-------------------
Basic run:
```powershell
python train.py --root ./OfficeHome --source_domain Art --target_domain Clipart --epochs 5
```
Adjust PGD strength:
```powershell
python train.py --pgd_iter 20 --pgd_eps 0.01
```
Increase clients:
```powershell
python train.py --num_clients 5 --batch_size 32
```

7. Extending / Notes
--------------------
• To evaluate target adaptation, introduce pseudo-labeling or domain validation logic (not implemented here).
• To add mixed precision: integrate `torch.cuda.amp.autocast` around forward passes.
• To log metrics: wrap training loop with your preferred logger (e.g., `wandb`).

8. Troubleshooting
------------------
• Empty batches: Ensure each domain folder contains images in all 65 classes.
• Memory issues: Lower `--batch_size` or reduce PGD iterations.
• Slow training: Remove PGD (`--pgd_iter 0` after adjusting code) or install `accelerate`.

9. Requirements
---------------
See `requirements.txt` for explicit dependencies (PyTorch, timm, scikit-learn, numpy, pillow, accelerate optional).


---
End of documentation.