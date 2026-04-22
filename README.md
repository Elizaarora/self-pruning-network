# Self-Pruning Neural Network on CIFAR-10

**Tredence AI Engineering Internship — Case Study Submission**

---

## What This Does

A feed-forward neural network that **learns to prune itself during training** using learnable sigmoid gates and a combined L1 + sharpening sparsity loss.

- Every weight $w_{ij}$ is multiplied by a learnable gate $g_{ij} = \sigma(\tau \cdot s_{ij})$
- An L1 sparsity penalty pushes gate scores negative during training → gates collapse to ~0
- The sharpening term $g(1-g)$ forces gates to commit to 0 or 1 — no undecided gates
- No post-training pruning step — sparsity emerges naturally from the loss

---

## Results

| Lambda (λ) | Test Accuracy (%) | Sparsity (%) |
|:----------:|:-----------------:|:------------:|
| 1e-4 | 55.60 | 41.63 |
| 5e-4 | 55.92 | 43.20 |
| 1e-3 | 55.85 | 43.31 |

**Over 40% of weights pruned with no accuracy loss.**

---

## Gate Distribution (Best Model)

The bimodal distribution confirms successful self-pruning:

![alt text](image-1.png)

- **Spike at ≈ 0** → pruned (redundant) weights
- **Cluster near 1** → active (necessary) weights
- **Empty middle** → sharpening term forced decisive gates

---

## Architecture

```
Input (3 × 32 × 32)
    ↓  Flatten → 3072
PrunableLinear(3072 → 1024)  →  ReLU
PrunableLinear(1024 → 512)   →  ReLU
PrunableLinear(512  → 256)   →  ReLU
PrunableLinear(256  → 10)
```

Approximately ~3.8M learnable parameters per weight-gate pair in the network.

---

## How to Run

```bash
pip install -r requirements.txt
python solution.py
```

**Google Colab (T4 GPU recommended):**
```python
exec(open("solution.py").read())
```

---

## Key Design Decisions

| Component | Choice |
|:--|:--|
| Gate activation | `sigmoid(20 × gate_score)` — high temp snaps gates to 0 or 1 |
| Sparsity loss | L1 + sharpening `g(1−g)` — prunes AND forces bimodal distribution |
| Gate init | zeros → gate starts at 0.5, neutral |
| Lambda annealing | None — constant λ from epoch 1 |
| Threshold | 0.01 — gates below 1% magnitude counted as pruned |

---

## Files

| File | Description |
|:-----|:------------|
| `solution.py` | Complete script — model, training, evaluation, plot |
| `report.md` | Full case study report with theory and analysis |
| `requirements.txt` | Python dependencies |