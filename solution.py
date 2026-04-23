# ============================================================
# SELF-PRUNING NETWORK 
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

# ── CONFIG ─────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 40
BATCH_SIZE = 128
LR = 1e-3

#  Proper lambda range
LAMBDAS = [1e-4, 5e-4, 1e-3]

THRESHOLD = 1e-2

#  STRONG sharpening
TEMP = 20.0

print("Device:", DEVICE)


# ── 1. Prunable Layer ──────────────────────────────────
class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) * 0.02
        )
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Neutral initialization
        self.gate_scores = nn.Parameter(
            torch.zeros(out_features, in_features)
        )

    def forward(self, x):
        gates = torch.sigmoid(TEMP * self.gate_scores)
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)

    def get_gates(self):
        return torch.sigmoid(TEMP * self.gate_scores)


# ── 2. Network ─────────────────────────────────────────
class SparseNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = PrunableLinear(3072, 1024)
        self.fc2 = PrunableLinear(1024, 512)
        self.fc3 = PrunableLinear(512, 256)
        self.fc4 = PrunableLinear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

    # STRONG sparsity + sharpening
    def sparsity_loss(self):
        total = 0
        count = 0

        for m in self.modules():
            if isinstance(m, PrunableLinear):
                g = m.get_gates()

                #  L1 sparsity
                total += g.sum()

                # Sharpening term (push to 0 or 1)
                total += (g * (1 - g)).sum()

                count += g.numel()

        # normalize + amplify
        return (total / count) * 5

    def get_all_gates(self):
        gates = []
        for m in self.modules():
            if isinstance(m, PrunableLinear):
                gates.append(
                    m.get_gates().detach().cpu().numpy().ravel()
                )
        return np.concatenate(gates)


# ── 3. Data ────────────────────────────────────────────
def get_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train = torchvision.datasets.CIFAR10(
        "./data", train=True, download=True, transform=transform
    )
    test = torchvision.datasets.CIFAR10(
        "./data", train=False, download=True, transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train, batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=256
    )

    return train_loader, test_loader


# ── 4. Training ────────────────────────────────────────
def train_model(lam, train_loader):
    model = SparseNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        correct, total = 0, 0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()

            out = model(x)
            ce_loss = F.cross_entropy(out, y)
            sp_loss = model.sparsity_loss()

            #  No annealing
            loss = ce_loss + lam * sp_loss

            loss.backward()
            optimizer.step()

            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        if epoch % 5 == 0:
            print(f"[λ={lam}] Epoch {epoch}: "
                  f"Train Acc={100*correct/total:.2f}% "
                  f"Sparsity Loss={sp_loss.item():.4f}")

    return model


# ── 5. Evaluation ──────────────────────────────────────
def evaluate(model, test_loader):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    acc = 100 * correct / total

    gates = model.get_all_gates()
    sparsity = (gates < THRESHOLD).mean() * 100

    return acc, sparsity, gates


# ── 6. Main ────────────────────────────────────────────
train_loader, test_loader = get_data()

results = []

for lam in LAMBDAS:
    print("\n==============================")
    print(f"Training with lambda = {lam}")
    print("==============================")

    model = train_model(lam, train_loader)
    acc, sparsity, gates = evaluate(model, test_loader)

    results.append((lam, acc, sparsity, gates))

    print(f"Test Acc: {acc:.2f}% | Sparsity: {sparsity:.2f}%")


# ── 7. Results ─────────────────────────────────────────
print("\nFINAL RESULTS")
print("Lambda\tAccuracy\tSparsity (%)")
for lam, acc, sp, _ in results:
    print(f"{lam}\t{acc:.2f}\t\t{sp:.2f}")


# ── 8. Plot ────────────────────────────────────────────
best = max(results, key=lambda x: x[1])
gates = best[3]

plt.figure(figsize=(8, 5))
plt.hist(gates, bins=100)
plt.axvline(THRESHOLD, color='r', linestyle='--', label="Threshold")
plt.yscale("log")
plt.xlabel("Gate Value")
plt.ylabel("Count (log scale)")
plt.title("Gate Distribution (Best Model)")
plt.legend()
plt.show()
