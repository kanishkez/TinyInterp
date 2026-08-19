import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class LinearProbe(nn.Module):
    def __init__(self, d_model: int, num_classes: int = 2):
        super().__init__()
        self.linear = nn.Linear(d_model, num_classes)
        
    def forward(self, x): return self.linear(x)

def train_probe(activations, labels, epochs=10, lr=1e-3, batch_size=32):
    d_model = activations.shape[-1]
    num_classes = len(torch.unique(labels))
    probe = LinearProbe(d_model, num_classes).to(activations.device)
    optimizer = optim.AdamW(probe.parameters(), lr=lr)
    dataset = TensorDataset(activations, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    probe.train()
    for _ in range(epochs):
        for bx, by in loader:
            optimizer.zero_grad()
            loss = nn.CrossEntropyLoss()(probe(bx), by)
            loss.backward()
            optimizer.step()
    return probe
