import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 

torch.manual_seed(0)

x_train = torch.tensor([[1],[2],[3]])
y_train = torch.tensor([[2],[4],[6]])

w = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

optimizer = optim.SGD([w, b], lr=0.01) 

nb_epochs = 1999
for epoch in range(nb_epochs + 1):
    hypothesis = x_train * w + b
    cost = torch.mean((hypothesis - y_train) ** 2)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch {epoch:4d}/ {nb_epochs} Cost: {cost.item():.6f} w: {w.item():.3f} b: {b.item():.3f}')