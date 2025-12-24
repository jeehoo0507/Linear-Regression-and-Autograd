import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

x_train = torch.FloatTensor([[1],[2],[3]])
y_train = torch.FloatTensor([[2],[4],[6]])

model = nn.Linear(1, 1)

print(list(model.parameters()))

optimizer = optim.SGD(model.parameters(), lr=0.01)  

nb_epochs = 1999
for epoch in range(nb_epochs + 1):
    prediction = model(x_train)
    cost = F.mse_loss(prediction, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        [w, b] = model.parameters()
        print(f'Epoch {epoch:4d}/ {nb_epochs} Cost: {cost.item():.6f} w: {w.item():.3f} b: {b.item():.3f}')