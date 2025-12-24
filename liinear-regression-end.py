import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class CustomDataset(Dataset):
    def __init__(self):
        self.x_data = torch.FloatTensor([[1],[2],[3]])
        self.y_data = torch.FloatTensor([[2],[4],[6]])

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self,idx):
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])
        return x,y
class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1,1)

    def forward(self,x):
        return self.linear(x)

model = LinearRegression()

dataset = CustomDataset()
dataloader = DataLoader(dataset,batch_size=2,shuffle = True)

optimizer = optim.SGD(model.parameters(), lr =0.01)

nb_epochs = 20000
for epoch in range(nb_epochs+1):
    for batch_idx ,samples in enumerate(dataloader):
        print(batch_idx)
        print(samples)

        x_train, y_train = samples

        prediction = model(x_train)

        cost = F.mse_loss(prediction,y_train)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'
        .format(epoch, nb_epochs, batch_idx+1, len(dataloader),
        cost.item()
        ))
        
print(list(model.parameters()))