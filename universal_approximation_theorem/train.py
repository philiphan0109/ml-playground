import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from model import relumodel

x_train = torch.tensor(np.arange(-50, 50, 1), dtype=torch.float32).view(-1, 1)
y_train = []
sign = 1
for i in range(len(x_train)):
    if i % 10 == 0:
        sign *= -1
    y_train.append(sign)
y_train = torch.tensor(y_train, dtype = torch.float32).view(-1, 1)
dataset = TensorDataset(x_train, y_train)
dataloader = DataLoader(dataset, 128, shuffle = True)

model = relumodel()
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(params=model.parameters(), lr = 0.01)


losses = []
model.train()
for epoch in range(0,10001):
    epoch_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        
        (x, y) = batch
        logits = model(x)
        loss = criterion(logits, y)
        epoch_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
    if epoch % 2500 == 0:
        print(f"epoch: {epoch} | loss: {epoch_loss}")
        
plt.plot(losses)
plt.show()

x_test = torch.tensor(np.arange(-50, 50, 0.01), dtype=torch.float32).view(-1, 1)
y_test = model(x_test).detach().numpy()

plt.plot(x_test, y_test)
plt.scatter(x_train, y_train)
plt.show()