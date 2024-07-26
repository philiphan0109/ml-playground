import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from model import approximator

x_train = torch.tensor(np.arange(-1000, 1000, 30), dtype=torch.float32).view(-1, 1)

def function(x):
    return x**3

y_train = function(x_train)

x_mean, x_std = x_train.mean(), x_train.std()
y_mean, y_std = y_train.mean(), y_train.std()

x_train_normalized = (x_train - x_mean) / x_std
y_train_normalized = (y_train - y_mean) / y_std

dataset = TensorDataset(x_train_normalized, y_train_normalized)
dataloader = DataLoader(dataset, 128, shuffle=True)

model = approximator()
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.001) 
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=100, factor=0.5)

losses = []
model.train()
for epoch in range(10000):
    epoch_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        
        (x, y) = batch
        logits = model(x)
        loss = criterion(logits, y)
        epoch_loss += loss.item()
        
        loss.backward()
        optimizer.step()
    
    # scheduler.step(epoch_loss)
    losses.append(epoch_loss / len(dataloader))
        
    if epoch % 2500 == 0: 
        print(f"epoch: {epoch} | loss: {epoch_loss / len(dataloader):.6f} | lr: {optimizer.param_groups[0]['lr']:.6f}")

plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')
plt.show()

model.eval()
with torch.no_grad():
    x_test = torch.tensor(np.arange(-1000, 1000, 1), dtype=torch.float32).view(-1, 1)
    x_test_normalized = (x_test - x_mean) / x_std
    y_test_normalized = model(x_test_normalized)
    y_test = y_test_normalized * y_std + y_mean  # Denormalize the output

# Plot true function for comparison
y_true = function(x_test)
plt.figure(figsize=(12, 6))
plt.plot(x_test, y_true, color='green', label='True Function', zorder=1)
plt.plot(x_test, y_test, color='blue', label='Model Prediction', zorder=2)
plt.scatter(x_train, y_train, color='red', label='Training Data', zorder=3, s=10)
plt.title('Model Prediction vs True Function')
plt.legend()
plt.show()