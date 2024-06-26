from data.generate_data import *
from model.pgd_approximator import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

target_matrix,dataset = generate_data()

# Split dataset
train_size = int(0.8 * len(dataset))S
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Instantiate the model
model = PGDAppproximator(t=3)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
train_losses = train_model(model, train_loader, criterion, optimizer, num_epochs=10)

# Test the model
test_loss = test_model(model, test_loader, criterion)

# Plot the losses
plot_losses(train_losses)