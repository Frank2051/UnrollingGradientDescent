import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt


class PGDAppproximator2(nn.Module):
    def __init__(self, t):
        super(PGDAppproximator2, self).__init__()
        self.t = t
        self.x_layers = nn.ModuleList()  # ModuleList to hold dynamically created x linear layers
        self.y_layers = nn.ModuleList()  # ModuleList to hold dynamically created y linear layers
        self.mus = nn.ParameterList()    # ParameterList to hold the scalar parameters
        self.relu = nn.ReLU()

        # Create t pairs of linear layers and a parameter for each pair
        for i in range(t):
            self.x_layers.append(nn.Linear(100, 100).double())  # Linear layer for x with double precision
            self.y_layers.append(nn.Linear(75, 100).double())   # Linear layer for y with double precision
            self.mus.append(nn.Parameter(torch.ones(1).double()))  # Scalar parameter

    def forward(self, x, y):
        for i in range(self.t):
            x_linear = self.x_layers[i](x)  # Select linear layer for x
            y_linear = self.y_layers[i](y)  # Select linear layer for y
            mu = self.mus[i]  # Access the scalar parameter directly
            x = x - mu * (x_linear - y_linear)  # Subtract y_linear from x_linear
            x = self.relu(x)  # Apply ReLU activation
        return x
    

# Function to train the model
def train_model(model, train_loader,test_loader, criterion, optimizer, num_epochs=10):
    train_losses = []
    test_losses = []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for y, x in train_loader:
            optimizer.zero_grad()
            output = model(x.double(),y.double())  # Pass y and x as double precision
            loss = criterion(output, x.double())    # Use x as the target for loss calculation with double precision
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_epoch_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_epoch_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}')
        # # Testing Phase
        # model.eval()
        # epoch_test_loss = 0.0
        # with torch.no_grad():
        #     for x, y in test_loader:
        #         output = model(x.double(), y.double())  # Pass y and x as double precision
        #         loss = criterion(output, x.double())   # Use x as the target for loss calculation with double precision
        #         epoch_test_loss += loss.item()
        
        # avg_test_loss = epoch_test_loss / len(test_loader)
        # test_losses.append(avg_test_loss)
    return train_losses, test_losses

# Function to test the model
def test_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for y, x in test_loader:
            output = model( x.double(),y.double())  # Pass y and x as double precision
            test_loss += criterion(output, x.double()).item()  # Calculate test loss with double precision
    avg_test_loss = test_loss / len(test_loader)
    print(f'Test Loss: {avg_test_loss:.4f}')
    return avg_test_loss

# Function to plot losses
def plot_losses(train_losses, test_losses=None):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss', marker='o')
    if test_losses:
        plt.plot(test_losses, label='Test Loss', marker='o')
    plt.title('Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_losses_iterations_k(losses, t_values, epoch_k):
    plt.figure(figsize=(10, 6))
    for i, t in enumerate(t_values):
        plt.plot(losses[i], label=f't={t}')
    plt.xlabel(f'Epoch {epoch_k}')
    plt.ylabel('MSE Loss')
    plt.title(f'MSE Loss at Epoch {epoch_k} for Different t Values')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_log_losses_iterations_k(losses, t_values, epoch_k):

    plt.figure(figsize=(10, 6))
    for i, t in enumerate(t_values):
        plt.loglog(losses[i], label=f't={t}')  
    plt.xlabel(f'Epoch {epoch_k}')
    plt.ylabel('Log MSE Loss')
    plt.title(f'Log MSE Loss at Epoch {epoch_k} for Different t Values')
    plt.legend()
    plt.grid(True)
    plt.show()


def predict(model, inputs):
    # Ensure the model is in evaluation mode
    model.eval()
    
    # Disable gradient calculations for inference
    with torch.no_grad():
        # Initialize an empty list to store predictions
        predictions = []
        
        for x in inputs:
            # Ensure the input is a tensor and move it to the correct device (CPU/GPU)
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x)
            x = x.to(next(model.parameters()).device)
            
            # Forward pass through the model to get the prediction
            y_pred = model(x)
            
            # Append the prediction to the list
            predictions.append(y_pred)
    
    return predictions