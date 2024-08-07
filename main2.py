from data.generate_data import *
from model.pgd_approximator2 import *
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import  DataLoader, random_split


target_matrix,dataset = generate_data(sample_size=10000)

# Split dataset
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

# _,first_iteration=dataset[0]
# print(first_iteration)
# _,second_iteration=dataset[2]
# print(np.mean((first_iteration.numpy()-second_iteration.numpy())**2))

# Instantiate the model
model = PGDAppproximator2(t=3)

class MeanAbsolutePercentageLoss(nn.Module):
    def __init__(self):
        super(MeanAbsolutePercentageLoss, self).__init__()

    def forward(self, y_pred, y_true):
        # Ensure y_true is not zero to avoid division by zero
        epsilon = 1e-8
        percentage_errors = torch.abs((y_true - y_pred) / (y_true + epsilon))
        return torch.mean(percentage_errors)
# # Define loss function and optimizer
# criterion = MeanAbsolutePercentageLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.0001)

# # Train the model
# train_losses = train_model(model, train_loader, criterion, optimizer, num_epochs=10)

# # Test the model
# test_loss = test_model(model, test_loader, criterion)

# # Plot the losses
# plot_losses(train_losses,test_loss)

# study effectivness of numbers of iterations

# Define range of t values to try
t_values = [1, 2, 3, 5, 10]

# Define other training parameters
num_epochs = 300
k = num_epochs # k-th epoch to plot

# Initialize lists to store losses for each t
losses_by_t = [[] for _ in range(len(t_values))]

# Loop over different t values
for i, t in enumerate(t_values):
    # Initialize the model
    model = PGDAppproximator2(t)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_losses = train_model(model, train_loader,test_loader, criterion, optimizer, num_epochs=num_epochs)[0]

    # Test the model
    test_loss = test_model(model, test_loader, criterion)

    # plot_losses(train_losses,test_loss)

    # Store losses for plotting
    losses_by_t[i] = train_losses

# Plot losses for each t at the k-th epoch
plot_losses_iterations_k(losses_by_t, t_values, epoch_k=k)

plot_log_losses_iterations_k(losses_by_t, t_values, epoch_k=k)
