import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data.dataset import random_split
import gpytorch
from sklearn.model_selection import train_test_split


class StandardScaler:
    def fit(self, x):
        self.mean = x.mean(0, keepdim=True)
        self.std = x.std(0, unbiased=False, keepdim=True)

    def transform(self, x):
        x -= self.mean
        x /= (self.std + 1e-7)
        return x

    def inverse_transform_mean(self, x):
        '''To transform mean'''
        return x * (self.std + 1e-7) + self.mean

    def inverse_transform_std(self, x):
        '''To transform standard deviation'''
        return x * (self.std + 1e-7)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

xdata = pd.read_csv('xdf.csv', index_col=0) #update with path
ydata = pd.read_csv('ydf.csv', index_col=0) #update with path
xdat = np.array(xdata)
ydat = np.array(ydata)
xtr, xte, ytr, yte = train_test_split(xdat, ydat, test_size=0.2, random_state=17)
scaler = StandardScaler()
scaler1 = StandardScaler()
xtr1 = torch.tensor(xtr)
xte1 = torch.tensor(xte)
ytr1 = torch.tensor(ytr)
yte1 = torch.tensor(yte)
scaler.fit(xtr1)
scaler1.fit(ytr1)
train_x = scaler.transform(xtr1).double()
test_x = scaler.transform(xte1).double()
train_y = scaler1.transform(ytr1).double()
test_y = scaler1.transform(yte1).double()

train_dataset = TensorDataset(train_x, train_y)
val_dataset = TensorDataset(test_x, test_y)

train_loader = DataLoader(dataset=train_dataset, batch_size=20, drop_last=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=20, drop_last=True)


class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=3
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=3, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


# Estimate a and b
torch.manual_seed(42)

x_batch1 = train_x[0:20]
y_batch1 = train_y[0:20]
likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=3).to(device)
model = MultitaskGPModel(x_batch1, y_batch1, likelihood).to(device).double()
# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
# optimizer = optim.SGD(model.parameters(), lr=1e-1)
optimizer = optim.Adam(model.parameters(), lr=0.5)  # Includes GaussianLikelihood parameters

n_epochs = 50
training_losses = []
validation_losses = []
# print(model.state_dict())

for epoch in range(n_epochs):
    batch_losses = []
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        model.train()
        likelihood.train()
        output = model(x_batch)
        loss = -mll(output, y_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        batch_losses.append(loss)
        print(batch_losses)
    training_loss = np.mean(batch_losses)
    training_losses.append(training_loss)

    with torch.no_grad():
        val_losses = []
        for x_val, y_val in val_loader:
            x_val = x_val.to(device)
            y_val = y_val.to(device)
            model.eval()
            yhat = model(x_val)
            val_loss = -mll(yhat, y_val)
            val_losses.append(val_loss)
        validation_loss = np.mean(val_losses)
        validation_losses.append(validation_loss)

    print(f"[{epoch + 1}] Training loss: {training_loss:.3f}\t Validation loss: {validation_loss:.3f}")

print(model.state_dict())