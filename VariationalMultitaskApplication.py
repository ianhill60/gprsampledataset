import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data.dataset import random_split
import gpytorch
from sklearn.model_selection import train_test_split
import tqdm


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


# device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_tasks = 3
num_latents = 3
xdata = pd.read_csv('xdf.csv', index_col=0)
ydata = pd.read_csv('ydf.csv', index_col=0)
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
train_x = scaler.transform(xtr1).double().cuda()
test_x = scaler.transform(xte1).double().cuda()
train_y = scaler1.transform(ytr1).double()
test_y = scaler1.transform(yte1).double()


class MultitaskGPModel(gpytorch.models.ApproximateGP):
    def __init__(self):
        # Let's use a different set of inducing points for each latent function
        inducing_points = torch.rand(num_latents, 20, 12)

        # We have to mark the CholeskyVariationalDistribution as batch
        # so that we learn a variational distribution for each task
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([num_latents])
        )

        # We have to wrap the VariationalStrategy in a LMCVariationalStrategy
        # so that the output will be a MultitaskMultivariateNormal rather than a batch output
        variational_strategy = gpytorch.variational.LMCVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            ),
            num_tasks=num_tasks,
            num_latents=num_latents,
            latent_dim=-1
        )

        super().__init__(variational_strategy)

        # The mean and covariance modules should be marked as batch
        # so we learn a different set of hyperparameters
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_latents]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_latents])),
            batch_shape=torch.Size([num_latents])
        )

    def forward(self, x):
        # The forward function should be written as if we were dealing with each output
        # dimension in batch
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# Estimate a and b
torch.manual_seed(42)
model = MultitaskGPModel().double().cuda()
likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=3).cuda()


class IndependentMultitaskGPModel(gpytorch.models.ApproximateGP):
    def __init__(self):
        # Let's use a different set of inducing points for each task
        inducing_points = torch.rand(num_tasks, 20, 12)

        # We have to mark the CholeskyVariationalDistribution as batch
        # so that we learn a variational distribution for each task
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([num_tasks])
        )

        variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            ),
            num_tasks=num_tasks,
        )

        super().__init__(variational_strategy)

        # The mean and covariance modules should be marked as batch
        # so we learn a different set of hyperparameters
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_tasks]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_tasks])),
            batch_shape=torch.Size([num_tasks])
        )

    def forward(self, x):
        # The forward function should be written as if we were dealing with each output
        # dimension in batch
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# smoke_test = ('CI' in os.environ)
num_epochs = 1  # if smoke_test else 500

model.train()
likelihood.train()

optimizer = torch.optim.Adam([
    {'params': model.parameters()},
    {'params': likelihood.parameters()},
], lr=0.1)

# Our loss object. We're using the VariationalELBO, which essentially just computes the ELBO
mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0)).cuda()

# We use more CG iterations here because the preconditioner introduced in the NeurIPS paper seems to be less
# effective for VI.
epochs_iter = tqdm.tqdm_notebook(range(num_epochs), desc="Epoch")
val_loss = []
for i in epochs_iter:
    # Within each iteration, we will go over each minibatch of data
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    epochs_iter.set_postfix(loss=loss.item())
    loss.backward()
    optimizer.step()

    val_losses.append(loss)
    validation_loss = np.mean(val_losses)
    validation_losses.append(validation_loss)

    print(f"[{i + 1}] Validation loss: {validation_loss:.3f}")

print(model.state_dict())
