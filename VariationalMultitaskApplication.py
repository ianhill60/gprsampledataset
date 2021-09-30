import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data.dataset import random_split
import gpytorch
import sklearn
from sklearn.model_selection import train_test_split
import tqdm
from scipy.cluster.vq import kmeans2

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

num_inducing_pts = 1000            # Number of inducing points in each hidden layer
num_tasks = 3
num_latents = 4 # how many linear functions are learned to correlate tasks. 
xdata = pd.read_csv('xdf.csv',index_col=0)
ydata = pd.read_csv('ydf.csv',index_col=0)
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
train_n = len(train_x) ##############################################################################333

train_dataset = TensorDataset(train_x, train_y)
val_dataset = TensorDataset(test_x, test_y)

train_loader = DataLoader(dataset=train_dataset, batch_size=1000, drop_last=False)
val_loader = DataLoader(dataset=val_dataset, batch_size=1000, drop_last=False)

class MultitaskGPModel(gpytorch.models.ApproximateGP):
    def __init__(self):

        # Let's use a different set of inducing points for each latent function
        # Use k-means to initialize inducing points (only helpful for the first layer)
        inducing_points = (train_x[torch.randperm(min(1000 * 100, train_n))[0:num_inducing_pts], :])
        inducing_points = inducing_points.clone().data.cpu().numpy()
        inducing_points = torch.tensor(kmeans2(train_x.data.cpu().numpy(),
                               inducing_points, minit='matrix')[0])

        if torch.cuda.is_available():
          inducing_points = inducing_points.cuda()


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

torch.manual_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = MultitaskGPModel().double().to(device)
likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=3).to(device)

#smoke_test = ('CI' in os.environ)
num_epochs = 1000 #if smoke_test else 500


model.train()
likelihood.train()

optimizer = torch.optim.Adam([
    {'params': model.parameters()},
    {'params': likelihood.parameters()},
], lr=.001)

# Our loss object. We're using the VariationalELBO, which essentially just computes the ELBO
mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0)).cuda()

# We use more CG iterations here because the preconditioner introduced in the NeurIPS paper seems to be less
# effective for VI.
epochs_iter = tqdm.tqdm_notebook(range(num_epochs), desc="Epoch")
val_loss = []
for i in epochs_iter:
  batch_losses = []
  for x_batch, y_batch in train_loader:
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)
    # Within each iteration, we will go over each minibatch of data
    optimizer.zero_grad()
    output = model(x_batch)
    loss = -mll(output, y_batch)
    epochs_iter.set_postfix(loss=loss.item())
    loss.backward()
    optimizer.step()
  
    batch_losses.append(loss)

 
# Set into eval mode
model.eval().to(device)
likelihood.eval().to(device)

# Make predictions
means = torch.empty((1,3)).to(device)
with torch.no_grad():
    for x_batch, y_batch in val_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        preds = likelihood(model(x_batch))
        means = torch.cat([means, preds.mean])
means = means[1:]
means = means.to('cpu')
test_y = test_y.to('cpu')
print(sklearn.metrics.mean_absolute_error(scaler1.inverse_transform_mean(test_y), scaler1.inverse_transform_mean(means), multioutput='raw_values'))
print(sklearn.metrics.r2_score(scaler1.inverse_transform_mean(test_y), scaler1.inverse_transform_mean(means)))

