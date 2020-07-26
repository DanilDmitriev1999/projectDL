import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def mse(x, y):
    y = y[0]
    loss = nn.MSELoss().to(device)
    return loss(x, y)


def KL_divergence(mu, logsigma):
    """
    часть функции потерь, которая отвечает за "близость" латентных представлений разных людей
    """
    loss = -0.5 * torch.sum(1 + 2 * logsigma - mu * mu - torch.exp(2 * logsigma))
    return loss


def log_likelihood(x, reconstruction):
    """
    часть функции потерь, которая отвечает за качество реконструкции (как mse в обычном autoencoder)
    """
    loss = F.binary_cross_entropy(reconstruction, x)
    return loss


def loss_vae(x, model_output):
    reconstruction = model_output[0]
    mu = model_output[1]
    logsigma = model_output[2]

    size = x.size(0) * x.size(1)
    return (KL_divergence(mu, logsigma) / size) + log_likelihood(x, reconstruction)