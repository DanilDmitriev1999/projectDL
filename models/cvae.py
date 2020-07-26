import torch.nn as nn
import torch
from torch.autograd import Variable

INPUT_DIM_CVAE_FACES = (64*64*3) + 2
INPUT_DIM_CVAE_FACES_UNCAT = 64*64*3
HIDDEN_DIM_CVAE_FACES = 100


class CVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.en1 = nn.Sequential(
            nn.Linear(INPUT_DIM_CVAE_FACES, 1024),
            nn.LeakyReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 200),
            nn.ReLU()
        )
        self.en_mu = nn.Linear(200, HIDDEN_DIM_CVAE_FACES)
        self.en_std = nn.Linear(200, HIDDEN_DIM_CVAE_FACES)

        self.de1 = nn.Linear(HIDDEN_DIM_CVAE_FACES + 2, 200)
        self.de2 = nn.Sequential(
            nn.Linear(200, 512),
            nn.LeakyReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 1024),
            nn.LeakyReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, INPUT_DIM_CVAE_FACES_UNCAT)

        )
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.en1(x))
        return self.en_mu(h1), self.en_std(h1)

    def decode(self, z, labels):
        z = torch.cat((z, labels), dim=-1)

        h2 = self.relu(self.de1(z))
        return self.sigmoid(self.de2(h2))

    def gaussian_sampler(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, labels):
        mu, logvar = self.encode(x.view(-1, INPUT_DIM_CVAE_FACES))
        z = self.gaussian_sampler(mu, logvar)
        return self.decode(z, labels), mu, logvar

    def get_latent_code(self, x):
        mu, logvar = self.encode(x.view(-1, INPUT_DIM_CVAE_FACES))
        return self.gaussian_sampler(mu, logvar)