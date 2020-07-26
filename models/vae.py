import torch.nn as nn
import torch


class vanila_VAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        """
        помните, у encoder должны быть два "хвоста",
        т.е. encoder должен кодировать картинку в 2 переменные -- mu и logsigma
        """
        super(vanila_VAE, self).__init__()
        self.encoder_fc1 = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.LeakyReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, hidden_size),
            nn.ReLU()
        )
        self.relu = nn.ReLU()

        self._mu = nn.Linear(hidden_size, latent_size)
        self._logsigma = nn.Linear(hidden_size, latent_size)

        self.decoder_fc1 = nn.Sequential(
            nn.Linear(latent_size, 512),
            nn.LeakyReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 1024),
            nn.LeakyReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, input_size)

        )

    def encode(self, x):
        """
        в качестве ваозвращаемых переменных -- mu и logsigma
        """
        x = self.encoder_fc1(x)

        mu = self._mu(x)
        logsigma = self._logsigma(x)

        return mu, logsigma

    def gaussian_sampler(self, mu, logsigma):
        """
        Функция сэмплирует латентные векторы из нормального распределения с параметрами mu и sigma
        """
        if self.training:
            std = torch.exp(logsigma * 0.5)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        """
        <реализуйте forward проход декодера
        в качестве ваозвращаемой переменной -- reconstruction>
        """
        reconstruction = self.decoder_fc1(z)

        return torch.sigmoid(reconstruction)

    def forward(self, x):
        """
        в качестве ваозвращаемых переменных -- mu, logsigma и reconstruction>
        """
        mu, logsigma = self.encode(x)
        z = self.gaussian_sampler(mu, logsigma)
        reconstruction = self.decode(z)

        return reconstruction, mu, logsigma

    def get_latent_code(self, x):
        mu, logsigma = self.encode(x)
        latent_code = self.gaussian_sampler(mu, logsigma)
        return latent_code