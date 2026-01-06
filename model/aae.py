import math
import torch
from torch import nn
from utils.config import config
from inspect import isfunction
from model.blocks import GroupNorm, Upsample, Downsample,ResidualBlock


class Encoder(nn.Module):
    def __init__(self, image_channels = 1, latent_dim =config.latent_dim):
        super(Encoder, self).__init__()
        channels = [16, 32, 64]
        num_res_blocks = 1
        layers = [nn.Conv3d(image_channels, channels[0], 3, 1, 1)]
        for i in range(len(channels)-1):
            in_channels = channels[i]
            out_channels = channels[i + 1]
            for j in range(num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
            if i != len(channels)-1:
                layers.append(Downsample(channels[i+1]))
        layers.append(ResidualBlock(channels[-1], channels[-1]))
        layers.append(GroupNorm(channels[-1]))
        layers.append(nn.LeakyReLU())
        layers.append(nn.Conv3d(channels[-1], latent_dim, 3, 1, 1))
        self.model = nn.Sequential(*layers)
       
    def forward(self, x):
        out = self.model(x)
        return out

class Decoder(nn.Module):
    def __init__(self, image_channels = 1, latent_dim = config.latent_dim):
        super(Decoder, self).__init__()
        channels = [64, 32, 16]
        num_res_blocks = 1

        in_channels = channels[0]
        layers = [nn.Conv3d(latent_dim, in_channels, 3, 1, 1),
                  ResidualBlock(in_channels, in_channels)]

        for i in range(len(channels)):
            out_channels = channels[i]
            for j in range(num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
            if i != 0:
                layers.append(Upsample(in_channels))

        layers.append(GroupNorm(in_channels))
        layers.append(nn.LeakyReLU())
        layers.append(nn.Conv3d(in_channels, image_channels, 3, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class AAE(nn.Module):
    def __init__(self):
        super(AAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, data):
        encoded_data = self.encoder(data)
        decoded_data = self.decoder(encoded_data)
        return decoded_data
    
class Discriminator(nn.Module):
    def __init__(self, image_channels = 1, channels = [16, 32, 64, 128]):
        super(Discriminator, self).__init__()

        layers = [nn.Conv3d(image_channels, channels[0], 4, 2, 1), nn.LeakyReLU(0.2)]
        layers += [
            nn.Conv3d(channels[0], channels[1], 4, 2, 1, bias=False),
            nn.BatchNorm3d(channels[1]),
            nn.LeakyReLU(0.2, True)
        ]
        layers += [
            nn.Conv3d(channels[1], channels[2], 4, 2, 1, bias=False),
            nn.BatchNorm3d(channels[2]),
            nn.LeakyReLU(0.2, True)
        ]
        layers += [
            nn.Conv3d(channels[2], channels[3], 4, 2, 1, bias=False),
            nn.BatchNorm3d(channels[3]),
            nn.LeakyReLU(0.2, True)
        ]

        layers.append(nn.Conv3d(channels[3], image_channels, 4, 2, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
