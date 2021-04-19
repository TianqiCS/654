# Copyright 2019 RBC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# architectures.py is used to create the generator and discriminator used in the generative models
import torch.nn as nn
import torch
import numpy as np

class Generator(nn.Module):
    def __init__(self, latent_size, output_size, conditional=True):
        super().__init__()
        z = latent_size
        d = output_size
        if conditional:
            z = z + 1
        else:
            d = d + 1
        self.main = nn.Sequential(
            nn.Linear(z, 2 * latent_size),
            nn.ReLU(),
            nn.Linear(2 * latent_size, d))

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self, input_size, wasserstein=False):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(input_size + 1, int(input_size / 2)),
            nn.ReLU(),
            nn.Linear(int(input_size / 2), 1))

        if not wasserstein:
            self.main.add_module(str(3), nn.Sigmoid())

    def forward(self, x):
        return self.main(x)


# For MNIST
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.zero_()
img_shape = (1, 28, 28)

class GeneratorCNN(nn.Module):
    def __init__(self):
        super(GeneratorCNN, self).__init__()

        self.label_emb = nn.Embedding(2, 2)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(100 + 2, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *img_shape)
        return img


class DiscriminatorCNN(nn.Module):
    def __init__(self):
        super(DiscriminatorCNN, self).__init__()

        self.label_embedding = nn.Embedding(2, 2)

        self.model = nn.Sequential(
            nn.Linear(2 + int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        return validity
