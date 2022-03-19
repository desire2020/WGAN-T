import torch
import torch.autograd as autograd
import torch.utils.data
import argparse
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import tqdm
from torch.optim import Adam
from modeling_euct import EuclideanTransformerPreTrainedModel, EuclideanTransformerModel, XLNetConfig, EuclideanTransformerTransposeLayer
import PIL.Image as image
from torchvision import datasets, transforms

def save_img(batch_of_img, batch_size, channel=1, file=None):
    imgs = np.transpose(batch_of_img, [0, 3, 1, 2])
    for i in range(batch_size):
        img = imgs[i]
        img = np.asarray(img, dtype=np.uint8)
        if channel == 3:
            r = image.fromarray(img[0]).convert('L')
            g = image.fromarray(img[1]).convert('L')
            b = image.fromarray(img[2]).convert('L')
        else:
            r = image.fromarray(img[0]).convert('L')
            g = r
            b = r
        img = image.merge("RGB", (r, g, b))
        if file is not None:
            img.save('./%s/sample%d.png' % (file, i), 'png')
    return img

def gradients(y, x):
    return autograd.grad(
                outputs=y, inputs=x, retain_graph=True,
                create_graph=True, grad_outputs=torch.ones_like(y), only_inputs=True)[0]

def build_convex_combination(x0, x1, critic):
    bsz = x0.size(0)
    myu = torch.rand(size=(bsz, 1, 1, 1), device=x0.device)
    x = (x0 * myu + x1 * (1.0 - myu)).detach()
    x.requires_grad_(True)
    y = critic(x)
    grad = gradients(y, x)[:, 1:-1, :].reshape((bsz, -1))
    grad_norm = torch.norm(grad, dim=-1)
    return grad_norm


class EuclideanTransformerGenerativeModel(EuclideanTransformerPreTrainedModel):
    def __init__(self, config: XLNetConfig, z_dim=128, starting_size=(7, 7)):
        super().__init__(config)
        self.config = config
        self.mem_len = config.mem_len
        self.reuse_len = config.reuse_len
        self.d_model = config.d_model
        self.same_length = config.same_length
        self.z_dim = z_dim
        self.attn_type = config.attn_type
        self.bi_data = config.bi_data
        self.clamp_len = config.clamp_len
        self.n_layer = config.n_layer
        self.input_proj = nn.Linear(z_dim, config.d_model)
        self.starting_size = starting_size
        self.layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.pool_emb = nn.Parameter(torch.FloatTensor(1, starting_size[0], starting_size[1], config.d_model))

        self.layer = nn.ModuleList([EuclideanTransformerTransposeLayer(config) for _ in range(config.n_layer)])
        self.dropout = nn.Dropout(config.dropout)
        self.output_layer = nn.Linear(config.d_model, 1)

        self.init_weights()

    def forward(self,
                z=None,
                batch_size=64,
                up_sampling=None
                ):
        if up_sampling is None:
            up_sampling = []
        if z is None:
            z = torch.randn(batch_size, self.z_dim, device=self.pool_emb.device)
        semantic_embeddings = self.input_proj(z).reshape(batch_size, 1, self.config.d_model)
        pixel_embeddings = self.pool_emb.expand(batch_size, self.starting_size[0], self.starting_size[1],
                                                self.config.d_model)

        for i, layer_module in enumerate(self.layer):
            batch_size, h_size, w_size, channel_size = pixel_embeddings.shape
            pixel_embeddings, semantic_embeddings = layer_module(
                pixel_embeddings=pixel_embeddings,
                semantic_embeddings=semantic_embeddings,
                up_sampling=(i in up_sampling)
            )
            pixel_embeddings = pixel_embeddings + semantic_embeddings.reshape(batch_size, 1, 1, self.config.d_model)
        y = self.output_layer(pixel_embeddings)

        return y


class Critic(EuclideanTransformerPreTrainedModel):
    def __init__(self, config: XLNetConfig):
        super().__init__(config)
        self.config = config
        self.model = EuclideanTransformerModel(config=config, in_channels=1)
        self.output_layer = nn.Linear(config.d_model, 1)
        self.init_weights()
        #####################################################################################

    def forward(self, x):
        ########################################Your Code####################################
        x = x.permute(0, 3, 1, 2)
        batch_size, C, H, W = x.shape
        pixel_embeddings, semantic_embeddings = self.model(x)
        semantic_embeddings = pixel_embeddings.mean(dim=1).mean(dim=1)
        x = self.output_layer(semantic_embeddings.reshape(batch_size, self.config.d_model))
        return x


class Settings(object):
    n_layers = 4
    d_model = 128
    batch_size = 16
    log_freq = 10


settings = Settings()


def main():
    iterations = np.float64(1.0)
    iterations_int = 0
    buffer_cleaning = 0
    transform = transforms.ToTensor()
    data_train = datasets.MNIST(root="./data/",
                                train=True,
                                transform=transform,
                                download=True)
    config = XLNetConfig.from_pretrained("xlnet-base-cased",
                                         n_layers=settings.n_layers, n_head=4,
                                         d_model=settings.d_model, d_inner=settings.d_model * 4
                                         )

    dataloader = DataLoader(
        data_train, batch_size=settings.batch_size, shuffle=True,
        num_workers=4, drop_last=True
    )
    critic = Critic(config)
    critic.cuda()
    generator = EuclideanTransformerGenerativeModel(config)
    generator.cuda()

    d_opt = Adam(lr=2e-5, betas=(0.5, 0.9), params=critic.parameters())
    g_opt = Adam(lr=2e-5, betas=(0.5, 0.9), params=generator.parameters())
    D_loss = []
    for epochs in range(100):
        iterator = tqdm.tqdm(dataloader)
        for i, (y_real, _) in enumerate(iterator):
            x0 = y_real[:, 0].cuda().unsqueeze(dim=-1)
            x1 = generator(batch_size=x0.size(0), up_sampling=[3]).sigmoid()
            iterations += 1.0
            iterations_int += 1
            real_score = critic(x0)
            fake_score = critic(x1)
            grad_norm = build_convex_combination(x0, x1, critic)
            lp = torch.relu(grad_norm - 1.0).sum()
            d_loss = -real_score.mean() + fake_score.mean()
            d_loss_train = d_loss + 1.0 * lp
            d_opt.zero_grad()
            d_loss_train.backward()
            d_opt.step()
            D_loss.append((-d_loss).detach().item())
            if len(D_loss) > 5000:
                D_loss.pop(0)
            if i % settings.log_freq == settings.log_freq - 1:
                iterations_int = 0
                iterator.write("at epochs #%d: d_loss=%f, lipschitz~%f, estimated Earth Mover's Distance: %f" % (
                epochs, d_loss.detach().item(), grad_norm.max().detach().item(), np.mean(D_loss)))
                x1 = generator(batch_size=settings.batch_size, up_sampling=[3]).sigmoid()
                fake_score = critic(x1)
                g_loss = -fake_score.mean()
                g_opt.zero_grad()
                g_loss.backward()
                g_opt.step()
            if i % 100 == 0:
                torch.save(generator.state_dict(), "data/g_model")
    torch.save(generator.state_dict(), "data/g_model")


if __name__ == "__main__":
    main()