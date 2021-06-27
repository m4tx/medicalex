from os.path import join

import numpy as np
import torch
from numpy.lib import math
from scipy import ndimage
from torch import nn

from model.load_pretrained import np2th, ATTENTION_Q, ATTENTION_K, ATTENTION_V, ATTENTION_OUT, FC_0, FC_1, \
    ATTENTION_NORM, MLP_NORM


class Attention(nn.Module):
    def __init__(self, n_heads, hidden_dim, dropout_rate):
        super(Attention, self).__init__()
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // n_heads
        self.all_dim = n_heads * self.head_dim

        self.query = nn.Linear(hidden_dim, self.all_dim)
        self.key = nn.Linear(hidden_dim, self.all_dim)
        self.value = nn.Linear(hidden_dim, self.all_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)

        self.attention_dropout = nn.Dropout(dropout_rate)
        self.projection_dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=-1)

    def transpose(self, x):
        return x.view(x.size()[:-1] + (self.n_heads, self.head_dim)).permute(0, 2, 1, 3)

    def forward(self, x):
        q = self.transpose(self.query(x))
        k = self.transpose(self.key(x))
        v = self.transpose(self.value(x))

        x = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        x = self.softmax(x)
        x = self.attention_dropout(x)
        x = torch.matmul(x, v).permute(0, 2, 1, 3).contiguous()
        x = x.view(x.size()[:-2] + (self.all_dim,))
        x = self.out(x)
        x = self.projection_dropout(x)
        return x


class MLP(nn.Module):
    def __init__(self, hidden_dim, mlp_dim, dropout_rate):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, mlp_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(mlp_dim, hidden_dim)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    def __init__(self, img_size, in_channels, hidden_dim, grid_size, dropout_rate):
        super(Embeddings, self).__init__()

        patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
        patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
        n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])

        self.patch_embeddings = nn.Conv2d(in_channels=in_channels,
                                          out_channels=hidden_dim,
                                          kernel_size=patch_size,
                                          stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, hidden_dim))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.patch_embeddings(x)
        x = x.flatten(2).transpose(-1, -2)
        x = x + self.position_embeddings
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, mlp_dim, n_heads, dropout_rate):
        super(TransformerBlock, self).__init__()
        self.hidden_dim = hidden_dim

        self.attention = Attention(n_heads, hidden_dim, dropout_rate)
        self.mlp = MLP(hidden_dim, mlp_dim, dropout_rate)

        self.attention_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.mlp_norm = nn.LayerNorm(hidden_dim, eps=1e-6)

    def forward(self, x):
        x_a = self.attention_norm(x)
        x_a = self.attention(x_a)
        x = x + x_a
        x_m = self.mlp_norm(x)
        x_m = self.mlp(x_m)
        x = x + x_m
        return x

    def load_from(self, weights, n_block):
        """https://github.com/jeonsworld/ViT-pytorch/blob/main/models/modeling.py"""
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[join(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_dim, self.hidden_dim).t()
            key_weight = np2th(weights[join(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_dim, self.hidden_dim).t()
            value_weight = np2th(weights[join(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_dim, self.hidden_dim).t()
            out_weight = np2th(weights[join(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_dim, self.hidden_dim).t()

            query_bias = np2th(weights[join(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[join(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[join(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[join(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attention.query.weight.copy_(query_weight)
            self.attention.key.weight.copy_(key_weight)
            self.attention.value.weight.copy_(value_weight)
            self.attention.out.weight.copy_(out_weight)
            self.attention.query.bias.copy_(query_bias)
            self.attention.key.bias.copy_(key_bias)
            self.attention.value.bias.copy_(value_bias)
            self.attention.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[join(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[join(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[join(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[join(ROOT, FC_1, "bias")]).t()

            self.mlp.fc1.weight.copy_(mlp_weight_0)
            self.mlp.fc2.weight.copy_(mlp_weight_1)
            self.mlp.fc1.bias.copy_(mlp_bias_0)
            self.mlp.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[join(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[join(ROOT, ATTENTION_NORM, "bias")]))
            self.mlp_norm.weight.copy_(np2th(weights[join(ROOT, MLP_NORM, "scale")]))
            self.mlp_norm.bias.copy_(np2th(weights[join(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Module):
    def __init__(self, hidden_dim, transformer_n_layers, mlp_dim, n_heads, dropout_rate):
        super(Encoder, self).__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, mlp_dim, n_heads, dropout_rate) for _ in range(transformer_n_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x


class ViT(nn.Module):
    def __init__(self, hidden_dim, transformer_n_layers, mlp_dim, n_heads, dropout_rate, img_size, in_channels,
                 grid_size):
        super(ViT, self).__init__()
        self.embeddings = Embeddings(img_size, in_channels, hidden_dim, grid_size, dropout_rate)
        self.encoder = Encoder(hidden_dim, transformer_n_layers, mlp_dim, n_heads, dropout_rate)

    def forward(self, x):
        x = self.embeddings(x)
        x = self.encoder(x)
        return x

    def load_from(self, weights):
        """https://github.com/jeonsworld/ViT-pytorch/blob/main/models/modeling.py"""
        with torch.no_grad():
            self.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.encoder.norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.encoder.norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))
            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.embeddings.position_embeddings.copy_(posemb)
            elif posemb.size()[1] - 1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.embeddings.position_embeddings.copy_(posemb)
            else:
                ntok_new = posemb_new.size(1)
                _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid
                self.embeddings.position_embeddings.copy_(np2th(posemb))
            for bname, block in self.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)
