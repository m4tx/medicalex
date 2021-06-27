import numpy as np
from torch import nn
from torchinfo import summary

from model.decoder import Decoder
from model.resnet import ResNetV2
from model.vit import ViT


class TransUNet(nn.Module):
    def __init__(self, out_classes=2, img_size=(224, 224), hidden_dim=768, transformer_n_layers=12, mlp_dim=3072,
                 n_heads=12, dropout_rate=0.1, resnet_layers=(3, 4, 9), patch_size=16, up_channels=(256, 128, 64, 16),
                 skip_channels=(512, 256, 64, 16), n_skip_channels=3):
        super(TransUNet, self).__init__()
        self.out_classes = self.n_classes = out_classes

        grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.resnet = ResNetV2(resnet_layers, 1)
        in_channels = self.resnet.width * 16

        self.vit = ViT(hidden_dim, transformer_n_layers, mlp_dim, n_heads, dropout_rate, img_size, in_channels,
                       grid_size)

        self.decoder = Decoder(hidden_dim, out_classes, up_channels, skip_channels, n_skip_channels)

    def forward(self, x):
        x, skips = self.resnet(x)
        x = self.vit(x)
        x = self.decoder(x, skips)
        return x

    def load_from(self, weights):
        self.vit.load_from(weights)
        self.resnet.load_from(weights)


if __name__ == '__main__':
    unet = TransUNet(2, (224, 224))
    unet.load_from(weights=np.load('../project_TransUNet/model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz'))
    summary(unet, input_size=(1, 3, 224, 224), depth=8)
