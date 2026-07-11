from typing import Mapping, Text, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# Gram Loss
class GramLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.feature_extractor = vgg16.features.eval()

        self.register_buffer("imagenet_mean", torch.Tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("imagenet_std", torch.Tensor([0.229, 0.224, 0.225])[None, :, None, None])

        for param in self.parameters():
            param.requires_grad = False

    @staticmethod
    def gram_matrix(features: torch.Tensor):
        """
        Compute Gram matrix for feature maps
        features: [N, C, H, W]
        returns: [N, C, C]
        """
        N, C, H, W = features.shape
        F_flat = features.view(N, C, H * W)  # [N, C, H*W]
        G = F_flat @ F_flat.transpose(1, 2)  # [N, C, C]
        return G / (C * H * W)  # normalize

    def forward(self, input: torch.Tensor, target: torch.Tensor,):
        """Computes the perceptual loss.

        Args:
            input: A tensor of shape (B, C, L), the input concatenated image. Normalized to [-1, 1].
            target: A tensor of shape (B, C, L), the target concatenated image. Normalized to [-1, 1].

        Returns:
            A scalar tensor, the perceptual loss.
        """
        # Always in eval mode.
        self.feature_extractor.eval()
        # [-1, 1] -> [0, 1]

        input_img = F.interpolate(input, size=(224, 224), mode="bilinear", align_corners=False)
        target_img = F.interpolate(target, size=(224, 224), mode="bilinear", align_corners=False)

        input_imgs = (input_img + 1.0) / 2.0
        target_imgs = (target_img + 1.0) / 2.0

        input_imgs = (input_imgs - self.imagenet_mean) / self.imagenet_std
        target_imgs = (target_imgs - self.imagenet_mean) / self.imagenet_std

        input_features = []
        target_features = []

        x_in = input_imgs
        x_tg = target_imgs
        for layer in self.feature_extractor:
            x_in = layer(x_in)
            x_tg = layer(x_tg)

            input_features.append(x_in)
            target_features.append(x_tg)

        total_loss = 0.0
        for f_in, f_tg in zip(input_features, target_features):
            G_in = self.gram_matrix(f_in)
            G_tg = self.gram_matrix(f_tg)
            total_loss += ((G_in - G_tg) ** 2).mean()

        return total_loss