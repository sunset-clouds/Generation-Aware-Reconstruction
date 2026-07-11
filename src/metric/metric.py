"""
Image quality metrics for reconstruction evaluation.

Same as VQ-Transplant/VAR/metric/metric.py
"""

import pyiqa
import torch
import torch.nn as nn
import piq


## PSNR - computed by pyiqa ()
## data range (0, 1)
class PSNR():
    def __init__(self, device=None):
        self.iqa_metric = pyiqa.create_metric('psnr', test_y_channel=True, color_space='ycbcr', device=device)
    
    def __call__(self, real, fake):
        return self.iqa_metric(real, fake)


## SSIM - computed by piq
## data range (0, 1)
class SSIM():
    def __call__(self, real, fake):
        return piq.ssim(real, fake, data_range=1., reduction='none')


## LPIPS - computed by pyiqa
## data range (0, 1) - note: different from original which expects (-1, 1)
class LPIPS():
    def __init__(self, device=None):
        self.iqa_metric = pyiqa.create_metric('lpips', device=device)
    
    def __call__(self, real, fake):
        return self.iqa_metric(real, fake)
