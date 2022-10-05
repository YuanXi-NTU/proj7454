import clip
import torch

import torch.nn as nn
from .utils import ori_classes, classes, simp_classes

class SegModel(nn.Module):
    pass

class Tmodel(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.seg=SegModel()
        self.clip,self.clip_pre=clip.load('ViT-B/32',args.device)

    def forward(self,x):
        with torch.no_grad():
            x=self.seg(x)
        x=tensor2PIL(x)

        return x
