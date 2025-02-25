
import torch
import torch.nn as nn
import math
import numpy as np
import random


class Necker(nn.Module):

    def __init__(self,
                  clip_model
                 ):
        super(Necker, self).__init__()
        self.clip_model=clip_model
        target = 32

        for i,size in enumerate(self.clip_model.token_size):
            self.add_module("{}_upsample".format(i),
                                nn.UpsamplingBilinear2d(scale_factor=target/size))


    @torch.no_grad()
    def forward(self, tokens):
        align_features=[]
        for i,token in enumerate(tokens):
            if len(token.shape) == 3:
                B, N, C=token.shape
                token = token[:, 1:, :]
                token=token.view((B,int(math.sqrt(N-1)),int(math.sqrt(N-1)),C)).permute(0, 3, 1, 2)

            align_features.append(token)
        return align_features


