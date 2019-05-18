from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import torch


from max_convolution2d import MaxConv2d

# test equivalence of cuda and c code

operator = MaxConv2d()

iC = 30
oC = 20
b = 12
H = 32
W = 32
kH = 2
kW = 2

fake_input = torch.rand((b,iC,H,W))
fake_weight = torch.rand((oC,iC,kW,kH))

c_result = operator(fake_input,fake_weight)
cuda_result = operator(fake_input.cuda(),fake_weight.cuda())

torch.all(torch.eq(c_result, cuda_result.cpu()))

# test if result is same as python fucntion

def python_fun(input, weights, padding=0.):

    kernel_size = (weights.size(2),weights.size(3))

    # input \in [batch_size x output_channels x width x height]
    full_message = F.unfold(input, kernel_size=kernel_size, stride=kernel_size,
                            padding=padding).transpose(2, 1)
    del input
    # input \in [batch_size x Z x output_channels * kernel_size^2]

    full_message = full_message.unsqueeze(2).expand(-1, -1, self.C, -1)

    full_message = full_message + F.log_softmax(weights, 1).view(weights.shape[0], -1)
    full_message = full_message \
        .view(self.batch_size, self.width, self.width, self.C, self.output_channels, self.kernel_size ** 2) \
        .transpose(3, 4)
    full_message, phi = full_message.max(3)
    full_message = full_message.sum(4)
    # input \in [batch_size x width x height x C]

    return full_message.permute(0, 3, 1, 2), phi