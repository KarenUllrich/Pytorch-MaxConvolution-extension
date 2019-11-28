import math
import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

import max_convolution2d_sampler_backend as max_convolution2d


def max_conv2d(input,
               weight,
               padding=0):
    """Apply max-convolution to the input.

    Every parameter except input and weight can be either single int
    or a pair of int.

    Args:
        input : The first parameter.
        weight : The second parameter.
        padding : padding applied to input1 and input2 before applying
            the correlation sampling, will modify output height and width

    Returns:
        Tensor: Result of max-convolution

    """
    kernel_size = (weight.size(2),weight.size(3))
    max_convolution_func = MaxConvolutionFunction(kernel_size,
                                                  padding)
    return max_convolution_func(input, weight)


class MaxConvolutionFunction(Function):
    def __init__(self,
                 kernel_size,
                 padding,):
        super(MaxConvolutionFunction, self).__init__()
        self.kernel_size = _pair(kernel_size)
        self.padding = _pair(padding)

    def forward(self, input, weight):

        # I do not think, we need this line.
        # self.save_for_backward(input, weight)
        kH, kW = self.kernel_size
        padH, padW = self.padding
        assert math.log2(input.size(2) + 2 * padH) % kH == 0, "Kernel and Padding do not fit striding requirement."
        assert math.log2(input.size(3) + 2 * padW) % kW == 0, "Kernel and Padding do not fit striding requirement."
        output = max_convolution2d.forward(input, weight, padH, padW)

        return output

    @once_differentiable
    def backward(self, grad_output):
        assert False, ("The Max-Convolution is not differentiable.")


class MaxConv2d(nn.Module):
    def __init__(self, padding=0):
        super(MaxConv2d, self).__init__()
        self.padding = padding

    def forward(self, input, weight):
        return max_conv2d(input, weight, self.padding)
