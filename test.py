from __future__ import division
from __future__ import print_function

import unittest
import torch


import MaxConvolutionCuda as MaxConv2d

class MaxConv2dTest(unittest.TestCase):
    def setUp(self):
        self.iC = 5
        self.oC = 3
        self.n = 2
        self.H = 4
        self.W = 4
        self.kH = 2
        self.kW = 2
        self.input = torch.zeros((self.n, self.iC, self.H, self.W))
        self.weight = torch.ones((self.oC, self.iC, self.kW, self.kH))

    def testPhi(self):
        torch.cuda.empty_cache()
        max_index = 2
        self.input[:,max_index,:,:] = 1
        [mu, phi] = MaxConv2d.forward(self.input.cuda(), self.weight.cuda(), 0, 0)
        self.assertEquals(phi[0,0,0,0,0,0], max_index)

if __name__ == '__main__':
    unittest.main()