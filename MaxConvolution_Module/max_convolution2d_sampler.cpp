#include <torch/extension.h>
#include <ATen/ATen.h>

#include <vector>
#include <iostream>

// declarations

std::pair<torch::Tensor, torch::Tensor> max_convolution2d_cuda_forward(
    torch::Tensor input,
    torch::Tensor weight,
    int kH, int kW,
    int padH, int padW);

std::pair<torch::Tensor, torch::Tensor> max_convolution2d_cpp_forward(
    torch::Tensor input,
    torch::Tensor weight,
    int kH, int kW,
    int padH, int padW);


// C++ interface

#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x, " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_CHECK(x.is_contiguous(), #x, " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::pair<torch::Tensor, torch::Tensor> max_convolution2d_sample_forward(
    torch::Tensor input,
    torch::Tensor weight,
    int kH, int kW,
    int padH, int padW) {
  if (input.type().is_cuda()){
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    
    return max_convolution2d_cuda_forward(input, weight,
                             kH, kW,
                             padH, padW);
  }else{
    return max_convolution2d_cpp_forward(input, weight,
                             kH, kW,
                             padH, padW);
  }
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &max_convolution2d_sample_forward, "Max-Convolution 2d Forward");
  }
