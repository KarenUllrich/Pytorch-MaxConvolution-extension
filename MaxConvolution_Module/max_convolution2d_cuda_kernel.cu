#include <torch/extension.h>
#include <ATen/ATen.h>
using namespace at;

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <iostream>

// Cuda tensor accessor definitions
// restrict pointer traits piroritize speed over memory consumption
#define TensorAcc3R PackedTensorAccessor<scalar_t,3,RestrictPtrTraits,int32_t>
#define TensorAcc4R PackedTensorAccessor<scalar_t,4,RestrictPtrTraits,int32_t>
#define TensorAcc5R PackedTensorAccessor<scalar_t,5,RestrictPtrTraits,int32_t>
#define TensorAcc6R PackedTensorAccessor<scalar_t,6,RestrictPtrTraits,int32_t>
#define WITHIN_BOUNDS(x, H) (x >= 0 && x < H)

#define THREADS_FORWARD 32 //should be multiple of 32

namespace {
template <typename scalar_t>
__global__ void max_convolution2d_cuda_forward_kernel(
    const TensorAcc4R rInput,
    const TensorAcc4R rWeight,
    TensorAcc4R output1,
    TensorAcc6R output2,
    int padH, int padW, int oW) {

  const int iC = rInput.size(1);
  const int iH = rInput.size(2);
  const int iW = rInput.size(3);
  const int kH = rWeight.size(2);
  const int kW = rWeight.size(3);

  // independent, large dimensions to be paralllized: oC, batch_size, oH, oW
  const int n = blockIdx.x;
  const int oc = blockIdx.y;
  const int h = blockIdx.z;
  const int thread = threadIdx.x;

  for (int w=thread; w<oW; w += THREADS_FORWARD){
    scalar_t max_p;
    scalar_t p;
    torch::Tensor interim_max = torch::zeros({kH,kW});
    torch::Tensor interim_argmax = torch::zeros({kH,kW});
    scalar_t interim_sum;
    interim_sum = 0;
    for (int i=0; i<kH; ++i){
      int ii = h * kH + i - padH;
      if WITHIN_BOUNDS(ii, iH){
        for (int j=0; j<kW; ++j){
          int ij = w * kW + j -padW;
          if WITHIN_BOUNDS(ij, iW){
            max_p = - std::numeric_limits<float>::infinity(); // TODO REPLace this!!!
            for (int c=0; c<iC; ++c){
              scalar_t inp = rInput[n][c][ii][ij];
              scalar_t wei = rWeight[oc][c][i][j];
              p = inp + wei;
              if (p > max_p){
                max_p = p;
                interim_max[i][j] = p;
                interim_argmax[i][j] = c;
              }
            }
          }
        }
      }
    }
    output2[n][oc][h][w] = interim_argmax.packed_accessor<scalar_t,2,RestrictPtrTraits,int32_t>();
    auto interim_max_acc = interim_max.packed_accessor<scalar_t,2,RestrictPtrTraits,int32_t>();
    for (int i=0; i<kH; ++i){
      for (int j=0; j<kW; ++j){
         interim_sum += interim_max_acc[i][j];
      }
    }
    output1[n][oc][h][w] =  interim_sum;
  }
  // accumulate
  __syncthreads();
}


std::tuple<torch::Tensor, torch::Tensor> max_convolution2d_cuda_forward(
    torch::Tensor input,
    torch::Tensor weight,
    int padH, int padW) {

  const int batch_size = input.size(0);
  const int iH = input.size(2);
  const int iW = input.size(3);

  const int oC = weight.size(0);
  const int kH = weight.size(2);
  const int kW = weight.size(3);

  const int oH = (iH + 2 * padH) / kH;
  const int oW = (iW + 2 * padW) / kW;
  auto output1 = torch::zeros({batch_size, oC, oH, oW}, input.options());
  auto output2 = torch::zeros({batch_size, oC, oH, oW, kH, kW}, input.options());

  auto rInput = input.contiguous();
  auto rWeight = weight.contiguous();

  const int threads = THREADS_FORWARD;
  const dim3 blocks(batch_size, oC, oH);

  AT_DISPATCH_FLOATING_TYPES(input.type(), "max_convolution2d_cuda_forward", ([&] {
    TensorAcc4R rInput_acc  = rInput.packed_accessor<scalar_t,4,RestrictPtrTraits,int32_t>();
    TensorAcc4R rWeight_acc = rWeight.packed_accessor<scalar_t,4,RestrictPtrTraits,int32_t>();
    TensorAcc4R output1_acc = output1.packed_accessor<scalar_t,4,RestrictPtrTraits,int32_t>();
    TensorAcc6R output2_acc = output2.packed_accessor<scalar_t,6,RestrictPtrTraits,int32_t>();
    max_convolution2d_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
        rInput_acc, rWeight_acc, output1_acc, output2_acc, padH, padW, oW);
  }));

  return std::make_pair (output1, output2);
}
