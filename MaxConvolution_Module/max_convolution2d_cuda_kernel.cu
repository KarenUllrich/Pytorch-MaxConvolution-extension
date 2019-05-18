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
#define WITHIN_BOUNDS(x, y, H, W) (x >= 0 && x < H && y >= 0 && y < W)

#define THREADS_FORWARD 32

namespace {
template <typename scalar_t>
__global__ void correlation_cuda_forward_kernel(
    const TensorAcc4R rInput1,
    const TensorAcc4R rInput2,
    TensorAcc5R output,
    int kH, int kW,
    int patchH, int patchW,
    int padH, int padW,
    int dilation_patchH, int dilation_patchW,
    int dH, int dW) {

  const int iH = rInput1.size(1);
  const int iW = rInput1.size(2);
  const int iC = rInput1.size(3);

  const int n = blockIdx.x;
  const int h = blockIdx.y;
  const int w = blockIdx.z;
  const int thread = threadIdx.x;

  const int start_i = -padH + h * dH;
  const int start_j = -padW + w * dW;

  const int patchRadH = dilation_patchH * (patchH - 1) / 2;
  const int patchRadW = dilation_patchW * (patchW - 1) / 2;

  __shared__ scalar_t prod_sum[THREADS_FORWARD];

  for(int ph = 0; ph < patchH; ++ph){
    int ph_dilated = ph * dilation_patchH - patchRadH;
    for(int pw = 0; pw < patchW; ++pw){
      int pw_dilated = pw * dilation_patchW - patchRadW;
      prod_sum[thread] = 0;
      for (int i=0; i<kH; ++i){
        int i1 = start_i + i;
        int i2 = i1 + ph_dilated;
        if WITHIN_BOUNDS(i1, i2, iH, iH){
          for (int j=0; j<kW; ++j){
            int j1 = start_j + j;
            int j2 = j1 + pw_dilated;
            if WITHIN_BOUNDS(j1, j2, iW, iW){
              for (int c=thread; c<C; c += THREADS_FORWARD){
                scalar_t v1 = rInput1[n][i1][j1][c];
                scalar_t v2 = rInput2[n][i2][j2][c];
                prod_sum[thread] += v1 * v2;
              }
            }
          }
        }
      }
      // accumulate
      __syncthreads();
      if (thread == 0) {
        scalar_t reduce_sum = 0;
        for (int index = 0; index < THREADS_FORWARD; ++index) {
          reduce_sum += prod_sum[index];
        }
        output[n][ph][pw][h][w] = reduce_sum;
      }
    }
  }
}


torch::Tensor correlation_cuda_forward(
    torch::Tensor input,
    torch::Tensor weight,
    int kH, int kW,
    int padH, int padW) {

  const int batch_size = input.size(0);
  const int iH = input.size(2);
  const int iW = input.size(3);

  const int oC = weight.size(0);
  const int kH = weight.size();
  const int kW = weight.size(2);

  const auto oH = (iH + 2 * padH) / kH;
  const auto oW = (iW + 2 * padW) / kW;
  auto output1 = torch::zeros({batch_size, oC, oH, oW}, input.options());
  auto output2 = torch::zeros({batch_size, oC, oH, oW, kH, kW}, input.options());

  auto trInput = input.permute({0, 2, 3, 1}).contiguous();
  auto trInput2 = input2.permute({0, 2, 3, 1}).contiguous();

  const int threads = THREADS_FORWARD;
  const dim3 blocks(batch_size, oH, oW);

  AT_DISPATCH_FLOATING_TYPES(input1.type(), "max_convolution2d_forward_cuda", ([&] {
    TensorAcc4R trInput1_acc  = trInput1.packed_accessor<scalar_t,4,RestrictPtrTraits,int32_t>();
    TensorAcc4R trInput2_acc = trInput2.packed_accessor<scalar_t,4,RestrictPtrTraits,int32_t>();
    TensorAcc5R output_acc = output.packed_accessor<scalar_t,5,RestrictPtrTraits,int32_t>();
    correlation_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
        trInput1_acc, trInput2_acc, output_acc,
        kH, kW, padH, padW);
  }));

  return output;
}
