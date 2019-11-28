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
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

// Use 1024 threads per block, which requires cuda sm_2x or above
constexpr int CUDA_NUM_THREADS = 1024;

// CUDA: number of blocks for threads.
inline int GET_BLOCKS(const int N)
{
  AT_ASSERTM(N > 0, "CUDA kernel launch blocks must be positive, but got N=", N);
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

namespace {
template <typename scalar_t>
__global__ void max_convolution2d_cuda_forward_kernel(
    const int total_threads,
    const TensorAcc4R rInput,
    const TensorAcc4R rWeight,
    TensorAcc4R output1,
    TensorAcc6R output2,
    int padH, int padW) {

  const int iC = rInput.size(1);
  const int iH = rInput.size(2);
  const int iW = rInput.size(3);

  const int oC = output2.size(1);
  const int oH = output2.size(2);
  const int oW = output2.size(3);
  const int kH = output2.size(4);
  const int kW = output2.size(5);

  CUDA_KERNEL_LOOP(index, total_threads) {
    const int oc = index % oC;
    const int h = (index / oC) % iH;
    const int w = index / (oC * iH) % iW;
    const int n = index / (oC * iH * iW);
    const int kh = h % kH;
    const int kw = w % kW;
    const int oh = h / kH;
    const int ow = w / kW;

    float max_p = - std::numeric_limits<scalar_t>::infinity(); // TODO REPLace this!!!
    for (int c=0; c<iC; ++c){
      const scalar_t p = rInput[n][c][h][w] + rWeight[oc][c][kh][kw];
      if (p > max_p){
        max_p = p;
        output2[n][oc][oh][ow][kh][kw] = c;
       }
     }

    // __syncthreads();
    atomicAdd(&output1[n][oc][oh][ow], max_p);
  }

    }
}


std::vector<torch::Tensor> max_convolution2d_cuda_forward(
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

  int total_threads = static_cast<int>(batch_size * oC * iW * iH);

  AT_DISPATCH_FLOATING_TYPES(input.type(), "max_convolution2d_cuda_forward", ([&] {
    TensorAcc4R rInput_acc  = rInput.packed_accessor<scalar_t,4,RestrictPtrTraits,int32_t>();
    TensorAcc4R rWeight_acc = rWeight.packed_accessor<scalar_t,4,RestrictPtrTraits,int32_t>();
    TensorAcc4R output1_acc = output1.packed_accessor<scalar_t,4,RestrictPtrTraits,int32_t>();
    TensorAcc6R output2_acc = output2.packed_accessor<scalar_t,6,RestrictPtrTraits,int32_t>();
    max_convolution2d_cuda_forward_kernel<scalar_t>
    <<<GET_BLOCKS(total_threads), CUDA_NUM_THREADS>>>(
        total_threads, rInput_acc, rWeight_acc, output1_acc, output2_acc, padH, padW);
  }));

  return {output1, output2};
}
// }