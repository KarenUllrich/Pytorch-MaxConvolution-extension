#include <algorithm>
#include <torch/extension.h>
#include <ATen/ATen.h>
using namespace at;

#include <vector>

#define WITHIN_BOUNDS(x, y, H, W) (x >= 0 && x < W && y >= 0 && y < H)
#define ARGMAX(x) (std::distance(x.begin(), std::max_element(x.begin(), x.end())))

template <typename scalar_t>
static void convolve_patch(
    TensorAccessor<scalar_t,3> input,
    TensorAccessor<scalar_t,2> weight,
    scalar_t *output1,
    TensorAccessor<scalar_t,2> output2,
    int h, int w){
  const int iC = input.size(0);
  const int iH = input.size(1);
  const int iW = input.size(2);
  const int kH = weight.size(0);
  const int kW = weight.size(1);

  auto interim1 = at::zeros({iC}, input.options());
  auto interim2 = at::zeros({kH, kW}, input.options());

  for (int i=0; i<kH; ++i){
    int ii = h * kH + i;
    if WITHIN_BOUNDS(i1, i2, iH, iH){
      for (int j=0; j<kW; ++j){
        int ij = w * kW + j;
        if WITHIN_BOUNDS(j1, j2, iW, iW){
          for (int c=0; c<iC; ++c){
            scalar_t inp = input[c][ii][ij];
            scalar_t w = weight[i][j];
            interim1[c] = inp + w;
          }
         iterim2[i][j] = max_element(iterim1)
         output2[i][j] = ARGMAX(iterim1)
        }
      }
    }
  }
  auto interim_sum = at::zeros({1}, input.options());
  for (int i=0; i<kH; ++i){
    for (int j=0; j<kW; ++j){
       iterim_sum += iterim2[i][j]
    }
  }
  *output1 =  iterim_sum;
}


torch::Tensor max_convolution2d_cpp_forward(
    torch::Tensor input,
    torch::Tensor weight,
    int padH, int padW) {

  const auto batch_size = input.size(0);
  const auto iH = input.size(2);
  const auto iW = input.size(3);

  const auto oC = weight.size(0);
  const auto kH = weight.size(1);
  const auto kW = weight.size(2);

  const auto oH = (iH + 2 * padH) / kH;
  const auto oW = (iW + 2 * padW) / kW;
  auto output1 = at::zeros({batch_size, oC, oH, oW}, input.options());
  auto output2 = at::zeros({batch_size, oC, oH, oW, kH, kW}, input.options());

  int n, c, h, w;
  #pragma omp parallel for private(n, c, h, w) collapse(2)
    for (n = 0; n < batch_size; ++n) {
      for(c = 0; c < oC; ++c){
          AT_DISPATCH_FLOATING_TYPES(input.type(), "max_convolution2d_forward_cpp", ([&] {
            auto input_acc = input.accessor<scalar_t, 4>();
            auto weight_acc = weight.accessor<scalar_t, 3>();
            auto output1_acc = output1.accessor<scalar_t, 4>();
            auto output2_acc = output2.accessor<scalar_t, 6>();
            for (h = 0; h < oH; ++h) {
              for (w = 0; w < oW; ++w) {
                convolve_patch(input_acc[n],
                               weight_acc[c],
                               &output1_acc[n][c][h][w],
                               output2_acc[n][c][h][w],
                               h, w);
              }
            }
          }));
      }
    }
  return output1, output2;
}
