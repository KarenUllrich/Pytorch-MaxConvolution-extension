#include <limits>
#include <algorithm>
#include <utility>      // std::pair
#include <torch/extension.h>
#include <ATen/ATen.h>
using namespace at;
using namespace std;

#include <vector>

#define WITHIN_BOUNDS(x, H) (x >= 0 && x < H)
#define KEEP_MAX()

template <typename scalar_t>
static void convolve_patch(
    TensorAccessor<scalar_t,3> input,
    TensorAccessor<scalar_t,3> weight,
    scalar_t *output1,
    TensorAccessor<scalar_t,2> output2,
    int h, int w,
    int padH, int padW){
  // get params
  const int iC = input.size(0);
  const int iH = input.size(1);
  const int iW = input.size(2);
  const int kH = weight.size(1);
  const int kW = weight.size(2);

  scalar_t max_p;
  scalar_t p;
  torch::Tensor interim_max = at::zeros({kH,kW});
  torch::Tensor interim_argmax = at::zeros({kH,kW});
  scalar_t interim_sum;
  interim_sum = 0;

  for (int i=0; i<kH; ++i){
    int ii = h * kH + i - padH;
    if WITHIN_BOUNDS(ii, iH){
      for (int j=0; j<kW; ++j){
        int ij = w * kW + j -padW;
        if WITHIN_BOUNDS(ij, iW){
          max_p = - std::numeric_limits<float>::infinity();
          for (int c=0; c<iC; ++c){
            scalar_t inp = input[c][ii][ij];
            scalar_t wei = weight[c][i][j];
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
  output2 = interim_argmax.accessor<scalar_t, 2>();
  auto interim_max_acc = interim_max.accessor<scalar_t, 2>();
  for (int i=0; i<kH; ++i){
    for (int j=0; j<kW; ++j){
       interim_sum += interim_max_acc[i][j];
    }
  }
  *output1 =  interim_sum;
}


std::tuple<torch::Tensor, torch::Tensor> max_convolution2d_cpp_forward(
    torch::Tensor input,
    torch::Tensor weight,
    int padH, int padW) {

  const auto batch_size = input.size(0);
  const auto iH = input.size(2);
  const auto iW = input.size(3);

  const auto oC = weight.size(0);
  const auto kH = weight.size(2);
  const auto kW = weight.size(3);

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
            auto weight_acc = weight.accessor<scalar_t, 4>();
            auto output1_acc = output1.accessor<scalar_t, 4>();
            auto output2_acc = output2.accessor<scalar_t, 6>();
            for (h = 0; h < oH; ++h) {
              for (w = 0; w < oW; ++w) {
                convolve_patch(input_acc[n],
                               weight_acc[c],
                               &output1_acc[n][c][h][w],
                               output2_acc[n][c][h][w],
                               h, w,
                               padH, padW);
              }
            }
          }));
      }
    }
  return std::make_pair (output1, output2);
}
